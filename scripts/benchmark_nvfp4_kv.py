#!/usr/bin/env python3
"""NVFP4 KV cache benchmark: FP8 vs NVFP4 comparison on SM121.

Runs:
  1. Sanity suite (5 prompts: math, code, reasoning, factual, instruction-following)
  2. Concurrency ramp (c1/c4/c8/c16/c32) with power telemetry
  3. KV cache / runtime facts from container logs

Usage:
  python3 benchmark_nvfp4_kv.py --base-url http://127.0.0.1:18080 --model Qwen3.6-27B-NVFP4
"""
import argparse, concurrent.futures as cf, json, time, urllib.request, subprocess, threading, statistics, datetime, re, sys
from pathlib import Path

SANITY_PROMPTS = [
    ("math",       "What is 17 * 23? Reply with only the number.", 32),
    ("code",       "Write a Python one-liner to reverse a string. Code only.", 64),
    ("reasoning",  "If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops definitely Lazzies? Answer yes or no with one sentence of reasoning.", 64),
    ("factual",    "What is the capital of Australia? One word answer.", 32),
    ("instruction","List exactly three colors of the rainbow, comma-separated, nothing else.", 32),
]

BENCH_PROMPT = 'Write a compact Python function that returns the nth Fibonacci number. Code only.'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://127.0.0.1:18080')
    ap.add_argument('--model', required=True)
    ap.add_argument('--output', default='benchmark_nvfp4_kv.json')
    ap.add_argument('--concurrency', nargs='*', type=int, default=[1, 4, 8, 16, 32])
    args = ap.parse_args()

    BASE = args.base_url
    MODEL = args.model

    def get(path):
        return urllib.request.urlopen(BASE + path, timeout=10).read().decode()

    def metrics():
        text = get('/metrics')
        out = {}
        for name in ['vllm:prompt_tokens_total', 'vllm:generation_tokens_total',
                      'vllm:spec_decode_num_draft_tokens_total',
                      'vllm:spec_decode_num_accepted_tokens_total']:
            m = re.search(re.escape(name) + r'\{[^}]*model_name="' + re.escape(MODEL) + r'"[^}]*\}\s+([0-9.eE+-]+)', text)
            out[name] = float(m.group(1)) if m else None
        return out

    def chat(prompt, max_tokens=256):
        payload = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                   'temperature': 0, 'max_tokens': max_tokens, 'stream': False}
        req = urllib.request.Request(BASE + '/v1/chat/completions',
              data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        t = time.time()
        with urllib.request.urlopen(req, timeout=300) as r:
            obj = json.loads(r.read().decode())
        dt = time.time() - t
        content = obj['choices'][0]['message'].get('content') or obj['choices'][0]['message'].get('reasoning_content') or ''
        return {'seconds': dt, 'completion_tokens': obj['usage']['completion_tokens'],
                'prompt_tokens': obj['usage']['prompt_tokens'],
                'response_text': content[:200]}

    def sample_power(stop, samples, interval=1.0):
        while not stop.is_set():
            t = time.time()
            try:
                s = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=power.draw,temperature.gpu,utilization.gpu',
                     '--format=csv,noheader,nounits'], text=True, timeout=5).strip().splitlines()[0]
                parts = [p.strip() for p in s.split(',')]
                samples.append({'t': t, 'power_w': float(parts[0]),
                                'temp_c': float(parts[1]), 'util_pct': float(parts[2])})
            except:
                pass
            stop.wait(interval)

    def run_row(c, prompt, max_tokens=256):
        samples = []; stop = threading.Event()
        th = threading.Thread(target=sample_power, args=(stop, samples), daemon=True)
        th.start(); t = time.time(); rows = []; errors = []
        with cf.ThreadPoolExecutor(max_workers=c) as ex:
            futs = [ex.submit(chat, prompt, max_tokens) for _ in range(c)]
            for fut in cf.as_completed(futs):
                try:
                    rows.append(fut.result())
                except Exception as e:
                    errors.append(repr(e))
        wall = time.time() - t; stop.set(); th.join(timeout=2)
        vals = [x['power_w'] for x in samples if 'power_w' in x]
        temps = [x['temp_c'] for x in samples if 'temp_c' in x]
        utils = [x['util_pct'] for x in samples if 'util_pct' in x]
        toks = sum(x['completion_tokens'] for x in rows)
        tok_s = toks / wall if wall else 0
        mean = statistics.mean(vals) if vals else 0
        return {'concurrency': c, 'requests_ok': len(rows), 'requests_error': len(errors),
                'seconds': wall, 'output_tokens': toks, 'output_tok_s': tok_s,
                'power_watts_mean': mean, 'power_watts_peak': max(vals) if vals else 0,
                'power_watts_min': min(vals) if vals else 0,
                'temp_c_mean': statistics.mean(temps) if temps else 0,
                'temp_c_peak': max(temps) if temps else 0,
                'util_pct_mean': statistics.mean(utils) if utils else 0,
                'tok_s_per_watt': tok_s / mean if mean else 0,
                'joules_per_1k_output_tokens': mean * wall / toks * 1000 if toks else 0,
                'errors': errors}

    # ─── Sanity Suite ───
    print("=" * 60)
    print("SANITY SUITE")
    print("=" * 60)
    sanity = []
    for name, prompt, max_tok in SANITY_PROMPTS:
        r = chat(prompt, max_tok)
        print(f"  [{name}] {r['completion_tokens']} tok in {r['seconds']:.2f}s → {r['response_text'][:80]}", flush=True)
        sanity.append({'name': name, **r})

    # ─── Concurrency Ramp ───
    print("\n" + "=" * 60)
    print("CONCURRENCY RAMP (prompt: 256 tok)")
    print("=" * 60)
    ramp = []
    for c in args.concurrency:
        r = run_row(c, BENCH_PROMPT, 256)
        print(f"  c{c:>2d}: {r['output_tok_s']:>8.2f} tok/s | {r['power_watts_mean']:>5.1f} W | "
              f"{r['temp_c_mean']:>4.1f}°C | {r['joules_per_1k_output_tokens']:>6.1f} J/1K tok | "
              f"{r['requests_ok']}/{c} ok", flush=True)
        ramp.append(r)

    # ─── KV/cache info from logs ───
    # Try multiple container names
    logs = ""
    for cname in ['sm121-vllm', 'sm121-vllm-kvexp']:
        try:
            r = subprocess.run(['docker', 'logs', '--tail', '2000', cname],
                              capture_output=True, text=True, timeout=30)
            logs += r.stderr + r.stdout
        except:
            pass

    facts = {}
    for k, p in {'kv_tokens': r'GPU KV cache size: ([0-9,]+) tokens',
                 'max_conc': r'Maximum concurrency for ([0-9,]+) tokens per request: ([0-9.]+)x',
                 'kv_mem': r'Available KV cache memory: ([0-9.]+) GiB',
                 'kv_dtype': r'kv_cache_dtype=(\S+)',
                 'cache_py_dtype': r'Using (\S+) data type to store kv cache',
                 'attn_backend': r'attention_backend.*?(\S+)',
                 'vllm_ver': r'vllm-([0-9.]+)'}.items():
        m = re.search(p, logs)
        if m:
            facts[k] = m.group(1) if k != 'max_conc' else f'{m.group(1)} @ {m.group(2)}x'

    # Metrics snapshot
    try:
        mtr = metrics()
    except:
        mtr = {}

    # GPU info
    try:
        gpu_info = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,driver_version',
             '--format=csv,noheader'], text=True, timeout=5).strip()
    except:
        gpu_info = "n/a"

    models = json.loads(get('/v1/models'))
    result = {
        'utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'base_url': BASE,
        'model': MODEL,
        'models_response': models,
        'gpu_info': gpu_info,
        'sanity_suite': sanity,
        'ramp': ramp,
        'kv_facts': facts,
        'metrics': mtr,
    }
    out = Path(args.output)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nResult saved to: {out}")
    print("\nKV Facts:")
    for k, v in facts.items():
        print(f"  {k}: {v}")

if __name__ == '__main__':
    main()
