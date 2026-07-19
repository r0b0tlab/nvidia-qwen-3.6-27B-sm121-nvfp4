# sparkrun

This repository is a sparkrun v2 recipe registry for the verified single-node GB10/SM121 image.

```bash
sparkrun registry add https://github.com/r0b0tlab/nvidia-qwen-3.6-27B-sm121-nvfp4
sparkrun recipe validate @r0b0tlab/qwen3.6-27b-nvfp4-vllm-r0b0tlab
sparkrun run @r0b0tlab/qwen3.6-27b-nvfp4-vllm-r0b0tlab --solo
```

The recipe pins the Hugging Face model revision and the versioned GHCR image. It uses FP8 KV as the production default. NVFP4 KV remains an explicit experimental profile until the matched semantic and quality gates pass.

Before publishing a recipe change:

```bash
sparkrun recipe validate sparkrun/recipes/qwen3.6-27b-nvfp4-vllm-r0b0tlab.yaml
python3 tests/test_launch_contract.py
```
