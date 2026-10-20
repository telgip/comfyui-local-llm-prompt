# ComfyUI Local LLM Prompt

A ComfyUI node that runs a local GGUF language model in-process with `llama-cpp-python`.
It needs no Ollama, LM Studio or other server. It is meant for expanding short ideas into full
Stable Diffusion prompts, and it works with any ChatML-format GGUF chat model.

## Install

1. Clone into `ComfyUI/custom_nodes/`:
   ```
   git clone https://github.com/telgip/comfyui-local-llm-prompt
   ```
2. Install `llama-cpp-python` into ComfyUI's Python environment. For NVIDIA GPUs use a CUDA wheel,
   for example (Windows, Python 3.12):
   ```
   pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.2.68-cu124/llama_cpp_python-0.2.68-cp312-cp312-win_amd64.whl diskcache
   ```
   Other builds: <https://github.com/abetlen/llama-cpp-python>. Pre-built CUDA wheels are listed at
   <https://abetlen.github.io/llama-cpp-python/whl/cu124/llama-cpp-python/>.
3. Put one or more `.gguf` files in `ComfyUI/models/LLM/`, then restart ComfyUI.

The node appears as **Local LLM Prompt** in the `llm` category.

## Inputs

| Input | Description |
| --- | --- |
| model | `.gguf` file from `models/LLM` |
| prompt | text sent to the model |
| system_prompt | defaults to the Promptmaster system prompt |
| temperature, max_tokens, seed | sampling settings |
| unload_after | free VRAM after each run so the diffusion model can use it (default on) |
| n_ctx | context length (default 4096) |
| gpu_layers | layers offloaded to GPU, `-1` = all |

Output: `text` (STRING), e.g. wired into a CLIP Text Encode node.

## Notes

- Tested with [Electrofried/Promptmaster-Mistral-7B-v1-GGUF](https://huggingface.co/Electrofried/Promptmaster-Mistral-7B-v1-GGUF) (Q8_0).
- The chat format is fixed to ChatML. Models that use another template need a code change.
- `llama-cpp-python` 0.2.68 does not support newer architectures (e.g. Gemma 3, Qwen3). Use a newer
  build for those.
