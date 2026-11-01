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
3. Put one or more `.gguf` files in `ComfyUI/models/llm-gguf/`, then restart ComfyUI.

The node appears as **Local LLM Prompt** in the `llm` category.

## Inputs

| Input | Description |
| --- | --- |
| model | `.gguf` file from `models/llm-gguf` |
| prompt | text sent to the model |
| system_prompt | defaults to the Promptmaster system prompt |
| temperature, seed | sampling settings |
| max_tokens | Ollama `num_predict`; `-1` = unlimited |
| unload_after | free VRAM after each run so the diffusion model can use it (default on, like `keep_alive: 0`) |
| n_ctx | context length, Ollama `num_ctx` (default 4096) |
| stop | extra stop strings, one per line |
| history | optional; the `history` output of a previous run, so the model remembers earlier turns (see below) |
| output_format | `text` or `json` (Ollama `format: json`) |
| top_k, top_p, min_p, typical_p, tfs_z | sampling filters |
| mirostat, mirostat_tau, mirostat_eta | Mirostat sampling (`0` = off) |
| repeat_penalty, repeat_last_n, presence_penalty, frequency_penalty | repetition control |
| gpu_layers | layers offloaded to GPU (Ollama `num_gpu`), `-1` = all |
| main_gpu, num_thread, num_batch, f16_kv, use_mmap, use_mlock, numa | hardware / memory options |
| preload_to_ram | read the `.gguf` file once before loading so the OS file cache holds it in RAM and the load into VRAM is fast. Not pinned: Windows may evict it under memory pressure. Needs free RAM about the size of the model |

Defaults follow Ollama's. The optional inputs are passed through to `llama-cpp-python`; any option
the installed build does not support is silently ignored.

Ollama options with no `llama-cpp-python` equivalent are not offered: `num_keep`,
`penalize_newline` and `low_vram`.

Outputs: `text` (STRING), e.g. wired into a CLIP Text Encode node, and `history` (STRING).

## Keeping context (storytelling)

`history` is the conversation so far as JSON. Wire the `history` output of one Local LLM Prompt node
into the `history` input of the next, and give each node the next instruction as its `prompt`. Each
node formats the earlier turns with the model's chat template, so they appear as real user and
assistant messages. Leave `history` unconnected to start a fresh story.

The history is stored in the workflow, not in the node, so it survives `unload_after` and restarts.
When it no longer fits in `n_ctx` (minus room for `max_tokens`), the oldest turns are dropped
automatically, so raise `n_ctx` for long stories. Each run re-processes the whole history, so runs
get slower as the story grows.

## Notes

- Tested with [Electrofried/Promptmaster-Mistral-7B-v1-GGUF](https://huggingface.co/Electrofried/Promptmaster-Mistral-7B-v1-GGUF) (Q8_0).
- The chat template is always taken from the model (`tokenizer.chat_template` in the `.gguf`).
  A model without an embedded template falls back to `llama-cpp-python`'s generic format (a warning
  is printed to the console), which may give poor output. Most instruct/chat GGUFs have a template.
- `llama-cpp-python` 0.2.68 does not support newer architectures (e.g. Gemma 3, Qwen3). Use a newer
  build for those.
