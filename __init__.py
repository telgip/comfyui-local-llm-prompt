import gc
import os

import torch  # noqa: F401  (puts torch's CUDA DLLs on the path for llama_cpp)
import folder_paths

LLM_DIR = os.path.join(folder_paths.models_dir, "LLM")
os.makedirs(LLM_DIR, exist_ok=True)
NONE = "(no .gguf files in models/LLM)"
DEFAULT_SYSTEM = (
    "You are a prompt generation AI. your task is to take a user input for a stable difusion "
    "prompt and output and expand the supplied prompt in a stable difusion format to provide "
    "better output. Do not deviate from the format. Do not output anything other than a "
    "stable diffusion prompt."
)

_cache = {"key": None, "llm": None}


def _local_models():
    found = sorted(f for f in os.listdir(LLM_DIR) if f.lower().endswith(".gguf"))
    return found or [NONE]


def _unload():
    _cache.update(key=None, llm=None)  # Llama.__del__ frees the VRAM once unreferenced
    gc.collect()


def _load(model, n_ctx, gpu_layers):
    key = (model, n_ctx, gpu_layers)
    if _cache["key"] == key:
        return _cache["llm"]
    _unload()
    if model == NONE:
        raise RuntimeError(f"Put a .gguf file in {LLM_DIR}")
    from llama_cpp import Llama
    llm = Llama(model_path=os.path.join(LLM_DIR, model), n_ctx=n_ctx,
                n_gpu_layers=gpu_layers, chat_format="chatml", verbose=False)
    _cache.update(key=key, llm=llm)
    return llm


class LocalLLMPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_local_models(),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "max_tokens": ("INT", {"default": 256, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "unload_after": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 512}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "llm"

    def run(self, model, prompt, system_prompt, temperature, max_tokens, seed,
            unload_after, n_ctx=4096, gpu_layers=-1):
        llm = _load(model, n_ctx, gpu_layers)
        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            out = llm.create_chat_completion(
                messages=messages, temperature=temperature, max_tokens=max_tokens,
                seed=seed, stop=["<|im_end|>", "<|im_start|>"],
            )
            text = out["choices"][0]["message"]["content"]
        finally:
            if unload_after:
                _unload()  # frees VRAM for the diffusion model
        return (text.strip(),)

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed


NODE_CLASS_MAPPINGS = {"LocalLLMPrompt": LocalLLMPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"LocalLLMPrompt": "Local LLM Prompt"}
