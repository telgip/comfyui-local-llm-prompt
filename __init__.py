import gc
import inspect
import json
import os

import torch  # noqa: F401  (puts torch's CUDA DLLs on the path for llama_cpp)
import folder_paths

LLM_DIR = os.path.join(folder_paths.models_dir, "llm-gguf")
os.makedirs(LLM_DIR, exist_ok=True)
NONE = "(no .gguf files in models/llm-gguf)"
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


def _supported(fn, kwargs):
    """Drop kwargs the installed llama-cpp-python build does not know about."""
    params = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _warm_file_cache(path, chunk=64 * 1024 * 1024):
    """Read the file once so the OS keeps it in RAM and the load below copies from memory."""
    buf = bytearray(chunk)
    with open(path, "rb", buffering=0) as f:
        while f.readinto(buf):
            pass


def _load(model, preload_to_ram=False, **ctor):
    key = (model, tuple(sorted(ctor.items())))
    if _cache["key"] == key:
        return _cache["llm"]
    _unload()
    if model == NONE:
        raise RuntimeError(f"Put a .gguf file in {LLM_DIR}")
    from llama_cpp import Llama
    path = os.path.join(LLM_DIR, model)
    if preload_to_ram:
        _warm_file_cache(path)
    kwargs = _supported(Llama.__init__, dict(
        model_path=path, verbose=False, **ctor))
    llm = Llama(**kwargs)
    if not llm.metadata.get("tokenizer.chat_template"):
        print(f"[LocalLLMPrompt] {model} has no embedded chat template; using the generic fallback format.")
    _cache.update(key=key, llm=llm)
    return llm


def _parse_history(raw):
    """History is a JSON list of {"role": "user"|"assistant", "content": str}."""
    if not raw or not raw.strip():
        return []
    try:
        history = json.loads(raw)
        ok = isinstance(history, list) and all(
            isinstance(m, dict) and m.get("role") in ("user", "assistant")
            and isinstance(m.get("content"), str) for m in history)
    except ValueError:
        ok = False
    if not ok:
        raise ValueError("history must be the JSON list produced by this node's history output")
    return history


def _trim_history(llm, history, fixed, budget):
    """Drop the oldest turns until the history fits in `budget` tokens."""
    def cost(msgs):  # +16 per message approximates the chat-template overhead
        return sum(len(llm.tokenize(m["content"].encode("utf-8"), add_bos=False)) + 16 for m in msgs)

    budget -= cost(fixed)
    history = list(history)
    while history and (cost(history) > budget or history[0]["role"] != "user"):
        history.pop(0)
    return history


def _f(default, lo, hi, step=0.05):
    return ("FLOAT", {"default": default, "min": lo, "max": hi, "step": step})


def _i(default, lo, hi, step=1):
    return ("INT", {"default": default, "min": lo, "max": hi, "step": step})


class LocalLLMPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (_local_models(),),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {"multiline": True, "default": DEFAULT_SYSTEM}),
                "temperature": _f(0.8, 0.0, 2.0),
                "max_tokens": _i(256, -1, 8192),  # num_predict, -1 = unlimited
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "unload_after": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # context
                "n_ctx": _i(4096, 512, 32768, 512),
                "stop": ("STRING", {"multiline": True, "default": ""}),  # one per line
                "output_format": (["text", "json"],),
                "history": ("STRING", {"forceInput": True}),
                # sampling
                "top_k": _i(40, 0, 1000),
                "top_p": _f(0.9, 0.0, 1.0),
                "min_p": _f(0.0, 0.0, 1.0),
                "typical_p": _f(1.0, 0.0, 1.0),
                "tfs_z": _f(1.0, 0.0, 1.0),
                "mirostat": _i(0, 0, 2),
                "mirostat_tau": _f(5.0, 0.0, 20.0, 0.1),
                "mirostat_eta": _f(0.1, 0.0, 1.0, 0.01),
                # repetition
                "repeat_penalty": _f(1.1, 0.0, 2.0),
                "repeat_last_n": _i(64, 0, 32768),
                "presence_penalty": _f(0.0, -2.0, 2.0),
                "frequency_penalty": _f(0.0, -2.0, 2.0),
                # hardware
                "gpu_layers": _i(-1, -1, 200),
                "main_gpu": _i(0, 0, 16),
                "num_thread": _i(0, 0, 256),  # 0 = auto
                "num_batch": _i(512, 1, 8192),
                "f16_kv": ("BOOLEAN", {"default": True}),
                "use_mmap": ("BOOLEAN", {"default": True}),
                "use_mlock": ("BOOLEAN", {"default": False}),
                "numa": ("BOOLEAN", {"default": False}),
                "preload_to_ram": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "history")
    FUNCTION = "run"
    CATEGORY = "llm"

    def run(self, model, prompt, system_prompt, temperature, max_tokens, seed, unload_after,
            n_ctx=4096, stop="", output_format="text", history="",
            top_k=40, top_p=0.9, min_p=0.0, typical_p=1.0, tfs_z=1.0,
            mirostat=0, mirostat_tau=5.0, mirostat_eta=0.1,
            repeat_penalty=1.1, repeat_last_n=64, presence_penalty=0.0, frequency_penalty=0.0,
            gpu_layers=-1, main_gpu=0, num_thread=0, num_batch=512,
            f16_kv=True, use_mmap=True, use_mlock=False, numa=False, preload_to_ram=False):
        llm = _load(model, preload_to_ram, n_ctx=n_ctx, n_gpu_layers=gpu_layers, main_gpu=main_gpu,
                    n_threads=num_thread or None, n_batch=num_batch, f16_kv=f16_kv,
                    use_mmap=use_mmap, use_mlock=use_mlock, numa=numa,
                    last_n_tokens_size=repeat_last_n)
        system = [{"role": "system", "content": system_prompt}] if system_prompt.strip() else []
        user = {"role": "user", "content": prompt}
        reserve = n_ctx // 4 if max_tokens < 0 else max_tokens
        past = _trim_history(llm, _parse_history(history), system + [user], n_ctx - reserve)
        messages = system + past + [user]
        stops = [x for x in stop.splitlines() if x]
        kwargs = dict(
            messages=messages, temperature=temperature, seed=seed, stop=stops or None,
            max_tokens=None if max_tokens < 0 else max_tokens,
            top_k=top_k, top_p=top_p, min_p=min_p, typical_p=typical_p, tfs_z=tfs_z,
            mirostat_mode=mirostat, mirostat_tau=mirostat_tau, mirostat_eta=mirostat_eta,
            repeat_penalty=repeat_penalty, presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
        )
        if output_format == "json":
            kwargs["response_format"] = {"type": "json_object"}
        try:
            out = llm.create_chat_completion(**_supported(llm.create_chat_completion, kwargs))
            text = out["choices"][0]["message"]["content"]
        finally:
            if unload_after:
                _unload()  # frees VRAM for the diffusion model
        text = text.strip()
        new_history = past + [user, {"role": "assistant", "content": text}]
        return (text, json.dumps(new_history, ensure_ascii=False))

    @classmethod
    def IS_CHANGED(cls, seed, **kwargs):
        return seed


NODE_CLASS_MAPPINGS = {"LocalLLMPrompt": LocalLLMPrompt}
NODE_DISPLAY_NAME_MAPPINGS = {"LocalLLMPrompt": "Local LLM Prompt"}
