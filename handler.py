import os
import runpod
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ["MODEL_ID"]
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

_tokenizer = None
_model = None

def _load_once():
    global _tokenizer, _model
    if _model is not None:
        return

    os.makedirs(os.environ.get("HF_HOME", "/runpod-volume/hf"), exist_ok=True)

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto"
    )
    _model.eval()

def handler(job):
    _load_once()

    inp = job.get("input", {})
    prompt = inp.get("prompt", "")
    max_new_tokens = int(inp.get("max_new_tokens", 256))

    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)

    with torch.no_grad():
        output = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=float(inp.get("temperature", 0.7)),
            top_p=float(inp.get("top_p", 0.9)),
            pad_token_id=_tokenizer.eos_token_id
        )

    text = _tokenizer.decode(output[0], skip_special_tokens=True)

    return {"text": text}

runpod.serverless.start({"handler": handler})
