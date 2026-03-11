import os
import traceback

import torch
import runpod
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_ID = os.environ.get("MODEL_ID")
HF_TOKEN = os.environ.get("HF_TOKEN")
DEFAULT_MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "512"))
DEFAULT_TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
DEFAULT_TOP_P = float(os.environ.get("TOP_P", "0.9"))
DEFAULT_DO_SAMPLE = os.environ.get("DO_SAMPLE", "true").lower() == "true"

if not MODEL_ID:
    raise RuntimeError("MODEL_ID is required for serverless worker")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available; GPU is required")

DEVICE = "cuda"
DTYPE = torch.float16

print(f"[boot] CUDA available: {torch.cuda.is_available()}")
print(f"[boot] CUDA device count: {torch.cuda.device_count()}")
print(f"[boot] Using device: {torch.cuda.get_device_name(0)}")
print(f"[boot] Loading model: {MODEL_ID}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
    trust_remote_code=True,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)
model.to(DEVICE)
model.eval()

print("[boot] Model loaded and moved to CUDA")


def build_prompt(job_input: dict) -> str:
    if not isinstance(job_input, dict):
        return ""

    prompt = job_input.get("prompt")
    if prompt:
        return str(prompt)

    messages = job_input.get("messages")
    if isinstance(messages, list) and messages:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            pass
        # Fallback string join
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)

    system = job_input.get("system")
    user = job_input.get("user")
    if system or user:
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system or ""},
                    {"role": "user", "content": user or ""},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )
        except Exception:
            return f"[system]\n{system or ''}\n\n[user]\n{user or ''}"

    return ""


def generate_text(prompt: str, job_input: dict) -> str:
    max_new_tokens = int(job_input.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    temperature = float(job_input.get("temperature", DEFAULT_TEMPERATURE))
    top_p = float(job_input.get("top_p", DEFAULT_TOP_P))
    do_sample = job_input.get("do_sample", DEFAULT_DO_SAMPLE)

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def handler(job):
    try:
        job_input = job.get("input", {}) if isinstance(job, dict) else {}
        prompt = build_prompt(job_input)
        if not prompt:
            return {"error": "prompt is empty", "traceback": ""}
        text = generate_text(prompt, job_input)
        return {"output": text}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "traceback": traceback.format_exc()}


runpod.serverless.start({"handler": handler})
