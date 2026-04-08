"""
LLaDA Quantized Inference
Load pre-quantized INT8 or INT4 weights and generate text.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from quantize import Int8Linear, Int4LinearPacked, quantize_int8, quantize_int4


def load_quantized(weight_path: str, mode: str = "int8", device: str = "cuda"):
    """
    Load a quantized LLaDA model from a saved .pt file.

    Args:
        weight_path: path to llada_int8_quantized.pt or llada_int4_quantized.pt
        mode:        "int8" or "int4"
        device:      "cuda", "cpu", or "mps"
    """
    assert mode in ("int8", "int4"), "mode must be 'int8' or 'int4'"

    print(f"Step 1: Loading architecture (CPU)...")
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Step 2: Building {mode.upper()} structure...")
    if mode == "int8":
        model, n = quantize_int8(model)
    else:
        model, n = quantize_int4(model)
    print(f"  Replaced {n} layers")

    print(f"Step 3: Loading weights from {weight_path}...")
    state = torch.load(weight_path, map_location="cpu")
    model.load_state_dict(state)
    del state

    print(f"Step 4: Moving to {device}...")
    model = model.to(device)
    model.eval()
    print("Model ready!")
    return model


@torch.no_grad()
def generate(model, tokenizer, prompt: str, steps: int = 64, gen_length: int = 128) -> str:
    """
    Generate text from a quantized LLaDA model.

    Args:
        model:      loaded quantized model
        tokenizer:  LLaDA tokenizer
        prompt:     user query string
        steps:      number of diffusion denoising steps
        gen_length: number of tokens to generate

    Returns:
        generated text string
    """
    MASK_ID = 126336
    device = next(model.parameters()).device

    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(device)

    prompt_len = input_ids.shape[1]
    x = torch.full(
        (1, prompt_len + gen_length),
        MASK_ID, dtype=torch.long, device=device
    )
    x[:, :prompt_len] = input_ids

    for step in range(steps):
        logits = model(x).logits
        masked = (x == MASK_ID)
        if masked.sum() == 0:
            break
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        confidence[~masked] = -float("inf")
        num_unmask = max(1, masked.sum().item() // max(1, steps - step))
        top_pos = confidence[0].topk(num_unmask).indices
        x[0, top_pos] = predicted[0, top_pos]

    return tokenizer.decode(x[0, prompt_len:], skip_special_tokens=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLaDA Quantized Inference")
    parser.add_argument("--weight_path", required=True, help="Path to quantized .pt file")
    parser.add_argument("--mode", choices=["int8", "int4"], default="int8")
    parser.add_argument("--prompt", type=str, default="What is artificial intelligence?")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )
    model = load_quantized(args.weight_path, args.mode, args.device)

    print(f"\nPrompt: {args.prompt}")
    output = generate(model, tokenizer, args.prompt, args.steps, args.gen_length)
    print(f"Output: {output}")
