"""
LLaDA Quantization — INT8 and INT4
World's first quantization for LLaDA diffusion language model.

Supports:
  - INT8 weight quantization (per-row scaling)
  - INT4 weight quantization (packed, per-row scaling)

Usage:
  from quantize import quantize_int8, quantize_int4
  model = quantize_int8(model)   # or quantize_int4(model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


# ─────────────────────────────────────────────
# INT8
# ─────────────────────────────────────────────

class Int8Linear(nn.Module):
    """INT8 weight-only quantization layer (per-row scaling)."""

    def __init__(self, original_layer: nn.Linear):
        super().__init__()
        w = original_layer.weight.data.float()
        scale = w.abs().max(dim=1, keepdim=True).values / 127.0
        scale = scale.clamp(min=1e-8)
        w_int8 = (w / scale).round().clamp(-128, 127).to(torch.int8)

        self.register_buffer("weight_int8", w_int8)
        self.register_buffer("scale", scale.to(torch.float32))
        if original_layer.bias is not None:
            self.register_buffer("bias", original_layer.bias.data.to(torch.float32))
        else:
            self.bias = None

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

    def forward(self, x):
        w = self.weight_int8.to(torch.float32) * self.scale
        w = w.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def quantize_int8(model: nn.Module) -> tuple:
    """
    Replace all nn.Linear layers with Int8Linear.
    Returns (quantized_model, num_layers_replaced).
    """
    replaced = 0

    def _replace(module):
        nonlocal replaced
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, Int8Linear(child))
                replaced += 1
            else:
                _replace(child)

    _replace(model)
    return model, replaced


# ─────────────────────────────────────────────
# INT4
# ─────────────────────────────────────────────

class Int4LinearPacked(nn.Module):
    """INT4 weight-only quantization layer (packed uint8, per-row scaling)."""

    def __init__(self, original_layer: nn.Linear):
        super().__init__()
        w = original_layer.weight.data.float()
        scale = w.abs().max(dim=1, keepdim=True).values / 7.0
        scale = scale.clamp(min=1e-8)
        w_int4 = (w / scale).round().clamp(-8, 7).to(torch.int8)

        # Pack two int4 values per byte
        w_flat = w_int4.reshape(-1)
        if w_flat.shape[0] % 2 != 0:
            w_flat = torch.cat([w_flat, torch.zeros(1, dtype=torch.int8)])
        even = w_flat[0::2] & 0x0F
        odd  = (w_flat[1::2] & 0x0F) << 4
        packed = (even | odd).to(torch.uint8)

        self.register_buffer("weight_packed", packed)
        self.register_buffer("scale", scale.to(torch.float32))
        self.original_shape = w_int4.shape

        if original_layer.bias is not None:
            self.register_buffer("bias", original_layer.bias.data.to(torch.float32))
        else:
            self.bias = None

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

    def forward(self, x):
        packed = self.weight_packed
        result = torch.empty(packed.numel() * 2, dtype=torch.int8, device=packed.device)
        result[0::2] = (packed & 0x0F).to(torch.int8)
        result[1::2] = ((packed >> 4) & 0x0F).to(torch.int8)
        mask = result > 7
        result[mask] = result[mask] - 16
        numel = self.original_shape[0] * self.original_shape[1]
        w = result[:numel].reshape(self.original_shape).float()
        w = w * self.scale
        w = w.to(x.dtype)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def quantize_int4(model: nn.Module) -> tuple:
    """
    Replace all nn.Linear layers with Int4LinearPacked.
    Returns (quantized_model, num_layers_replaced).
    """
    replaced = 0

    def _replace(module):
        nonlocal replaced
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, Int4LinearPacked(child))
                replaced += 1
            else:
                _replace(child)

    _replace(model)
    return model, replaced


# ─────────────────────────────────────────────
# Convenience: run quantization and save
# ─────────────────────────────────────────────

def run_and_save(mode: str = "int8", save_path: str = None):
    """
    Load LLaDA-8B-Instruct, quantize, and save weights.

    Args:
        mode:      "int8" or "int4"
        save_path: where to save the .pt file
    """
    assert mode in ("int8", "int4"), "mode must be 'int8' or 'int4'"

    if save_path is None:
        save_path = f"llada_{mode}_quantized.pt"

    print(f"Loading LLaDA-8B-Instruct (bfloat16, CPU)...")
    model = AutoModel.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Quantizing to {mode.upper()}...")
    if mode == "int8":
        model, n = quantize_int8(model)
    else:
        model, n = quantize_int4(model)
    print(f"  Replaced {n} Linear layers")

    print(f"Saving to {save_path} ...")
    torch.save(model.state_dict(), save_path)
    print("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLaDA Quantization")
    parser.add_argument("--mode", choices=["int8", "int4"], default="int8")
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

    run_and_save(args.mode, args.save_path)
