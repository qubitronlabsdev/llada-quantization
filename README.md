# LLaDA Quantization — INT8 & INT4

**World's first INT8 and INT4 quantization for [LLaDA](https://github.com/ML-GSAI/LLaDA) — a diffusion-based large language model.**

---

## Results

| Model | Size | Memory Saved | Speed (A100) |
|---|---|---|---|
| LLaDA-8B (original) | 16.13 GB | — | ~5 tok/s |
| LLaDA-8B INT8 | 8.54 GB | **47%** | **9.64 tok/s** |
| LLaDA-8B INT4 | 5.82 GB | **64%** | 3.39 tok/s |

- INT8 = **2x smaller** + faster inference
- INT4 = **3x smaller** — fits on consumer GPUs

---

## What is LLaDA?

LLaDA is a diffusion language model trained from scratch at 8B scale, competitive with LLaMA3-8B. Unlike autoregressive models (GPT, LLaMA), LLaDA generates tokens in **parallel** using masked diffusion.

Paper: [Large Language Diffusion Models](https://arxiv.org/abs/2502.09992)

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/llada-quantization
cd llada-quantization
pip install -r requirements.txt
```

---

## Usage

### Option 1 — Quantize from scratch

```python
from quantize import run_and_save

# Create INT8 weights
run_and_save(mode="int8", save_path="llada_int8_quantized.pt")

# Create INT4 weights
run_and_save(mode="int4", save_path="llada_int4_quantized.pt")
```

Or from command line:

```bash
python quantize.py --mode int8 --save_path llada_int8_quantized.pt
python quantize.py --mode int4 --save_path llada_int4_quantized.pt
```

### Option 2 — Use pre-quantized weights

Download weights from Google Drive (links below), then:

```python
from inference import load_quantized, generate
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
)

# Load INT8
model = load_quantized("llada_int8_quantized.pt", mode="int8", device="cuda")

# Generate
output = generate(model, tokenizer, "What is machine learning?")
print(output)
```

Or from command line:

```bash
python inference.py \
  --weight_path llada_int8_quantized.pt \
  --mode int8 \
  --prompt "What is artificial intelligence?" \
  --device cuda
```

### Option 3 — Google Colab demo

Open `llada_drive_test.ipynb` in Google Colab for an interactive INT8 vs INT4 comparison.

---

## Pre-quantized Weights

| File | Size | Link |
|---|---|---|
| llada_int8_quantized.pt | 8.54 GB | *(add Drive link)* |
| llada_int4_quantized.pt | 5.82 GB | *(add Drive link)* |

---

## How It Works

Standard LLaDA uses bfloat16 (2 bytes per weight). We replace all `nn.Linear` layers:

**INT8:** Each weight row scaled to `[-127, 127]` integers. Per-row scale factor stored in float32. ~1 byte per weight.

**INT4:** Each weight row scaled to `[-8, 7]` integers. Two values packed per byte (`uint8`). ~0.5 bytes per weight.

Both methods dequantize on-the-fly during the forward pass — no changes to model architecture or generation logic needed.

---

## Tested On

- NVIDIA A100 80GB
- NVIDIA H100
- Google Colab T4 (INT4 only — INT8 may OOM on T4)

---

## Citation

If you use this work, please cite:

```bibtex
@misc{llada-quantization-2026,
  title  = {LLaDA Quantization: INT8 and INT4 for Diffusion Language Models},
  author = {YOUR NAME},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/llada-quantization}
}
```

Original LLaDA paper:

```bibtex
@article{nie2025large,
  title  = {Large Language Diffusion Models},
  author = {Nie, Shen and others},
  year   = {2025},
  url    = {https://arxiv.org/abs/2502.09992}
}
```

---

## License

Apache 2.0 — same as original LLaDA.
