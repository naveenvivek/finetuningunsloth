# Fine-Tuning Qwen2.5-Coder-7B for SonarQube Code Fixes

Fine-tune [Qwen/Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) using **QLoRA** (4-bit quantization + LoRA) to automatically fix common code quality issues detected by SonarQube.

## What It Does

The model learns to take a SonarQube issue description + problematic code as input and produce the corrected code as output. The training data covers 30 common issue types including:

- Unused variables, imports, and parameters
- Missing null checks and type annotations
- Security issues (SQL injection, hardcoded passwords, `eval()` usage)
- Code smells (magic numbers, global variables, mutable default arguments)
- Dead code, broad exception handling, and more

## Project Structure

```
├── train.py                  # Training script
├── requirements.txt          # Python dependencies
├── data/
│   └── training_data.jsonl   # 30 SonarQube fix examples (instruction/input/output)
└── output/
    └── qwen-coder-finetuned/ # Saved LoRA adapter after training
        ├── adapter_config.json
        ├── adapter_model.safetensors
        └── ...
```

## Requirements

- **GPU**: NVIDIA GPU with CUDA support (tested with CUDA 12.8)
- **VRAM**: ~8-10 GB minimum (4-bit quantization + gradient checkpointing)
- **Python**: 3.10+

## Setup

```bash
# Create virtual environment and install dependencies
uv venv .venv
.venv\Scripts\activate
uv pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `torch` (CUDA 12.8) | GPU tensor operations |
| `transformers` | Model loading and tokenizer |
| `peft` | LoRA / QLoRA adapter training |
| `bitsandbytes` | 4-bit quantization (NF4) |
| `trl` | SFTTrainer for supervised fine-tuning |
| `datasets` | JSONL dataset loading |
| `accelerate` | Multi-GPU / mixed precision support |

## Training

```bash
python train.py
```

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Quantization | 4-bit NF4 with double quantization |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs | 3 |
| Batch size | 1 (effective 2 with gradient accumulation) |
| Learning rate | 2e-4 |
| Optimizer | `paged_adamw_8bit` |
| Max sequence length | 1024 |
| Precision | bf16 |
| Gradient checkpointing | Enabled |

### Trainable Parameters

Only ~1-2% of model parameters are trainable thanks to LoRA — the rest of the 7B model stays frozen in 4-bit precision.

## Dataset Format

Each example in `data/training_data.jsonl` follows this structure:

```json
{
  "instruction": "Fix the following SonarQube issue: Unused variable",
  "input": "def process():\n    x = 10\n    return 42",
  "output": "def process():\n    return 42"
}
```

During training, examples are formatted into the Qwen chat template:

```
<|im_start|>system
You are a code assistant that fixes code issues.<|im_end|>
<|im_start|>user
{instruction}

```{input}```<|im_end|>
<|im_start|>assistant
```{output}```<|im_end|>
```

## Output

After training, the LoRA adapter is saved to `output/qwen-coder-finetuned/`. This adapter can be loaded on top of the base Qwen model for inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
model = PeftModel.from_pretrained(base_model, "./output/qwen-coder-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./output/qwen-coder-finetuned")
```

## Framework Versions

- PEFT 0.18.1
- TRL 0.29.0
- Transformers 5.2.0
- PyTorch 2.10.0+cu128
- Datasets 4.6.1
