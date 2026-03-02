import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# ============ CONFIG ============
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_DIR = "./output/qwen-coder-1.5b-full"
DATA_PATH = "./data/training_data.jsonl"
MAX_SEQ_LENGTH = 1024

# ============ CUDA CHECK ============
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA not available! Install PyTorch with CUDA support.")
print(f"Using GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)")

# ============ LOAD MODEL (FULL PRECISION bf16) ============
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,} (100%)")

# ============ LOAD DATASET ============
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATA_PATH, split="train")


def format_prompt(example):
    """Format each example into the Qwen chat template."""
    text = f"""<|im_start|>system
You are a code assistant that fixes code issues.<|im_end|>
<|im_start|>user
{example['instruction']}

```
{example['input']}
```<|im_end|>
<|im_start|>assistant
```
{example['output']}
```<|im_end|>"""
    return {"text": text}


dataset = dataset.map(format_prompt)

# ============ TRAINING ============
print("Starting full fine-tuning...")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,        # Effective batch size = 4
    gradient_checkpointing=True,          # Saves ~30% VRAM
    learning_rate=2e-5,                   # Lower LR for full fine-tuning
    bf16=True,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    warmup_steps=10,
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    report_to="none",
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    args=training_args,
)

trainer.train()

# ============ SAVE ============
print("Saving model...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Done! Full model saved to {OUTPUT_DIR}")
