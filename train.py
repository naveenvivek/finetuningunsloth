import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ============ CONFIG ============
# Change to 14B if you want to push it
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR = "./output/qwen-coder-finetuned"
DATA_PATH = "./data/training_data.jsonl"
MAX_SEQ_LENGTH = 1024  # Reduce to 512 if OOM

# ============ CUDA CHECK ============
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available! Install PyTorch with CUDA support.")
print(f"Using GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB VRAM)")

# ============ LOAD MODEL IN 4-BIT ============
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============ APPLY LoRA ============
print("Applying LoRA...")
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Should show ~1-2% trainable

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
print("Starting training...")
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,       # Keep at 1 for VRAM
    gradient_accumulation_steps=2,        # Effective batch size = 2
    gradient_checkpointing=True,          # Saves ~30% VRAM
    learning_rate=2e-4,
    bf16=True,
    logging_steps=1,
    save_steps=50,
    save_total_limit=2,
    warmup_steps=10,
    optim="paged_adamw_8bit",            # Memory-efficient optimizer
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
print(f"Done! Model saved to {OUTPUT_DIR}")
