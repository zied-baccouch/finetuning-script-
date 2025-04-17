from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# 1. Load dataset
dataset = load_dataset('json', data_files={
    'train': 'train.jsonl',
    'validation': 'val.jsonl'
})

# 2. Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenization function
def tokenize_function(examples):
    texts = [
        f"<|user|>\n{instruction}<|end|>\n<|assistant|>\n{response}<|end|>"
        for instruction, response in zip(examples["instruction"], examples["response"])
    ]
    return tokenizer(texts, truncation=True, max_length=128, padding=False)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# 4. Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# 5. Load model
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16
)

# 6. Critical preparation steps
model = prepare_model_for_kbit_training(model)  # Prepares 4-bit model for training
model.config.use_cache = False  # Disables cache to save VRAM
model.enable_input_require_grads()  # Ensures gradients are computed

# 7. LoRA config
peft_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# 8. Verify trainable parameters
model.print_trainable_parameters()

# 9. Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# 10. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=3,
    learning_rate=1e-5,
    optim="adamw_torch_fused",
    logging_steps=10,
    save_steps=500,
    fp16=True,
    eval_strategy="steps",
    eval_steps=200,
    report_to="none",
    remove_unused_columns=True,
    label_names=["input_ids"],  # Explicitly set label names
)

# 11. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# 12. Start training
print("Starting training...")
trainer.train()