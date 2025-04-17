from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import torch

# Your fine-tuned adapter path
adapter_path = "./results/checkpoint-84"  # <- update if needed

# Load LoRA config
config = PeftConfig.from_pretrained(adapter_path)
base_model_name = config.base_model_name_or_path

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# BitsAndBytes config for 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    trust_remote_code=True,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
)

# Apply fine-tuned LoRA adapter
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

print("🧠 Fine-Tuned Phi-3 is ready! Type 'exit' to quit.")

# Predefined starter question
default_question = "What is the formula for the area of a circle?"

while True:
    user_input = input(f"\nYou [{default_question if default_question else ''}]: ") or default_question
    if user_input.lower() in ['exit', 'quit']:
        print("👋 Exiting chat. Goodbye!")
        break

    # Prompt formatting: same as fine-tuning dataset
    prompt = f"<|user|>\n{user_input}<|end|>\n<|assistant|>\n"

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False  # Disable cache for this generation step
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove prompt from output and trim at end marker
    reply = generated_text.replace(prompt, "").split("<|end|>")[0].strip()
    print(f"Model: {reply}")

    # Reset for next round
    default_question = ""
