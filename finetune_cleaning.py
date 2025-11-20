import json
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments

# ---------------------------------------------------------
# 1. SETUP
# ---------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATA_FILE = "synthetic_autodc_kp.jsonl"
OUTPUT_DIR = "checkpoints/cleaning_agent_7b_final"
MAX_SEQ_LENGTH = 4096 
TRAIN_SPLIT = 0.9

print(f"⬇️ Loading {MODEL_NAME}...")

# Load Model in 4-bit (Efficient & High Quality)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None,
    load_in_4bit = True,
)

# Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none", 
    use_gradient_checkpointing = "unsloth", 
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

# ---------------------------------------------------------
# 2. DATA PROCESSING
# ---------------------------------------------------------
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You are an autonomous Data Cleaning Agent.
Analyze the user's goal and the raw CSV provided.
Output a JSON list of OpenRefine operations to clean the data.

### Input:
GOAL: {}
DATA:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    instructions = examples["purpose"]
    inputs       = examples["raw_table"]
    outputs      = examples["cleaning_workflow"]
    texts = []
    
    for instruction, input_data, output_data in zip(instructions, inputs, outputs):
        # Convert JSON object to string
        output_str = json.dumps(output_data, indent=2)
        
        # Format prompt
        text = alpaca_prompt.format(instruction, input_data, output_str) + EOS_TOKEN
        texts.append(text)
        
    return { "text" : texts, }

# Loss Masking: The Model ignores the prompt and only learns the JSON output
response_template = "### Response:\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

print("📊 Processing Dataset...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# Split Train/Test to prevent overfitting
split_dataset = dataset.train_test_split(test_size=1-TRAIN_SPLIT, seed=3407)
train_dataset = split_dataset["train"].map(formatting_prompts_func, batched=True)
eval_dataset = split_dataset["test"].map(formatting_prompts_func, batched=True)

print(f"📈 Train Size: {len(train_dataset)} | Eval Size: {len(eval_dataset)}")

# ---------------------------------------------------------
# 3. TRAINING (A100 OPTIMIZED)
# ---------------------------------------------------------
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    data_collator = collator,
    packing = False, 
    args = TrainingArguments(
        per_device_train_batch_size = 16, # A100 Optimization: Higher batch = Better Gradients
        per_device_eval_batch_size = 16,
        gradient_accumulation_steps = 2,
        warmup_steps = 10,
        max_steps = 300,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUTPUT_DIR,
        eval_strategy = "steps",
        eval_steps = 50,
        save_strategy = "steps",
        save_steps = 50,
        load_best_model_at_end = True,
        metric_for_best_model = "eval_loss",
        report_to = "none", 
    ),
)

print("🚀 Starting Fine-Tuning...")
trainer_stats = trainer.train()

# ---------------------------------------------------------
# 4. TESTING & SAVING
# ---------------------------------------------------------
print("\n🔍 Running Inference Test...")
FastLanguageModel.for_inference(model)

test_instruction = "Standardize city names and date formats."
test_csv = """City,Date
new_york,01-15-2023
LOS-ANGELES,2023/02/20"""

input_prompt = alpaca_prompt.format(test_instruction, test_csv, "")

inputs = tokenizer([input_prompt], return_tensors = "pt").to("cuda")

# Generate JSON
outputs = model.generate(
    **inputs, 
    max_new_tokens = 1024, 
    use_cache = True, 
    temperature = 0.1 # Low temp for deterministic code
)
result = tokenizer.batch_decode(outputs)

generated_json = result[0].split("### Response:\n")[-1].replace(tokenizer.eos_token, "")

print("Generated Cleaning Workflow:")
print(generated_json)

print(f"\n💾 Saving to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
# Save GGUF for local use if needed
# model.save_pretrained_gguf(OUTPUT_DIR, tokenizer, quantization_method = "q4_k_m")

print("\n✅ Training Complete!")
print(f"Final Train Loss: {trainer_stats.training_loss:.4f}")