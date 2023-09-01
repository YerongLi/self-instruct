from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the GPT-J model and tokenizer
model_name = "EleutherAI/gpt-j-6B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the device (CPU or GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Define a list of prompts (text) for batch generation
prompts = [
    "Prompt 1: This is the first prompt.",
    "Prompt 2: This is the second prompt.",
    "Prompt 3: Another prompt to generate text.",
    "Prompt 4: Yet another prompt for GPT-J.",
    "Prompt 5: A fifth prompt to see what it generates.",
    "Prompt 6: GPT-J can generate text creatively.",
    "Prompt 7: Let's test another prompt.",
    "Prompt 8: The final prompt for this batch.",
]

# Encode the prompts and generate text in batch
input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=100).to(device)
with torch.no_grad():
    output = model.generate(input_ids, max_length=150, num_return_sequences=8)

# Decode and print the generated text for each prompt in the batch
generated_text = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output]
for i, text in enumerate(generated_text):
    print(f"Generated Text {i+1}:\n{text}\n")
