import torch
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator

# Define the folder path where the model is saved
model_folder = "/scratch/yerong/.cache/pyllama/Llama-2-7B-GPTQ"

# Load the model and tokenizer
model = ExLlama.load_from_folder(model_folder)
tokenizer = ExLlamaTokenizer.from_folder(model_folder)

# Create a cache for the model
cache = ExLlamaCache(model)

# Define your prompt
prompt = "who is elon musk"

# Tokenize the prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

# Generate text using the model
generator = ExLlamaGenerator(model, tokenizer, cache)
output_text = generator.generate(input_ids)

# Print the generated text
print(output_text)
