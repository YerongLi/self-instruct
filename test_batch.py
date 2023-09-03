import torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer

# Load the AutoGPTQForCausalLM model and tokenizer
model_name_or_path = "/scratch/yerong/.cache/pyllama/Llama-2-70B-GPTQ"
model = AutoGPTQForCausalLM.from_quantized(model_name_or_path, model_basename="model", inject_fused_attention=False, use_safetensors=True, trust_remote_code=False, device_map="auto", use_triton=False, quantize_config=None)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)

# Set the padding token to be equal to the end of sequence token (eos_token)
tokenizer.pad_token = tokenizer.eos_token

# Set the device (CPU or GPU)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# Define a list of prompts (text) for batch generation
sentences = [
    "Prompt 1: This is the first prompt.",
    "Prompt 2: This is the second prompt.",
    "Prompt 3: Another prompt to generate text.",
    # "Prompt 4: Yet another prompt for AutoGPT.",
    # "Prompt 5: A fifth prompt to see what it generates.",
    # "Prompt 6: AutoGPT can generate text creatively.",
    # "Prompt 7: Let's test another prompt.",
    # "Prompt 8: The final prompt for this batch.",
]

input_ids = tokenizer(sentences, return_tensors='pt', truncation=True, padding="max_length", max_length=512).input_ids.cuda()

with torch.no_grad():
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=float(0.01),
        top_k=40
    )

print(tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], max_new_tokens=256, skip_special_tokens=True))