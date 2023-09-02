from llama_cpp import Llama
llm = Llama(model_path="/scratch/yerong/.cache/pyllama/Llama-2-7b/ggml-model-q4_0.gguf", n_gpu_layers= 100, n_ctx=4096)
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32, stop=["Q:", "\n"], echo=True)
print(output)
