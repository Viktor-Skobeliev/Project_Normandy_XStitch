from llama_cpp import Llama
import os

model_path = os.path.join(os.getcwd(), "models", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")

print("Загружаю Llama... (это может занять минуту)")
llm = Llama(model_path=model_path, verbose=False)

output = llm("Q: Why does my cross-stitch program calculate 600 meters for a small bee? A:", max_tokens=50)
print("\nОтвет Llama:")
print(output['choices'][0]['text'])