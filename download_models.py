import os
from huggingface_hub import hf_hub_download

# Твой путь к папке в проекте
base_path = os.path.join(os.getcwd(), "models")
os.makedirs(base_path, exist_ok=True)

print(f"Downloading stable models to: {base_path}")

# 1. Llama-3.1-8B (Логика и код) - уже должна быть у тебя, но проверим
try:
    print("Checking Llama-3.1-8B-Instruct...")
    hf_hub_download(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        local_dir=base_path
    )
    print("Llama 3.1 is ready.")
except Exception as e:
    print(f"Llama download error: {e}")

# 2. Qwen2-VL-2B-Instruct (Зрение/Vision вместо Moondream)
# Эта модель отлично понимает графику и схемы.
try:
    print("Downloading Qwen2-VL-2B-Instruct-GGUF (Vision model)...")
    hf_hub_download(
        repo_id="bartowski/Qwen2-VL-2B-Instruct-GGUF",
        filename="Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        local_dir=base_path
    )
    print("Qwen2-VL (Vision) downloaded successfully!")
except Exception as e:
    print(f"Vision model error: {e}")

print("\nSetup complete. Use these GGUF files with llama-cpp-python.")