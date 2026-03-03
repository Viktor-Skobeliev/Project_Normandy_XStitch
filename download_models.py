import os
from huggingface_hub import hf_hub_download

base_path = os.path.join(os.getcwd(), "models")
os.makedirs(base_path, exist_ok=True)

print(f"Downloading stable models to: {base_path}")


# ── Llama 3.1 8B ─────────────────────────────────────────────────────────────
try:
    print("\nChecking Llama-3.1-8B-Instruct...")
    hf_hub_download(
        repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        filename="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        local_dir=base_path
    )
    print("Llama 3.1 is ready.")
except Exception as e:
    print(f"Llama download error: {e}")


# ── Qwen2-VL 2B Vision ───────────────────────────────────────────────────────
try:
    print("\nDownloading Qwen2-VL-2B-Instruct-GGUF (Vision model)...")
    hf_hub_download(
        repo_id="bartowski/Qwen2-VL-2B-Instruct-GGUF",
        filename="Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
        local_dir=base_path
    )
    print("Qwen2-VL (Vision) downloaded successfully!")
except Exception as e:
    print(f"Vision model error: {e}")


# ── MobileSAM weights ─────────────────────────────────────────────────────────
try:
    print("\nDownloading MobileSAM weights...")
    hf_hub_download(
        repo_id="dhkim2810/MobileSAM",
        filename="mobile_sam.pt",
        local_dir=base_path
    )
    print("MobileSAM is ready.")
except Exception as e:
    print(f"MobileSAM download error: {e}")


# ── U2Net (rembg background removal) ─────────────────────────────────────────
try:
    print("\nDownloading U2Net ONNX for background removal...")
    hf_hub_download(
        repo_id="vishnusureshperumbavoor/u2net_onnx",
        filename="u2net.onnx",
        local_dir=base_path
    )
    print("U2Net ONNX downloaded successfully!")
except Exception as e:
    print(f"U2Net download error: {e}")


print("\n✅ Setup complete.")
print(f"Models location: {base_path}")