#!/usr/bin/env python3
"""
Script để tải ColBERT model về local một lần duy nhất.
Sau khi tải xong, code sẽ dùng model từ local thay vì download từ HuggingFace.

Usage:
    python download_colbert_model.py

Model sẽ được lưu vào: ./models/colbert/
"""

import os
import sys
from pathlib import Path


def download_colbert_model():
    """Tải ColBERT model về local directory."""

    print("=" * 80)
    print("ColBERT Model Downloader")
    print("=" * 80)
    print()

    # Load HF token from .env if available
    hf_token = None
    try:
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print(f"🔑 Using HF token from .env: {hf_token[:10]}...")
        else:
            print("ℹ️  No HF token found (not needed for public models)")
    except ImportError:
        print("ℹ️  python-dotenv not installed, skipping token check")
    print()

    # Định nghĩa paths
    models_dir = Path("models/colbert")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_name = "colbert-ir/colbertv2.0"
    local_path = models_dir / "colbertv2.0"

    print(f"📥 Downloading model: {model_name}")
    print(f"📁 Save to: {local_path.absolute()}")
    print()

    try:
        # Import here để chỉ cần khi chạy script này
        from transformers import AutoModel, AutoTokenizer

        print("1️⃣  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=None,
            token=hf_token,  # Use token if available
        )
        tokenizer.save_pretrained(local_path)
        print(f"   ✅ Saved to {local_path}")
        print()

        print("2️⃣  Downloading model weights...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=None,
            token=hf_token,  # Use token if available
        )
        model.save_pretrained(local_path)
        print(f"   ✅ Saved to {local_path}")
        print()

        print("=" * 80)
        print("✅ DOWNLOAD COMPLETED!")
        print("=" * 80)
        print()
        print("Next steps:")
        print("1. Model đã được lưu tại:", local_path.absolute())
        print("2. Code sẽ tự động dùng model này thay vì download")
        print("3. Không cần commit model vào Git (đã có trong .gitignore)")
        print()
        print("To use in code:")
        print(f'   model = ColbertELModel(model_name_or_path="{local_path}")')
        print()

        return True

    except ImportError:
        print("❌ Error: transformers library not found")
        print("   Run: pip install transformers")
        return False

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("Common issues:")
        print("1. Network/proxy problems → Check internet connection")
        print("2. Disk space → Model cần ~450MB")
        print("3. Authentication → Một số models cần HF token")
        return False


if __name__ == "__main__":
    success = download_colbert_model()
    sys.exit(0 if success else 1)
