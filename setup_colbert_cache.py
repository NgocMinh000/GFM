#!/usr/bin/env python3
"""
Script để tải ColBERT model về cache một lần duy nhất.
RAGatouille sẽ tự động cache model sau lần đầu download.

Usage:
    python setup_colbert_cache.py
"""

import os
import sys
from pathlib import Path


def setup_colbert_cache():
    """
    Tải ColBERT model vào cache của RAGatouille.
    Sau lần đầu này, model sẽ được reuse từ cache.
    """

    print("=" * 80)
    print("ColBERT Model Cache Setup")
    print("=" * 80)
    print()

    # Setup HF token nếu có
    try:
        from dotenv import load_dotenv
        load_dotenv()

        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            print(f"🔑 HF token found: {hf_token[:10]}...")
            os.environ["HF_TOKEN"] = hf_token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        print()
    except ImportError:
        pass

    try:
        from ragatouille import RAGPretrainedModel

        model_name = "colbert-ir/colbertv2.0"

        print(f"📥 Loading ColBERT model: {model_name}")
        print("   (This will download on first run, then cache for reuse)")
        print()

        # RAGatouille tự động cache model
        model = RAGPretrainedModel.from_pretrained(
            model_name,
            index_root="tmp/colbert_cache_test",
        )

        print()
        print("=" * 80)
        print("✅ SETUP COMPLETED!")
        print("=" * 80)
        print()
        print("Model has been cached by RAGatouille.")
        print("Future runs will reuse the cached model automatically.")
        print()
        print("Cache location:")
        print("  ~/.cache/huggingface/  (HuggingFace cache)")
        print("  Or RAGatouille internal cache")
        print()

        return True

    except Exception as e:
        print()
        print("=" * 80)
        print("❌ SETUP FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        print()
        print("Common issues:")
        print("1. Network/proxy → Model cannot be downloaded")
        print("2. RAGatouille not installed → pip install ragatouille")
        print()
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = setup_colbert_cache()
    sys.exit(0 if success else 1)
