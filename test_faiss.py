#!/usr/bin/env python3
"""
Test FAISS GPU/CPU detection
"""

print("Testing FAISS import and GPU detection...")
print("=" * 60)

try:
    import faiss
    print(f"✅ FAISS imported successfully")
    print(f"   FAISS version: {faiss.__version__ if hasattr(faiss, '__version__') else 'unknown'}")

    # Check GPU availability
    try:
        num_gpus = faiss.get_num_gpus()
        if num_gpus > 0:
            print(f"✅ GPU available: {num_gpus} GPU(s) detected")

            # Get GPU properties
            for i in range(num_gpus):
                try:
                    res = faiss.StandardGpuResources()
                    print(f"   GPU {i}: Available")
                except Exception as e:
                    print(f"   GPU {i}: Error - {e}")
        else:
            print(f"⚠️  No GPU available (using CPU)")

    except AttributeError:
        print(f"⚠️  faiss.get_num_gpus() not available (CPU-only build)")
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")

    # Test creating a simple index
    print("\nTesting index creation...")
    dim = 128
    index = faiss.IndexFlatL2(dim)
    print(f"✅ Created IndexFlatL2(dim={dim})")

    # Test with some dummy data
    import numpy as np
    vectors = np.random.randn(10, dim).astype('float32')
    index.add(vectors)
    print(f"✅ Added {vectors.shape[0]} vectors to index")

    # Test search
    D, I = index.search(vectors[:1], 5)
    print(f"✅ Search completed: found {len(I[0])} neighbors")

    print("\n" + "=" * 60)
    print("✅ All FAISS tests passed!")

except ImportError as e:
    print(f"❌ Failed to import FAISS: {e}")
    print("\nPlease install FAISS:")
    print("  For GPU: pip install faiss-gpu-cu12")
    print("  For CPU: pip install faiss-cpu")

except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
