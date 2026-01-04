"""Debug ColBERT API structure"""
from ragatouille import RAGPretrainedModel

print("Loading ColBERT model...")
model = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

print("\n=== RAGPretrainedModel attributes ===")
print(dir(model))

print("\n=== model.model attributes ===")
if hasattr(model, 'model'):
    print(dir(model.model))

    if hasattr(model.model, 'model'):
        print("\n=== model.model.model attributes ===")
        print(dir(model.model.model))

        checkpoint = model.model.model
        print(f"\nCheckpoint type: {type(checkpoint)}")
        print(f"Checkpoint class: {checkpoint.__class__.__name__}")

        # Check for encoding methods
        print("\n=== Looking for encoding methods ===")
        for attr in dir(checkpoint):
            if 'encode' in attr.lower() or 'query' in attr.lower() or 'doc' in attr.lower():
                print(f"  - {attr}")
