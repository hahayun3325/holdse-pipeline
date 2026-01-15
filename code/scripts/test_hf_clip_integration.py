#!/usr/bin/env python3
"""Test Hugging Face CLIP integration with HOLDSE."""

import torch
from transformers import CLIPTokenizer, CLIPTextModel


def test_clip_dimensions():
    """Verify CLIP output matches GHOP U-Net expectations."""
    print("=" * 70)
    print("Testing Hugging Face CLIP Integration")
    print("=" * 70)

    model_name = "openai/clip-vit-large-patch14"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"\n1. Loading CLIP model: {model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device)
    model.eval()

    print(f"   ✅ Model loaded to {device}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ Parameters: {num_params:,}")

    # Test with sample prompts
    print("\n2. Testing text encoding...")
    prompts = [
        "a hand grasping a bottle",
        "a hand holding a mug"
    ]

    with torch.no_grad():
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        outputs = model(**inputs)
        embeddings = outputs.pooler_output  # (B, 768)
        embeddings_with_seq = embeddings.unsqueeze(1)  # (B, 1, 768)

    print(f"   ✅ Input prompts: {len(prompts)}")
    print(f"   ✅ Pooler output shape: {embeddings.shape}")
    print(f"   ✅ With sequence dim: {embeddings_with_seq.shape}")

    # Validate dimensions
    print("\n3. Validating compatibility with GHOP U-Net...")
    batch_size = len(prompts)
    expected_shape = (batch_size, 1, 768)

    checks = [
        (embeddings.shape[-1] == 768, "Embedding dimension is 768 ✅"),
        (embeddings_with_seq.shape == expected_shape, f"Shape matches U-Net input {expected_shape} ✅"),
        (embeddings_with_seq.requires_grad == False, "Gradients disabled (frozen) ✅"),
    ]

    all_passed = True
    for check, message in checks:
        if check:
            print(f"   {message}")
        else:
            print(f"   ❌ {message}")
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED - Ready for training!")
    else:
        print("❌ SOME TESTS FAILED - Check configuration")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = test_clip_dimensions()
    exit(0 if success else 1)