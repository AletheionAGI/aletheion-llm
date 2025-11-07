#!/usr/bin/env python3
"""
Local testing script for Aletheion HuggingFace Space.
Run this before deploying to verify everything works.
"""

import sys
from pathlib import Path


def test_imports():
    """Test that all required imports work."""
    print("🧪 Testing imports...")

    try:
        import torch

        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False

    try:
        from transformers import GPT2Tokenizer as _GPT2Tokenizer  # noqa: F401

        print("✅ Transformers (GPT2Tokenizer)")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False

    try:
        import gradio as gr

        print(f"✅ Gradio {gr.__version__}")
    except ImportError as e:
        print(f"❌ Gradio import failed: {e}")
        return False

    try:
        import numpy as np

        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False

    return True


def test_file_structure():
    """Test that all required files exist."""
    print("\n📁 Testing file structure...")

    required_files = [
        "app.py",
        "requirements.txt",
        "README.md",
        ".gitattributes",
        "model/config.json",
        "src/model.py",
        "src/attention.py",
        "src/aletheion/pyramidal_model.py",
        "src/aletheion/gates.py",
        "src/aletheion/loss.py",
        "src/aletheion/pyramid.py",
    ]

    optional_files = [
        "model/pytorch_model.bin",
        "assets/paper.pdf",
        "assets/training_curves.png",
        "assets/comparison_plot.png",
        "assets/calibration_plots.png",
    ]

    all_found = True

    for filepath in required_files:
        if Path(filepath).exists():
            print(f"✅ {filepath}")
        else:
            print(f"❌ {filepath} (REQUIRED)")
            all_found = False

    print("\nOptional files:")
    for filepath in optional_files:
        if Path(filepath).exists():
            print(f"✅ {filepath}")
        else:
            print(f"⚠️  {filepath} (optional)")

    return all_found


def test_aletheion_imports():
    """Test that Aletheion modules can be imported."""
    print("\n🗡️ Testing Aletheion imports...")

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent / "src"))

    try:
        from aletheion.pyramidal_model import AletheionPyramidalTransformer as _APT  # noqa: F401

        print("✅ AletheionPyramidalTransformer")
    except ImportError as e:
        print(f"❌ AletheionPyramidalTransformer import failed: {e}")
        return False

    try:
        from aletheion.gates import PyramidalEpistemicGates as _PEG  # noqa: F401

        print("✅ PyramidalEpistemicGates")
    except ImportError as e:
        print(f"❌ PyramidalEpistemicGates import failed: {e}")
        return False

    try:
        from model import BaselineTransformer as _BT  # noqa: F401

        print("✅ BaselineTransformer")
    except ImportError as e:
        print(f"❌ BaselineTransformer import failed: {e}")
        return False

    return True


def test_model_initialization():
    """Test that the model can be initialized."""
    print("\n🤖 Testing model initialization...")

    sys.path.insert(0, str(Path(__file__).parent / "src"))

    try:
        from aletheion.pyramidal_model import AletheionPyramidalTransformer

        model = AletheionPyramidalTransformer(
            vocab_size=50257,
            d_model=512,
            n_layers=6,
            n_heads=8,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.1,
        )

        print("✅ Model initialized successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True

    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_tokenizer():
    """Test that the tokenizer works."""
    print("\n📝 Testing tokenizer...")

    try:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        test_text = "Hello, world!"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)

        print("✅ Tokenizer works")
        print(f"   Input: {test_text}")
        print(f"   Tokens: {tokens}")
        print(f"   Decoded: {decoded}")
        return True

    except Exception as e:
        print(f"❌ Tokenizer test failed: {e}")
        return False


def test_config():
    """Test that the model config is valid JSON."""
    print("\n⚙️  Testing model config...")

    try:
        import json

        with open("model/config.json") as f:
            config = json.load(f)

        print("✅ Config is valid JSON")
        print(f"   Keys: {list(config.keys())}")
        return True

    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("🗡️ Aletheion HuggingFace Space - Local Testing")
    print("=" * 60)

    tests = [
        ("Imports", test_imports),
        ("File Structure", test_file_structure),
        ("Aletheion Imports", test_aletheion_imports),
        ("Model Config", test_config),
        ("Tokenizer", test_tokenizer),
        ("Model Initialization", test_model_initialization),
    ]

    results = []

    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name} test crashed: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready for deployment.")
        print("\n📋 Next steps:")
        print("   1. Review DEPLOYMENT.md")
        print("   2. cd hf_space && git init")
        print("   3. git lfs install")
        print("   4. git remote add origin https://huggingface.co/spaces/USERNAME/aletheion-llm")
        print("   5. git add . && git commit -m 'Initial deployment'")
        print("   6. git push -u origin main")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
