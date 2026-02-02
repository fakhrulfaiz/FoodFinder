import torch
import sys

print("="*60)
print("PyTorch GPU Diagnostic")
print("="*60)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print("\n✓ GPU is available and ready!")
else:
    print("\n⚠️  CUDA NOT AVAILABLE")
    print("\nPossible reasons:")
    print("1. CPU-only PyTorch installed")
    print("2. CUDA drivers not installed")
    print("3. Wrong PyTorch version for your CUDA")
    
    print("\nTo fix:")
    print("1. Uninstall current PyTorch:")
    print("   pip uninstall torch torchvision -y")
    print("\n2. Install GPU version:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    sys.exit(1)
