#!/usr/bin/env python3
"""
Pre-flight validation for Apple Silicon ColPali
Checks system requirements before running main application
"""

import sys
import os
import platform
import subprocess
import importlib.util

def check_system():
    """Check if running on Apple Silicon"""
    print("🖥️  System Check:")
    
    # Check architecture
    arch = platform.machine()
    print(f"   Architecture: {arch}")
    
    if arch != "arm64":
        print("   ❌ Not Apple Silicon (arm64)")
        return False
    else:
        print("   ✅ Apple Silicon detected")
    
    # Check macOS version
    version = platform.mac_ver()[0]
    print(f"   macOS Version: {version}")
    
    if version and float('.'.join(version.split('.')[:2])) < 11.0:
        print("   ⚠️  macOS 11.0+ recommended for best MPS support")
    else:
        print("   ✅ macOS version compatible")
    
    return True

def check_python():
    """Check Python version"""
    print("\n🐍 Python Check:")
    
    version = sys.version_info
    print(f"   Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("   ❌ Python 3.9+ required")
        return False
    else:
        print("   ✅ Python version compatible")
    
    return True

def check_poppler():
    """Check if poppler is installed"""
    print("\n📦 Poppler Check:")
    
    try:
        result = subprocess.run(['pdftoppm', '-h'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("   ✅ Poppler installed")
            return True
        else:
            print("   ❌ Poppler not working")
            return False
    except FileNotFoundError:
        print("   ❌ Poppler not found")
        print("   💡 Install with: brew install poppler")
        return False

def check_dependencies():
    """Check if key dependencies can be imported"""
    print("\n📚 Dependencies Check:")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("PIL", "Pillow"),
        ("pdf2image", "PDF2Image"),
        ("numpy", "NumPy"),
        ("psutil", "PSUtil"),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                print(f"   ✅ {name}")
            else:
                print(f"   ❌ {name} not found")
                all_good = False
        except ImportError:
            print(f"   ❌ {name} import error")
            all_good = False
    
    return all_good

def check_pytorch_mps():
    """Check PyTorch MPS support"""
    print("\n🔥 PyTorch MPS Check:")
    
    try:
        import torch
        print(f"   PyTorch Version: {torch.__version__}")
        
        mps_available = torch.backends.mps.is_available()
        print(f"   MPS Available: {mps_available}")
        
        if mps_available:
            try:
                # Test MPS tensor creation
                test_tensor = torch.tensor([1.0, 2.0]).to('mps')
                print("   ✅ MPS working correctly")
                return True
            except Exception as e:
                print(f"   ⚠️  MPS available but not working: {e}")
                return False
        else:
            print("   ⚠️  MPS not available - will use CPU")
            return False
            
    except ImportError:
        print("   ❌ PyTorch not installed")
        return False

def check_test_pdf():
    """Check if test PDF exists"""
    print("\n📄 Test PDF Check:")
    
    test_pdf = "/Users/ernest/Documents/Scribd/282739699-Flight-Training-Manual-A330-pdf.pdf"
    
    if os.path.exists(test_pdf):
        size_mb = os.path.getsize(test_pdf) / (1024 * 1024)
        print(f"   ✅ Test PDF found ({size_mb:.1f}MB)")
        return True
    else:
        print(f"   ❌ Test PDF not found: {test_pdf}")
        print("   💡 Update the path in test files if PDF is elsewhere")
        return False

def check_memory():
    """Check available memory"""
    print("\n💾 Memory Check:")
    
    try:
        import psutil
        
        # Get total memory
        total_memory = psutil.virtual_memory().total / (1024**3)
        available_memory = psutil.virtual_memory().available / (1024**3)
        
        print(f"   Total Memory: {total_memory:.1f}GB")
        print(f"   Available Memory: {available_memory:.1f}GB")
        
        if total_memory < 16:
            print("   ⚠️  Less than 16GB total memory")
        else:
            print("   ✅ Sufficient total memory")
        
        if available_memory < 3.5:
            print("   ⚠️  Less than 3.5GB available memory")
            print("   💡 Close other applications for best performance")
        else:
            print("   ✅ Sufficient available memory")
            
        return available_memory >= 3  # Minimum 3GB required for 3.5GB limit
        
    except ImportError:
        print("   ❌ Cannot check memory (psutil not available)")
        return False

def check_disk_space():
    """Check available disk space"""
    print("\n💿 Disk Space Check:")
    
    try:
        import shutil
        
        # Check current directory space
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        print(f"   Free Space: {free_gb:.1f}GB")
        
        if free_gb < 5:
            print("   ⚠️  Less than 5GB free space")
            print("   💡 Model and cache need ~4GB")
            return False
        else:
            print("   ✅ Sufficient disk space")
            return True
            
    except Exception:
        print("   ❌ Cannot check disk space")
        return False

def main():
    """Run all validation checks"""
    print("🍎 Apple Silicon ColPali Pre-Flight Check")
    print("=" * 50)
    
    checks = [
        ("System", check_system),
        ("Python", check_python),
        ("Poppler", check_poppler),
        ("Dependencies", check_dependencies),
        ("PyTorch MPS", check_pytorch_mps),
        ("Test PDF", check_test_pdf),
        ("Memory", check_memory),
        ("Disk Space", check_disk_space),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"   ❌ Error during {name} check: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 Validation Summary:")
    
    passed = 0
    total = len(results)
    critical_failed = []
    
    for name, result in results:
        if result:
            print(f"   ✅ {name}")
            passed += 1
        else:
            print(f"   ❌ {name}")
            if name in ["System", "Python", "Dependencies"]:
                critical_failed.append(name)
    
    print(f"\n📈 Score: {passed}/{total} checks passed")
    
    if critical_failed:
        print(f"\n🚨 Critical failures: {', '.join(critical_failed)}")
        print("   These must be fixed before proceeding.")
        return False
    elif passed == total:
        print("\n🎉 All checks passed! Ready to run ColPali.")
        print("\n🚀 Next steps:")
        print("   1. Run: python test_apple_silicon.py")
        print("   2. If successful: python app_apple_silicon.py")
        return True
    else:
        print("\n⚠️  Some checks failed but system may still work.")
        print("   Try running the tests and see if issues occur.")
        return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
