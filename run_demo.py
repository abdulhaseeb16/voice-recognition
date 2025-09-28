#!/usr/bin/env python3
"""
Voice Emotion AI - Demo Runner
Enhanced UI with modern design and interactive features
"""

import subprocess
import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'plotly', 'tensorflow', 'scikit-learn', 
        'pandas', 'numpy', 'soundfile', 'joblib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Install with: pip install -r requirements_full.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def run_streamlit_app():
    """Run the Streamlit application"""
    app_path = os.path.join("streamlit_app", "app_full.py")
    
    if not os.path.exists(app_path):
        print(f"❌ App file not found: {app_path}")
        return False
    
    print("🚀 Starting Voice Emotion AI...")
    print("🌐 The app will open in your default browser")
    print("📱 Access the app at: http://localhost:8501")
    print("\n🎯 New UI Features:")
    print("  • Modern gradient design")
    print("  • Interactive charts with Plotly")
    print("  • Enhanced emotion cards")
    print("  • Real-time confidence visualization")
    print("  • Improved audio analysis")
    print("  • About section with project details")
    print("\n⏹️  Press Ctrl+C to stop the application")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🎭 Voice Emotion AI - Enhanced UI Demo")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return
    
    # Check if models exist
    models_dir = "models"
    if not os.path.exists(models_dir):
        print("⚠️  Models directory not found!")
        print("🔧 Run the training pipeline first:")
        print("   python src/full_pipeline.py")
        print("\n🎯 You can still explore the UI, but predictions won't work")
        input("Press Enter to continue anyway...")
    
    # Run the app
    run_streamlit_app()

if __name__ == "__main__":
    main()