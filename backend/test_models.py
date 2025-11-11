"""
Test script to verify model loading and paths
Run this from the backend directory to check if models can be loaded
"""
import sys
from pathlib import Path

# Add parent directory to path (same as main.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("Testing Model Loading")
print("=" * 60)

# Test paths
backend_dir = Path(__file__).parent
models_dir = backend_dir.parent / "models" / "audio" / "cough" / "models"
preproc_dir = backend_dir.parent / "models" / "audio" / "cough" / "preprocessor"

print(f"\nBackend directory: {backend_dir}")
print(f"Models directory: {models_dir}")
print(f"Models directory exists: {models_dir.exists()}")
print(f"Preprocessor directory: {preproc_dir}")
print(f"Preprocessor directory exists: {preproc_dir.exists()}")

# Check model files
yamnet_keras = models_dir / "yamnet_88.keras"
yamnet_saved = models_dir / "yamnet_88_savedmodel"
preprocessor = preproc_dir / "preprocessor_saved.pkl"

print(f"\nYAMNet Keras file exists: {yamnet_keras.exists()}")
print(f"YAMNet SavedModel exists: {yamnet_saved.exists()}")
print(f"Preprocessor file exists: {preprocessor.exists()}")

# Try loading models
print("\n" + "=" * 60)
print("Attempting to load models...")
print("=" * 60)

try:
    import tensorflow as tf
    print("\n✓ TensorFlow imported successfully")
    print(f"  TensorFlow version: {tf.__version__}")
except Exception as e:
    print(f"\n✗ Failed to import TensorFlow: {e}")
    sys.exit(1)

try:
    from keras.models import load_model as keras_load_model
    print("✓ Keras imported successfully")
except Exception as e:
    print(f"✗ Failed to import Keras: {e}")

try:
    import joblib
    print("✓ Joblib imported successfully")
except Exception as e:
    print(f"✗ Failed to import Joblib: {e}")
    sys.exit(1)

# Load YAMNet SavedModel
print("\n1. Loading YAMNet SavedModel...")
try:
    yamnet_model = tf.saved_model.load(str(yamnet_saved))
    print("   ✓ YAMNet SavedModel loaded successfully!")
except Exception as e:
    print(f"   ✗ Failed to load YAMNet SavedModel: {e}")
    yamnet_model = None

# Load Cough Classifier
print("\n2. Loading Cough Classifier...")
try:
    cough_classifier = keras_load_model(str(yamnet_keras))
    print("   ✓ Cough classifier loaded successfully!")
except Exception as e1:
    print(f"   ✗ Keras load failed: {e1}")
    try:
        cough_classifier = tf.keras.models.load_model(str(yamnet_keras), compile=False)
        print("   ✓ Cough classifier loaded successfully (TF method)!")
    except Exception as e2:
        print(f"   ✗ TF load also failed: {e2}")
        cough_classifier = None

# Load Preprocessor
print("\n3. Loading Preprocessor...")
try:
    preprocessor = joblib.load(str(preprocessor))
    print("   ✓ Preprocessor loaded successfully!")
except Exception as e:
    print(f"   ✗ Failed to load preprocessor: {e}")
    preprocessor = None

# Summary
print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"YAMNet Model: {'✓ Loaded' if yamnet_model else '✗ Failed'}")
print(f"Cough Classifier: {'✓ Loaded' if cough_classifier else '✗ Failed'}")
print(f"Preprocessor: {'✓ Loaded' if preprocessor else '✗ Failed'}")

if yamnet_model and cough_classifier and preprocessor:
    print("\n✓ All models loaded successfully! Predictions should work.")
else:
    print("\n✗ Some models failed to load. Check the errors above.")
    sys.exit(1)
