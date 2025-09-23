from ultralytics import YOLO

# Load model
model = YOLO('best.pt')

# Export with proper settings for segmentation
print("Exporting model to TensorFlow.js format...")
model.export(
    format='tfjs',
    imgsz=640,      # Must be 640 as shown in inference
    half=False,     # No FP16 for TFJS
    simplify=True,  # Simplify the model
    optimize=False  # Don't optimize (can break TFJS)
)

print("\nExport complete!")
print("Files should be in: best_web_model/")
print("  - model.json")
print("  - *.bin files")