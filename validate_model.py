from ultralytics import YOLO
import cv2
import numpy as np

# Load your model
model = YOLO('best.pt')

# Check model type
print("=" * 50)
print("MODEL INFO:")
print(f"Task type: {model.task}")
print(f"Model type: {model.type}")
print(f"Number of classes: {len(model.names)}")
print(f"Class names: {model.names}")

# Test on an image to see output structure
img = cv2.imread('your_test_image.jpg')  # Use same test image
results = model(img)

print("\n" + "=" * 50)
print("INFERENCE OUTPUTS:")
for r in results:
    print(f"Has masks: {r.masks is not None}")
    if r.masks:
        print(f"Masks shape: {r.masks.data.shape}")
        print(f"Masks dtype: {r.masks.data.dtype}")
    print(f"Boxes shape: {r.boxes.data.shape}")
    print(f"Number of detections: {len(r.boxes)}")