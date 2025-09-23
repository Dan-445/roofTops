from ultralytics import YOLO
import cv2

# Load model
model = YOLO('best.pt')

# Use an actual rooftop image
img_path = 'staticmap.png'  # Use the same image that works in your Python API
results = model(img_path)

print(f"Testing on: {img_path}")
for r in results:
    print(f"Number of detections: {len(r.boxes) if r.boxes else 0}")
    if r.masks:
        print(f"Masks shape: {r.masks.data.shape}")
        print(f"Masks dtype: {r.masks.data.dtype}")
    if r.boxes:
        print(f"Boxes shape: {r.boxes.data.shape}")
        for i, box in enumerate(r.boxes.data):
            print(f"  Detection {i}: confidence={box[4]:.3f}, class={int(box[5])}")