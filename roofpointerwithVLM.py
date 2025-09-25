import cv2
import numpy as np
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Set device to CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. Load pretrained models once (outside the loop)
yolo_model = YOLO("runs/segment/train/weights/best.pt")
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
owlvit_model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16").to(device)

# 2. List of image paths to process
image_paths = ["/content/121.jpg"]  # Add more paths like ["/content/121.jpg", "/content/122.jpg"] if needed

for image_path in image_paths:
    # 3. Input rooftop image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print(f"Failed to load image: {image_path}")
        continue
    orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

    # 4. Run YOLO inference for initial mask
    yolo_results = yolo_model(image_path, task="segment", conf=0.55, retina_masks=True)

    # 5. Process YOLO masks
    mask = None
    for r in yolo_results:
        masks = r.masks
        if masks is not None and len(masks.data) > 0:
            mask = masks.data[0].cpu().numpy().astype(np.uint8) * 255  # Move to CPU for NumPy
            break

    if mask is None:
        print(f"No valid mask detected by YOLO for {image_path}")
        continue

    # Smooth YOLO mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 6. Run OWL-ViT for refinement with combined prompt
    combined_prompts = ["rooftop outline, building roof edges, top-down view", "house roof top-view, exclude trees vegetation pool driveway"]
    inputs = processor(text=combined_prompts, images=orig_img_rgb, return_tensors="pt").to(device)
    outputs = owlvit_model(**inputs)
    target_sizes = torch.Tensor([orig_img_rgb.shape[:2]]).to(device)
    results = processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes, threshold=0.02)[0]

    # 7. Extract OWL-ViT bounding box to refine mask
    refined_mask = mask  # Default to YOLO mask
    if 'scores' in results and results['scores'].numel() > 0:
        scores_np = results['scores'].detach().cpu().numpy()
        if scores_np.max() > 0.02:
            best_idx = np.argmax(scores_np)
            box = results['boxes'][best_idx].detach().cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            refined_mask = np.zeros_like(mask)
            refined_mask[y1:y2, x1:x2] = mask[y1:y2, x1:x2]  # Apply YOLO mask within OWL-ViT bounds
    else:
        print(f"OWL-ViT detected no valid objects for {image_path}; using YOLO mask.")

    # 8. Detect corners directly on the refined mask using contours
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)  # Take the largest contour (main rooftop)
        epsilon = 0.005 * cv2.arcLength(cnt, True)  # Adjust epsilon for smoother approximation
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Limit to a reasonable number of corners (e.g., 8-12)
        if len(approx) > 12:
            approx = approx[::max(1, len(approx) // 12)]  # Downsample to ~12 points

        # Print pixel coordinates of detected corners
        print(f"Pixel coordinates for {image_path}:")
        for i, point in enumerate(approx):
            x, y = point[0]
            print(f"Corner {i + 1}: ({x}, {y})")

        # Draw lines and dots on the mask's contour
        for i in range(len(approx)):
            p1 = tuple(approx[i][0])
            p2 = tuple(approx[(i + 1) % len(approx)][0])
            cv2.line(orig_img, p1, p2, (0, 255, 0), 2)
            cv2.circle(orig_img, p1, 6, (255, 0, 0), -1)
            cv2.putText(orig_img, f"({p1[0]},{p1[1]})", (p1[0] + 5, p1[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # 9. Display results for this image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(refined_mask, cmap="gray")
    plt.title(f"Refined Segmentation Mask - {image_path}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Detected Corners and Outlines - {image_path}")
    plt.axis("off")
    plt.show()

print("Processing complete for all images.")
