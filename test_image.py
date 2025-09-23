# import cv2
# import os
# import numpy as np
# from RooftopCornerExtractor import RooftopCornerExtractor

# model_path = 'best.pt'
# extractor = RooftopCornerExtractor(model_path)

# # Paths
# image_path = "image.png"
# output_path = "result.png"

# # Extract corners
# corners_data = extractor.extract_rooftop_corners(image_path=image_path)

# # Load original image
# image = cv2.imread(image_path)

# for roof_id, data in corners_data.items():
#     pts = np.array(data["corners"], dtype=np.int32)

#     if pts is None or len(pts) < 3:
#         print(f"{roof_id}: Not enough points for polygon")
#         continue

#     # Draw red corner points
#     for (x, y) in pts:
#         cv2.circle(image, (x, y), 4, (0, 0, 255), -1)

#     # --- ✅ Step 1: Build a mask from points ---
#     mask = np.zeros(image.shape[:2], dtype=np.uint8)
#     cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)

#     # --- ✅ Step 2: Extract clean contour from mask ---
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         continue

#     # Use largest contour (roof area)
#     contour = max(contours, key=cv2.contourArea)

#     # --- ✅ Step 3: Approximate polygon ---
#     epsilon = 0.005 * cv2.arcLength(contour, True)  # smaller epsilon = more precise
#     approx = cv2.approxPolyDP(contour, epsilon, True)

#     # --- ✅ Step 4: Draw final polygon ---
#     cv2.polylines(image, [approx], True, (0, 255, 0), 2)

#     # Replace corners with clean polygon
#     corners_data[roof_id]["corners"] = approx.reshape(-1, 2).tolist()

# # Save result
# cv2.imwrite(output_path, image)
# print(f"Visualization saved at: {output_path}")



import cv2
import os
from RooftopCornerExtractor import RooftopCornerExtractor
import numpy as np

model_path = 'best.pt'
extractor = RooftopCornerExtractor(model_path)

# Path to local image
image_path = "image.png"   # change to your local folder path
output_path = "result.png"

# Extract corners with fewer points (tune simplify_factor)
corners_data = extractor.extract_rooftop_corners(image_path=image_path, simplify_factor=0.002)

# Load original image
image = cv2.imread(image_path)

# Draw polygons and dots
for roof_id, data in corners_data.items():
    pts = data["corners"]

    if pts is not None and len(pts) > 0:
        pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        # Draw polygon (green)
        cv2.polylines(image, [pts_array], True, (0, 255, 0), 2)

        # Draw each corner as red dot
        for (x, y) in pts:
            cv2.circle(image, (x, y), 6, (0, 0, 255), -1)  # red dot

    else:
        print("No points detected, skipping drawing.")

# Save result
cv2.imwrite(output_path, image)
print(f"Visualization saved at: {output_path}")
