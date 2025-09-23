from fastapi import FastAPI, File, UploadFile, HTTPException
import uvicorn
import shutil
import os
import threading
import time
import asyncio
import nest_asyncio
from RooftopCornerExtractor import RooftopCornerExtractor

# Apply nest_asyncio to allow running asyncio event loop in a thread
nest_asyncio.apply()

# class RooftopCornerExtractor:
#     def __init__(self, model_path):
#         """
#         Initialize the rooftop corner extractor with YOLOv11 model

#         Args:
#             model_path: Path to your YOLOv11 segmentation model
#         """
#         self.model = YOLO(model_path)

#     def extract_rooftop_corners(self, image_path, rooftop_class_id=None,
#                               corner_method='contour_approx', visualize=False):
#         """
#         Extract corner pixels from rooftop segmentation masks

#         Args:
#             image_path: Path to input image
#             rooftop_class_id: Class ID for rooftop (None to auto-detect)
#             corner_method: 'harris', 'shi_tomasi', or 'contour_approx'
#             visualize: Whether to display results (not applicable in API context)

#         Returns:
#             Dictionary with corner coordinates for each rooftop
#         """
#         # Load and process image
#         image = cv2.imread(image_path)
#         if image is None:
#             raise ValueError(f"Could not load image: {image_path}")

#         # Run YOLOv11 segmentation
#         results = self.model(image)

#         rooftop_corners = {}

#         for idx, result in enumerate(results):
#             if result.masks is None:
#                 print("No segmentation masks found")
#                 continue

#             masks = result.masks.data.cpu().numpy()
#             boxes = result.boxes.data.cpu().numpy()

#             for i, mask in enumerate(masks):
#                 # Get class ID
#                 class_id = int(boxes[i][5])

#                 # Filter for rooftop class if specified
#                 if rooftop_class_id is not None and class_id != rooftop_class_id:
#                     continue

#                 # Resize mask to image dimensions
#                 mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
#                 mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

#                 # Extract corners based on method
#                 corners = self._extract_corners(mask_binary, method=corner_method)

#                 rooftop_corners[f"rooftop_{i}"] = {
#                     'corners': corners,
#                     'class_id': class_id,
#                     'mask': mask_binary # Mask is not typically returned in API response
#                 }

#         # Visualization is skipped in the API context
#         # if visualize:
#         #     self._visualize_results(image, rooftop_corners)

#         return rooftop_corners

#     def _extract_corners(self, mask, method='harris'):
#         """
#         Extract corners from binary mask using specified method
#         """
#         if method == 'harris':
#             return self._harris_corners(mask)
#         elif method == 'shi_tomasi':
#             return self._shi_tomasi_corners(mask)
#         elif method == 'contour_approx':
#             return self._contour_approximation_corners(mask)
#         else:
#             raise ValueError(f"Unknown method: {method}")

#     def _harris_corners(self, mask):
#         """Extract corners using Harris corner detection"""
#         # Convert to float32
#         mask_float = np.float32(mask)

#         # Harris corner detection
#         corners = cv2.cornerHarris(mask_float, 2, 3, 0.04)

#         # Threshold and find corner coordinates
#         corners = cv2.dilate(corners, None)
#         corner_coords = np.where(corners > 0.01 * corners.max())

#         return list(zip(corner_coords[1], corner_coords[0]))  # (x, y) format

#     def _shi_tomasi_corners(self, mask):
#         """Extract corners using Shi-Tomasi corner detection"""
#         corners = cv2.goodFeaturesToTrack(
#             mask,
#             maxCorners=20,
#             qualityLevel=0.01,
#             minDistance=10,
#             blockSize=3
#         )

#         if corners is not None:
#             return [(int(x), int(y)) for [[x, y]] in corners]
#         return []

#     def _contour_approximation_corners(self, mask):
#         """Extract corners using contour approximation"""
#         # Find contours
#         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#         all_corners = []
#         for contour in contours:
#             # Approximate contour to polygon
#             epsilon = 0.001 * cv2.arcLength(contour, True) # Epsilon value from last successful run
#             approx = cv2.approxPolyDP(contour, epsilon, True)

#             # Extract corner points
#             corners = [(point[0][0], point[0][1]) for point in approx]
#             all_corners.extend(corners)

#         return all_corners

#     # Visualization method is not included in the API script
#     # def _visualize_results(self, image, rooftop_corners):
#     #     """Visualize the extracted corners on the image"""
#     #     fig, axes = plt.subplots(1, 2, figsize=(15, 7))
#     #     # ... visualization code ...
#     #     plt.show()

#     # Save method is not included in the API script
#     # def save_corner_coordinates(self, rooftop_corners, output_file):
#     #     """Save corner coordinates to file"""
#     #     # ... save code ...
#     #     pass


app = FastAPI()

# Initialize the extractor globally
# Update this path to your trained model file
model_path = 'best.pt'
extractor = RooftopCornerExtractor(model_path)


@app.post("/extract_corners/")
async def extract_corners(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PNG, JPG, or JPEG image.")

    # Create a temporary file to save the uploaded image
    temp_file_path = f"temp_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Integrate the RooftopCornerExtractor here
        corners_data = extractor.extract_rooftop_corners(image_path=temp_file_path, visualize=False)

        # Format the corners_data into the desired JSON structure
        formatted_corners = {}
        for roof_id, data in corners_data.items():
             formatted_corners[roof_id] = {
                 "class_id": data["class_id"],
                 "corners": [[int(c[0]), int(c[1])] for c in data["corners"]] # Ensure corners are list of integers
             }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {e}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return formatted_corners


# Fixed async function to run FastAPI
async def run_fastapi_async():
    config = uvicorn.Config(app, host="0.0.0.0", port=10000)
    server = uvicorn.Server(config)
    await server.serve()


def run_fastapi():
    """Run FastAPI server in a thread with proper asyncio event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_fastapi_async())


if __name__ == "__main__":
    print("Starting FastAPI app...")
    # Run FastAPI in a separate thread
    thread = threading.Thread(target=run_fastapi)
    thread.daemon = True  # Make thread daemon so it exits when main thread exits
    thread.start()

    print("FastAPI app is running. You can access it at:")
    print("http://localhost:8000/docs for the OpenAPI documentation.")
    print("http://localhost:8000/extract_corners/ to test the endpoint.")

    # Keep the main thread alive to prevent the script from exiting
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("FastAPI app stopped.")
        pass
