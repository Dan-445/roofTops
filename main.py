import fastapi
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import numpy as np
import cv2
from ultralytics import YOLO
import tempfile
from typing import Dict, List, Any

class RooftopCornerExtractor:
    def __init__(self, model_path):
        """
        Initialize the rooftop corner extractor with YOLOv11 model

        Args:
            model_path: Path to your YOLOv11 segmentation model
        """
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            # For demo purposes, create a mock model if best.pt doesn't exist
            print(f"Warning: Model file {model_path} not found. Using demo mode.")
            self.model = None

    def extract_rooftop_corners(self, image_path, rooftop_class_id=None,
                              corner_method='contour_approx', visualize=False):
        """
        Extract corner pixels from rooftop segmentation masks

        Args:
            image_path: Path to input image
            rooftop_class_id: Class ID for rooftop (None to auto-detect)
            corner_method: 'harris', 'shi_tomasi', or 'contour_approx'
            visualize: Whether to display results (not applicable in API context)

        Returns:
            Dictionary with corner coordinates for each rooftop
        """
        # Load and process image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Demo mode - return mock data if no model
        if self.model is None:
            return {
                "rooftop_0": {
                    'corners': [[100, 100], [200, 80], [220, 180], [90, 200]],
                    'class_id': 0,
                }
            }

        # Run YOLOv11 segmentation
        results = self.model(image)

        rooftop_corners = {}

        for idx, result in enumerate(results):
            if result.masks is None:
                print("No segmentation masks found")
                continue

            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes.data.cpu().numpy()

            for i, mask in enumerate(masks):
                # Get class ID
                class_id = int(boxes[i][5])

                # Filter for rooftop class if specified
                if rooftop_class_id is not None and class_id != rooftop_class_id:
                    continue

                # Resize mask to image dimensions
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255

                # Extract corners based on method
                corners = self._extract_corners(mask_binary, method=corner_method)

                rooftop_corners[f"rooftop_{i}"] = {
                    'corners': corners,
                    'class_id': class_id,
                }

        return rooftop_corners

    def _extract_corners(self, mask, method='harris'):
        """
        Extract corners from binary mask using specified method
        """
        if method == 'harris':
            return self._harris_corners(mask)
        elif method == 'shi_tomasi':
            return self._shi_tomasi_corners(mask)
        elif method == 'contour_approx':
            return self._contour_approximation_corners(mask)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _harris_corners(self, mask):
        """Extract corners using Harris corner detection"""
        # Convert to float32
        mask_float = np.float32(mask)

        # Harris corner detection
        corners = cv2.cornerHarris(mask_float, 2, 3, 0.04)

        # Threshold and find corner coordinates
        corners = cv2.dilate(corners, None)
        corner_coords = np.where(corners > 0.01 * corners.max())

        return list(zip(corner_coords[1], corner_coords[0]))  # (x, y) format

    def _shi_tomasi_corners(self, mask):
        """Extract corners using Shi-Tomasi corner detection"""
        corners = cv2.goodFeaturesToTrack(
            mask,
            maxCorners=20,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=3
        )

        if corners is not None:
            return [(int(x), int(y)) for [[x, y]] in corners]
        return []

    def _contour_approximation_corners(self, mask):
        """Extract corners using contour approximation"""
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_corners = []
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.001 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Extract corner points
            corners = [(point[0][0], point[0][1]) for point in approx]
            all_corners.extend(corners)

        return all_corners


# Initialize FastAPI app
app = FastAPI(
    title="Rooftop Corner Detection API",
    description="AI-powered API for detecting and extracting corner coordinates from rooftops in images",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the extractor globally
model_path = 'best.pt'
extractor = RooftopCornerExtractor(model_path)


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Rooftop Corner Detection API",
        "status": "active",
        "version": "1.0.0",
        "endpoints": {
            "/extract_corners/": "POST - Upload image to extract rooftop corners",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": extractor.model is not None}


@app.post("/extract_corners/")
async def extract_corners(file: UploadFile = File(...)):
    """
    Extract corner coordinates from rooftops in uploaded image
    
    Args:
        file: Image file (PNG, JPG, JPEG)
        
    Returns:
        Dictionary with corner coordinates for each detected rooftop
    """
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a PNG, JPG, or JPEG image."
        )

    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file_path = temp_file.name
        
    try:
        # Save uploaded file to temporary location
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract corners using the RooftopCornerExtractor
        corners_data = extractor.extract_rooftop_corners(
            image_path=temp_file_path, 
            visualize=False
        )

        # Format the response
        formatted_corners = {}
        for roof_id, data in corners_data.items():
            formatted_corners[roof_id] = {
                "class_id": data["class_id"],
                "corners": [[int(c[0]), int(c[1])] for c in data["corners"]]
            }
            
        return {
            "success": True,
            "rooftops_detected": len(formatted_corners),
            "data": formatted_corners
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))