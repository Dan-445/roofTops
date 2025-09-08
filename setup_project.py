#!/usr/bin/env python3
"""
Setup script to create all necessary files for the Rooftop Corner Detection API
Run this script in your project directory: python setup_project.py
"""

import os

def create_requirements_txt():
    content = """fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
opencv-python-headless==4.8.1.78
numpy==1.24.3
ultralytics==8.0.200
torch==2.1.0
torchvision==0.16.0
matplotlib==3.7.2
nest-asyncio==1.5.8
Pillow==10.0.1"""
    
    with open('requirements.txt', 'w') as f:
        f.write(content)
    print("✅ Created requirements.txt")

def create_main_py():
    content = '''import fastapi
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))'''
    
    with open('main.py', 'w') as f:
        f.write(content)
    print("✅ Created main.py")

def create_dockerfile():
    content = """FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    libglib2.0-0 \\
    libgtk-3-0 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]"""
    
    with open('Dockerfile', 'w') as f:
        f.write(content)
    print("✅ Created Dockerfile")

def create_gitignore():
    content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Temporary files
temp_*
*.tmp

# Logs
*.log

# Model files (add your model file here if it's large)
# best.pt

# Test images
test_images/
uploads/"""
    
    with open('.gitignore', 'w') as f:
        f.write(content)
    print("✅ Created .gitignore")

def create_readme():
    content = """# Rooftop Corner Detection API 🏠

AI-powered FastAPI service for detecting and extracting corner coordinates from rooftops in images using YOLOv11 and computer vision techniques.

## 🚀 Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```

3. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs

## 📡 API Endpoints

### `POST /extract_corners/`
Upload an image to extract rooftop corner coordinates.

### `GET /health`
Check API health status.

## 🌐 Deployment

### Deploy to Render
1. Push to GitHub
2. Connect repo to Render
3. Use build command: `pip install -r requirements.txt`
4. Use start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## 📋 Requirements
- Python 3.11+
- YOLOv11 model file (best.pt) - optional for demo mode
"""
    
    with open('README.md', 'w') as f:
        f.write(content)
    print("✅ Created README.md")

def main():
    print("🚀 Setting up Rooftop Corner Detection API project...")
    print("=" * 50)
    
    # Create all files
    create_requirements_txt()
    create_main_py()
    create_dockerfile()
    create_gitignore()
    create_readme()
    
    print("\n" + "=" * 50)
    print("✅ All files created successfully!")
    print("\n📋 Next steps:")
    print("1. Copy your model file (best.pt) to this directory (optional)")
    print("2. Test locally: uvicorn main:app --reload")
    print("3. Push to GitHub:")
    print("   git init")
    print("   git add .")
    print("   git commit -m 'Initial commit'")
    print("   git remote add origin YOUR_REPO_URL")
    print("   git push -u origin main")
    print("4. Deploy to Render using the GitHub repo")
    print("\n🎉 Your API is ready for deployment!")

if __name__ == "__main__":
    main()