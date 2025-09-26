from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import tempfile
import os
from typing import List, Dict

app = FastAPI(title="Rooftop Segmentation API")

# Set device to CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 1. Load pretrained YOLO model
yolo_model = YOLO("updated-roof-segment.pt")

@app.post("/segment-rooftop/")
async def segment_rooftop(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_file_path = tmp_file.name
    
    try:
        # 2. Input rooftop image
        image_path = tmp_file_path
        orig_img = cv2.imread(image_path)
        
        # Check if the image was loaded successfully
        if orig_img is None:
            raise HTTPException(status_code=400, detail=f"Error: Could not load image from {image_path}")
        
        orig_img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # 3. Run YOLO inference for initial mask
        yolo_results = yolo_model(image_path, task="segment", conf=0.55, retina_masks=True)
        
        # 4. Process YOLO masks
        mask = None
        for r in yolo_results:
            masks = r.masks
            if masks is not None and len(masks.data) > 0:
                mask = masks.data[0].cpu().numpy().astype(np.uint8) * 255  # Move to CPU for NumPy
                break
        
        if mask is None:
            raise HTTPException(status_code=400, detail="No valid mask detected by YOLO")
        
        # Smooth YOLO mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 5. Detect corners directly on the refined mask using contours
        corners_list = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = max(contours, key=cv2.contourArea)  # Take the largest contour (main rooftop)
            epsilon = 0.005 * cv2.arcLength(cnt, True)  # Adjust epsilon for smoother approximation
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            
            # Limit to a reasonable number of corners (e.g., 8-12)
            if len(approx) > 12:
                approx = approx[::max(1, len(approx) // 12)]  # Downsample to ~12 points
            
            # Draw lines and dots on the mask's contour
            for i in range(len(approx)):
                p1 = tuple(approx[i][0])
                p2 = tuple(approx[(i + 1) % len(approx)][0])
                cv2.line(orig_img, p1, p2, (0, 255, 0), 2)
                cv2.circle(orig_img, p1, 6, (255, 0, 0), -1)
                cv2.putText(orig_img, f"({p1[0]},{p1[1]})", (p1[0] + 5, p1[1] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Add corner to list
                corners_list.append({
                    "index": i,
                    "x": int(p1[0]),
                    "y": int(p1[1])
                })
        
        # 6. Display results (keeping the visualization logic but not returning it)
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap="gray")
        plt.title("Segmentation Mask")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Corners and Outlines on Rooftop")
        plt.axis("off")
        plt.close()  # Close the plot without showing
        
        # Return JSON response with corners
        return JSONResponse(content={
            "status": "success",
            "corners_count": len(corners_list),
            "corners": corners_list
        })
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

@app.get("/")
async def root():
    return {"message": "Rooftop Segmentation API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
