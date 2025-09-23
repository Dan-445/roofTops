from ultralytics import YOLO
import cv2
import numpy as np
import math
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
import tempfile
import os
import uvicorn

class RoofEdgeDetector:
    def __init__(self, model_path):
        """
        Initialize the roof edge detector with trained YOLO model
        Following the recommended pipeline: AI returns pixels, we convert to lat/lng
        
        Args:
            model_path (str): Path to your trained YOLO11 segmentation model
        """
        self.model = YOLO(model_path)
    
    def get_roof_edge_pixels(self, image_path, confidence_threshold=0.5):
        """
        Step 2: Send image to AI model and get roof edge pixels
        AI model returns polylines/polygons as image pixels in native pixel grid (0,0 = top-left)
        
        Args:
            image_path (str): Path to the satellite image
            confidence_threshold (float): Minimum confidence for detections
            
        Returns:
            list: List of roof detections with pixel coordinates only
        """
        # Run inference on the image
        results = self.model(image_path)
        
        roof_detections = []
        
        for result in results:
            if result.masks is not None:
                boxes = result.boxes
                masks = result.masks.data.cpu().numpy()
                
                for i, (box, mask) in enumerate(zip(boxes.data, masks)):
                    confidence = float(box[4])
                    
                    if confidence >= confidence_threshold:
                        # Extract edge points from mask (pixel coordinates only)
                        edge_pixels = self._extract_roof_edges_from_mask(mask)
                        
                        if len(edge_pixels) >= 3:  # Valid polygon
                            roof_detection = {
                                'roof_id': i,
                                'confidence': confidence,
                                'bbox_pixels': box[:4].tolist(),  # [x1, y1, x2, y2]
                                'edge_pixels': edge_pixels,  # [[x, y], [x, y], ...] in image coordinates
                                'num_edges': len(edge_pixels)
                            }
                            roof_detections.append(roof_detection)
        
        return roof_detections
    
    def _extract_roof_edges_from_mask(self, mask, epsilon_factor=0.015):
        """
        Extract roof edge points from segmentation mask
        Returns pixel coordinates in image's native grid (0,0 = top-left)
        
        Args:
            mask (numpy.ndarray): Binary mask of the roof
            epsilon_factor (float): Douglas-Peucker approximation factor
            
        Returns:
            list: Edge points as [[x, y], [x, y], ...] in pixel coordinates
        """
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Get the largest contour (main roof)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Use Douglas-Peucker algorithm to get key edge points
        epsilon = epsilon_factor * cv2.arcLength(largest_contour, True)
        approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Convert to list of [x, y] coordinates (pixel coordinates)
        edge_pixels = []
        for point in approx_contour:
            x, y = point[0]
            edge_pixels.append([int(x), int(y)])
        
        return edge_pixels
    
    def process_static_map_image(self, image_path, confidence_threshold=0.5):
        """
        Complete pipeline following the recommended approach:
        1. We have static map with known metadata (center, zoom, size, scale)
        2. Send image to AI model → get roof edges as pixel coordinates
        3. Convert pixels to lat/lng using Web Mercator formulas
        
        Args:
            image_path (str): Path to the static map image
            center_lat (float): Center latitude from static map request
            center_lng (float): Center longitude from static map request
            zoom (int): Zoom level from static map request
            img_width (int): Image width from static map request
            img_height (int): Image height from static map request
            scale (int): Scale factor from static map request
            confidence_threshold (float): Minimum confidence for detections
            
        Returns:
            dict: Complete results with both pixel and lat/lng coordinates
        """
        
        print("Step 2: Sending to AI model...")
        # Get roof edge pixels from AI model
        roof_pixels = self.get_roof_edge_pixels(image_path, confidence_threshold)
        print(f"  AI model found {len(roof_pixels)} roofs with edge pixels")
        
        
        return {
            'image_path': image_path,
            'roofs': roof_pixels,
            'total_roofs': len(roof_pixels),
            'pipeline_followed': "AI returns pixels",
            'visualization_url': '/static/roof_edges_visualization.jpg'
        }
    
    def visualize_results(self, image_path, results, output_path=None):
        """
        Visualize the detected roof edges with pixel coordinates
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
            
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, roof in enumerate(results['roofs']):
            color = colors[i % len(colors)]
            edge_pixels = roof['edge_pixels']
            
            # Draw edge points
            for j, (x, y) in enumerate(edge_pixels):
                cv2.circle(image, (x, y), 6, color, -1)
                cv2.putText(image, str(j+1), (x+8, y-8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw polygon connecting edge points
            if len(edge_pixels) > 2:
                pts = np.array(edge_pixels, np.int32)
                cv2.polylines(image, [pts], True, color, 3)
            
            # Add roof info
            if edge_pixels:
                text = f"Roof {roof['roof_id']}: {roof['confidence']:.3f} ({roof['num_edges']} edges)"
                cv2.putText(image, text, (edge_pixels[0][0], edge_pixels[0][1]-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Visualization saved to: {output_path}")
        
        return image
    

app = FastAPI()

detector = RoofEdgeDetector("seg-best.pt")

app.mount("/static", StaticFiles(directory="."), name="static")

@app.post("/process")
async def process_roof(
    image: UploadFile = File(...),
    confidence_threshold: float = Form(0.5)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpeg") as tmp:
        contents = await image.read()
        tmp.write(contents)
        image_path = tmp.name

    try:
        results = detector.process_static_map_image(
            image_path, confidence_threshold
        )
        detector.visualize_results(image_path, results, "roof_edges_visualization.jpg")
        return results
    finally:
        os.unlink(image_path)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
