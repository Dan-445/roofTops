import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from ultralytics import YOLO

class RooftopCornerExtractor:
    def __init__(self, model_path="best.pt"):
        # load YOLO model once during initialization
        self.model = YOLO(model_path)

    def filter_shadow_vegetation(self, image, mask):
        """Remove shadow and vegetation areas from the mask"""
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create shadow mask (low value channel)
        shadow_mask = cv2.inRange(hsv[:,:,2], 0, 80)  # Very dark areas

        # Create vegetation mask (green areas)
        vegetation_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))

        # Combine unwanted areas
        unwanted_mask = cv2.bitwise_or(shadow_mask, vegetation_mask)

        # Remove unwanted areas from roof mask
        cleaned_mask = cv2.bitwise_and(mask, cv2.bitwise_not(unwanted_mask))

        return cleaned_mask

    def filter_by_confidence_and_position(self, results, box_index, result_index):
        """Filter detections based on YOLO confidence and position"""
        r = results[result_index]

        # Check if we have confidence scores
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes.conf) > box_index:
            confidence = float(r.boxes.conf[box_index])
            # Only keep high-confidence detections (roofs should be confident)
            if confidence < 0.70:  # Only keep detections with >70% confidence
                return False

        # Check position - roads are typically at image edges or bottom
        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes.xyxy) > box_index:
            x1, y1, x2, y2 = r.boxes.xyxy[box_index]
            img_height, img_width = r.orig_img.shape[:2]

            # Calculate detection position
            center_y = (y1 + y2) / 2
            detection_height = y2 - y1
            detection_width = x2 - x1

            # Filter out detections that are likely roads:
            # 1. Very close to bottom edge (likely road at bottom)
            if center_y > img_height * 0.85:
                return False

            # 2. Very wide and low (likely horizontal road)
            aspect_ratio = detection_width / detection_height
            if aspect_ratio > 3.0 and center_y > img_height * 0.7:
                return False

        return True

    def preprocess_image_for_rooftops(self, image):
        """Apply image preprocessing to highlight rooftop structures"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply histogram equalization to enhance contrast
        equalized = cv2.equalizeHist(gray)

        # Invert the image - this can make rooftops stand out more
        inverted = cv2.bitwise_not(equalized)

        # Apply stronger sharpening kernel
        kernel = np.array([[-1,-1,-1,-1,-1],
                          [-1,-1,-1,-1,-1],
                          [-1,-1,25,-1,-1],
                          [-1,-1,-1,-1,-1],
                          [-1,-1,-1,-1,-1]]) / 9
        sharpened = cv2.filter2D(inverted, -1, kernel)

        # Apply additional unsharp masking for extra sharpness
        gaussian = cv2.GaussianBlur(sharpened, (0, 0), 2.0)
        unsharp = cv2.addWeighted(sharpened, 2.0, gaussian, -1.0, 0)

        # Convert back to BGR for YOLO
        enhanced = cv2.cvtColor(unsharp, cv2.COLOR_GRAY2BGR)

        return enhanced

    def remove_close_points(self, points, min_distance=10):
        """Remove points that are too close to each other"""
        if len(points) <= 2:
            return points

        cleaned_points = [points[0]]  # Always keep first point

        for i in range(1, len(points)):
            current_point = points[i]
            too_close = False

            # Check distance to all already accepted points
            for accepted_point in cleaned_points:
                distance = np.sqrt((current_point[0] - accepted_point[0])**2 +
                                 (current_point[1] - accepted_point[1])**2)
                if distance < min_distance:
                    too_close = True
                    break

            if not too_close:
                cleaned_points.append(current_point)

        return cleaned_points

    def refine_roof_mask(self, mask):
        """Apply morphological operations to clean up the mask"""
        # Remove small noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)

        # Fill small holes
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_medium)

        # Smooth edges
        mask = cv2.medianBlur(mask, 5)

        return mask

    def extract_rooftop_corners(self, image_path, simplify_factor=0.002, min_area=1000, use_preprocessing=False):
        # Load original image
        original_image = cv2.imread(image_path)

        # Optionally preprocess image to highlight rooftops
        if use_preprocessing:
            processed_image = self.preprocess_image_for_rooftops(original_image)
            # Save preprocessed image temporarily
            temp_path = image_path.replace('.', '_processed.')
            cv2.imwrite(temp_path, processed_image)
            results = self.model(temp_path)
        else:
            results = self.model(image_path)

        all_roofs = {}

        # Use original image for color-based filtering
        image = original_image

        for i, r in enumerate(results):
            if not hasattr(r, 'masks') or r.masks is None:
                continue

            for j, box in enumerate(r.masks.xy):
                # Filter by confidence and position first (before processing)
                if not self.filter_by_confidence_and_position(results, j, i):
                    continue

                poly = Polygon(box)

                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)

                # Convert polygon to integer points
                pts = np.array(poly.exterior.coords, dtype=np.int32)

                # --- Create initial mask ---
                mask = np.zeros((r.orig_img.shape[0], r.orig_img.shape[1]), dtype=np.uint8)
                cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)

                # --- Apply filtering and refinement ---
                # Filter out shadows and vegetation
                cleaned_mask = self.filter_shadow_vegetation(image, mask)

                # Refine the mask with morphological operations
                refined_mask = self.refine_roof_mask(cleaned_mask)

                # Find contours on refined mask
                contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    continue

                # Filter out small contours
                contours = [c for c in contours if cv2.contourArea(c) > min_area]
                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)

                # --- Simplify (reduce number of points) ---
                epsilon = simplify_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # Reshape and remove close points
                polygon_points = approx.reshape(-1, 2).tolist()
                cleaned_points = self.remove_close_points(polygon_points, min_distance=15)

                final_polygon = cleaned_points

                # Store corners
                all_roofs[f"rooftop_{i}_{j}"] = {
                    "corners": final_polygon,
                    "class_id": 0,
                    "area": cv2.contourArea(contour)
                }

        return all_roofs

    def visualize(self, image_path, all_roofs):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for roof in all_roofs.values():
            pts = np.array(roof["corners"], dtype=np.int32)
            cv2.polylines(img_rgb, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

        cv2.imshow("Rooftops", img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()