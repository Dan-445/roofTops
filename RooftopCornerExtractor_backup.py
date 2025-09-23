import cv2
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from ultralytics import YOLO

class RooftopCornerExtractor:
    def __init__(self, model_path="best.pt"):
        # load YOLO model once during initialization
        self.model = YOLO(model_path)

    def extract_rooftop_corners(self, image_path, simplify_factor=0.002):
        results = self.model(image_path)
        all_roofs = {}

        for i, r in enumerate(results):
            for box in r.masks.xy:
                poly = Polygon(box)

                if isinstance(poly, MultiPolygon):
                    poly = max(poly.geoms, key=lambda p: p.area)

                # Convert polygon to integer points
                pts = np.array(poly.exterior.coords, dtype=np.int32)

                # --- Full contour mask ---
                mask = np.zeros((r.orig_img.shape[0], r.orig_img.shape[1]), dtype=np.uint8)
                cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], 255)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)

                # --- Simplify (reduce number of points) ---
                epsilon = simplify_factor * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)

                final_polygon = approx.reshape(-1, 2)

                # Store corners
                all_roofs[f"rooftop_{i}"] = {
                    "corners": final_polygon.tolist(),
                    "class_id": 0
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