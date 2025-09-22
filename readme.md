# Rooftop Corner Detection API 🏠

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
   ```bash
   uvicorn main:app --loop asyncio
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
