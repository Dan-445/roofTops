# Rooftop Corner Detection API 🏠

AI-powered FastAPI service for detecting and extracting corner coordinates from rooftops in images using YOLOv11 and computer vision techniques.

## 🚀 Features

- **AI-Powered Detection**: Uses YOLOv11 deep learning model for accurate rooftop segmentation
- **Multiple Corner Detection Methods**: Harris, Shi-Tomasi, and Contour Approximation algorithms
- **RESTful API**: Easy-to-use HTTP endpoints with comprehensive documentation
- **Real-time Processing**: Fast image processing and corner extraction
- **Production Ready**: Containerized with Docker, deployable on cloud platforms

## 🛠️ Technology Stack

- **FastAPI**: Modern, fast web framework for Python APIs
- **YOLOv11**: State-of-the-art object detection and segmentation
- **OpenCV**: Computer vision library for image processing
- **NumPy**: Numerical computing for efficient array operations
- **Docker**: Containerization for easy deployment

## 📋 Requirements

- Python 3.11+
- YOLOv11 trained model (`best.pt`)
- Docker (for containerized deployment)

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/rooftop-corner-detection-api.git
   cd rooftop-corner-detection-api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your trained model**
   - Place your YOLOv11 model file as `best.pt` in the root directory
   - Or update the `model_path` in `main.py`

4. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```

5. **Access the API**
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t rooftop-corner-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 rooftop-corner-api
   ```

## 📡 API Endpoints

### `POST /extract_corners/`
Extract corner coordinates from rooftops in uploaded images.

**Request**: Upload image file (PNG, JPG, JPEG)

**Response**:
```json
{
  "success": true,
  "rooftops_detected": 2,
  "data": {
    "rooftop_0": {
      "class_id": 0,
      "corners": [[150, 200], [300, 180], [320, 350], [140, 370]]
    },
    "rooftop_1": {
      "class_id": 0,
      "corners": [[450, 120], [600, 100], [620, 250], [430, 270]]
    }
  }
}
```

### `GET /health`
Health check endpoint

### `GET /`
API information and available endpoints

## 🧪 Testing

### Using curl
```bash
curl -X POST "http://localhost:8000/extract_corners/" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

### Using Python
```python
import requests

url = "http://localhost:8000/extract_corners/"
files = {"file": open("your_image.jpg", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

## 🌐 Deployment

### Deploy to Render

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Render**
   - Go to [Render.com](https://render.com)
   - Create new Web Service
   - Connect your GitHub repository
   - Use these settings:
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Deploy to Other Platforms

- **Heroku**: Use `Procfile` with `web: uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Railway**: Connect GitHub repo, Railway auto-detects Python app
- **Google Cloud Run**: Deploy using Docker image
- **AWS Lambda**: Use Mangum adapter for serverless deployment

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 8000)
- `MODEL_PATH`: Path to YOLOv11 model file (default: best.pt)

### Corner Detection Methods
- `harris`: Harris Corner Detection
- `shi_tomasi`: Shi-Tomasi Corner Detection
- `contour_approx`: Contour Approximation (default)

## 📊 Use Cases

- **Architecture & Construction**: Building measurement and analysis
- **Real Estate**: Property assessment automation
- **Solar Panel Installation**: Rooftop area calculation
- **Insurance**: Damage assessment
- **Urban Planning**: City development analysis
- **Drone/Satellite Imagery**: Automated building detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Issues & Support

If you encounter any issues or need support:
- Open an issue on GitHub
- Check the [API documentation](http://localhost:8000/docs)
- Review the health check endpoint for system status

## 📈 Roadmap

- [ ] Batch image processing
- [ ] Real-time video processing
- [ ] Additional corner detection algorithms
- [ ] Model training pipeline
- [ ] Performance optimization
- [ ] WebSocket support for real-time updates