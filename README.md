# EdgeVision QC

An AI-powered industrial quality inspection web application designed to enhance product imagery and identify six industrial surface-defect classes using computer vision and a custom CNN.

## Features
- **Defect Detection**: Identifies 6 classes (crazing, inclusion, patches, pitted surface, rolled in scale, scratches).
- **Image Enhancement**: Contrast enhancement and denoising using OpenCV.
- **Pass/Fail Logic**: Automatically flags products.
- **History & Analytics**: Local SQLite storage for traceability and a professional dashboard for statistics.
- **Responsive UI**: Works across desktop, tablet, and mobile.

## Technology Stack
- **Frontend**: HTML5, CSS3, JavaScript (Bootstrap 5)
- **Backend**: Python 3.11, Flask
- **AI**: TensorFlow, Keras, CNN
- **Computer Vision**: OpenCV
- **Database**: SQLite

## Project Architecture
- `app.py`: Flask application routes, database setup, and overall logic.
- `predict.py`: Image preprocessing, inference, and model loading utilities.
- `model/`: Directory containing the Keras model (`model.keras`) and metadata (`class_names.json`, `classification_report.txt`).
- `templates/`: HTML templates for the UI.
- `static/`: CSS and JavaScript files, plus uploaded/enhanced images.
- `database/`: SQLite database storage.

## Installation Instructions

1. **Virtual Environment Setup**:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Requirements Installation**:
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Database Setup**:
   The database is created automatically upon first run.

## How to run
Execute the application from the root directory:
```bash
python app.py
```

## How to use
1. Open your browser and go to `http://127.0.0.1:5000/`.
2. Navigate to the **Dashboard** and upload a product image (JPG/PNG).
3. Click **Analyze Product**.
4. View the resulting classification, confidence score, and Pass/Fail status.
5. Review past inspections on the **History** page.

## CNN Architecture and Pipeline
The AI uses a Convolutional Neural Network trained on the NEU-DET dataset.
**Prediction Pipeline**:
Input Image -> Image Enhancement (OpenCV) -> Preprocessing (224x224, Normalized) -> CNN Model -> Six-Class Classification -> Confidence Score -> Inspection Result -> History Storage.

## Future Enhancements
- Fine-tune model for specific production lines.
- Implement automated retrain loop based on false-positives.
- Integrate active learning for edge cases.
