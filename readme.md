# Helmet Detection System

A Streamlit-based application for detecting helmets in images and real-time video streams using YOLO object detection.

## Features

- **Image Upload Mode**: Upload JPG/PNG images for helmet detection with side-by-side comparison
- **Real-Time Camera Mode**: Live webcam detection with bounding box overlays
- **Detection Statistics**: View counts of detected helmets and violations
- **Adjustable Confidence**: Configure detection sensitivity via sidebar slider

## Prerequisites

- Python 3.8 or higher
- A trained YOLO model file (`.pt` format) for helmet detection
- Webcam (optional, for real-time detection)

## Installation

1. **Clone or download the project**

2. **Install dependencies**

   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Configure the model path**

   Open `app.py` and update the `MODEL_PATH` variable at the top of the file:

   \`\`\`python
   MODEL_PATH = "path/to/your/helmet_model.pt"
   \`\`\`

## Usage

1. **Run the Streamlit app**

   \`\`\`bash
   cd scripts
   streamlit run app.py
   \`\`\`

2. **Open your browser**

   The app will automatically open at `http://localhost:8501`

3. **Select detection mode**

   - **Upload Image**: Click "Browse files" to upload an image for analysis
   - **Real-Time Camera**: Click "START" to begin webcam detection

## Configuration

| Setting | Description | Default |
|---------|-------------|---------|
| `MODEL_PATH` | Path to YOLO model file | `best.pt` |
| Confidence Threshold | Minimum detection confidence (adjustable in sidebar) | 0.5 |

## Detection Classes

- **Helmet** (Green box): Person wearing a helmet
- **No Helmet** (Red box): Person not wearing a helmet

## Troubleshooting

- **Model not loading**: Verify `MODEL_PATH` points to a valid `.pt` file
- **Webcam not working**: Ensure browser has camera permissions enabled
- **Slow performance**: Lower the confidence threshold or use a smaller YOLO model

## Requirements

See `requirements.txt` for full dependency list:

- streamlit
- ultralytics
- opencv-python
- numpy
- Pillow
- streamlit-webrtc