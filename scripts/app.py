import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

MODEL_PATH = "helmet_detector_yolo11s_v2.pt"


def apply_custom_styles():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #0d2137 100%);
        border-right: 1px solid #2dd4bf33;
    }
    
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #2dd4bf !important;
    }
    
    .main-header {
        background: linear-gradient(90deg, #0d9488 0%, #2dd4bf 50%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .section-header {
        color: #2dd4bf;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #2dd4bf33;
    }
    
    .custom-card {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #2dd4bf33;
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(45, 212, 191, 0.1);
    }
    
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #2dd4bf33;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 15px rgba(45, 212, 191, 0.08);
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #2dd4bf !important;
        font-weight: 700;
    }
    
    [data-testid="stFileUploader"] {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border: 2px dashed #2dd4bf55;
        border-radius: 16px;
        padding: 2rem;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #2dd4bf;
        box-shadow: 0 0 20px rgba(45, 212, 191, 0.2);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #0d9488 0%, #2dd4bf 100%);
        color: #0f172a;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(45, 212, 191, 0.4);
    }
    
    [data-testid="stRadio"] label {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stRadio"] label:hover {
        color: #2dd4bf !important;
    }
    
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #2dd4bf33;
    }
    
    .stAlert {
        background: linear-gradient(145deg, #1e293b 0%, #0f172a 100%);
        border-left: 4px solid #2dd4bf;
        border-radius: 8px;
    }
    
    .stSpinner > div {
        border-top-color: #2dd4bf !important;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0;
        color: #e2e8f0;
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    .legend-dot.green {
        background: #22c55e;
        box-shadow: 0 0 8px #22c55e88;
    }
    
    .legend-dot.red {
        background: #ef4444;
        box-shadow: 0 0 8px #ef444488;
    }
    
    hr {
        border-color: #2dd4bf33;
        margin: 1.5rem 0;
    }
    
    [data-testid="column"] {
        padding: 0 0.5rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = YOLO(MODEL_PATH)
    return model


def draw_detections(image, results):
    annotated_image = image.copy()
    
    img_height, img_width = annotated_image.shape[:2]
    font_scale = max(0.35, min(img_width, img_height) / 1200)
    thickness = max(1, int(font_scale * 2))
    box_thickness = max(1, int(font_scale * 3))
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = result.names[cls]
            
            if class_name.lower() in ['helmet', 'with_helmet', 'wearing_helmet']:
                color = (0, 255, 0)
                label = f"Helmet:{conf:.0%}"
            else:
                color = (0, 0, 255)
                label = f"No Helmet:{conf:.0%}"
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, box_thickness)
            
            label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            label_padding = 4
            
            label_y = y1 + label_size[1] + label_padding * 2
            
            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x1 + label_size[0] + label_padding * 2, label_y),
                color,
                -1
            )
            
            cv2.putText(
                annotated_image,
                label,
                (x1 + label_padding, y1 + label_size[1] + label_padding),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
    
    return annotated_image


class HelmetVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_model()
    
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        results = self.model(img, verbose=False)
        
        annotated_frame = draw_detections(img, results)
        
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")


def main():
    st.set_page_config(
        page_title="Helmet Detection System",
        layout="wide"
    )
    
    apply_custom_styles()
    
    st.markdown('<h1 class="main-header">Helmet Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Helmet detection using YOLOv11 for images and real time video</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## Detection Mode")
    mode = st.sidebar.radio(
        "Select Mode:",
        ["Image Upload", "Real Time Camera"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Detection Legend")
    st.sidebar.markdown("""
    <div class="legend-item">
        <div class="legend-dot green"></div>
        <span>Helmet Detected</span>
    </div>
    <div class="legend-item">
        <div class="legend-dot red"></div>
        <span>No Helmet Detected</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6;">
    This system uses YOLOv11 for accurate helmet detection in workplace safety monitoring.
    </div>
    """, unsafe_allow_html=True)
    
    if mode == "Image Upload":
        st.markdown('<h2 class="section-header">Image Upload Detection</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: #94a3b8;">Upload an image (JPG/PNG) to analyze helmet compliance.</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<p style="color: #2dd4bf; font-weight: 600; margin-bottom: 0.5rem;">Original Image</p>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            
            with st.spinner("Analyzing image..."):
                model = load_model()
                results = model(image_bgr, verbose=False)
                annotated_image = draw_detections(image_bgr, results)
                
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.markdown('<p style="color: #2dd4bf; font-weight: 600; margin-bottom: 0.5rem;">Detection Results</p>', unsafe_allow_html=True)
                st.image(annotated_image_rgb, use_container_width=True)
            
            st.markdown("---")
            st.markdown('<h2 class="section-header">Detection Statistics</h2>', unsafe_allow_html=True)
            
            helmet_count = 0
            no_helmet_count = 0
            
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    class_name = result.names[cls]
                    if class_name.lower() in ['helmet', 'with_helmet', 'wearing_helmet']:
                        helmet_count += 1
                    else:
                        no_helmet_count += 1
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            col_stat1.metric("Helmets Detected", helmet_count)
            col_stat2.metric("No Helmet", no_helmet_count)
            col_stat3.metric("Total Detections", helmet_count + no_helmet_count)
    
    else:
        st.markdown('<h2 class="section-header">Real time Camera Detection</h2>', unsafe_allow_html=True)
        st.markdown('<p style="color: #94a3b8;">Enable your webcam to detect helmets in real time.</p>', unsafe_allow_html=True)
        
        st.warning("Make sure to allow camera access when prompted by your browser.")
        
        webrtc_streamer(
            key="helmet detection",
            video_processor_factory=HelmetVideoProcessor,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": True,
                "audio": False
            },
        )
        
        st.info("Click 'START' to begin the webcam feed and helmet detection.")


if __name__ == "__main__":
    main()
