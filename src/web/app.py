"""
Streamlit web application for multimodal fake news detection
"""
import streamlit as st
import torch
import numpy as np
from PIL import Image
import io
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import config
from src.models.encoders import MultimodalEncoder
from src.models.fusion import MultimodalFusion
from src.models.classifier import FakeNewsClassifier, MultimodalFakeNewsModel
from src.explainability.explainer import MultimodalExplainer
from transformers import AutoTokenizer
import cv2
import albumentations as A

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Multimodal Fake News Detection System")
st.markdown("Detect fake news by analyzing both article text and images")

# Initialize session state
if 'model' not in st.session_state:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    encoders = MultimodalEncoder(config)
    fusion = MultimodalFusion(
        text_dim=config.text.hidden_size,
        image_dim=config.image.hidden_size,
        hidden_dim=config.fusion.hidden_size,
        fusion_type=config.fusion.fusion_type,
    )
    classifier = FakeNewsClassifier(
        input_dim=config.fusion.hidden_size,
        num_classes=2,
        hidden_dim=256,
        dropout=config.fusion.dropout,
    )
    
    model = MultimodalFakeNewsModel(encoders, fusion, classifier)
    model.to(device)
    model.eval()
    
    st.session_state.model = model
    st.session_state.device = device
    st.session_state.tokenizer = AutoTokenizer.from_pretrained(config.text.model_name)
    st.session_state.explainer = MultimodalExplainer(model, device)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    explanation_type = st.multiselect(
        "Explanation Methods",
        ["Text Attention", "Image Saliency", "Fusion Gates"],
        default=["Text Attention", "Image Saliency"]
    )
    
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Article Text")
    article_text = st.text_area(
        "Paste article text here:",
        height=200,
        placeholder="Enter the article text to analyze..."
    )

with col2:
    st.subheader("🖼️ Associated Image")
    uploaded_image = st.file_uploader(
        "Upload image:",
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_image:
        image = Image.open(uploaded_image).convert('RGB')
        st.image(image, use_column_width=True)

# Prediction button
if st.button("🔍 Analyze", type="primary", use_container_width=True):
    if not article_text or not uploaded_image:
        st.error("❌ Please provide both text and image")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Preprocess text
                tokenizer = st.session_state.tokenizer
                encodings = tokenizer(
                    article_text,
                    max_length=config.text.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Preprocess image
                image = Image.open(uploaded_image).convert('RGB')
                image_np = np.array(image)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                
                transforms = A.Compose([
                    A.Resize(config.image.image_size, config.image.image_size),
                    A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ])
                
                augmented = transforms(image=image_np)
                image_tensor = torch.from_numpy(augmented['image']).float().permute(2, 0, 1).unsqueeze(0)
                
                # Move to device
                device = st.session_state.device
                input_ids = encodings['input_ids'].to(device)
                attention_mask = encodings['attention_mask'].to(device)
                image_tensor = image_tensor.to(device)
                
                # Forward pass
                model = st.session_state.model
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask, image_tensor)
                
                # Get predictions
                probs = outputs['probs'][0].cpu().numpy()
                confidence = float(np.max(probs))
                prediction = int(np.argmax(probs))
                
                # Display results
                st.markdown("---")
                
                if prediction == 1:
                    st.error(f"🚨 **LIKELY FAKE NEWS** (Confidence: {confidence:.2%})")
                else:
                    st.success(f"✅ **LIKELY REAL NEWS** (Confidence: {confidence:.2%})")
                
                # Probabilities
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Real Probability", f"{probs[0]:.2%}")
                with col2:
                    st.metric("Fake Probability", f"{probs[1]:.2%}")
                
                # Confidence warning
                if confidence < confidence_threshold:
                    st.warning(f"⚠️ Low confidence prediction (below {confidence_threshold:.0%}). Result should be verified.")
                
                # Explanations
                if explanation_type:
                    st.markdown("---")
                    st.subheader("📊 Explanations")
                    
                    if "Text Attention" in explanation_type:
                        with st.expander("📝 Text Attention Analysis"):
                            explainer = st.session_state.explainer
                            text_attention = explainer.get_text_attention(input_ids, attention_mask, image_tensor)
                            st.info("Text attention visualization shows which parts of the article the model focuses on.")
                            # Placeholder for visualization
                            st.text("Text attention weights extracted (visualization coming)")
                    
                    if "Image Saliency" in explanation_type:
                        with st.expander("🖼️ Image Saliency Analysis"):
                            explainer = st.session_state.explainer
                            image_saliency = explainer.get_image_gradient(input_ids, attention_mask, image_tensor)
                            st.info("Image saliency map highlights important regions in the image.")
                            # Placeholder for visualization
                            st.text("Image saliency map extracted (visualization coming)")
                    
                    if "Fusion Gates" in explanation_type:
                        with st.expander("🔀 Fusion Gates Analysis"):
                            if 'text_gate' in outputs:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Text Gate Weight", f"{outputs['text_gate'].item():.3f}")
                                with col2:
                                    st.metric("Image Gate Weight", f"{outputs['image_gate'].item():.3f}")
                                st.info("Gate weights show the importance of each modality in the final prediction.")
            
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")

# About section
with st.expander("ℹ️ About"):
    st.markdown("""
    ### Multimodal Fake News Detection System
    
    This system uses advanced deep learning to detect fake news by analyzing:
    - **Article Text**: Using RoBERTa transformer
    - **Associated Images**: Using EfficientNet
    - **Multimodal Fusion**: Advanced fusion architecture
    
    **Features:**
    - Binary classification (Real/Fake)
    - Confidence scores
    - Explainability (Attention, Saliency, Gates)
    - Real-time analysis
    
    **Tech Stack:**
    - PyTorch
    - Transformers
    - FastAPI
    - Streamlit
    """)
