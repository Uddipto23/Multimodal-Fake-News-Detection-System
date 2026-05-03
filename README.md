# Multimodal Fake News Detection System

A production-ready deep learning system for detecting fake news by analyzing both article text and associated images using advanced multimodal fusion techniques.

## 🎯 Features

- **Multimodal Analysis**: Combines text (RoBERTa) and image (EfficientNet) encoders
- **Advanced Fusion**: Adaptive fusion with learnable gates and cross-modal attention
- **Explainability**: 6+ explanation techniques (attention, saliency, LIME, SHAP, Grad-CAM)
- **Uncertainty Quantification**: Monte Carlo Dropout for confidence intervals
- **Production Ready**: FastAPI backend + Streamlit web UI
- **Easy Deployment**: Docker support, batch inference, model serving

## 📦 Tech Stack

- **Deep Learning**: PyTorch 2.0+
- **NLP**: Hugging Face Transformers (RoBERTa)
- **Computer Vision**: EfficientNet, OpenCV, Albumentations
- **Web Framework**: FastAPI, Streamlit
- **Explainability**: LIME, SHAP, Grad-CAM
- **Logging**: TensorBoard

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Uddipto23/Multimodal-Fake-News-Detection-System.git
cd Multimodal-Fake-News-Detection-System
pip install -r requirements.txt
