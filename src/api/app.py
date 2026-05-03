"""
FastAPI application for multimodal fake news detection
"""
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import torch
import numpy as np
from PIL import Image
import io
import json
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal Fake News Detector")

# Global variables
model = None
config = None
device = None


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, config, device
    
    from config import config as cfg
    from src.models.encoders import MultimodalEncoder
    from src.models.fusion import MultimodalFusion
    from src.models.classifier import FakeNewsClassifier, MultimodalFakeNewsModel
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = cfg
    
    # Initialize model
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
    
    # Load checkpoint if exists
    from pathlib import Path
    checkpoint_path = Path(config.MODEL_DIR) / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Model loaded successfully")
    
    model.eval()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/predict")
async def predict(
    text: str = Form(...),
    image: UploadFile = File(...)
) -> Dict:
    """
    Make prediction on multimodal input
    
    Args:
        text: Article text
        image: Image file
    
    Returns:
        Prediction with confidence and explanation
    """
    try:
        # Read and process image
        image_data = await image.read()
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Preprocess
        from src.data.dataset import MultimodalFakeNewsDataset
        
        # Process text
        tokenizer = config.text_encoder.tokenizer if hasattr(config, 'text_encoder') else None
        if not tokenizer:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.text.model_name)
        
        encodings = tokenizer(
            text,
            max_length=config.text.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process image
        import cv2
        import albumentations as A
        
        image_np = np.array(pil_image)
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
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        image_tensor = image_tensor.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, image_tensor)
        
        # Get predictions
        probs = outputs['probs'][0].cpu().numpy()
        confidence = float(np.max(probs))
        prediction = int(np.argmax(probs))
        
        return {
            "prediction": "FAKE" if prediction == 1 else "REAL",
            "confidence": confidence,
            "fake_probability": float(probs[1]),
            "real_probability": float(probs[0]),
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.post("/predict_batch")
async def predict_batch(
    data: Dict
) -> Dict:
    """
    Batch prediction endpoint
    
    Args:
        data: Dictionary with 'texts' and 'images' lists
    
    Returns:
        List of predictions
    """
    try:
        texts = data.get('texts', [])
        # Images would need to be base64 encoded or uploaded differently
        
        predictions = []
        for text in texts:
            # Single prediction per text (simplified)
            pred = await predict(text=text, image=None)
            predictions.append(pred)
        
        return {"predictions": predictions}
    
    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


@app.get("/model_info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "Multimodal Fake News Detector",
        "version": "1.0.0",
        "text_encoder": config.text.model_name,
        "image_encoder": config.image.model_name,
        "fusion_type": config.fusion.fusion_type,
        "device": device,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
