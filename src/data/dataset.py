"""
Custom Dataset class for multimodal fake news detection
"""
import os
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import cv2
import albumentations as A


class MultimodalFakeNewsDataset(Dataset):
    """
    Custom dataset for multimodal fake news detection
    
    Expected data structure:
    data/
    ├── train/
    │   ├── images/
    │   │   └── *.jpg
    │   ├── articles.jsonl
    └── test/
        ├── images/
        │   └── *.jpg
        └── articles.jsonl
    
    JSONL format:
    {"id": "1", "text": "article text", "image_id": "1", "label": 0}
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        text_config=None,
        image_config=None,
        augment: bool = True,
    ):
        """
        Args:
            data_dir: Root data directory
            split: 'train', 'val', or 'test'
            text_config: Text configuration
            image_config: Image configuration
            augment: Whether to apply augmentations
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.text_config = text_config
        self.image_config = image_config
        self.augment = augment and split == "train"
        
        # Initialize tokenizer
        model_name = text_config.model_name if text_config else "roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load data
        self.samples = self._load_samples()
        
        # Image augmentations
        self.train_transforms = self._get_train_transforms()
        self.val_transforms = self._get_val_transforms()
        
    def _load_samples(self) -> list:
        """Load samples from JSONL file"""
        samples = []
        jsonl_file = self.data_dir / "articles.jsonl"
        
        if not jsonl_file.exists():
            print(f"Warning: {jsonl_file} not found. Dataset will be empty.")
            return samples
        
        with open(jsonl_file, 'r') as f:
            for line in f:
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError:
                    continue
        
        return samples
    
    def _get_train_transforms(self):
        """Get training augmentations"""
        return A.Compose([
            A.RandomResizedCrop(
                self.image_config.image_size,
                self.image_config.image_size,
                scale=(0.8, 1.0),
                p=0.8
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=15, p=0.5),
            A.GaussNoise(p=0.2),
            A.GaussianBlur(p=0.2),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ], bbox_params=None)
    
    def _get_val_transforms(self):
        """Get validation augmentations (minimal)"""
        return A.Compose([
            A.Resize(
                self.image_config.image_size,
                self.image_config.image_size
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single sample"""
        sample = self.samples[idx]
        
        # Process text
        text = sample.get('text', '')
        text_encodings = self._process_text(text)
        
        # Process image
        image_path = self.data_dir / "images" / f"{sample['image_id']}.jpg"
        image_tensor = self._process_image(image_path)
        
        # Get label
        label = torch.tensor(sample.get('label', 0), dtype=torch.long)
        
        return {
            'input_ids': text_encodings['input_ids'],
            'attention_mask': text_encodings['attention_mask'],
            'image': image_tensor,
            'label': label,
            'sample_id': sample.get('id', idx),
        }
    
    def _process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Process and tokenize text"""
        encodings = self.tokenizer(
            text,
            max_length=self.text_config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(0),
            'attention_mask': encodings['attention_mask'].squeeze(0),
        }
    
    def _process_image(self, image_path: Path) -> torch.Tensor:
        """Process and augment image"""
        try:
            # Read image
            image = cv2.imread(str(image_path))
            if image is None:
                # Return black image if file doesn't exist
                image = np.zeros(
                    (self.image_config.image_size, self.image_config.image_size, 3),
                    dtype=np.uint8
                )
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply transforms
            if self.augment:
                augmented = self.train_transforms(image=image)
            else:
                augmented = self.val_transforms(image=image)
            
            image = augmented['image']
            
            # Convert to tensor
            image_tensor = torch.from_numpy(image).float()
            # Change from HxWxC to CxHxW
            image_tensor = image_tensor.permute(2, 0, 1)
            
            return image_tensor
        
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            # Return black tensor
            return torch.zeros(
                (3, self.image_config.image_size, self.image_config.image_size),
                dtype=torch.float32
            )


class FakeNewsDataLoader:
    """Helper class for creating data loaders"""
    
    @staticmethod
    def create_dataloaders(
        data_dir: str,
        text_config,
        image_config,
        batch_size: int = 16,
        num_workers: int = 4,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Create train, val, and test data loaders"""
        
        train_dataset = MultimodalFakeNewsDataset(
            data_dir,
            split='train',
            text_config=text_config,
            image_config=image_config,
            augment=True,
        )
        
        val_dataset = MultimodalFakeNewsDataset(
            data_dir,
            split='val',
            text_config=text_config,
            image_config=image_config,
            augment=False,
        )
        
        test_dataset = MultimodalFakeNewsDataset(
            data_dir,
            split='test',
            text_config=text_config,
            image_config=image_config,
            augment=False,
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        return train_loader, val_loader, test_loader
