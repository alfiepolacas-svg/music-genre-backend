"""
Training schemas
"""
from pydantic import BaseModel
from typing import Dict, List, Optional

class TrainingConfig(BaseModel):
    """Training configuration"""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2

class TrainingStatus(BaseModel):
    """Training status response"""
    status: str
    epoch: int
    total_epochs: int
    loss: float
    accuracy: float
    val_loss: float
    val_accuracy: float