"""
Transformer-based Market Prediction Model

This module implements a Transformer architecture for multi-variate time series
prediction with attention mechanisms for better long-range dependency capture.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
import logging
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerPredictor(nn.Module):
    """Transformer model for time series prediction"""
    
    def __init__(self, input_dim: int, d_model: int = 512, nhead: int = 8,
                 num_encoder_layers: int = 6, num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 prediction_length: int = 5, max_seq_length: int = 100):
        super(TransformerPredictor, self).__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        self.prediction_length = prediction_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        
        # Output layers
        self.decoder_projection = nn.Linear(d_model, d_model // 2)
        self.output_projection = nn.Linear(d_model // 2, 1)  # Predict price
        
        # Additional outputs for trading signals
        self.signal_head = nn.Linear(d_model // 2, 3)  # Buy/Hold/Sell
        self.confidence_head = nn.Linear(d_model // 2, 1)  # Confidence score
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate mask for decoder self-attention"""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src: torch.Tensor, tgt: Optional[torch.Tensor] = None,
                src_mask: Optional[torch.Tensor] = None,
                tgt_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            src: Source sequence (seq_len, batch, input_dim)
            tgt: Target sequence for teacher forcing (optional)
            src_mask: Source mask
            tgt_mask: Target mask
        
        Returns:
            Dictionary with predictions, signals, and confidence
        """
        # Project input to d_model dimensions
        src = self.input_projection(src)
        src = self.pos_encoder(src)
        src = self.layer_norm(src)
        
        # If no target is provided, use the last value of source
        if tgt is None:
            # Use the last encoded state to predict future
            tgt = src[-1:, :, :].repeat(self.prediction_length, 1, 1)
        else:
            tgt = self.input_projection(tgt)
            tgt = self.pos_encoder(tgt)
        
        # Generate masks if not provided
        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        
        # Transformer forward pass
        output = self.transformer(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)
        
        # Decode output
        decoded = F.relu(self.decoder_projection(output))
        
        # Generate predictions
        price_predictions = self.output_projection(decoded).squeeze(-1)
        signal_logits = self.signal_head(decoded[-1, :, :])  # Use last time step
        confidence = torch.sigmoid(self.confidence_head(decoded[-1, :, :])).squeeze(-1)
        
        return {
            'predictions': price_predictions,
            'signal_logits': signal_logits,
            'confidence': confidence
        }


class TimeSeriesTransformerDataset(Dataset):
    """Dataset for transformer time series prediction"""
    
    def __init__(self, data: np.ndarray, sequence_length: int = 60,
                 prediction_length: int = 5, stride: int = 1):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.stride = stride
        
        # Calculate valid indices
        self.indices = []
        for i in range(0, len(data) - sequence_length - prediction_length + 1, stride):
            self.indices.append(i)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Source sequence
        src = self.data[start_idx:end_idx]
        
        # Target sequence (for teacher forcing during training)
        tgt_start = end_idx
        tgt_end = tgt_start + self.prediction_length
        tgt = self.data[tgt_start:tgt_end]
        
        # Labels for signal classification
        current_price = self.data[end_idx - 1, 0]  # Close price
        future_price = self.data[tgt_end - 1, 0]
        
        return_pct = (future_price - current_price) / current_price
        
        # Signal label
        if return_pct < -0.02:
            signal_label = 0  # Sell
        elif return_pct > 0.02:
            signal_label = 2  # Buy
        else:
            signal_label = 1  # Hold
        
        return (torch.FloatTensor(src), 
                torch.FloatTensor(tgt),
                torch.LongTensor([signal_label]),
                torch.FloatTensor([return_pct]))


class TransformerTradingSystem:
    """Complete transformer-based trading system"""
    
    def __init__(self, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 4, sequence_length: int = 60,
                 prediction_length: int = 5, learning_rate: float = 0.0001,
                 use_cuda: bool = True):
        """
        Initialize Transformer Trading System
        
        Args:
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_layers: Number of encoder/decoder layers
            sequence_length: Input sequence length
            prediction_length: Output prediction length
            learning_rate: Learning rate
            use_cuda: Whether to use GPU
        """
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.learning_rate = learning_rate
        
        # Device
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model components
        self.model = None
        self.optimizer = None
        self.scaler = StandardScaler()
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features from dataframe"""
        features = []
        
        # Price features
        features.append(df['Close'].values)
        features.append(df['Open'].values)
        features.append(df['High'].values)
        features.append(df['Low'].values)
        features.append(df['Volume'].values)
        
        # Price ratios and returns
        features.append(df['Close'].pct_change().fillna(0).values)
        features.append((df['High'] / df['Low'] - 1).values)
        features.append((df['Close'] / df['Open'] - 1).values)
        
        # Volume features
        features.append(df['Volume'].pct_change().fillna(0).values)
        volume_ma = df['Volume'].rolling(20).mean()
        features.append((df['Volume'] / volume_ma).fillna(1).values)
        
        # Technical indicators if available
        indicator_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'adx', 'ema_9', 'ema_21', 'ema_50',
            'supertrend', 'squeeze_momentum'
        ]
        
        for col in indicator_columns:
            if col in df.columns:
                features.append(df[col].values)
        
        # Stack and handle NaN
        features = np.column_stack(features)
        features = pd.DataFrame(features).fillna(method='ffill').fillna(0).values
        
        return features
    
    def create_model(self, input_dim: int):
        """Create transformer model"""
        self.model = TransformerPredictor(
            input_dim=input_dim,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            prediction_length=self.prediction_length,
            max_seq_length=self.sequence_length * 2
        ).to(self.device)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        logger.info(f"Created Transformer model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, early_stopping_patience: int = 15):
        """Train the transformer model"""
        logger.info("Starting Transformer training...")
        
        # Prepare features
        features = self.prepare_features(df)
        features_scaled = self.scaler.fit_transform(features)
        
        # Create dataset
        dataset = TimeSeriesTransformerDataset(
            features_scaled,
            sequence_length=self.sequence_length,
            prediction_length=self.prediction_length,
            stride=1
        )
        
        # Split dataset
        train_size = int(len(dataset) * (1 - validation_split))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model
        if self.model is None:
            self.create_model(features_scaled.shape[1])
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for src, tgt, signal_labels, returns in train_loader:
                # Move to device and transpose for transformer
                src = src.transpose(0, 1).to(self.device)
                tgt = tgt.transpose(0, 1).to(self.device)
                signal_labels = signal_labels.to(self.device)
                returns = returns.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(src, tgt)
                
                # Calculate losses
                price_loss = self.mse_loss(outputs['predictions'][-1, :], returns)
                signal_loss = self.ce_loss(outputs['signal_logits'], signal_labels.squeeze())
                
                # Combined loss
                total_loss = price_loss + 0.5 * signal_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Track metrics
                train_loss += total_loss.item()
                _, predicted = torch.max(outputs['signal_logits'], 1)
                train_correct += (predicted == signal_labels.squeeze()).sum().item()
                train_total += signal_labels.size(0)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for src, tgt, signal_labels, returns in val_loader:
                    src = src.transpose(0, 1).to(self.device)
                    tgt = tgt.transpose(0, 1).to(self.device)
                    signal_labels = signal_labels.to(self.device)
                    returns = returns.to(self.device)
                    
                    outputs = self.model(src)
                    
                    price_loss = self.mse_loss(outputs['predictions'][-1, :], returns)
                    signal_loss = self.ce_loss(outputs['signal_logits'], signal_labels.squeeze())
                    total_loss = price_loss + 0.5 * signal_loss
                    
                    val_loss += total_loss.item()
                    _, predicted = torch.max(outputs['signal_logits'], 1)
                    val_correct += (predicted == signal_labels.squeeze()).sum().item()
                    val_total += signal_labels.size(0)
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            # Update learning rate
            self.scheduler.step(avg_val_loss)
            
            # Update history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['train_accuracy'].append(train_acc)
            self.training_history['val_accuracy'].append(val_acc)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.save_model(f"dl_models/best_transformer_model.pth")
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions using the transformer"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        
        # Prepare features
        features = self.prepare_features(df)
        features_scaled = self.scaler.transform(features)
        
        # Create sequences
        predictions = []
        confidences = []
        signal_probs = []
        
        with torch.no_grad():
            for i in range(len(features_scaled) - self.sequence_length + 1):
                # Get sequence
                seq = features_scaled[i:i + self.sequence_length]
                seq_tensor = torch.FloatTensor(seq).unsqueeze(1).to(self.device)
                seq_tensor = seq_tensor.transpose(0, 1)
                
                # Predict
                outputs = self.model(seq_tensor)
                
                # Get predictions
                price_pred = outputs['predictions'][-1, 0].cpu().item()
                signal_probs_batch = F.softmax(outputs['signal_logits'], dim=1).cpu().numpy()[0]
                confidence = outputs['confidence'].cpu().item()
                
                predictions.append(price_pred)
                signal_probs.append(signal_probs_batch)
                confidences.append(confidence)
        
        # Create results dataframe
        start_idx = self.sequence_length - 1
        results = pd.DataFrame(index=df.index[start_idx:start_idx + len(predictions)])
        
        results['price_prediction'] = predictions
        results['confidence'] = confidences
        results['sell_prob'] = [p[0] for p in signal_probs]
        results['hold_prob'] = [p[1] for p in signal_probs]
        results['buy_prob'] = [p[2] for p in signal_probs]
        
        # Generate signals
        results['signal'] = 0
        buy_mask = (results['buy_prob'] > 0.6) & (results['confidence'] > 0.7)
        sell_mask = (results['sell_prob'] > 0.6) & (results['confidence'] > 0.7)
        
        results.loc[buy_mask, 'signal'] = 1
        results.loc[sell_mask, 'signal'] = -1
        
        return results
    
    def save_model(self, path: str):
        """Save model and configuration"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_config': {
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'sequence_length': self.sequence_length,
                'prediction_length': self.prediction_length,
                'input_dim': self.model.input_dim
            },
            'scaler': self.scaler,
            'training_history': self.training_history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and configuration"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load config
        config = checkpoint['model_config']
        self.d_model = config['d_model']
        self.nhead = config['nhead']
        self.num_layers = config['num_layers']
        self.sequence_length = config['sequence_length']
        self.prediction_length = config['prediction_length']
        
        # Create model
        self.create_model(config['input_dim'])
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load other components
        self.scaler = checkpoint['scaler']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {path}")


def main():
    """Test the transformer trading system"""
    import sys
    sys.path.append('..')
    from utils.csv_data_manager import CSVDataManager
    
    # Initialize data manager
    data_manager = CSVDataManager()
    
    # Load data
    symbol = "GARAN"
    df = data_manager.load_data(symbol, "1d")
    
    if df is not None and len(df) > 200:
        # Initialize transformer system
        transformer = TransformerTradingSystem(
            d_model=256,
            nhead=8,
            num_layers=4,
            sequence_length=60,
            prediction_length=5,
            learning_rate=0.0001
        )
        
        # Train model
        transformer.train(df, epochs=50, batch_size=32)
        
        # Generate predictions
        predictions = transformer.predict(df)
        
        # Display results
        print("\nRecent Predictions:")
        print(predictions.tail(10))
        
        # Signal distribution
        print("\nSignal Distribution:")
        print(predictions['signal'].value_counts())
        
        # Save model
        transformer.save_model(f"dl_models/{symbol}_transformer_model.pth")
    else:
        print(f"Insufficient data for {symbol}")


if __name__ == "__main__":
    main()