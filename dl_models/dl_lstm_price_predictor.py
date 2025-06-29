"""
LSTM-based Price Prediction Model for Trading System

This module implements a deep learning price predictor using LSTM/GRU networks
to capture temporal patterns in stock price movements.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockPriceDataset(Dataset):
    """Custom Dataset for stock price time series"""
    
    def __init__(self, data: np.ndarray, sequence_length: int = 60, 
                 prediction_horizon: int = 5):
        """
        Args:
            data: Numpy array of features
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of time steps to predict ahead
        """
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_horizon, 0]  # Predict close price
        return torch.FloatTensor(x), torch.FloatTensor(y)


class LSTMPricePredictor(nn.Module):
    """LSTM model for price prediction"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2,
                 prediction_horizon: int = 5, use_attention: bool = True):
        super(LSTMPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        self.use_attention = use_attention
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(hidden_size)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, prediction_horizon)
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size // 2)
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention if enabled
        if self.use_attention:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.attention_norm(lstm_out + attn_out)
        
        # Use the last hidden state
        out = lstm_out[:, -1, :]
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.batch_norm(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class GRUPricePredictor(nn.Module):
    """GRU model for price prediction (alternative to LSTM)"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 3, dropout: float = 0.2,
                 prediction_horizon: int = 5):
        super(GRUPricePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, prediction_horizon)
        )
        
    def forward(self, x):
        gru_out, hidden = self.gru(x)
        out = gru_out[:, -1, :]
        out = self.fc(out)
        return out


class DeepLearningPricePredictor:
    """Main class for DL-based price prediction"""
    
    def __init__(self, model_type: str = 'lstm', sequence_length: int = 60,
                 prediction_horizon: int = 5, hidden_size: int = 128,
                 num_layers: int = 3, dropout: float = 0.2,
                 learning_rate: float = 0.001, use_cuda: bool = True):
        """
        Initialize the Deep Learning Price Predictor
        
        Args:
            model_type: 'lstm' or 'gru'
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of time steps to predict ahead
            hidden_size: Size of hidden layers
            num_layers: Number of LSTM/GRU layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimizer
            use_cuda: Whether to use GPU if available
        """
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        
        # Device configuration
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Scalers
        self.feature_scaler = RobustScaler()
        self.target_scaler = StandardScaler()
        
        # Model
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'best_val_loss': float('inf')
        }
        
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features from raw dataframe"""
        features = []
        
        # Price features
        features.append(df['Close'].values)
        features.append(df['High'].values)
        features.append(df['Low'].values)
        features.append(df['Open'].values)
        features.append(df['Volume'].values)
        
        # Technical indicators (if available)
        indicator_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'adx', 'plus_di', 'minus_di',
            'ema_9', 'ema_21', 'ema_50',
            'supertrend', 'squeeze_momentum'
        ]
        
        for col in indicator_columns:
            if col in df.columns:
                features.append(df[col].values)
        
        # Price ratios
        features.append((df['Close'] / df['Open']).values)
        features.append((df['High'] / df['Low']).values)
        features.append((df['Close'] / df['Close'].shift(1)).fillna(1).values)
        
        # Volume features
        features.append((df['Volume'] / df['Volume'].rolling(20).mean()).fillna(1).values)
        
        # Stack features
        features = np.column_stack(features)
        
        # Handle NaN values
        features = pd.DataFrame(features).fillna(method='ffill').fillna(0).values
        
        return features
    
    def create_model(self, input_size: int):
        """Create the neural network model"""
        if self.model_type == 'lstm':
            self.model = LSTMPricePredictor(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                prediction_horizon=self.prediction_horizon,
                use_attention=True
            ).to(self.device)
        elif self.model_type == 'gru':
            self.model = GRUPricePredictor(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
                prediction_horizon=self.prediction_horizon
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        logger.info(f"Created {self.model_type.upper()} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32,
              validation_split: float = 0.2, early_stopping_patience: int = 10):
        """Train the model"""
        logger.info("Starting model training...")
        
        # Prepare features
        features = self.prepare_features(df)
        
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Create dataset
        dataset = StockPriceDataset(
            features_scaled, 
            self.sequence_length, 
            self.prediction_horizon
        )
        
        # Split dataset
        train_size = int(len(dataset) * (1 - validation_split))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create model if not exists
        if self.model is None:
            self.create_model(features_scaled.shape[1])
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Update training history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['val_loss'].append(avg_val_loss)
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                self.training_history['best_val_loss'] = best_val_loss
                patience_counter = 0
                # Save best model
                self.save_model(f"dl_models/best_{self.model_type}_model.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        logger.info(f"Training completed. Best validation loss: {best_val_loss:.6f}")
    
    def predict(self, df: pd.DataFrame, return_confidence: bool = True) -> Dict[str, np.ndarray]:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        
        # Prepare features
        features = self.prepare_features(df)
        features_scaled = self.feature_scaler.transform(features)
        
        # Create sequences
        sequences = []
        for i in range(len(features_scaled) - self.sequence_length + 1):
            seq = features_scaled[i:i + self.sequence_length]
            sequences.append(seq)
        
        if not sequences:
            raise ValueError("Not enough data to create sequences")
        
        sequences = torch.FloatTensor(np.array(sequences)).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(sequences).cpu().numpy()
        
        # Calculate prediction statistics
        results = {
            'predictions': predictions,
            'mean_prediction': np.mean(predictions, axis=1),
            'prediction_std': np.std(predictions, axis=1)
        }
        
        if return_confidence:
            # Calculate confidence based on prediction consistency
            results['confidence'] = 1 / (1 + results['prediction_std'])
        
        return results
    
    def generate_trading_signals(self, df: pd.DataFrame, 
                               threshold_percentile: float = 70) -> pd.DataFrame:
        """Generate trading signals based on predictions"""
        predictions = self.predict(df)
        
        # Calculate expected returns
        current_prices = df['Close'].values[self.sequence_length-1:]
        mean_predictions = predictions['mean_prediction']
        
        # Ensure arrays have same length
        min_len = min(len(current_prices), len(mean_predictions))
        current_prices = current_prices[:min_len]
        mean_predictions = mean_predictions[:min_len]
        
        expected_returns = (mean_predictions - current_prices) / current_prices
        
        # Generate signals
        buy_threshold = np.percentile(expected_returns[expected_returns > 0], threshold_percentile)
        sell_threshold = np.percentile(expected_returns[expected_returns < 0], 100 - threshold_percentile)
        
        signals = pd.DataFrame(index=df.index[self.sequence_length-1:self.sequence_length-1+min_len])
        signals['expected_return'] = expected_returns
        signals['confidence'] = predictions['confidence'][:min_len]
        signals['signal'] = 0
        
        # Buy signals
        buy_mask = (expected_returns > buy_threshold) & (predictions['confidence'][:min_len] > 0.7)
        signals.loc[buy_mask, 'signal'] = 1
        
        # Sell signals
        sell_mask = (expected_returns < sell_threshold) & (predictions['confidence'][:min_len] > 0.7)
        signals.loc[sell_mask, 'signal'] = -1
        
        return signals
    
    def save_model(self, path: str):
        """Save model and configuration"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'model_type': self.model_type,
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'hidden_size': self.hidden_size,
                'num_layers': self.num_layers,
                'dropout': self.dropout,
                'input_size': self.model.lstm.input_size if hasattr(self.model, 'lstm') else self.model.gru.input_size
            },
            'training_history': self.training_history,
            'feature_scaler': self.feature_scaler,
            'target_scaler': self.target_scaler
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and configuration"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load configuration
        config = checkpoint['model_config']
        self.model_type = config['model_type']
        self.sequence_length = config['sequence_length']
        self.prediction_horizon = config['prediction_horizon']
        self.hidden_size = config['hidden_size']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        
        # Create model
        self.create_model(config['input_size'])
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scalers and history
        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {path}")


def main():
    """Test the DL price predictor"""
    import sys
    sys.path.append('..')
    from utils.csv_data_manager import CSVDataManager
    
    # Initialize data manager
    data_manager = CSVDataManager()
    
    # Load data for a symbol
    symbol = "ASELS"
    df = data_manager.load_data(symbol, "1d")
    
    if df is not None and len(df) > 200:
        # Initialize predictor
        predictor = DeepLearningPricePredictor(
            model_type='lstm',
            sequence_length=60,
            prediction_horizon=5,
            hidden_size=128,
            num_layers=3,
            dropout=0.2,
            learning_rate=0.001
        )
        
        # Train model
        predictor.train(df, epochs=50, batch_size=32)
        
        # Generate signals
        signals = predictor.generate_trading_signals(df)
        
        # Display recent signals
        print("\nRecent Trading Signals:")
        print(signals.tail(10))
        
        # Save model
        predictor.save_model(f"dl_models/{symbol}_lstm_model.pth")
        
        # Test loading
        new_predictor = DeepLearningPricePredictor()
        new_predictor.load_model(f"dl_models/{symbol}_lstm_model.pth")
        
        # Make predictions with loaded model
        test_signals = new_predictor.generate_trading_signals(df)
        print("\nTest signals from loaded model:")
        print(test_signals.tail(5))
    else:
        print(f"Insufficient data for {symbol}")


if __name__ == "__main__":
    main()