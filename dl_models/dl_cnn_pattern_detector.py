"""
CNN-based Candlestick Pattern Detection Model

This module implements a Convolutional Neural Network for detecting
candlestick patterns and price action patterns in trading data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CandlestickImageGenerator:
    """Generate candlestick chart images for CNN input"""
    
    def __init__(self, window_size: int = 20, image_size: Tuple[int, int] = (128, 128)):
        """
        Args:
            window_size: Number of candlesticks in each image
            image_size: Output image size (height, width)
        """
        self.window_size = window_size
        self.image_size = image_size
        
    def create_candlestick_image(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """Create candlestick chart image from OHLCV data"""
        if len(ohlcv_data) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} data points")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(4, 3), dpi=32)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        # Get data
        opens = ohlcv_data['Open'].values
        highs = ohlcv_data['High'].values
        lows = ohlcv_data['Low'].values
        closes = ohlcv_data['Close'].values
        
        # Normalize prices
        price_min = min(lows.min(), opens.min(), closes.min())
        price_max = max(highs.max(), opens.max(), closes.max())
        price_range = price_max - price_min
        
        if price_range == 0:
            price_range = 1
        
        # Draw candlesticks
        for i in range(len(ohlcv_data)):
            open_price = (opens[i] - price_min) / price_range
            high_price = (highs[i] - price_min) / price_range
            low_price = (lows[i] - price_min) / price_range
            close_price = (closes[i] - price_min) / price_range
            
            # Determine color
            color = 'green' if closes[i] >= opens[i] else 'red'
            
            # Draw high-low line
            ax.plot([i, i], [low_price, high_price], color=color, linewidth=1)
            
            # Draw body
            body_height = abs(close_price - open_price)
            body_bottom = min(open_price, close_price)
            
            rect = patches.Rectangle(
                (i - 0.3, body_bottom), 0.6, body_height,
                linewidth=0, edgecolor=color, facecolor=color
            )
            ax.add_patch(rect)
        
        # Set limits
        ax.set_xlim(-1, len(ohlcv_data))
        ax.set_ylim(-0.05, 1.05)
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        # Get image as numpy array
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        
        # Convert to RGB and resize
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        image = cv2.resize(image, self.image_size)
        
        plt.close(fig)
        
        # Convert to tensor format (C, H, W)
        image = image.transpose(2, 0, 1) / 255.0
        
        return image.astype(np.float32)
    
    def create_volume_profile(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """Create volume profile image"""
        volumes = ohlcv_data['Volume'].values
        prices = ohlcv_data['Close'].values
        
        # Create volume profile
        fig, ax = plt.subplots(figsize=(4, 3), dpi=32)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # Normalize volume
        vol_normalized = volumes / volumes.max() if volumes.max() > 0 else volumes
        
        # Create bars
        ax.bar(range(len(volumes)), vol_normalized, color='cyan', alpha=0.7)
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        # Convert to image
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)
        
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        image = cv2.resize(image, self.image_size)
        
        plt.close(fig)
        
        return image.astype(np.float32) / 255.0


class PatternDataset(Dataset):
    """Dataset for candlestick patterns"""
    
    def __init__(self, data: pd.DataFrame, window_size: int = 20,
                 prediction_window: int = 5, image_size: Tuple[int, int] = (128, 128)):
        self.data = data
        self.window_size = window_size
        self.prediction_window = prediction_window
        self.image_generator = CandlestickImageGenerator(window_size, image_size)
        
        # Pre-compute valid indices
        self.valid_indices = list(range(len(data) - window_size - prediction_window))
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.window_size
        
        # Get window data
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Generate candlestick image
        candle_image = self.image_generator.create_candlestick_image(window_data)
        
        # Generate volume profile
        volume_image = self.image_generator.create_volume_profile(window_data)
        
        # Stack images (4 channels: RGB + Volume)
        image = np.concatenate([candle_image, volume_image[np.newaxis, :, :]], axis=0)
        
        # Calculate label (future return)
        current_price = self.data.iloc[end_idx]['Close']
        future_price = self.data.iloc[end_idx + self.prediction_window]['Close']
        future_return = (future_price - current_price) / current_price
        
        # Convert to classification (0: sell, 1: hold, 2: buy)
        if future_return < -0.02:
            label = 0  # Sell
        elif future_return > 0.02:
            label = 2  # Buy
        else:
            label = 1  # Hold
        
        return torch.FloatTensor(image), torch.LongTensor([label])


class CNNPatternDetector(nn.Module):
    """CNN model for pattern detection"""
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super(CNNPatternDetector, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Conv block 4
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(-1, 256 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class ResNetBlock(nn.Module):
    """Residual block for deeper networks"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetPatternDetector(nn.Module):
    """ResNet-based pattern detector for better performance"""
    
    def __init__(self, num_classes: int = 3, dropout: float = 0.3):
        super(ResNetPatternDetector, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResNetBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class CNNPatternDetectorSystem:
    """Main system for CNN-based pattern detection"""
    
    def __init__(self, model_type: str = 'cnn', window_size: int = 20,
                 prediction_window: int = 5, image_size: Tuple[int, int] = (128, 128),
                 learning_rate: float = 0.001, use_cuda: bool = True):
        """
        Initialize the CNN Pattern Detection System
        
        Args:
            model_type: 'cnn' or 'resnet'
            window_size: Number of candlesticks in each pattern
            prediction_window: Number of periods to predict ahead
            image_size: Size of generated images
            learning_rate: Learning rate for optimizer
            use_cuda: Whether to use GPU if available
        """
        self.model_type = model_type
        self.window_size = window_size
        self.prediction_window = prediction_window
        self.image_size = image_size
        self.learning_rate = learning_rate
        
        # Device configuration
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'best_val_acc': 0.0
        }
        
    def create_model(self):
        """Create the CNN model"""
        if self.model_type == 'cnn':
            self.model = CNNPatternDetector(num_classes=3, dropout=0.3).to(self.device)
        elif self.model_type == 'resnet':
            self.model = ResNetPatternDetector(num_classes=3, dropout=0.3).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        logger.info(f"Created {self.model_type.upper()} model with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def train(self, df: pd.DataFrame, epochs: int = 50, batch_size: int = 32,
              validation_split: float = 0.2, early_stopping_patience: int = 10):
        """Train the model"""
        logger.info("Starting CNN pattern detector training...")
        
        # Create dataset
        dataset = PatternDataset(
            df, 
            window_size=self.window_size,
            prediction_window=self.prediction_window,
            image_size=self.image_size
        )
        
        # Split dataset
        train_size = int(len(dataset) * (1 - validation_split))
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, len(dataset) - train_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        # Create model if not exists
        if self.model is None:
            self.create_model()
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.squeeze().to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(self.device), labels.squeeze().to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            train_acc = 100 * train_correct / train_total
            avg_val_loss = val_loss / len(val_loader)
            val_acc = 100 * val_correct / val_total
            
            # Update history
            self.training_history['train_loss'].append(avg_train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(avg_val_loss)
            self.training_history['val_acc'].append(val_acc)
            
            # Log progress
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}], "
                          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.training_history['best_val_acc'] = best_val_acc
                patience_counter = 0
                # Save best model
                self.save_model(f"dl_models/best_{self.model_type}_pattern_model.pth")
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
    
    def detect_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect patterns in the data"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        
        # Create dataset
        dataset = PatternDataset(
            df,
            window_size=self.window_size,
            prediction_window=self.prediction_window,
            image_size=self.image_size
        )
        
        # Create data loader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Make predictions
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Create results dataframe
        start_idx = self.window_size
        end_idx = start_idx + len(all_predictions)
        
        results = pd.DataFrame(index=df.index[start_idx:end_idx])
        results['pattern_signal'] = all_predictions
        results['sell_prob'] = [p[0] for p in all_probabilities]
        results['hold_prob'] = [p[1] for p in all_probabilities]
        results['buy_prob'] = [p[2] for p in all_probabilities]
        results['confidence'] = results[['sell_prob', 'hold_prob', 'buy_prob']].max(axis=1)
        
        # Convert to trading signals
        results['signal'] = 0
        results.loc[results['pattern_signal'] == 0, 'signal'] = -1  # Sell
        results.loc[results['pattern_signal'] == 2, 'signal'] = 1   # Buy
        
        return results
    
    def save_model(self, path: str):
        """Save model and configuration"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'model_type': self.model_type,
                'window_size': self.window_size,
                'prediction_window': self.prediction_window,
                'image_size': self.image_size
            },
            'training_history': self.training_history
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and configuration"""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load configuration
        config = checkpoint['model_config']
        self.model_type = config['model_type']
        self.window_size = config['window_size']
        self.prediction_window = config['prediction_window']
        self.image_size = config['image_size']
        
        # Create model
        self.create_model()
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load history
        self.training_history = checkpoint['training_history']
        
        logger.info(f"Model loaded from {path}")


def main():
    """Test the CNN pattern detector"""
    import sys
    sys.path.append('..')
    from utils.csv_data_manager import CSVDataManager
    
    # Initialize data manager
    data_manager = CSVDataManager()
    
    # Load data for a symbol
    symbol = "THYAO"
    df = data_manager.load_data(symbol, "1d")
    
    if df is not None and len(df) > 100:
        # Initialize detector
        detector = CNNPatternDetectorSystem(
            model_type='cnn',
            window_size=20,
            prediction_window=5,
            image_size=(128, 128),
            learning_rate=0.001
        )
        
        # Train model
        detector.train(df, epochs=30, batch_size=16)
        
        # Detect patterns
        patterns = detector.detect_patterns(df)
        
        # Display recent patterns
        print("\nRecent Pattern Detections:")
        print(patterns.tail(10))
        
        # Show signal distribution
        print("\nSignal Distribution:")
        print(patterns['signal'].value_counts())
        
        # Save model
        detector.save_model(f"dl_models/{symbol}_cnn_pattern_model.pth")
    else:
        print(f"Insufficient data for {symbol}")


if __name__ == "__main__":
    main()