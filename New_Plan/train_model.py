"""
Train GRU Model for Hybrid Trading System
Multi-timeframe model training with proper validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import argparse
from loguru import logger
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths
import sys
sys.path.append(str(Path(__file__).parent))

from core.csv_data_manager import CSVDataManager
from indicators.indicator_calculator import IndicatorCalculator
from core.feature_engineering import FeatureEngineering
from models.simple_gru_model import SimpleMultiTimeframeGRU


class TradingDataset(Dataset):
    """Dataset for multi-timeframe trading data"""
    
    def __init__(self, features_dict, labels, sequence_length=30):
        self.features = features_dict
        self.labels = labels
        self.sequence_length = sequence_length
        self.timeframes = list(features_dict.keys())
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Get sequences for each timeframe
        sequences = {}
        for tf in self.timeframes:
            # Get data as numpy array
            data = self.features[tf].values if hasattr(self.features[tf], 'values') else self.features[tf]
            
            if idx < self.sequence_length:
                # Pad with zeros if not enough history
                pad_length = self.sequence_length - idx - 1
                padding = np.zeros((pad_length, data.shape[1]))
                sequence = data[:idx+1]
                sequences[tf] = np.vstack([padding, sequence])
            else:
                sequences[tf] = data[idx-self.sequence_length+1:idx+1]
        
        # Convert to tensors
        tensors = {tf: torch.FloatTensor(seq) for tf, seq in sequences.items()}
        label = torch.FloatTensor([self.labels[idx]])
        
        return tensors, label


class ModelTrainer:
    """Model training and validation"""
    
    def __init__(self, config_path: str = 'config.json'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.csv_manager = CSVDataManager()
        self.indicator_calc = IndicatorCalculator()
        self.feature_engineer = FeatureEngineering(self.config)
        
        # Model parameters
        self.sequence_length = 30
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 100
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
    
    def prepare_training_data(self, symbols: List[str], start_date: str, end_date: str):
        """Prepare training data from multiple symbols"""
        logger.info(f"Preparing data for {len(symbols)} symbols...")
        
        all_features = {tf: [] for tf in self.config['timeframes']['analysis']}
        all_labels = []
        
        for symbol in tqdm(symbols, desc="Processing symbols"):
            try:
                # Get multi-timeframe data
                data = self._load_symbol_data(symbol, start_date, end_date)
                if not data:
                    continue
                
                # Create features
                features = self.feature_engineer.create_features(data, symbol)
                if not features:
                    continue
                
                # Generate labels (next day return)
                labels = self._generate_labels(data)
                if labels is None:
                    continue
                
                # Align features and labels
                min_length = min(len(labels), min(len(features[tf]) for tf in features))
                
                # Append to collection
                for tf in features:
                    if tf in all_features:
                        all_features[tf].append(features[tf].iloc[:min_length])
                
                all_labels.extend(labels[:min_length])
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Combine all data
        combined_features = {}
        for tf in all_features:
            if all_features[tf]:
                combined_features[tf] = pd.concat(all_features[tf], axis=0)
                logger.info(f"{tf}: {combined_features[tf].shape}")
        
        logger.info(f"Total samples: {len(all_labels)}")
        
        return combined_features, np.array(all_labels)
    
    def _load_symbol_data(self, symbol: str, start_date: str, end_date: str) -> Dict:
        """Load and prepare data for a symbol"""
        data = {'indicators': {}}
        
        # Load data for each timeframe
        for tf in self.config['timeframes']['analysis']:  # Use all configured timeframes
            df = self.csv_manager.get_raw_data(symbol, tf)
            if df is None:
                continue
            
            # Filter date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            if len(df) < 100:  # Need sufficient data
                continue
            
            # Calculate indicators
            indicators = self.indicator_calc.calculate_all_indicators(symbol, tf, save=False)
            if not indicators.empty:
                # Merge with raw data
                df = pd.concat([df, indicators], axis=1)
            
            data['indicators'][tf] = df
        
        # Add placeholder for macro/sentiment
        data['macro'] = {'vix': 20, 'usdtry': 30}
        data['sentiment'] = {'score': 0, 'count': 0}
        
        # Ensure we have all required timeframes
        required_timeframes = ['1h', '4h', '1d']
        has_all = all(tf in data['indicators'] for tf in required_timeframes)
        
        return data if has_all else None
    
    def _generate_labels(self, data: Dict) -> Optional[np.ndarray]:
        """Generate trading labels from price data"""
        # Use daily data for labels
        if not data or '1d' not in data.get('indicators', {}):
            return None
        
        df = data['indicators']['1d']
        
        if 'close' not in df.columns:
            return None
        
        # Calculate next day returns
        returns = df['close'].pct_change(fill_method=None).shift(-1)  # Next day return
        
        # Convert to binary labels (1 for positive, 0 for negative)
        # Using 0.5% threshold to filter noise
        labels = (returns > 0.005).astype(int).values[:-1]  # Remove last NaN
        
        return labels if len(labels) > 0 else None
    
    def train_model(self, features: Dict, labels: np.ndarray, val_split: float = 0.2):
        """Train the GRU model"""
        logger.info("Starting model training...")
        
        # Split data
        train_size = int(len(labels) * (1 - val_split))
        
        train_features = {tf: features[tf].iloc[:train_size] for tf in features}
        val_features = {tf: features[tf].iloc[train_size:] for tf in features}
        
        train_labels = labels[:train_size]
        val_labels = labels[train_size:]
        
        # Normalize features
        scalers = {}
        for tf in features:
            scaler = StandardScaler()
            train_features[tf] = pd.DataFrame(
                scaler.fit_transform(train_features[tf]),
                columns=train_features[tf].columns,
                index=train_features[tf].index
            )
            val_features[tf] = pd.DataFrame(
                scaler.transform(val_features[tf]),
                columns=val_features[tf].columns,
                index=val_features[tf].index
            )
            scalers[tf] = scaler
        
        # Create datasets
        train_dataset = TradingDataset(train_features, train_labels, self.sequence_length)
        val_dataset = TradingDataset(val_features, val_labels, self.sequence_length)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # Initialize model
        input_size = list(train_features.values())[0].shape[1]
        model = SimpleMultiTimeframeGRU(
            input_size=input_size,
            hidden_size=50,
            num_layers=1,
            dropout=0.2
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        # Training loop
        for epoch in range(self.epochs):
            # Train
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                tensors, labels_batch = batch
                
                # Prepare inputs for different timeframes
                x_1h = tensors.get('1h', None).to(self.device) if '1h' in tensors else None
                x_4h = tensors.get('4h', None).to(self.device) if '4h' in tensors else None
                x_1d = tensors.get('1d', None).to(self.device) if '1d' in tensors else None
                
                # Fill missing timeframes with None
                x_15m = None  # Not used in training
                x_1w = None   # Not used in training
                
                labels_batch = labels_batch.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs, _ = model(x_15m, x_1h, x_4h, x_1d, x_1w)
                loss = criterion(outputs, labels_batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                # Calculate accuracy
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predictions == labels_batch).sum().item()
                train_total += labels_batch.size(0)
            
            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    tensors, labels_batch = batch
                    
                    x_1h = tensors.get('1h', None).to(self.device) if '1h' in tensors else None
                    x_4h = tensors.get('4h', None).to(self.device) if '4h' in tensors else None
                    x_1d = tensors.get('1d', None).to(self.device) if '1d' in tensors else None
                    x_15m = None
                    x_1w = None
                    
                    labels_batch = labels_batch.to(self.device)
                    
                    outputs, _ = model(x_15m, x_1h, x_4h, x_1d, x_1w)
                    loss = criterion(outputs, labels_batch)
                    
                    val_loss += loss.item()
                    
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (predictions == labels_batch).sum().item()
                    val_total += labels_batch.size(0)
            
            # Calculate metrics
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100
            
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Print progress
            logger.info(f"Epoch {epoch+1}/{self.epochs} - "
                       f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.1f}% - "
                       f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.1f}%")
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                self._save_model(model, scalers, history)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Plot training history
        self._plot_training_history(history)
        
        return model, history
    
    def _save_model(self, model, scalers, history):
        """Save model and training artifacts"""
        save_dir = Path('models/saved')
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'input_size': model.input_size,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers
            },
            'scalers': scalers,
            'history': history,
            'timestamp': datetime.now().isoformat()
        }, save_dir / 'gru_multi_timeframe.pth')
        
        logger.info("Model saved successfully")
    
    def _plot_training_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training History - Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Acc')
        ax2.plot(history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training History - Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        logger.info("Training history plot saved")


def main():
    """Main training function"""
    # Load config to get default symbols
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    parser = argparse.ArgumentParser(description='Train GRU model for trading')
    parser.add_argument('--symbols', nargs='+', 
                       default=None,
                       help='Symbols to use for training (default: all available symbols)')
    parser.add_argument('--start', default='2022-01-01', help='Start date')
    parser.add_argument('--end', default='2023-12-31', help='End date')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # If no symbols specified, use all available symbols
    if args.symbols is None:
        csv_manager = CSVDataManager()
        args.symbols = [s for s in csv_manager.get_available_symbols() if not s.endswith('.IS')]
        logger.info(f"Using all {len(args.symbols)} available symbols")
    
    # Initialize trainer
    trainer = ModelTrainer()
    trainer.epochs = args.epochs
    trainer.batch_size = args.batch_size
    trainer.learning_rate = args.lr
    
    logger.info("="*60)
    logger.info("GRU MODEL TRAINING")
    logger.info("="*60)
    logger.info(f"Symbols: {', '.join(args.symbols)}")
    logger.info(f"Period: {args.start} to {args.end}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch Size: {args.batch_size}")
    logger.info(f"Learning Rate: {args.lr}")
    
    # Prepare data
    features, labels = trainer.prepare_training_data(args.symbols, args.start, args.end)
    
    if len(labels) == 0:
        logger.error("No training data available")
        return
    
    logger.info(f"\nTraining data prepared:")
    logger.info(f"Total samples: {len(labels)}")
    logger.info(f"Positive samples: {np.sum(labels)} ({np.mean(labels)*100:.1f}%)")
    
    # Train model
    model, history = trainer.train_model(features, labels)
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Final Train Accuracy: {history['train_acc'][-1]:.1f}%")
    logger.info(f"Final Val Accuracy: {history['val_acc'][-1]:.1f}%")
    logger.info(f"Best Val Loss: {min(history['val_loss']):.4f}")
    
    logger.info("\nModel saved to: models/saved/gru_multi_timeframe.pth")


if __name__ == "__main__":
    main()