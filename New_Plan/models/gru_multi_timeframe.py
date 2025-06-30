"""
Multi-Timeframe GRU Model
CPU üzerinde verimli çalışan hibrit model
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from loguru import logger


class MultiTimeframeGRU(nn.Module):
    """Multi-timeframe GRU modeli"""
    
    def __init__(self, config: Dict):
        super(MultiTimeframeGRU, self).__init__()
        
        self.config = config
        self.input_size = 30  # Özellik sayısı (indikatörler + fiyat)
        self.hidden_size = config['models']['gru']['hidden_size']
        self.num_layers = config['models']['gru']['num_layers']
        self.dropout = config['models']['gru']['dropout']
        self.sequence_length = config['models']['gru']['sequence_length']
        
        # Her timeframe için ayrı GRU
        self.gru_15m = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        self.gru_1h = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        self.gru_4h = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        self.gru_1d = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        self.gru_1w = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mekanizması (basit)
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_size * 5, 128),
            nn.Tanh(),
            nn.Linear(128, 5),
            nn.Softmax(dim=1)
        )
        
        # Fusion katmanları
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_size * 5, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Çıkış katmanı: 3 sınıf (Al/Tut/Sat) + confidence
        self.output = nn.Linear(128, 4)
        
        # Sınıflandırma için softmax
        self.softmax = nn.Softmax(dim=1)
        
        logger.info(f"MultiTimeframeGRU model oluşturuldu - Hidden: {self.hidden_size}")
    
    def forward(self, x_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        x_dict: Her timeframe için tensor dict
        Returns: (predictions, confidence)
        """
        # Her timeframe için GRU çıktısı
        _, h_15m = self.gru_15m(x_dict.get('15m', torch.zeros(1, self.sequence_length, self.input_size)))
        _, h_1h = self.gru_1h(x_dict.get('1h', torch.zeros(1, self.sequence_length, self.input_size)))
        _, h_4h = self.gru_4h(x_dict.get('4h', torch.zeros(1, self.sequence_length, self.input_size)))
        _, h_1d = self.gru_1d(x_dict.get('1d', torch.zeros(1, self.sequence_length, self.input_size)))
        _, h_1w = self.gru_1w(x_dict.get('1w', torch.zeros(1, self.sequence_length, self.input_size)))
        
        # Son hidden state'leri al (num_layers, batch, hidden)
        h_15m = h_15m[-1]  # (batch, hidden)
        h_1h = h_1h[-1]
        h_4h = h_4h[-1]
        h_1d = h_1d[-1]
        h_1w = h_1w[-1]
        
        # Hidden state'leri birleştir
        combined = torch.cat([h_15m, h_1h, h_4h, h_1d, h_1w], dim=1)  # (batch, hidden*5)
        
        # Attention weights hesapla
        attention_weights = self.attention(combined)  # (batch, 5)
        
        # Weighted combination
        hidden_states = torch.stack([h_15m, h_1h, h_4h, h_1d, h_1w], dim=1)  # (batch, 5, hidden)
        attended = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1)  # (batch, hidden)
        
        # Fusion ve output
        fused = self.fusion(combined)
        output = self.output(fused)
        
        # İlk 3 çıktı sınıflandırma, 4. çıktı confidence
        predictions = self.softmax(output[:, :3])
        confidence = torch.sigmoid(output[:, 3])
        
        return predictions, confidence
    
    def prepare_features(self, data: Dict[str, pd.DataFrame], features: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """
        Raw data'dan model inputu hazırla
        """
        prepared = {}
        
        for timeframe in ['15m', '1h', '4h', '1d', '1w']:
            if timeframe in data and timeframe in features:
                # Son sequence_length kadar veriyi al
                df = features[timeframe].tail(self.sequence_length)
                
                if len(df) < self.sequence_length:
                    # Eksik veri varsa pad yap
                    padding = self.sequence_length - len(df)
                    df = pd.concat([
                        pd.DataFrame(np.zeros((padding, df.shape[1])), columns=df.columns),
                        df
                    ])
                
                # Tensor'a çevir
                tensor = torch.FloatTensor(df.values).unsqueeze(0)  # (1, seq_len, features)
                prepared[timeframe] = tensor
        
        return prepared
    
    def predict(self, data: Dict[str, pd.DataFrame], features: pd.DataFrame) -> Dict[str, float]:
        """
        Tahmin yap
        """
        self.eval()
        
        with torch.no_grad():
            # Özellikleri hazırla
            x_dict = self.prepare_features(data, features)
            
            # Tahmin
            predictions, confidence = self.forward(x_dict)
            
            # Numpy'a çevir
            pred_probs = predictions.squeeze().numpy()
            conf = confidence.item()
            
            # Sınıf tahminleri
            pred_class = np.argmax(pred_probs)
            
            # Sonuç
            result = {
                'prediction': ['SAT', 'TUT', 'AL'][pred_class],
                'probabilities': {
                    'SAT': float(pred_probs[0]),
                    'TUT': float(pred_probs[1]),
                    'AL': float(pred_probs[2])
                },
                'confidence': float(conf),
                'signal_strength': float(pred_probs[pred_class] * conf)
            }
            
            return result


class GRUTrainer:
    """GRU model eğitici"""
    
    def __init__(self, model: MultiTimeframeGRU, config: Dict):
        self.model = model
        self.config = config
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['models']['gru']['learning_rate']
        )
        
        # Loss functions
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_conf = nn.MSELoss()
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
    
    def train_epoch(self, train_loader) -> float:
        """Bir epoch eğitim"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (x_dict, y_class, y_conf) in enumerate(train_loader):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions, confidence = self.model(x_dict)
            
            # Loss hesapla
            loss_class = self.criterion_class(predictions, y_class)
            loss_conf = self.criterion_conf(confidence, y_conf)
            
            # Toplam loss (weighted)
            loss = loss_class + 0.5 * loss_conf
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (exploding gradient önlemi)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """Validasyon"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x_dict, y_class, y_conf in val_loader:
                # Tahmin
                predictions, confidence = self.model(x_dict)
                
                # Loss
                loss_class = self.criterion_class(predictions, y_class)
                loss_conf = self.criterion_conf(confidence, y_conf)
                loss = loss_class + 0.5 * loss_conf
                
                total_loss += loss.item()
                
                # Accuracy
                _, predicted = torch.max(predictions, 1)
                total += y_class.size(0)
                correct += (predicted == y_class).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, epochs: int = 50):
        """Model eğitimi"""
        logger.info(f"GRU eğitimi başlıyor - {epochs} epoch")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Model kaydet
                torch.save(self.model.state_dict(), 'models/best_gru_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= 10:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Log
            if epoch % 5 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}"
                )
        
        logger.success("GRU eğitimi tamamlandı")