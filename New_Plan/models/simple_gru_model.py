"""
Simplified Multi-Timeframe GRU Model
Direct parameter version for easier initialization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class SimpleMultiTimeframeGRU(nn.Module):
    """Simplified multi-timeframe GRU model"""
    
    def __init__(self, input_size: int = 50, hidden_size: int = 50, 
                 num_layers: int = 1, dropout: float = 0.2):
        super(SimpleMultiTimeframeGRU, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Single shared GRU for all timeframes (simpler)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention layer for timeframe weighting
        self.attention = nn.Linear(hidden_size, 1)
        
        # Output layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x_15m, x_1h, x_4h, x_1d, x_1w):
        """Forward pass for all timeframes"""
        # Process each timeframe
        outputs = []
        hidden_states = []
        
        for x in [x_15m, x_1h, x_4h, x_1d, x_1w]:
            if x is not None:
                out, hidden = self.gru(x)
                # Take last hidden state
                last_hidden = hidden[-1]  # (batch, hidden_size)
                outputs.append(out)
                hidden_states.append(last_hidden)
        
        if not hidden_states:
            raise ValueError("No valid timeframe data provided")
        
        # Stack hidden states
        hidden_stack = torch.stack(hidden_states, dim=1)  # (batch, num_timeframes, hidden_size)
        
        # Calculate attention weights
        attention_scores = self.attention(hidden_stack)  # (batch, num_timeframes, 1)
        attention_weights = torch.softmax(attention_scores.squeeze(-1), dim=1)  # (batch, num_timeframes)
        
        # Apply attention
        weighted_hidden = torch.bmm(
            attention_weights.unsqueeze(1), 
            hidden_stack
        ).squeeze(1)  # (batch, hidden_size)
        
        # Output layers
        x = self.relu(self.fc1(weighted_hidden))
        x = self.dropout_layer(x)
        output = self.fc2(x)
        
        return output, attention_weights
    
    def predict(self, x_15m, x_1h, x_4h, x_1d, x_1w):
        """Make prediction with model in eval mode"""
        self.eval()
        with torch.no_grad():
            output, attention = self.forward(x_15m, x_1h, x_4h, x_1d, x_1w)
            # Apply sigmoid for probability
            prob = torch.sigmoid(output)
            return prob, attention