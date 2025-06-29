"""
Hybrid ML+DL Ensemble Trading System

This module combines traditional ML models with deep learning models
to create a powerful ensemble trading system.
"""

import numpy as np
import pandas as pd
import torch
import logging
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime
import joblib
import sys
sys.path.append('..')

from utils.csv_data_manager import CSVDataManager
from ml_models.ml_trading_system_fixed import MLTradingSystem
from ml_models.feature_engineering import FeatureEngineer
from dl_lstm_price_predictor import DeepLearningPricePredictor
from dl_cnn_pattern_detector import CNNPatternDetectorSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridMLDLEnsemble:
    """Hybrid ensemble combining ML and DL models"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the hybrid ensemble system
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self.data_manager = CSVDataManager()
        self.feature_engineer = FeatureEngineer()
        
        # ML models
        self.ml_trading_system = MLTradingSystem()
        
        # DL models
        self.lstm_predictor = DeepLearningPricePredictor(
            model_type='lstm',
            sequence_length=self.config['lstm']['sequence_length'],
            prediction_horizon=self.config['lstm']['prediction_horizon'],
            hidden_size=self.config['lstm']['hidden_size'],
            num_layers=self.config['lstm']['num_layers']
        )
        
        self.cnn_detector = CNNPatternDetectorSystem(
            model_type='cnn',
            window_size=self.config['cnn']['window_size'],
            prediction_window=self.config['cnn']['prediction_window'],
            image_size=tuple(self.config['cnn']['image_size'])
        )
        
        # Ensemble weights
        self.model_weights = self.config['ensemble_weights']
        self.adaptive_weights = self.config.get('use_adaptive_weights', True)
        
        # Performance tracking
        self.model_performance = {
            'ml_ensemble': {'correct': 0, 'total': 0},
            'lstm': {'correct': 0, 'total': 0},
            'cnn': {'correct': 0, 'total': 0}
        }
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'lstm': {
                'sequence_length': 60,
                'prediction_horizon': 5,
                'hidden_size': 128,
                'num_layers': 3
            },
            'cnn': {
                'window_size': 20,
                'prediction_window': 5,
                'image_size': [128, 128]
            },
            'ensemble_weights': {
                'ml_ensemble': 0.4,
                'lstm': 0.35,
                'cnn': 0.25
            },
            'use_adaptive_weights': True,
            'min_confidence_threshold': 0.6,
            'signal_threshold': 0.7
        }
    
    def train_all_models(self, symbols: List[str], timeframe: str = '1d',
                        epochs: int = 50, save_models: bool = True):
        """Train all models in the ensemble"""
        logger.info("Starting hybrid ensemble training...")
        
        for symbol in symbols:
            logger.info(f"Training models for {symbol}...")
            
            # Load data
            df = self.data_manager.load_data(symbol, timeframe)
            if df is None or len(df) < 200:
                logger.warning(f"Insufficient data for {symbol}, skipping...")
                continue
            
            try:
                # Train ML models
                logger.info(f"Training ML models for {symbol}...")
                features = self.feature_engineer.create_features(df, symbol, timeframe)
                self.ml_trading_system.train(features)
                
                # Train LSTM
                logger.info(f"Training LSTM for {symbol}...")
                self.lstm_predictor.train(df, epochs=epochs)
                
                # Train CNN
                logger.info(f"Training CNN for {symbol}...")
                self.cnn_detector.train(df, epochs=epochs)
                
                # Save models if requested
                if save_models:
                    self.save_models(symbol)
                    
            except Exception as e:
                logger.error(f"Error training models for {symbol}: {str(e)}")
                continue
        
        logger.info("Hybrid ensemble training completed!")
    
    def generate_signals(self, symbol: str, timeframe: str = '1d') -> pd.DataFrame:
        """Generate trading signals using the hybrid ensemble"""
        # Load data
        df = self.data_manager.load_data(symbol, timeframe)
        if df is None or len(df) < 100:
            logger.warning(f"Insufficient data for {symbol}")
            return pd.DataFrame()
        
        # Get ML signals
        ml_signals = self._get_ml_signals(df, symbol, timeframe)
        
        # Get LSTM signals
        lstm_signals = self._get_lstm_signals(df)
        
        # Get CNN signals
        cnn_signals = self._get_cnn_signals(df)
        
        # Combine signals
        combined_signals = self._combine_signals(ml_signals, lstm_signals, cnn_signals)
        
        return combined_signals
    
    def _get_ml_signals(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get signals from ML ensemble"""
        try:
            features = self.feature_engineer.create_features(df, symbol, timeframe)
            predictions = self.ml_trading_system.predict(features)
            
            signals = pd.DataFrame(index=features.index)
            signals['ml_signal'] = predictions['signal']
            signals['ml_confidence'] = predictions['confidence']
            
            return signals
        except Exception as e:
            logger.error(f"Error getting ML signals: {str(e)}")
            return pd.DataFrame()
    
    def _get_lstm_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get signals from LSTM model"""
        try:
            lstm_results = self.lstm_predictor.generate_trading_signals(df)
            
            signals = pd.DataFrame(index=lstm_results.index)
            signals['lstm_signal'] = lstm_results['signal']
            signals['lstm_confidence'] = lstm_results['confidence']
            signals['lstm_expected_return'] = lstm_results['expected_return']
            
            return signals
        except Exception as e:
            logger.error(f"Error getting LSTM signals: {str(e)}")
            return pd.DataFrame()
    
    def _get_cnn_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get signals from CNN pattern detector"""
        try:
            patterns = self.cnn_detector.detect_patterns(df)
            
            signals = pd.DataFrame(index=patterns.index)
            signals['cnn_signal'] = patterns['signal']
            signals['cnn_confidence'] = patterns['confidence']
            signals['cnn_buy_prob'] = patterns['buy_prob']
            signals['cnn_sell_prob'] = patterns['sell_prob']
            
            return signals
        except Exception as e:
            logger.error(f"Error getting CNN signals: {str(e)}")
            return pd.DataFrame()
    
    def _combine_signals(self, ml_signals: pd.DataFrame, 
                        lstm_signals: pd.DataFrame, 
                        cnn_signals: pd.DataFrame) -> pd.DataFrame:
        """Combine signals from all models"""
        # Align indices
        common_index = ml_signals.index
        if not lstm_signals.empty:
            common_index = common_index.intersection(lstm_signals.index)
        if not cnn_signals.empty:
            common_index = common_index.intersection(cnn_signals.index)
        
        # Create combined dataframe
        combined = pd.DataFrame(index=common_index)
        
        # Add individual signals
        if not ml_signals.empty:
            combined['ml_signal'] = ml_signals.loc[common_index, 'ml_signal']
            combined['ml_confidence'] = ml_signals.loc[common_index, 'ml_confidence']
        else:
            combined['ml_signal'] = 0
            combined['ml_confidence'] = 0
        
        if not lstm_signals.empty:
            combined['lstm_signal'] = lstm_signals.loc[common_index, 'lstm_signal']
            combined['lstm_confidence'] = lstm_signals.loc[common_index, 'lstm_confidence']
        else:
            combined['lstm_signal'] = 0
            combined['lstm_confidence'] = 0
        
        if not cnn_signals.empty:
            combined['cnn_signal'] = cnn_signals.loc[common_index, 'cnn_signal']
            combined['cnn_confidence'] = cnn_signals.loc[common_index, 'cnn_confidence']
        else:
            combined['cnn_signal'] = 0
            combined['cnn_confidence'] = 0
        
        # Calculate weights
        if self.adaptive_weights:
            weights = self._calculate_adaptive_weights()
        else:
            weights = self.model_weights
        
        # Calculate weighted ensemble signal
        combined['weighted_signal'] = (
            weights['ml_ensemble'] * combined['ml_signal'] * combined['ml_confidence'] +
            weights['lstm'] * combined['lstm_signal'] * combined['lstm_confidence'] +
            weights['cnn'] * combined['cnn_signal'] * combined['cnn_confidence']
        )
        
        # Calculate ensemble confidence
        combined['ensemble_confidence'] = (
            weights['ml_ensemble'] * combined['ml_confidence'] +
            weights['lstm'] * combined['lstm_confidence'] +
            weights['cnn'] * combined['cnn_confidence']
        )
        
        # Generate final signal
        signal_threshold = self.config['signal_threshold']
        confidence_threshold = self.config['min_confidence_threshold']
        
        combined['signal'] = 0
        buy_mask = (combined['weighted_signal'] > signal_threshold) & \
                   (combined['ensemble_confidence'] > confidence_threshold)
        sell_mask = (combined['weighted_signal'] < -signal_threshold) & \
                    (combined['ensemble_confidence'] > confidence_threshold)
        
        combined.loc[buy_mask, 'signal'] = 1
        combined.loc[sell_mask, 'signal'] = -1
        
        # Add metadata
        combined['model_agreement'] = self._calculate_model_agreement(combined)
        combined['signal_strength'] = abs(combined['weighted_signal'])
        
        return combined
    
    def _calculate_adaptive_weights(self) -> Dict[str, float]:
        """Calculate adaptive weights based on recent performance"""
        weights = {}
        total_performance = 0
        
        # Calculate performance scores
        for model, perf in self.model_performance.items():
            if perf['total'] > 0:
                accuracy = perf['correct'] / perf['total']
                performance_score = accuracy * np.sqrt(perf['total'])  # Favor models with more samples
                weights[model] = performance_score
                total_performance += performance_score
            else:
                weights[model] = self.model_weights[model]
                total_performance += self.model_weights[model]
        
        # Normalize weights
        if total_performance > 0:
            for model in weights:
                weights[model] /= total_performance
        
        # Apply smoothing with default weights
        alpha = 0.7  # Smoothing factor
        for model in weights:
            weights[model] = alpha * weights[model] + (1 - alpha) * self.model_weights[model]
        
        return weights
    
    def _calculate_model_agreement(self, signals: pd.DataFrame) -> pd.Series:
        """Calculate agreement between models"""
        ml_direction = np.sign(signals['ml_signal'])
        lstm_direction = np.sign(signals['lstm_signal'])
        cnn_direction = np.sign(signals['cnn_signal'])
        
        agreement = (ml_direction == lstm_direction).astype(int) + \
                   (ml_direction == cnn_direction).astype(int) + \
                   (lstm_direction == cnn_direction).astype(int)
        
        return agreement / 3.0
    
    def update_performance(self, symbol: str, timeframe: str, lookback_days: int = 30):
        """Update model performance based on recent results"""
        df = self.data_manager.load_data(symbol, timeframe)
        if df is None or len(df) < lookback_days:
            return
        
        # Get recent signals
        signals = self.generate_signals(symbol, timeframe)
        if signals.empty:
            return
        
        # Calculate actual returns
        df = df.loc[signals.index]
        df['returns'] = df['Close'].pct_change().shift(-1)
        
        # Evaluate each model
        for model in ['ml', 'lstm', 'cnn']:
            signal_col = f'{model}_signal'
            if signal_col in signals.columns:
                correct = ((signals[signal_col] > 0) & (df['returns'] > 0)) | \
                         ((signals[signal_col] < 0) & (df['returns'] < 0))
                
                total = (signals[signal_col] != 0).sum()
                correct_count = correct.sum()
                
                # Update performance
                model_key = model if model != 'ml' else 'ml_ensemble'
                self.model_performance[model_key]['correct'] += correct_count
                self.model_performance[model_key]['total'] += total
    
    def save_models(self, symbol: str):
        """Save all models"""
        # Save ML models
        self.ml_trading_system.save_models(f"models/{symbol}_ml_ensemble")
        
        # Save LSTM
        self.lstm_predictor.save_model(f"dl_models/{symbol}_lstm_hybrid.pth")
        
        # Save CNN
        self.cnn_detector.save_model(f"dl_models/{symbol}_cnn_hybrid.pth")
        
        # Save ensemble config and performance
        ensemble_data = {
            'config': self.config,
            'model_performance': self.model_performance,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f"dl_models/{symbol}_hybrid_ensemble_config.json", 'w') as f:
            json.dump(ensemble_data, f, indent=4)
    
    def load_models(self, symbol: str):
        """Load all models"""
        try:
            # Load ML models
            self.ml_trading_system.load_models(f"models/{symbol}_ml_ensemble")
            
            # Load LSTM
            self.lstm_predictor.load_model(f"dl_models/{symbol}_lstm_hybrid.pth")
            
            # Load CNN
            self.cnn_detector.load_model(f"dl_models/{symbol}_cnn_hybrid.pth")
            
            # Load ensemble config
            with open(f"dl_models/{symbol}_hybrid_ensemble_config.json", 'r') as f:
                ensemble_data = json.load(f)
                self.config = ensemble_data['config']
                self.model_performance = ensemble_data['model_performance']
            
            logger.info(f"Successfully loaded hybrid ensemble for {symbol}")
            
        except Exception as e:
            logger.error(f"Error loading models for {symbol}: {str(e)}")


class HybridTradingStrategy:
    """Trading strategy using the hybrid ensemble"""
    
    def __init__(self, ensemble: HybridMLDLEnsemble, 
                 risk_per_trade: float = 0.02,
                 max_positions: int = 10):
        self.ensemble = ensemble
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.positions = {}
        
    def evaluate_symbols(self, symbols: List[str], timeframe: str = '1d') -> pd.DataFrame:
        """Evaluate multiple symbols and rank by signal strength"""
        results = []
        
        for symbol in symbols:
            try:
                signals = self.ensemble.generate_signals(symbol, timeframe)
                if signals.empty or len(signals) < 1:
                    continue
                
                latest = signals.iloc[-1]
                
                if latest['signal'] != 0:
                    results.append({
                        'symbol': symbol,
                        'signal': latest['signal'],
                        'confidence': latest['ensemble_confidence'],
                        'strength': latest['signal_strength'],
                        'model_agreement': latest['model_agreement'],
                        'ml_confidence': latest.get('ml_confidence', 0),
                        'lstm_confidence': latest.get('lstm_confidence', 0),
                        'cnn_confidence': latest.get('cnn_confidence', 0)
                    })
                    
            except Exception as e:
                logger.error(f"Error evaluating {symbol}: {str(e)}")
                continue
        
        # Create dataframe and sort by strength
        df_results = pd.DataFrame(results)
        if not df_results.empty:
            df_results = df_results.sort_values('strength', ascending=False)
        
        return df_results
    
    def generate_orders(self, symbols: List[str], capital: float, 
                       timeframe: str = '1d') -> List[Dict]:
        """Generate trading orders based on ensemble signals"""
        # Evaluate all symbols
        candidates = self.evaluate_symbols(symbols, timeframe)
        
        if candidates.empty:
            logger.info("No trading signals found")
            return []
        
        # Filter by available positions
        available_positions = self.max_positions - len(self.positions)
        if available_positions <= 0:
            logger.info("Maximum positions reached")
            return []
        
        # Select top candidates
        selected = candidates.head(available_positions)
        
        # Generate orders
        orders = []
        position_size = capital * self.risk_per_trade
        
        for _, row in selected.iterrows():
            order = {
                'symbol': row['symbol'],
                'side': 'buy' if row['signal'] > 0 else 'sell',
                'size': position_size,
                'confidence': row['confidence'],
                'strength': row['strength'],
                'model_agreement': row['model_agreement'],
                'timestamp': datetime.now().isoformat()
            }
            orders.append(order)
            
            # Update positions
            self.positions[row['symbol']] = order
        
        return orders


def main():
    """Test the hybrid ensemble system"""
    # Configuration
    symbols = ['ASELS', 'THYAO', 'SISE', 'GARAN', 'KCHOL']
    
    # Initialize ensemble
    ensemble = HybridMLDLEnsemble()
    
    # Train models (commented out for testing)
    # ensemble.train_all_models(symbols, epochs=30)
    
    # Test signal generation
    for symbol in symbols[:2]:
        logger.info(f"\nTesting signals for {symbol}...")
        
        signals = ensemble.generate_signals(symbol)
        if not signals.empty:
            print(f"\nLatest signals for {symbol}:")
            print(signals.tail())
            
            # Show signal distribution
            print(f"\nSignal distribution for {symbol}:")
            print(signals['signal'].value_counts())
    
    # Test trading strategy
    strategy = HybridTradingStrategy(ensemble)
    
    # Evaluate symbols
    rankings = strategy.evaluate_symbols(symbols)
    print("\nSymbol Rankings:")
    print(rankings)
    
    # Generate orders
    orders = strategy.generate_orders(symbols, capital=100000)
    print("\nGenerated Orders:")
    for order in orders:
        print(order)


if __name__ == "__main__":
    main()