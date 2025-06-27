#!/usr/bin/env python3
"""
ML Prediction Module
Eğitilmiş modellerle tahmin yapar
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import pickle
import json
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Proje imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_models.feature_engineering import FeatureEngineering
from ml_models.model_trainer import ModelTrainer
from utils.csv_data_manager import CSVDataManager
from indicators.indicator_calculator import IndicatorCalculator


class Predictor:
    """ML tahmin sınıfı"""
    
    def __init__(self):
        self.feature_engineering = FeatureEngineering()
        self.csv_manager = CSVDataManager()
        self.indicator_calculator = IndicatorCalculator()
        
        # Yüklü modeller
        self.loaded_models = {}
        self.models_dir = Path("ml_models/trained")
        
        # Tahmin sonuçları
        self.predictions_dir = Path("data/predictions")
        self.predictions_dir.mkdir(exist_ok=True, parents=True)
    
    def load_latest_model(self, symbol: str, timeframe: str, 
                         model_type: str = None) -> Tuple[Any, Dict]:
        """En güncel modeli yükle"""
        try:
            # Model dosyalarını bul
            pattern = f"{symbol}_{timeframe}_*"
            if model_type:
                pattern = f"{symbol}_{timeframe}_{model_type}_*"
            
            model_files = list(self.models_dir.glob(f"{pattern}.pkl"))
            
            if not model_files:
                logger.error(f"No model found for {symbol} {timeframe}")
                return None, {}
            
            # En yeni modeli seç (timestamp'e göre)
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            
            # Modeli yükle
            with open(latest_model, 'rb') as f:
                model_data = pickle.load(f)
            
            # Metadata'yı yükle
            metadata_path = str(latest_model).replace('.pkl', '_metadata.json')
            if Path(metadata_path).exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            logger.info(f"Loaded model: {latest_model.name}")
            return model_data, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, {}
    
    def prepare_live_features(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Canlı tahmin için özellikleri hazırla"""
        try:
            # Son veriyi al
            df = self.csv_manager.load_raw_data(symbol, timeframe)
            if df is None or len(df) < 200:
                logger.error(f"Insufficient data for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # İndikatörleri hesapla (eğer yoksa)
            # Not: İndikatörler zaten hesaplanmış olmalı, yoksa feature engineering hesaplar
            
            # Özellikleri oluştur
            features, _ = self.feature_engineering.create_feature_matrix(symbol, timeframe)
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def predict_single(self, model_data: Dict, features: pd.DataFrame, 
                      last_n: int = 1) -> pd.DataFrame:
        """Tek model ile tahmin yap"""
        try:
            model = model_data['model']
            scaler = model_data.get('scaler')
            model_type = model_data.get('model_type', 'classification')
            
            # Son N satırı al
            X = features.iloc[-last_n:]
            
            # Ölçekleme (eğer scaler varsa)
            if scaler:
                X_scaled = pd.DataFrame(
                    scaler.transform(X),
                    index=X.index,
                    columns=X.columns
                )
            else:
                X_scaled = X
            
            # Tahmin yap
            predictions = model.predict(X_scaled)
            
            # Olasılıkları al (classification için)
            if model_type == 'classification' and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
                # En yüksek olasılık
                max_proba = np.max(probabilities, axis=1)
            else:
                probabilities = None
                max_proba = None
            
            # Sonuçları DataFrame'e dönüştür
            results = pd.DataFrame({
                'timestamp': X.index,
                'prediction': predictions,
                'confidence': max_proba if max_proba is not None else 1.0,
                'model_type': model_type
            })
            
            # Olasılıkları ekle (varsa)
            if probabilities is not None:
                for i in range(probabilities.shape[1]):
                    results[f'proba_class_{i}'] = probabilities[:, i]
            
            return results
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return pd.DataFrame()
    
    def predict_ensemble(self, symbol: str, timeframe: str, 
                        last_n: int = 1) -> pd.DataFrame:
        """Birden fazla model ile ensemble tahmin"""
        try:
            # Mevcut tüm modelleri yükle
            model_files = list(self.models_dir.glob(f"{symbol}_{timeframe}_*.pkl"))
            
            if not model_files:
                logger.error(f"No models found for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Özellikleri hazırla
            features = self.prepare_live_features(symbol, timeframe)
            if features.empty:
                return pd.DataFrame()
            
            # Her model için tahmin yap
            all_predictions = []
            
            for model_file in model_files:
                # Modeli yükle
                with open(model_file, 'rb') as f:
                    model_data = pickle.load(f)
                
                # Tahmin yap
                pred = self.predict_single(model_data, features, last_n)
                if not pred.empty:
                    pred['model_name'] = model_data.get('model_name', 'unknown')
                    all_predictions.append(pred)
            
            if not all_predictions:
                logger.error("No successful predictions")
                return pd.DataFrame()
            
            # Tahminleri birleştir
            ensemble_df = pd.concat(all_predictions, ignore_index=True)
            
            # Ensemble tahmin (çoğunluk oyu veya ortalama)
            if model_data.get('model_type') == 'classification':
                # Çoğunluk oyu
                ensemble_pred = ensemble_df.groupby('timestamp')['prediction'].agg(
                    lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
                )
                # Ortalama güven
                ensemble_conf = ensemble_df.groupby('timestamp')['confidence'].mean()
            else:
                # Ortalama tahmin
                ensemble_pred = ensemble_df.groupby('timestamp')['prediction'].mean()
                ensemble_conf = ensemble_df.groupby('timestamp')['confidence'].mean()
            
            # Final sonuç
            result = pd.DataFrame({
                'timestamp': ensemble_pred.index,
                'ensemble_prediction': ensemble_pred.values,
                'ensemble_confidence': ensemble_conf.values,
                'num_models': len(model_files)
            })
            
            # Model detaylarını ekle
            for model_name in ensemble_df['model_name'].unique():
                model_preds = ensemble_df[ensemble_df['model_name'] == model_name]
                result[f'{model_name}_pred'] = model_preds.set_index('timestamp')['prediction']
                result[f'{model_name}_conf'] = model_preds.set_index('timestamp')['confidence']
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return pd.DataFrame()
    
    def generate_signals(self, predictions: pd.DataFrame, 
                        threshold: float = 0.6) -> pd.DataFrame:
        """Tahminlerden trading sinyalleri üret"""
        signals = predictions.copy()
        
        # Ensemble tahmin varsa onu kullan
        if 'ensemble_prediction' in signals.columns:
            pred_col = 'ensemble_prediction'
            conf_col = 'ensemble_confidence'
        else:
            pred_col = 'prediction'
            conf_col = 'confidence'
        
        # Sinyal üret (güven eşiğini de kullan)
        signals['signal'] = 0  # Nötr
        
        # Yüksek güvenli alım sinyali
        buy_condition = (signals[pred_col] > 0) & (signals[conf_col] >= threshold)
        signals.loc[buy_condition, 'signal'] = 1
        
        # Yüksek güvenli satım sinyali
        sell_condition = (signals[pred_col] < 0) & (signals[conf_col] >= threshold)
        signals.loc[sell_condition, 'signal'] = -1
        
        # Sinyal gücü
        signals['signal_strength'] = signals[conf_col] * abs(signals[pred_col])
        
        return signals
    
    def predict_and_save(self, symbol: str, timeframe: str, 
                        use_ensemble: bool = True) -> Dict:
        """Tahmin yap ve kaydet"""
        try:
            logger.info(f"Making predictions for {symbol} {timeframe}")
            
            if use_ensemble:
                # Ensemble tahmin
                predictions = self.predict_ensemble(symbol, timeframe)
            else:
                # Tek model tahmin
                model_data, metadata = self.load_latest_model(symbol, timeframe)
                if model_data is None:
                    logger.error("No model available")
                    return {}
                
                features = self.prepare_live_features(symbol, timeframe)
                if features.empty:
                    return {}
                
                predictions = self.predict_single(model_data, features)
            
            if predictions.empty:
                logger.error("Prediction failed")
                return {}
            
            # Sinyalleri üret
            signals = self.generate_signals(predictions)
            
            # Sonuçları kaydet
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Tahminler
            pred_file = self.predictions_dir / f"{symbol}_{timeframe}_predictions_{timestamp}.csv"
            predictions.to_csv(pred_file, index=False)
            
            # Sinyaller
            signal_file = self.predictions_dir / f"{symbol}_{timeframe}_signals_{timestamp}.csv"
            signals.to_csv(signal_file, index=False)
            
            # Son sinyali al
            last_signal = signals.iloc[-1]
            
            result = {
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': last_signal['timestamp'],
                'signal': int(last_signal['signal']),
                'signal_strength': float(last_signal['signal_strength']),
                'confidence': float(last_signal.get('ensemble_confidence', last_signal.get('confidence', 0))),
                'prediction_file': str(pred_file),
                'signal_file': str(signal_file)
            }
            
            # Sinyal tipini belirle
            if result['signal'] == 1:
                result['action'] = 'BUY'
            elif result['signal'] == -1:
                result['action'] = 'SELL'
            else:
                result['action'] = 'HOLD'
            
            logger.info(f"Prediction complete: {result['action']} (confidence: {result['confidence']:.2%})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in predict_and_save: {e}")
            return {}
    
    def batch_predict(self, symbols: List[str] = None, 
                     timeframes: List[str] = None) -> List[Dict]:
        """Toplu tahmin yap"""
        if symbols is None:
            # Trained modelleri bul
            model_files = list(self.models_dir.glob("*.pkl"))
            symbols = list(set([f.name.split('_')[0] for f in model_files]))
        
        if timeframes is None:
            timeframes = ['15m', '1h', '4h', '1d']
        
        results = []
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Model var mı kontrol et
                if not list(self.models_dir.glob(f"{symbol}_{timeframe}_*.pkl")):
                    continue
                
                result = self.predict_and_save(symbol, timeframe)
                if result:
                    results.append(result)
        
        # Özet rapor
        if results:
            logger.info(f"\nBatch prediction completed: {len(results)} predictions")
            
            # Sinyal özeti
            buy_signals = sum(1 for r in results if r['signal'] == 1)
            sell_signals = sum(1 for r in results if r['signal'] == -1)
            hold_signals = sum(1 for r in results if r['signal'] == 0)
            
            logger.info(f"Signals: {buy_signals} BUY, {sell_signals} SELL, {hold_signals} HOLD")
            
            # En güçlü sinyaller
            sorted_results = sorted(results, key=lambda x: x['signal_strength'], reverse=True)
            
            logger.info("\nTop 5 strongest signals:")
            for i, result in enumerate(sorted_results[:5]):
                logger.info(f"{i+1}. {result['symbol']} {result['timeframe']}: "
                          f"{result['action']} (strength: {result['signal_strength']:.3f})")
        
        return results


def main():
    """Test prediction"""
    predictor = Predictor()
    
    # Test için tek tahmin
    symbol = "AKBNK"
    timeframe = "1h"
    
    result = predictor.predict_and_save(symbol, timeframe, use_ensemble=True)
    
    if result:
        print(f"\nPrediction Result:")
        print(f"Symbol: {result['symbol']}")
        print(f"Timeframe: {result['timeframe']}")
        print(f"Action: {result['action']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Signal Strength: {result['signal_strength']:.3f}")
    
    # Toplu tahmin testi
    print("\nRunning batch predictions...")
    batch_results = predictor.batch_predict(
        symbols=["AKBNK", "THYAO", "GARAN"],
        timeframes=["1h", "4h"]
    )
    
    print(f"\nBatch completed: {len(batch_results)} predictions generated")


if __name__ == "__main__":
    main()