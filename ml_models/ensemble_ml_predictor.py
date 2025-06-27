#!/usr/bin/env python3
"""
Ensemble ML Tahmin Sistemi
XGBoost, LightGBM ve Neural Network kombinasyonu
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Proje imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.csv_data_manager import CSVDataManager
from ml_models.feature_engineering import FeatureEngineering


class EnsembleMLPredictor:
    """Çoklu ML modeli kullanan tahmin sistemi"""
    
    def __init__(self):
        self.csv_manager = CSVDataManager()
        self.feature_engineer = FeatureEngineering()
        
        # Model havuzu
        self.models = {
            'xgboost': None,
            'lightgbm': None,
            'random_forest': None,
            'voting_ensemble': None
        }
        
        # Model konfigürasyonları
        self.model_configs = {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.01,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 200,
                'num_leaves': 31,
                'learning_rate': 0.01,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'random_state': 42,
                'verbose': -1
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42,
                'n_jobs': -1
            }
        }
        
        # Scaler'lar
        self.feature_scaler = RobustScaler()
        self.is_fitted = False
        
        # Model performans takibi
        self.model_performance = {
            'xgboost': {'accuracy': 0, 'predictions': []},
            'lightgbm': {'accuracy': 0, 'predictions': []},
            'random_forest': {'accuracy': 0, 'predictions': []},
            'ensemble': {'accuracy': 0, 'predictions': []}
        }
        
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gelişmiş özellik mühendisliği"""
        features = pd.DataFrame(index=df.index)
        
        # Fiyat özellikleri
        features['returns_1'] = df['close'].pct_change(1)
        features['returns_5'] = df['close'].pct_change(5)
        features['returns_20'] = df['close'].pct_change(20)
        
        # Logaritmik getiriler
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatilite özellikleri
        features['volatility_20'] = features['returns_1'].rolling(20).std()
        features['volatility_ratio'] = features['volatility_20'] / features['volatility_20'].rolling(60).mean()
        
        # Price action
        features['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        features['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
        
        # Volume özellikleri
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        # Momentum özellikleri
        features['roc_10'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
        features['rsi_14'] = self._calculate_rsi(df['close'], 14)
        
        # Trend özellikleri
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        features['price_to_sma20'] = df['close'] / sma_20
        features['price_to_sma50'] = df['close'] / sma_50
        features['sma_cross'] = (sma_20 - sma_50) / df['close']
        
        # Market mikro yapısı
        features['spread'] = (df['high'] - df['low']) / df['close']
        features['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
        features['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, 1)
        
        # İndikatör özellikleri ekle
        indicator_features = self._load_indicator_features(df)
        if indicator_features is not None:
            features = pd.concat([features, indicator_features], axis=1)
        
        # Temporal özellikler
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        
        # Temizlik
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _load_indicator_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """İndikatör özelliklerini yükle"""
        # Bu fonksiyon gerçek uygulamada indikatör verilerini yükleyecek
        # Şimdilik basit bir örnek
        indicator_features = pd.DataFrame(index=df.index)
        
        # Dummy indikatör özellikleri
        indicator_features['macd_signal'] = 0
        indicator_features['supertrend_signal'] = 0
        indicator_features['wavetrend_signal'] = 0
        
        return indicator_features
    
    def prepare_training_data(self, symbol: str, timeframe: str = '1h', 
                            lookback_days: int = 365) -> Tuple[np.ndarray, np.ndarray]:
        """Eğitim verisi hazırla"""
        # Ham veri yükle
        df = self.csv_manager.load_raw_data(symbol, timeframe)
        
        if df is None or len(df) < 200:
            raise ValueError(f"{symbol} için yeterli veri yok")
        
        # Son lookback_days günlük veriyi al
        end_date = df.index[-1]
        start_date = end_date - timedelta(days=lookback_days)
        df = df[df.index >= start_date]
        
        # Özellikler oluştur
        features = self.create_advanced_features(df)
        
        # Hedef değişken oluştur (5 bar sonraki getiri)
        future_returns = df['close'].shift(-5) / df['close'] - 1
        
        # Sınıflandırma için kategorize et
        # 0: Düşüş, 1: Yatay, 2: Yükseliş
        y = pd.cut(future_returns, 
                   bins=[-np.inf, -0.01, 0.01, np.inf], 
                   labels=[0, 1, 2]).astype(int)
        
        # NaN'ları temizle
        mask = ~(features.isna().any(axis=1) | y.isna())
        X = features[mask].values
        y = y[mask].values
        
        return X, y
    
    def train_models(self, symbols: List[str], timeframe: str = '1h'):
        """Modelleri eğit"""
        logger.info("Model eğitimi başlıyor...")
        
        # Tüm semboller için veri topla
        all_X = []
        all_y = []
        
        for symbol in symbols:
            try:
                X, y = self.prepare_training_data(symbol, timeframe)
                all_X.append(X)
                all_y.append(y)
                logger.info(f"{symbol} verisi hazırlandı: {len(X)} örnek")
            except Exception as e:
                logger.error(f"{symbol} veri hazırlama hatası: {e}")
        
        if not all_X:
            raise ValueError("Hiç eğitim verisi hazırlanamadı")
        
        # Verileri birleştir
        X = np.vstack(all_X)
        y = np.hstack(all_y)
        
        logger.info(f"Toplam eğitim verisi: {len(X)} örnek")
        
        # Veriyi ölçekle
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Train/validation split (time series için)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # XGBoost
        logger.info("XGBoost eğitiliyor...")
        self.models['xgboost'] = xgb.XGBClassifier(**self.model_configs['xgboost'])
        self.models['xgboost'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # LightGBM
        logger.info("LightGBM eğitiliyor...")
        self.models['lightgbm'] = lgb.LGBMClassifier(**self.model_configs['lightgbm'])
        self.models['lightgbm'].fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Random Forest
        logger.info("Random Forest eğitiliyor...")
        self.models['random_forest'] = RandomForestClassifier(**self.model_configs['random_forest'])
        self.models['random_forest'].fit(X_train, y_train)
        
        # Voting Ensemble
        logger.info("Ensemble oluşturuluyor...")
        self.models['voting_ensemble'] = VotingClassifier(
            estimators=[
                ('xgb', self.models['xgboost']),
                ('lgb', self.models['lightgbm']),
                ('rf', self.models['random_forest'])
            ],
            voting='soft'
        )
        self.models['voting_ensemble'].fit(X_train, y_train)
        
        # Model performanslarını değerlendir
        self._evaluate_models(X_val, y_val)
        
        self.is_fitted = True
        logger.info("Model eğitimi tamamlandı")
    
    def _evaluate_models(self, X_val: np.ndarray, y_val: np.ndarray):
        """Model performanslarını değerlendir"""
        for model_name, model in self.models.items():
            if model is not None:
                y_pred = model.predict(X_val)
                accuracy = (y_pred == y_val).mean()
                self.model_performance[model_name]['accuracy'] = accuracy
                logger.info(f"{model_name} doğruluk: {accuracy:.3f}")
    
    def predict(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Tahmin yap"""
        if not self.is_fitted:
            raise ValueError("Modeller henüz eğitilmedi")
        
        # Veri hazırla
        df = self.csv_manager.load_raw_data(symbol, timeframe)
        
        if df is None or len(df) < 100:
            raise ValueError(f"{symbol} için yeterli veri yok")
        
        # Son 100 bar
        df = df.iloc[-100:]
        
        # Özellikler
        features = self.create_advanced_features(df)
        X = features.iloc[-1:].values
        X_scaled = self.feature_scaler.transform(X)
        
        # Tahminler
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models.items():
            if model is not None:
                pred = model.predict(X_scaled)[0]
                proba = model.predict_proba(X_scaled)[0]
                
                predictions[model_name] = int(pred)
                probabilities[model_name] = proba.tolist()
        
        # Ensemble kararı
        ensemble_pred = predictions['voting_ensemble']
        ensemble_proba = probabilities['voting_ensemble']
        
        # Sinyal çevir
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        signal = signal_map[ensemble_pred]
        confidence = max(ensemble_proba)
        
        # Detaylı sonuç
        result = {
            'signal': signal,
            'confidence': confidence,
            'class_probabilities': {
                'sell': ensemble_proba[0],
                'hold': ensemble_proba[1],
                'buy': ensemble_proba[2]
            },
            'model_predictions': predictions,
            'model_probabilities': probabilities,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, symbols: List[str], timeframe: str = '1h') -> Dict[str, Dict]:
        """Birden fazla sembol için tahmin"""
        results = {}
        
        for symbol in symbols:
            try:
                results[symbol] = self.predict(symbol, timeframe)
            except Exception as e:
                logger.error(f"{symbol} tahmin hatası: {e}")
                results[symbol] = {
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'error': str(e)
                }
        
        return results
    
    def get_feature_importance(self) -> Dict:
        """Özellik önemlerini al"""
        if not self.is_fitted:
            return {}
        
        importance_dict = {}
        
        # XGBoost
        if self.models['xgboost'] is not None:
            importance_dict['xgboost'] = self.models['xgboost'].feature_importances_
        
        # LightGBM
        if self.models['lightgbm'] is not None:
            importance_dict['lightgbm'] = self.models['lightgbm'].feature_importances_
        
        # Random Forest
        if self.models['random_forest'] is not None:
            importance_dict['random_forest'] = self.models['random_forest'].feature_importances_
        
        return importance_dict
    
    def save_models(self, directory: str = 'ml_models/trained/ensemble'):
        """Modelleri kaydet"""
        save_dir = Path(directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Modelleri kaydet
        for model_name, model in self.models.items():
            if model is not None:
                model_path = save_dir / f"{model_name}_model.pkl"
                joblib.dump(model, model_path)
                logger.info(f"{model_name} kaydedildi: {model_path}")
        
        # Scaler'ı kaydet
        scaler_path = save_dir / "feature_scaler.pkl"
        joblib.dump(self.feature_scaler, scaler_path)
        
        # Metadata kaydet
        metadata = {
            'model_configs': self.model_configs,
            'model_performance': self.model_performance,
            'training_date': datetime.now().isoformat(),
            'is_fitted': self.is_fitted
        }
        
        metadata_path = save_dir / "ensemble_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_models(self, directory: str = 'ml_models/trained/ensemble'):
        """Modelleri yükle"""
        load_dir = Path(directory)
        
        if not load_dir.exists():
            raise ValueError(f"Model dizini bulunamadı: {directory}")
        
        # Modelleri yükle
        for model_name in self.models.keys():
            model_path = load_dir / f"{model_name}_model.pkl"
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                logger.info(f"{model_name} yüklendi")
        
        # Scaler'ı yükle
        scaler_path = load_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            self.feature_scaler = joblib.load(scaler_path)
        
        # Metadata yükle
        metadata_path = load_dir / "ensemble_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_performance = metadata.get('model_performance', {})
                self.is_fitted = metadata.get('is_fitted', False)


def main():
    """Test fonksiyonu"""
    predictor = EnsembleMLPredictor()
    
    # Test sembolleri
    symbols = ['THYAO', 'GARAN', 'AKBNK']
    
    print("Model eğitimi başlıyor...")
    try:
        predictor.train_models(symbols[:2])  # İlk 2 sembol ile eğit
        
        # Modelleri kaydet
        predictor.save_models()
        
        # Test tahmini
        print("\nTest tahminleri:")
        predictions = predictor.predict_batch(symbols)
        
        for symbol, pred in predictions.items():
            if 'error' not in pred:
                print(f"\n{symbol}:")
                print(f"  Sinyal: {pred['signal']}")
                print(f"  Güven: {pred['confidence']:.3f}")
                print(f"  Olasılıklar: Buy={pred['class_probabilities']['buy']:.3f}, "
                      f"Hold={pred['class_probabilities']['hold']:.3f}, "
                      f"Sell={pred['class_probabilities']['sell']:.3f}")
            else:
                print(f"\n{symbol}: Hata - {pred['error']}")
        
    except Exception as e:
        logger.error(f"Test hatası: {e}")


if __name__ == "__main__":
    main()