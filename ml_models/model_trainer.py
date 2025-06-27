#!/usr/bin/env python3
"""
ML Model Training Module
Farklı ML algoritmalarını eğitir ve en iyi modeli seçer
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

# ML Kütüphaneleri
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score,
                           classification_report, confusion_matrix)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                            GradientBoostingClassifier, GradientBoostingRegressor,
                            AdaBoostClassifier, AdaBoostRegressor)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
import xgboost as xgb
import lightgbm as lgb

# Proje imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ml_models.feature_engineering import FeatureEngineering


class ModelTrainer:
    """ML Model eğitim sınıfı"""
    
    def __init__(self, model_type: str = 'classification'):
        """
        Args:
            model_type: 'classification' veya 'regression'
        """
        self.model_type = model_type
        self.feature_engineering = FeatureEngineering()
        self.models_dir = Path("ml_models/trained")
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # Model havuzu
        if model_type == 'classification':
            self.models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boost': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False),
                'lightgbm': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
                'logistic': LogisticRegression(max_iter=1000, random_state=42),
                'svm': SVC(probability=True, random_state=42),
                'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            }
        else:
            self.models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'xgboost': xgb.XGBRegressor(n_estimators=100, random_state=42),
                'lightgbm': lgb.LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
                'ridge': Ridge(alpha=1.0, random_state=42),
                'lasso': Lasso(alpha=0.1, random_state=42),
                'svr': SVR(kernel='rbf'),
                'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
            }
        
        self.scaler = None
        self.best_model = None
        self.best_model_name = None
        self.training_history = []
    
    def prepare_data(self, features: pd.DataFrame, target: pd.Series,
                    test_size: float = 0.2, scale_features: bool = True) -> Dict:
        """Veriyi eğitim ve test setlerine ayır"""
        
        # Time series split - zaman sırasını koru
        split_index = int(len(features) * (1 - test_size))
        
        X_train = features.iloc[:split_index]
        X_test = features.iloc[split_index:]
        y_train = target.iloc[:split_index]
        y_test = target.iloc[split_index:]
        
        # Özellik ölçekleme
        if scale_features:
            self.scaler = RobustScaler()  # Outlier'lara karşı dayanıklı
            X_train_scaled = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                index=X_train.index,
                columns=X_train.columns
            )
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                index=X_test.index,
                columns=X_test.columns
            )
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'train_index': X_train.index,
            'test_index': X_test.index
        }
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Model performansını değerlendir"""
        y_pred = model.predict(X_test)
        
        if self.model_type == 'classification':
            # Sınıflandırma metrikleri
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Olasılık tahminleri (varsa)
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            else:
                y_proba = None
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
        else:
            # Regresyon metrikleri
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mean_pred': y_pred.mean(),
                'std_pred': y_pred.std()
            }
        
        return metrics
    
    def train_single_model(self, model_name: str, model, 
                          X_train: pd.DataFrame, y_train: pd.Series,
                          X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Tek bir modeli eğit ve değerlendir"""
        logger.info(f"Training {model_name}...")
        
        start_time = datetime.now()
        
        try:
            # Modeli eğit
            model.fit(X_train, y_train)
            
            # Eğitim süresi
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Model değerlendirmesi
            train_metrics = self.evaluate_model(model, X_train, y_train)
            test_metrics = self.evaluate_model(model, X_test, y_test)
            
            # Cross-validation (time series için)
            tscv = TimeSeriesSplit(n_splits=5)
            
            if self.model_type == 'classification':
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='f1_weighted')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
                cv_scores = -cv_scores  # MSE'yi pozitif yap
            
            result = {
                'model_name': model_name,
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores.tolist(),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'feature_importance': self.get_feature_importance(model, X_train.columns)
            }
            
            logger.success(f"{model_name} trained successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return None
    
    def get_feature_importance(self, model, feature_names: List[str]) -> Optional[Dict]:
        """Model özellik önemlerini al"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
            else:
                return None
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df.to_dict('records')
            
        except:
            return None
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:
        """Tüm modelleri eğit"""
        results = {}
        
        for model_name, model in self.models.items():
            result = self.train_single_model(
                model_name, model, X_train, y_train, X_test, y_test
            )
            if result:
                results[model_name] = result
                self.training_history.append(result)
        
        return results
    
    def select_best_model(self, results: Dict[str, Dict]) -> Tuple[str, Any]:
        """En iyi modeli seç"""
        if self.model_type == 'classification':
            # F1 score'a göre sırala
            best_model_name = max(results.keys(), 
                                key=lambda k: results[k]['test_metrics']['f1_score'])
        else:
            # R2 score'a göre sırala
            best_model_name = max(results.keys(), 
                                key=lambda k: results[k]['test_metrics']['r2'])
        
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        return best_model_name, self.best_model
    
    def save_model(self, model_name: str, model: Any, metadata: Dict, 
                  symbol: str, timeframe: str):
        """Modeli kaydet"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Model dosyası
        model_file = self.models_dir / f"{symbol}_{timeframe}_{model_name}_{timestamp}.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump({
                'model': model,
                'scaler': self.scaler,
                'model_type': self.model_type,
                'model_name': model_name
            }, f)
        
        # Metadata dosyası
        metadata['timestamp'] = timestamp
        metadata['model_file'] = str(model_file)
        metadata_file = self.models_dir / f"{symbol}_{timeframe}_{model_name}_{timestamp}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved: {model_file}")
        return model_file
    
    def load_model(self, model_path: Path) -> Tuple[Any, Dict]:
        """Modeli yükle"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Metadata'yı yükle
        metadata_path = str(model_path).replace('.pkl', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return model_data, metadata
    
    def train_and_save(self, symbol: str, timeframe: str, 
                      target_column: str = 'target_direction_1'):
        """Tam eğitim pipeline'ı"""
        logger.info(f"Starting training pipeline for {symbol} {timeframe}")
        
        # Özellikleri yükle
        features, targets = self.feature_engineering.load_features(symbol, timeframe)
        
        if features.empty:
            # Özellikleri oluştur
            logger.info("Creating features...")
            features, targets = self.feature_engineering.create_feature_matrix(symbol, timeframe)
            
            if features.empty:
                logger.error("Could not create features")
                return None
            
            # Özellikleri kaydet
            self.feature_engineering.save_features(features, targets, symbol, timeframe)
        
        # Hedef değişkeni seç
        if target_column not in targets.columns:
            logger.error(f"Target column {target_column} not found")
            return None
        
        target = targets[target_column]
        
        # Veriyi hazırla
        data_split = self.prepare_data(features, target)
        
        # Tüm modelleri eğit
        results = self.train_all_models(
            data_split['X_train'], data_split['y_train'],
            data_split['X_test'], data_split['y_test']
        )
        
        # En iyi modeli seç
        best_model_name, best_model = self.select_best_model(results)
        
        logger.info(f"\nBest model: {best_model_name}")
        logger.info(f"Test metrics: {results[best_model_name]['test_metrics']}")
        
        # Modeli kaydet
        metadata = {
            'symbol': symbol,
            'timeframe': timeframe,
            'target_column': target_column,
            'features': features.columns.tolist(),
            'train_size': len(data_split['X_train']),
            'test_size': len(data_split['X_test']),
            'results': results[best_model_name]
        }
        
        model_path = self.save_model(best_model_name, best_model, metadata, symbol, timeframe)
        
        return {
            'model_path': model_path,
            'best_model_name': best_model_name,
            'results': results,
            'metadata': metadata
        }


def main():
    """Test model training"""
    # Classification örneği
    trainer = ModelTrainer(model_type='classification')
    
    symbol = "AKBNK"
    timeframe = "1h"
    
    # Eğit ve kaydet
    result = trainer.train_and_save(symbol, timeframe, target_column='target_direction_1')
    
    if result:
        print(f"\nTraining completed!")
        print(f"Best model: {result['best_model_name']}")
        print(f"Model saved to: {result['model_path']}")
        
        # Sonuçları göster
        print("\nModel comparison:")
        for model_name, model_result in result['results'].items():
            if self.model_type == 'classification':
                metric = model_result['test_metrics']['f1_score']
                print(f"{model_name:15} - F1 Score: {metric:.4f}")
            else:
                metric = model_result['test_metrics']['r2']
                print(f"{model_name:15} - R2 Score: {metric:.4f}")


if __name__ == "__main__":
    main()