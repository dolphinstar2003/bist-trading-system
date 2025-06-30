"""
Feature Engineering Module
ML/DL modeller için özellik hazırlama
Multi-timeframe ve indicator bazlı özellikler
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Try to import talib, use fallback if not available
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available, using fallback implementations")
    TALIB_AVAILABLE = False


class FeatureEngineering:
    """Özellik mühendisliği ve hazırlama"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Feature grupları
        self.feature_groups = {
            'price': ['returns', 'log_returns', 'price_position', 'price_momentum'],
            'trend': ['macd_signal', 'ema_signals', 'adx_signal', 'trend_strength'],
            'momentum': ['rsi_signal', 'dmi_signal', 'stoch_signal', 'momentum_score'],
            'volatility': ['atr_signal', 'bb_signal', 'volatility_regime'],
            'volume': ['obv_signal', 'volume_signal', 'mfi_signal'],
            'pattern': ['candle_patterns', 'support_resistance', 'price_action'],
            'multi_tf': ['tf_alignment', 'tf_divergence', 'trend_consistency'],
            'macro': ['market_regime', 'correlation_features']
        }
        
        # Scalers for normalization
        self.scalers = {}
        
        logger.info("FeatureEngineering modülü başlatıldı")
    
    def create_features(self, data: Dict[str, pd.DataFrame], symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Ana feature oluşturma fonksiyonu
        
        Args:
            data: Multi-timeframe veri dict'i
            symbol: Sembol adı
            
        Returns:
            Dict[timeframe, features_df]
        """
        features = {}
        
        # Her timeframe için özellik oluştur
        for tf in self.config['timeframes']['analysis']:
            if 'indicators' in data and tf in data['indicators']:
                # Raw + indicator data
                df = data['indicators'][tf]
                
                # Temel özellikler
                tf_features = pd.DataFrame(index=df.index)
                
                # Price features
                price_feats = self._create_price_features(df)
                tf_features = pd.concat([tf_features, price_feats], axis=1)
                
                # Trend features
                trend_feats = self._create_trend_features(df)
                tf_features = pd.concat([tf_features, trend_feats], axis=1)
                
                # Momentum features
                momentum_feats = self._create_momentum_features(df)
                tf_features = pd.concat([tf_features, momentum_feats], axis=1)
                
                # Volatility features
                volatility_feats = self._create_volatility_features(df)
                tf_features = pd.concat([tf_features, volatility_feats], axis=1)
                
                # Volume features
                volume_feats = self._create_volume_features(df)
                tf_features = pd.concat([tf_features, volume_feats], axis=1)
                
                # Pattern features
                pattern_feats = self._create_pattern_features(df)
                tf_features = pd.concat([tf_features, pattern_feats], axis=1)
                
                features[tf] = tf_features
        
        # Multi-timeframe features
        if len(features) > 1:
            mtf_features = self._create_multi_timeframe_features(features)
            # Her timeframe'e MTF özellikleri ekle
            for tf in features:
                features[tf] = pd.concat([features[tf], mtf_features], axis=1)
        
        # Macro features ekle
        if 'macro' in data:
            macro_features = self._create_macro_features(data['macro'])
            for tf in features:
                for col, val in macro_features.items():
                    features[tf][col] = val
        
        # Sentiment features ekle
        if 'sentiment' in data:
            sentiment_features = self._create_sentiment_features(data['sentiment'])
            for tf in features:
                for col, val in sentiment_features.items():
                    features[tf][col] = val
        
        # NaN handling
        for tf in features:
            features[tf] = features[tf].ffill().fillna(0)
        
        logger.info(f"{symbol} için {len(features)} timeframe'de özellikler oluşturuldu")
        
        return features
    
    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fiyat bazlı özellikler"""
        features = pd.DataFrame(index=df.index)
        
        # Returns
        features['returns_1'] = df['close'].pct_change(1, fill_method=None)
        features['returns_5'] = df['close'].pct_change(5, fill_method=None)
        features['returns_10'] = df['close'].pct_change(10, fill_method=None)
        features['returns_20'] = df['close'].pct_change(20, fill_method=None)
        
        # Log returns
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Price position (0-1 arası, günlük range'de nerede)
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Price momentum
        features['price_momentum_5'] = df['close'] / df['close'].shift(5) - 1
        features['price_momentum_10'] = df['close'] / df['close'].shift(10) - 1
        
        # High/Low ratios
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_to_high'] = df['close'] / df['high']
        features['close_to_low'] = df['close'] / df['low']
        
        # OHLC patterns
        features['body_size'] = abs(df['close'] - df['open']) / df['open']
        features['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        features['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        
        return features
    
    def _create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trend indikatör özellikleri"""
        features = pd.DataFrame(index=df.index)
        
        # MACD features (raporda en önemli)
        if 'macd' in df.columns:
            features['macd_signal'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
            features['macd_hist_norm'] = df['macd_hist'] / df['close']
            features['macd_divergence'] = df['macd'] - df['macd'].shift(5)
            
            # MACD histogram trend
            features['macd_hist_increasing'] = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(int)
        
        # EMA features
        if 'ema_9' in df.columns:
            features['ema_9_21_signal'] = np.where(df['ema_9'] > df['ema_21'], 1, -1)
            features['ema_21_50_signal'] = np.where(df['ema_21'] > df['ema_50'], 1, -1)
            features['ema_50_200_signal'] = np.where(df['ema_50'] > df['ema_200'], 1, -1)
            
            # Price distance from EMAs
            features['price_to_ema_9'] = (df['close'] - df['ema_9']) / df['ema_9']
            features['price_to_ema_50'] = (df['close'] - df['ema_50']) / df['ema_50']
            
            # EMA slopes
            features['ema_9_slope'] = (df['ema_9'] - df['ema_9'].shift(5)) / df['ema_9'].shift(5)
            features['ema_50_slope'] = (df['ema_50'] - df['ema_50'].shift(5)) / df['ema_50'].shift(5)
        
        # ADX features
        if 'adx' in df.columns:
            features['trend_strength'] = df['adx'] / 100  # Normalize
            features['strong_trend'] = (df['adx'] > 25).astype(int)
            features['adx_increasing'] = (df['adx'] > df['adx'].shift(1)).astype(int)
        
        # DMI features
        if 'plus_di' in df.columns:
            features['dmi_bull_signal'] = (df['plus_di'] > df['minus_di']).astype(int)
            features['dmi_strength'] = abs(df['plus_di'] - df['minus_di'])
        
        return features
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum indikatör özellikleri"""
        features = pd.DataFrame(index=df.index)
        
        # RSI features
        if 'rsi' in df.columns:
            features['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            features['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            features['rsi_neutral'] = ((df['rsi'] >= 30) & (df['rsi'] <= 70)).astype(int)
            features['rsi_divergence'] = df['rsi'] - df['rsi'].shift(5)
            
            # RSI momentum
            features['rsi_increasing'] = (df['rsi'] > df['rsi'].shift(1)).astype(int)
        
        # Stochastic features
        if 'stoch_k' in df.columns:
            features['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
            features['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
            features['stoch_cross'] = (df['stoch_k'] > df['stoch_d']).astype(int)
        
        # CCI features
        if 'cci' in df.columns:
            features['cci_overbought'] = (df['cci'] > 100).astype(int)
            features['cci_oversold'] = (df['cci'] < -100).astype(int)
        
        # Combined momentum score
        momentum_signals = []
        if 'rsi' in df.columns:
            momentum_signals.append((df['rsi'] - 50) / 50)
        if 'stoch_k' in df.columns:
            momentum_signals.append((df['stoch_k'] - 50) / 50)
        if 'cci' in df.columns:
            momentum_signals.append(df['cci'] / 100)
        
        if momentum_signals:
            features['momentum_score'] = np.mean(momentum_signals, axis=0)
        
        return features
    
    def _create_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatilite özellikleri"""
        features = pd.DataFrame(index=df.index)
        
        # ATR features
        if 'atr' in df.columns:
            features['atr_normalized'] = df['atr'] / df['close']
            features['volatility_regime'] = pd.cut(
                df['atr_percent'], 
                bins=[0, 1, 2, 100], 
                labels=['low', 'medium', 'high']
            ).cat.codes
            
            # ATR expansion/contraction
            features['atr_expanding'] = (df['atr'] > df['atr'].shift(5)).astype(int)
        
        # Bollinger Band features
        if 'bb_upper' in df.columns:
            features['bb_position'] = df['bb_percent']
            features['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(20).mean()).astype(int)
            features['bb_expansion'] = (df['bb_width'] > df['bb_width'].shift(1)).astype(int)
            
            # Band touches
            features['bb_upper_touch'] = (df['high'] >= df['bb_upper']).astype(int)
            features['bb_lower_touch'] = (df['low'] <= df['bb_lower']).astype(int)
        
        # Squeeze indicator
        if 'squeeze' in df.columns:
            features['in_squeeze'] = df['squeeze'].astype(int)
            features['squeeze_momentum'] = features['in_squeeze'].diff()
        
        return features
    
    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Hacim özellikleri"""
        features = pd.DataFrame(index=df.index)
        
        # Volume features
        if 'volume_ratio' in df.columns:
            features['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
            features['low_volume'] = (df['volume_ratio'] < 0.5).astype(int)
            features['volume_trend'] = df['volume_sma_20'].pct_change(5)
        
        # OBV features
        if 'obv' in df.columns:
            features['obv_trend'] = (df['obv'] > df['obv'].shift(5)).astype(int)
            features['obv_divergence'] = np.sign(df['obv'].diff()) != np.sign(df['close'].diff())
        
        # MFI features
        if 'mfi' in df.columns:
            features['mfi_overbought'] = (df['mfi'] > 80).astype(int)
            features['mfi_oversold'] = (df['mfi'] < 20).astype(int)
        
        # AD features
        if 'ad' in df.columns:
            features['ad_trend'] = (df['ad'] > df['ad'].shift(5)).astype(int)
        
        return features
    
    def _create_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mum pattern özellikleri"""
        features = pd.DataFrame(index=df.index)
        
        if TALIB_AVAILABLE:
            # Candlestick patterns
            features['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close']) / 100
            features['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close']) / 100
            features['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close']) / 100
            features['harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close']) / 100
        else:
            # Simple pattern detection without talib
            # Doji: Open and close are very close
            body_size = abs(df['close'] - df['open']) / df['open']
            features['doji'] = (body_size < 0.001).astype(int)
            
            # Hammer: Small body at top, long lower shadow
            lower_shadow = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
            upper_shadow = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
            features['hammer'] = ((lower_shadow > body_size * 2) & (upper_shadow < body_size)).astype(int)
            
            # Simplified patterns
            features['engulfing'] = 0
            features['harami'] = 0
        
        # Support/Resistance (basit)
        rolling_high = df['high'].rolling(20).max()
        rolling_low = df['low'].rolling(20).min()
        
        features['near_resistance'] = (df['close'] > rolling_high * 0.98).astype(int)
        features['near_support'] = (df['close'] < rolling_low * 1.02).astype(int)
        
        # Price action
        features['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        features['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        features['inside_bar'] = ((df['high'] < df['high'].shift(1)) & 
                                 (df['low'] > df['low'].shift(1))).astype(int)
        
        return features
    
    def _create_multi_timeframe_features(self, features_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Multi-timeframe özellikler"""
        # En küçük timeframe'i baz al
        base_tf = '15m'
        if base_tf not in features_dict:
            base_tf = list(features_dict.keys())[0]
        
        base_index = features_dict[base_tf].index
        mtf_features = pd.DataFrame(index=base_index)
        
        # Timeframe alignment
        for tf in ['1h', '4h', '1d', '1w']:
            if tf in features_dict:
                # Key features from higher timeframes
                for col in ['macd_signal', 'rsi_overbought', 'trend_strength', 'momentum_score']:
                    if col in features_dict[tf].columns:
                        # Reindex to base timeframe
                        aligned = features_dict[tf][col].reindex(base_index, method='ffill')
                        mtf_features[f'{tf}_{col}'] = aligned
        
        # Trend consistency across timeframes
        trend_signals = []
        for tf in features_dict:
            if 'macd_signal' in features_dict[tf].columns:
                aligned = features_dict[tf]['macd_signal'].reindex(base_index, method='ffill')
                trend_signals.append(aligned)
        
        if trend_signals:
            mtf_features['trend_consistency'] = np.mean(trend_signals, axis=0)
            mtf_features['trend_agreement'] = (np.std(trend_signals, axis=0) < 0.5).astype(int)
        
        return mtf_features
    
    def _create_macro_features(self, macro_data: Dict) -> Dict[str, float]:
        """Makro ekonomik özellikler"""
        features = {}
        
        # USD/TRY momentum
        if 'usdtry' in macro_data:
            features['usdtry_level'] = macro_data['usdtry']
            # Normalize etmek gerekirse historical data ile karşılaştır
        
        # VIX fear gauge
        if 'vix' in macro_data:
            features['market_fear'] = macro_data['vix'] / 30  # Normalize around 30
            features['high_fear'] = int(macro_data['vix'] > 30)
        
        # Gold as safe haven
        if 'gold' in macro_data:
            features['gold_momentum'] = 0  # Historical comparison needed
        
        # Market regime
        if 'xu100' in macro_data:
            features['market_level'] = macro_data['xu100']
        
        return features
    
    def _create_sentiment_features(self, sentiment_data: Dict) -> Dict[str, float]:
        """Haber sentiment özellikleri"""
        features = {}
        
        features['sentiment_score'] = sentiment_data.get('score', 0)
        features['news_count'] = sentiment_data.get('count', 0)
        features['positive_ratio'] = sentiment_data.get('positive', 0) / max(sentiment_data.get('count', 1), 1)
        features['negative_ratio'] = sentiment_data.get('negative', 0) / max(sentiment_data.get('count', 1), 1)
        
        # Sentiment strength
        features['sentiment_strength'] = abs(sentiment_data.get('score', 0))
        
        return features
    
    def prepare_model_input(self, features: Dict[str, pd.DataFrame], 
                          timeframe: str, sequence_length: int = 30) -> np.ndarray:
        """Model için input hazırla"""
        if timeframe not in features:
            raise ValueError(f"Timeframe {timeframe} not found in features")
        
        df = features[timeframe]
        
        # Son sequence_length kadar veriyi al
        if len(df) < sequence_length:
            # Padding
            pad_length = sequence_length - len(df)
            padding = pd.DataFrame(
                np.zeros((pad_length, df.shape[1])),
                columns=df.columns
            )
            df = pd.concat([padding, df])
        else:
            df = df.tail(sequence_length)
        
        # Normalize
        if timeframe not in self.scalers:
            self.scalers[timeframe] = StandardScaler()
            scaled = self.scalers[timeframe].fit_transform(df)
        else:
            scaled = self.scalers[timeframe].transform(df)
        
        return scaled
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Feature importance analizi"""
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance
        
        return pd.DataFrame()