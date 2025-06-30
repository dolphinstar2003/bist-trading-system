"""
Indicator Calculator
Teknik indikatörleri hesaplar ve CSV olarak kaydeder
Raporda önerilen MACD, RSI, ADX, ATR, Bollinger Bands vs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger
from pathlib import Path
import sys

# Try to import talib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    logger.warning("TA-Lib not available, using pandas-ta instead")
    TALIB_AVAILABLE = False
    try:
        import pandas_ta as ta
    except ImportError:
        logger.warning("pandas-ta also not available, using basic implementations")

# Parent path ekle
sys.path.append(str(Path(__file__).parent.parent))
from core.csv_data_manager import CSVDataManager


class IndicatorCalculator:
    """Teknik indikatör hesaplayıcı"""
    
    def __init__(self):
        self.csv_manager = CSVDataManager()
        
        # Raporda önerilen indikatörler
        self.indicators = {
            'trend': ['macd', 'ema_cross', 'adx'],
            'momentum': ['rsi', 'dmi', 'stochastic'],
            'volatility': ['atr', 'bollinger'],
            'volume': ['obv', 'volume_ratio']
        }
        
        logger.info("IndicatorCalculator başlatıldı")
    
    def calculate_all_indicators(self, symbol: str, timeframe: str, save: bool = True) -> pd.DataFrame:
        """Tüm indikatörleri hesapla"""
        # Raw data al
        df = self.csv_manager.get_raw_data(symbol, timeframe)
        if df is None or df.empty:
            logger.error(f"{symbol} {timeframe} raw data bulunamadı")
            return pd.DataFrame()
        
        # Her kategori için hesapla
        results = {}
        
        # Trend indicators
        results.update(self.calculate_trend_indicators(df))
        
        # Momentum indicators
        results.update(self.calculate_momentum_indicators(df))
        
        # Volatility indicators
        results.update(self.calculate_volatility_indicators(df))
        
        # Volume indicators
        results.update(self.calculate_volume_indicators(df))
        
        # DataFrame'e çevir
        indicator_df = pd.DataFrame(results, index=df.index)
        
        # Kaydet
        if save:
            for indicator_name, indicator_data in results.items():
                # Her indikatör için ayrı dosya
                ind_df = pd.DataFrame({indicator_name: indicator_data}, index=df.index)
                self.csv_manager.save_indicator_data(
                    symbol, timeframe, indicator_name, ind_df, use_new_path=True
                )
            
            logger.info(f"{symbol} {timeframe} için {len(results)} indikatör kaydedildi")
        
        return indicator_df
    
    def calculate_trend_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Trend indikatörlerini hesapla"""
        results = {}
        
        if TALIB_AVAILABLE:
            # MACD (Raporda en güvenli bulunmuş)
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'], 
                fastperiod=12, 
                slowperiod=26, 
                signalperiod=9
            )
            results['macd'] = macd
            results['macd_signal'] = macd_signal
            results['macd_hist'] = macd_hist
            
            # EMA'lar ve kesişimleri
            ema_9 = talib.EMA(df['close'], timeperiod=9)
            ema_21 = talib.EMA(df['close'], timeperiod=21)
            ema_50 = talib.EMA(df['close'], timeperiod=50)
            ema_200 = talib.EMA(df['close'], timeperiod=200)
            
            # ADX (Trend strength)
            results['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        else:
            # Manual MACD calculation
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - macd_signal
            
            results['macd'] = macd
            results['macd_signal'] = macd_signal
            results['macd_hist'] = macd_hist
            
            # EMA calculations
            ema_9 = df['close'].ewm(span=9, adjust=False).mean()
            ema_21 = df['close'].ewm(span=21, adjust=False).mean()
            ema_50 = df['close'].ewm(span=50, adjust=False).mean()
            ema_200 = df['close'].ewm(span=200, adjust=False).mean()
            
            # Simple ADX calculation
            results['adx'] = self._calculate_adx(df, 14)
        
        results['ema_9'] = ema_9
        results['ema_21'] = ema_21
        results['ema_50'] = ema_50
        results['ema_200'] = ema_200
        
        # EMA Cross signals
        results['ema_cross_9_21'] = (ema_9 > ema_21).astype(int)
        results['ema_cross_21_50'] = (ema_21 > ema_50).astype(int)
        results['ema_cross_50_200'] = (ema_50 > ema_200).astype(int)
        
        return results
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Momentum indikatörlerini hesapla"""
        results = {}
        
        if TALIB_AVAILABLE:
            # RSI
            results['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # DMI (Directional Movement Index)
            results['plus_di'] = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            results['minus_di'] = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Stochastic
            slowk, slowd = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            results['stoch_k'] = slowk
            results['stoch_d'] = slowd
            
            # CCI (Commodity Channel Index)
            results['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        else:
            # Manual calculations
            results['rsi'] = self._calculate_rsi(df['close'], 14)
            
            # Simple DMI
            high = df['high']
            low = df['low']
            up = high - high.shift(1)
            down = low.shift(1) - low
            
            plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
            minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
            
            atr = self._calculate_atr(df, 14)
            results['plus_di'] = 100 * (plus_dm.rolling(14).mean() / atr)
            results['minus_di'] = 100 * (minus_dm.rolling(14).mean() / atr)
            
            # Simple Stochastic
            low_min = df['low'].rolling(window=14).min()
            high_max = df['high'].rolling(window=14).max()
            k = 100 * ((df['close'] - low_min) / (high_max - low_min))
            results['stoch_k'] = k.rolling(window=3).mean()
            results['stoch_d'] = results['stoch_k'].rolling(window=3).mean()
            
            # Simple CCI
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            sma = typical_price.rolling(window=20).mean()
            mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
            results['cci'] = (typical_price - sma) / (0.015 * mad)
        
        results['dmi_diff'] = results['plus_di'] - results['minus_di']
        
        return results
    
    def calculate_volatility_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Volatilite indikatörlerini hesapla"""
        results = {}
        
        if TALIB_AVAILABLE:
            # ATR (Average True Range) - Stop loss için kritik
            results['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Bollinger Bands
            upper, middle, lower = talib.BBANDS(
                df['close'], 
                timeperiod=20, 
                nbdevup=2, 
                nbdevdn=2
            )
            results['bb_upper'] = upper
            results['bb_middle'] = middle
            results['bb_lower'] = lower
            
            # Keltner Channel için
            ema20 = talib.EMA(df['close'], timeperiod=20)
            atr20 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=20)
        else:
            # Manual calculations
            results['atr'] = self._calculate_atr(df, 14)
            
            # Bollinger Bands
            middle = df['close'].rolling(window=20).mean()
            std = df['close'].rolling(window=20).std()
            upper = middle + (2 * std)
            lower = middle - (2 * std)
            
            results['bb_upper'] = upper
            results['bb_middle'] = middle
            results['bb_lower'] = lower
            
            # Keltner Channel için
            ema20 = df['close'].ewm(span=20, adjust=False).mean()
            atr20 = self._calculate_atr(df, 20)
        
        # ATR yüzdesi
        results['atr_percent'] = (results['atr'] / df['close']) * 100
        
        # BB width ve percent
        results['bb_width'] = results['bb_upper'] - results['bb_lower']
        results['bb_percent'] = (df['close'] - results['bb_lower']) / (results['bb_upper'] - results['bb_lower'])
        
        # Keltner Channel
        results['kc_upper'] = ema20 + (2 * atr20)
        results['kc_lower'] = ema20 - (2 * atr20)
        
        # Squeeze detector (BB içinde KC)
        results['squeeze'] = (results['bb_upper'] < results['kc_upper']) & \
                           (results['bb_lower'] > results['kc_lower'])
        
        return results
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Hacim indikatörlerini hesapla"""
        results = {}
        
        if TALIB_AVAILABLE:
            # OBV (On Balance Volume)
            results['obv'] = talib.OBV(df['close'], df['volume'])
            
            # Volume SMA
            volume_sma = talib.SMA(df['volume'], timeperiod=20)
            
            # MFI (Money Flow Index)
            results['mfi'] = talib.MFI(
                df['high'], df['low'], df['close'], df['volume'], 
                timeperiod=14
            )
            
            # AD (Accumulation/Distribution)
            results['ad'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
            
            # ADOSC (Accumulation/Distribution Oscillator)
            results['adosc'] = talib.ADOSC(
                df['high'], df['low'], df['close'], df['volume'],
                fastperiod=3, slowperiod=10
            )
        else:
            # Manual OBV
            obv = pd.Series(index=df.index, dtype=float)
            obv.iloc[0] = df['volume'].iloc[0]
            
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            results['obv'] = obv
            
            # Volume SMA
            volume_sma = df['volume'].rolling(window=20).mean()
            
            # Simple MFI
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * df['volume']
            
            positive_flow = pd.Series(index=df.index, dtype=float)
            negative_flow = pd.Series(index=df.index, dtype=float)
            
            for i in range(1, len(df)):
                if typical_price.iloc[i] > typical_price.iloc[i-1]:
                    positive_flow.iloc[i] = money_flow.iloc[i]
                    negative_flow.iloc[i] = 0
                else:
                    positive_flow.iloc[i] = 0
                    negative_flow.iloc[i] = money_flow.iloc[i]
            
            positive_mf = positive_flow.rolling(window=14).sum()
            negative_mf = negative_flow.rolling(window=14).sum()
            
            mfi = 100 - (100 / (1 + positive_mf / negative_mf))
            results['mfi'] = mfi
            
            # Simple AD
            clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            clv = clv.fillna(0)
            ad = (clv * df['volume']).cumsum()
            results['ad'] = ad
            
            # ADOSC
            ad_fast = ad.ewm(span=3, adjust=False).mean()
            ad_slow = ad.ewm(span=10, adjust=False).mean()
            results['adosc'] = ad_fast - ad_slow
        
        results['volume_sma_20'] = volume_sma
        
        # Volume ratio (current / average)
        results['volume_ratio'] = df['volume'] / volume_sma
        
        return results
    
    def calculate_custom_indicators(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Özel indikatörler (Rapordan)"""
        results = {}
        
        # Get MACD hist (from previous calculations or calculate new)
        if hasattr(self, '_last_macd_hist'):
            macd_hist = self._last_macd_hist
        else:
            # Calculate MACD manually if needed
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = macd - macd_signal
        
        price_highs = df['high'].rolling(window=20).max()
        macd_highs = macd_hist.rolling(window=20).max()
        
        # Bearish divergence: Price yeni zirve, MACD yapmıyor
        results['macd_bearish_div'] = (
            (df['high'] == price_highs) & 
            (macd_hist < macd_highs.shift(1))
        ).astype(int)
        
        # Bullish divergence: Price yeni dip, MACD yapmıyor
        price_lows = df['low'].rolling(window=20).min()
        macd_lows = macd_hist.rolling(window=20).min()
        results['macd_bullish_div'] = (
            (df['low'] == price_lows) & 
            (macd_hist > macd_lows.shift(1))
        ).astype(int)
        
        # Trend Quality Index (özel)
        # Get ADX from previous calculations or calculate
        if hasattr(self, '_last_adx'):
            adx = self._last_adx
        else:
            adx = self._calculate_adx(df, 14)
        
        trend_quality = adx / 100  # Normalize
        
        # Volatility adjusted trend
        atr = self._calculate_atr(df, 14)
        atr_pct = (atr / df['close']) * 100
        results['trend_quality'] = trend_quality * (1 / (1 + atr_pct/10))
        
        return results
    
    def process_all_symbols(self, symbols: List[str], timeframes: List[str]):
        """Tüm semboller için indikatörleri hesapla"""
        total = len(symbols) * len(timeframes)
        processed = 0
        
        logger.info(f"{len(symbols)} sembol, {len(timeframes)} timeframe için indikatör hesaplanıyor...")
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    # Hesapla ve kaydet
                    self.calculate_all_indicators(symbol, timeframe, save=True)
                    processed += 1
                    
                    if processed % 10 == 0:
                        logger.info(f"İlerleme: {processed}/{total} ({processed/total*100:.1f}%)")
                        
                except Exception as e:
                    logger.error(f"{symbol} {timeframe} hesaplama hatası: {e}")
        
        logger.success(f"Toplam {processed} indikatör seti hesaplandı")
    
    def update_latest_indicators(self, symbol: str, timeframe: str):
        """Son verileri güncelle (real-time için)"""
        # Raw data'nın son kısmını al
        df = self.csv_manager.get_latest_data(symbol, timeframe, periods=200)
        
        if df.empty:
            return None
        
        # Sadece son bar için hesapla
        indicators = self.calculate_all_indicators(symbol, timeframe, save=False)
        
        # Son değerleri döndür
        latest = {}
        for col in indicators.columns:
            latest[col] = indicators[col].iloc[-1]
        
        return latest
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Simple ADX calculation without talib"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Directional movements
        up = high - high.shift(1)
        down = low.shift(1) - low
        
        plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0), index=df.index)
        minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0), index=df.index)
        
        # Smoothed DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Simple RSI calculation"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Simple ATR calculation"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        hl = high - low
        hc = abs(high - close.shift(1))
        lc = abs(low - close.shift(1))
        
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr


def main():
    """Test ve batch processing"""
    calculator = IndicatorCalculator()
    
    # Mevcut sembolleri al
    symbols = calculator.csv_manager.get_available_symbols()
    
    print(f"Toplam {len(symbols)} sembol bulundu")
    
    # Test için ilk 5 sembol
    test_symbols = symbols[:150]
    timeframes = ['15m', '1h', '4h', '1d']
    
    calculator.process_all_symbols(test_symbols, timeframes)


if __name__ == "__main__":
    main()