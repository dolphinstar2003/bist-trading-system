"""
Unified Data Collector
Multi-timeframe veri toplama ve yönetim modülü
Mevcut CSV veri yapısını kullanır
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import yfinance as yf
import finnhub
from alpha_vantage.timeseries import TimeSeries
from loguru import logger
import redis
import json
import pickle
from .csv_data_manager import CSVDataManager


class UnifiedDataCollector:
    """Çoklu kaynaklardan veri toplama ve yönetim"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # CSV Data Manager - mevcut veri yapısını kullan
        self.csv_manager = CSVDataManager()
        
        # Timeframe mapping for Yahoo Finance
        self.tf_mapping = {
            '15m': '15m',
            '1h': '60m',
            '4h': '1h',  # Will be resampled
            '1d': '1d',
            '1w': '1wk'
        }
        
        # API istemcileri (sadece eksik veri veya real-time için)
        self.yf_client = yf
        if config['api']['finnhub']['api_key']:
            self.finnhub_client = finnhub.Client(api_key=config['api']['finnhub']['api_key'])
        else:
            self.finnhub_client = None
            
        if config['api']['alpha_vantage']['api_key']:
            self.alpha_vantage = TimeSeries(key=config['api']['alpha_vantage']['api_key'])
        else:
            self.alpha_vantage = None
        
        # Redis cache
        try:
            self.cache = redis.Redis(
                host='localhost', 
                port=6379, 
                decode_responses=False,  # Binary data için
                db=0
            )
            self.cache.ping()
            self.cache_enabled = True
        except:
            logger.warning("Redis bağlantısı kurulamadı, cache devre dışı")
            self.cache_enabled = False
            
        self.cache_ttl = config['data']['cache_ttl']  # 15 dakika
        
        logger.info("UnifiedDataCollector başlatıldı (CSV veri kullanımı)")
    
    async def collect_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Bir sembol için tüm timeframe verilerini topla - Önce CSV'den oku"""
        cache_key = f"mtf_data:{symbol}"
        
        # Cache kontrolü
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            logger.debug(f"{symbol} verisi cache'den alındı")
            return cached_data
        
        data = {}
        
        # Önce CSV'den okumayı dene
        for tf in self.config['timeframes']['analysis']:
            df = self.csv_manager.get_raw_data(symbol, tf)
            if df is not None:
                data[tf] = df
                logger.debug(f"{symbol} {tf} CSV'den yüklendi")
            else:
                # CSV'de yoksa API'den çekmeyi dene
                logger.warning(f"{symbol} {tf} CSV'de yok, API deneniyor...")
                df = await self._collect_single_timeframe(symbol, tf)
                if df is not None:
                    data[tf] = df
        
        # Eksik 4h verisini 1h'den oluştur
        if '4h' not in data and '1h' in data:
            data['4h'] = self._resample_to_4h(data['1h'])
        
        # İndikatör verilerini ekle
        data['indicators'] = {}
        for tf in data.keys():
            if tf not in ['macro', 'sentiment', 'indicators']:
                # Mevcut hesaplanmış indikatörleri al
                combined = self.csv_manager.get_all_indicators(symbol, tf)
                if not combined.empty:
                    data['indicators'][tf] = combined
        
        # Makro verileri ekle
        data['macro'] = await self._collect_macro_data()
        
        # Haber sentiment ekle
        data['sentiment'] = await self._collect_sentiment_data(symbol)
        
        # Cache'e kaydet
        if data and self.cache_enabled:
            self._save_to_cache(cache_key, data)
        
        return data
    
    async def _collect_single_timeframe(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Tek bir timeframe için veri topla"""
        try:
            # BIST sembolleri için .IS ekle
            yahoo_symbol = f"{symbol}.IS" if not symbol.endswith('.IS') else symbol
            
            # Period hesapla
            periods = {
                '15m': '60d',   # 60 gün
                '1h': '730d',   # 2 yıl
                '4h': '730d',   # 2 yıl (1h'den hesaplanacak)
                '1d': '5y',     # 5 yıl
                '1w': '10y'     # 10 yıl
            }
            
            # Yahoo Finance'den veri çek
            interval = self.tf_mapping.get(timeframe, timeframe)
            ticker = yf.Ticker(yahoo_symbol)
            
            df = ticker.history(
                period=periods.get(timeframe, '1y'),
                interval=interval,
                auto_adjust=True
            )
            
            if df.empty:
                raise ValueError(f"Veri bulunamadı: {symbol} {timeframe}")
            
            # Kolon isimlerini standartlaştır
            df.columns = [col.lower() for col in df.columns]
            
            # Sadece OHLCV kolonları
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            # NaN değerleri temizle
            df = df.dropna()
            
            logger.debug(f"{symbol} {timeframe}: {len(df)} bar toplandı")
            
            return df
            
        except Exception as e:
            logger.error(f"{symbol} {timeframe} veri toplama hatası: {e}")
            raise
    
    def _resample_to_4h(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """1 saatlik veriyi 4 saatliğe dönüştür"""
        try:
            df_4h = df_1h.resample('4H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            df_4h = df_4h.dropna()
            logger.debug(f"1h -> 4h dönüşüm: {len(df_4h)} bar")
            
            return df_4h
            
        except Exception as e:
            logger.error(f"4h resample hatası: {e}")
            return pd.DataFrame()
    
    async def _collect_macro_data(self) -> Dict[str, float]:
        """Makroekonomik verileri topla"""
        macro_data = {}
        
        try:
            # USD/TRY kuru
            usdtry = yf.Ticker("USDTRY=X")
            usdtry_data = usdtry.history(period="1d")
            if not usdtry_data.empty:
                macro_data['usdtry'] = float(usdtry_data['Close'].iloc[-1])
            
            # Altın (Ons/USD)
            gold = yf.Ticker("GC=F")
            gold_data = gold.history(period="1d")
            if not gold_data.empty:
                macro_data['gold'] = float(gold_data['Close'].iloc[-1])
            
            # VIX (Volatilite Endeksi)
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            if not vix_data.empty:
                macro_data['vix'] = float(vix_data['Close'].iloc[-1])
            
            # BIST 100 Endeksi
            xu100 = yf.Ticker("XU100.IS")
            xu100_data = xu100.history(period="1d")
            if not xu100_data.empty:
                macro_data['xu100'] = float(xu100_data['Close'].iloc[-1])
            
            logger.debug(f"Makro veriler toplandı: {list(macro_data.keys())}")
            
        except Exception as e:
            logger.error(f"Makro veri toplama hatası: {e}")
        
        return macro_data
    
    async def _collect_sentiment_data(self, symbol: str) -> Dict[str, Any]:
        """Haber sentiment verisi topla"""
        sentiment_data = {
            'score': 0.0,
            'count': 0,
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }
        
        try:
            # Finnhub'dan haberler
            news = self.finnhub_client.company_news(
                symbol, 
                _from=datetime.now() - timedelta(days=1),
                to=datetime.now()
            )
            
            if news:
                sentiment_data['count'] = len(news)
                
                # Basit sentiment analizi (geliştirilmesi gerekir)
                for article in news:
                    # Finnhub sentiment sağlıyorsa kullan
                    if 'sentiment' in article:
                        score = article['sentiment']['score']
                        if score > 0.3:
                            sentiment_data['positive'] += 1
                        elif score < -0.3:
                            sentiment_data['negative'] += 1
                        else:
                            sentiment_data['neutral'] += 1
                
                # Ortalama sentiment skoru
                if sentiment_data['count'] > 0:
                    sentiment_data['score'] = (
                        sentiment_data['positive'] - sentiment_data['negative']
                    ) / sentiment_data['count']
            
            logger.debug(f"{symbol} sentiment: {sentiment_data['score']:.2f} ({sentiment_data['count']} haber)")
            
        except Exception as e:
            logger.error(f"Sentiment veri hatası: {e}")
        
        return sentiment_data
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Cache'den veri al"""
        if not self.cache_enabled:
            return None
            
        try:
            data = self.cache.get(key)
            if data:
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Cache okuma hatası: {e}")
        return None
    
    def _save_to_cache(self, key: str, data: Any):
        """Cache'e veri kaydet"""
        if not self.cache_enabled:
            return
            
        try:
            serialized = pickle.dumps(data)
            self.cache.setex(key, self.cache_ttl, serialized)
        except Exception as e:
            logger.error(f"Cache yazma hatası: {e}")
    
    async def get_realtime_data(self, symbol: str) -> Dict[str, float]:
        """Gerçek zamanlı fiyat verisi (Algolab'dan alınacak)"""
        try:
            # Yahoo Finance'den anlık veri
            ticker = yf.Ticker(f"{symbol}.IS")
            info = ticker.info
            
            return {
                'price': info.get('regularMarketPrice', 0),
                'volume': info.get('regularMarketVolume', 0),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'spread': info.get('ask', 0) - info.get('bid', 0)
            }
            
        except Exception as e:
            logger.error(f"Gerçek zamanlı veri hatası: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
        """Belirli sayıda bar için historical veri"""
        cache_key = f"hist:{symbol}:{timeframe}:{periods}"
        
        # Cache kontrolü
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        
        try:
            yahoo_symbol = f"{symbol}.IS" if not symbol.endswith('.IS') else symbol
            ticker = yf.Ticker(yahoo_symbol)
            
            # Period hesapla
            period_map = {
                '15m': f"{periods}d",
                '1h': f"{periods*2}d",
                '4h': f"{periods*8}d",
                '1d': f"{periods*30}d",
                '1w': f"{periods*52}d"
            }
            
            df = ticker.history(
                period=period_map.get(timeframe, f"{periods}d"),
                interval=self.tf_mapping.get(timeframe, timeframe)
            )
            
            if not df.empty:
                df.columns = [col.lower() for col in df.columns]
                df = df[['open', 'high', 'low', 'close', 'volume']].tail(periods)
                
                # Cache'e kaydet
                self._save_to_cache(cache_key, df)
                
                return df
                
        except Exception as e:
            logger.error(f"Historical veri hatası: {e}")
            
        return pd.DataFrame()