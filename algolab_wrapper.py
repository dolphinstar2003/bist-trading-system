import sys
import os
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import threading
from pathlib import Path
from dotenv import load_dotenv

# .env dosyasını yükle
env_path = Path(__file__).parent / "config" / ".env"
load_dotenv(env_path)

# Proje modülleri
from algolab.algolab import AlgoLab
from algolab.algolab_socket import AlgoLabSocket
from utils.logger import get_logger, log_trade

logger = get_logger(__name__)


class AlgoLabWrapper:
    """AlgoLab API için wrapper sınıfı"""
    
    def __init__(self):
        # Settings'den yapılandırmayı yükle
        with open("settings.json", "r") as f:
            self.settings = json.load(f)
        
        # .env'den API bilgilerini al
        self.api_key = os.getenv("ALGOLAB_API_KEY", "")
        self.username = os.getenv("ALGOLAB_USERNAME", "")
        self.password = os.getenv("ALGOLAB_PASSWORD", "")
        
        if not all([self.api_key, self.username, self.password]):
            logger.warning("AlgoLab credentials not found in .env file!")
        
        # API bağlantısını başlat
        self.api = None
        self.socket = None
        self.is_connected = False
        self.hash = None
        self.token = None
        
        # Hisse listesi
        self.symbols = self.settings["trading"]["symbols"]
        
        # Session refresh thread
        self.refresh_thread = None
        self.stop_refresh = False
        
        # Önce varolan session'ı kontrol et
        self._check_existing_session()
        
        logger.info("AlgoLab Wrapper initialized")
    
    def _check_existing_session(self):
        """Varolan session'ı kontrol et"""
        try:
            import os
            # Önce .algolab_session.json'ı kontrol et (simple_login.py'den)
            if os.path.exists('.algolab_session.json'):
                with open('.algolab_session.json', 'r') as f:
                    session_data = json.load(f)
                
                expires_at = datetime.fromisoformat(session_data['expires_at'])
                remaining = (expires_at - datetime.now()).total_seconds() / 60
                
                if remaining > 0:
                    self.hash = session_data['hash']
                    self.token = session_data['token']
                    logger.info(f"Found existing session ({remaining:.1f} minutes remaining)")
                    return True
            # Eski format data.json
            elif os.path.exists("data.json"):
                with open("data.json", "r") as f:
                    data = json.load(f)
                    # Session zamanını kontrol et
                    session_time = datetime.strptime(data["date"], "%Y-%m-%d %H:%M:%S")
                    elapsed = (datetime.now() - session_time).total_seconds() / 60
                    
                    if elapsed < 15:  # 15 dakikadan yeni
                        self.hash = data.get("hash")
                        self.token = data.get("token")
                        logger.info(f"Found existing session ({15-elapsed:.1f} minutes remaining)")
                        return True
        except Exception as e:
            logger.debug(f"No valid existing session: {e}")
        return False
    
    def connect(self, sms_code: Optional[str] = None) -> bool:
        """AlgoLab API'ye bağlan"""
        try:
            # API bağlantısı
            self.api = AlgoLab(
                api_key=self.api_key,
                username=self.username,
                password=self.password,
                auto_login=False,  # Otomatik login'i kapatıyoruz
                keep_alive=False,  # Test için keep_alive kapatıldı
                verbose=True  # Debug için açık
            )
            
            # Önce manuel session kontrolü
            if self.hash and self.token:
                # Session var, API'ye yükle
                self.api.hash = self.hash
                self.api.token = self.token
                self.is_connected = True
                logger.info("Using existing session from wrapper")
                return True
            
            # API'nin kendi session kontrolü
            elif self.api.load_settings():
                # Session yüklendi, geçerli mi kontrol et
                if self.api.is_alive:
                    self.hash = self.api.hash
                    self.is_connected = True
                    logger.info("Using existing session from API")
                    return True
                else:
                    logger.info("Saved session expired, need new login")
            
            # Yeni login gerekiyor
            if not sms_code:
                # İlk adım: SMS gönder
                login_result = self.api.LoginUser()
                
                if login_result == True:
                    # Token api nesnesinde saklanıyor
                    self.token = self.api.token
                    logger.info("SMS sent to your registered phone. Please provide SMS code.")
                    return False  # SMS bekleniyor
                else:
                    logger.error(f"AlgoLab login failed")
                    return False
            else:
                # İkinci adım: SMS ile login
                if not self.token:
                    logger.error("No token found. Please login first.")
                    return False
                
                # API'nin token ve sms_code değerlerini ayarla
                self.api.token = self.token
                self.api.sms_code = sms_code
                
                # LoginUserControl metodunu doğrudan çağırıyoruz
                # AlgoLab API'si kendi içinde input() çağırıyor
                # stdin'i geçici olarak değiştirerek SMS kodunu veriyoruz
                import io
                import sys
                
                # stdin'i yedekle ve değiştir
                old_stdin = sys.stdin
                sys.stdin = io.StringIO(sms_code + '\n')
                
                try:
                    control_result = self.api.LoginUserControl()
                finally:
                    # stdin'i geri yükle
                    sys.stdin = old_stdin
                
                if control_result == True:
                    # Hash api nesnesinde saklanıyor
                    if hasattr(self.api, 'hash') and self.api.hash:
                        self.hash = self.api.hash
                        self.is_connected = True
                        logger.info("AlgoLab API connected successfully with SMS verification")
                        
                        # Hash'i kaydet (15 dakika boyunca geçerli)
                        self.api.save_settings()
                        
                        # Session refresh thread'i başlat
                        self._start_session_refresh()
                        
                        return True
                    else:
                        logger.error("SMS verification successful but no hash found")
                        return False
                else:
                    logger.error("SMS verification failed - check SMS code")
                    return False
            
        except Exception as e:
            logger.error(f"Error connecting to AlgoLab: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def connect_socket(self) -> bool:
        """WebSocket bağlantısını başlat"""
        try:
            if not self.is_connected:
                logger.error("API not connected. Connect API first.")
                return False
            
            self.socket = AlgoLabSocket(
                api_key=self.api_config["api_key"],
                socket_url=self.api_config.get("socket_url", "wss://api.algolab.com.tr/socket")
            )
            
            # Socket'i başlat
            self.socket.connect()
            
            logger.info("AlgoLab WebSocket connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting WebSocket: {str(e)}")
            return False
    
    def get_market_data(self, symbol: str, period: str = "1d", 
                       bar_count: int = 100) -> Optional[pd.DataFrame]:
        """Piyasa verilerini getir"""
        try:
            if not self.is_connected:
                logger.error("API not connected")
                return None
            
            # Period mapping (bizim format -> AlgoLab format)
            period_mapping = {
                "1m": "1",      # 1 dakika
                "5m": "5",      # 5 dakika
                "15m": "15",    # 15 dakika
                "30m": "30",    # 30 dakika
                "1h": "60",     # 1 saat
                "4h": "240",    # 4 saat
                "1d": "1440",   # 1 gün
                "1w": "10080",  # 1 hafta
            }
            
            # Period'u AlgoLab formatına çevir
            algolab_period = period_mapping.get(period, period)
            logger.debug(f"Period mapping: {period} -> {algolab_period}")
            
            # AlgoLab'dan veri çek (GetCandleData kullanarak)
            result = self.api.GetCandleData(
                symbol=symbol,
                period=algolab_period
            )
            
            if not result or not result.get("success"):
                logger.warning(f"No data received for {symbol} - {period}")
                return None
            
            data = result.get("content", [])
            
            if not data:
                logger.warning(f"No data received for {symbol}")
                return None
            
            # DataFrame'e dönüştür
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                
                # Tarih sütununu kontrol et
                if 'date' in df.columns:
                    # ISO8601 formatını kullan (timezone bilgisi ile)
                    df['timestamp'] = pd.to_datetime(df['date'], format='ISO8601', utc=True)
                elif 'timestamp' in df.columns:
                    # ISO8601 formatını kullan (timezone bilgisi ile)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', utc=True)
                else:
                    logger.error(f"No date column found for {symbol}")
                    return None
                
                df.set_index('timestamp', inplace=True)
                
                # Sütun isimlerini standartlaştır
                column_mapping = {
                    'o': 'open', 'O': 'open',
                    'h': 'high', 'H': 'high', 
                    'l': 'low', 'L': 'low',
                    'c': 'close', 'C': 'close',
                    'v': 'volume', 'V': 'volume'
                }
                
                df.rename(columns=column_mapping, inplace=True)
            else:
                logger.warning(f"Empty or invalid data for {symbol}")
                return None
            
            logger.info(f"Market data received: {symbol} - {len(df)} bars")
            return df
            
        except Exception as e:
            logger.error(f"Error getting market data for {symbol}: {str(e)}")
            return None
    
    def get_account_info(self) -> Optional[Dict]:
        """Hesap bilgilerini getir (CashFlow kullanarak)"""
        try:
            if not self.is_connected:
                logger.error("API not connected")
                return None
            
            # CashFlow ile bakiye bilgisi al
            cash_flow = self.api.CashFlow()
            
            if cash_flow and cash_flow.get("success"):
                logger.info(f"Account info received")
                # Debug için tüm response'u logla
                logger.debug(f"CashFlow full response: {cash_flow}")
                
                # Eğer content dict değilse veya boşsa, diğer API'leri dene
                content = cash_flow.get("content", {})
                if not content or (content.get("t0") == "0.00" and content.get("t1") == "0.00"):
                    logger.warning("CashFlow returned zero balances, trying alternative methods")
                    
                    # GetEquitySubAccounts deneyelim
                    try:
                        sub_accounts = self.api.GetEquitySubAccounts()
                        if sub_accounts and sub_accounts.get("success"):
                            logger.info(f"SubAccounts info: {sub_accounts}")
                    except:
                        pass
                
                return cash_flow
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None
    
    def get_positions(self) -> Optional[List[Dict]]:
        """Açık pozisyonları getir"""
        try:
            if not self.is_connected:
                logger.error("API not connected")
                return None
            
            positions = self.api.GetInstantPosition()
            
            # Debug için veri yapısını görelim
            logger.debug(f"Positions raw data type: {type(positions)}")
            
            if positions and isinstance(positions, dict):
                if positions.get("success"):
                    # API başarılı dönüş, content'i kontrol et
                    content = positions.get("content", [])
                    logger.info(f"Positions received: {len(content)} open positions")
                    return content
                else:
                    logger.error(f"GetInstantPosition failed: {positions}")
                    return None
            else:
                # Direkt liste dönüyor olabilir
                logger.info(f"Positions received: {len(positions) if positions else 0} open positions")
                return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return None
    
    def get_portfolio(self) -> Optional[Dict]:
        """Portföy bilgilerini getir"""
        try:
            if not self.is_connected:
                logger.error("API not connected")
                return None
            
            portfolio = self.api.Portfolio()
            
            if portfolio:
                logger.info("Portfolio data received")
                return portfolio
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting portfolio: {str(e)}")
            return None
    
    def get_equity_order_history(self, start_date: Optional[str] = None, 
                                end_date: Optional[str] = None) -> Optional[List[Dict]]:
        """Hisse emir geçmişini getir"""
        try:
            if not self.is_connected:
                logger.error("API not connected")
                return None
            
            # GetEquityOrderHistory metodu API'de yok, TodaysTransaction kullan
            result = self.api.GetTodaysTransaction()
            
            if result and result.get("success"):
                orders = result.get("content", [])
                logger.info(f"Order history received: {len(orders)} orders")
                return orders
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting order history: {str(e)}")
            return []
    
    def get_account_extre(self, start_date: str, end_date: str, 
                         sub_account: str = "") -> Optional[List[Dict]]:
        """Hesap ekstresini getir"""
        try:
            if not self.is_connected:
                logger.error("API not connected")
                return None
            
            extre = self.api.AccountExtre(start_date, end_date, sub_account)
            
            if extre:
                logger.info(f"Account extre received for period {start_date} - {end_date}")
                return extre
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting account extre: {str(e)}")
            return None
    
    def get_cash_flow(self, sub_account: str = "") -> Optional[Dict]:
        """Nakit akışı bilgilerini getir"""
        try:
            if not self.is_connected:
                logger.error("API not connected")
                return None
            
            cash_flow = self.api.CashFlow(sub_account)
            
            if cash_flow:
                logger.info("Cash flow data received")
                return cash_flow
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cash flow: {str(e)}")
            return None
    
    def place_order(self, symbol: str, side: str, quantity: float, 
                   order_type: str = "market", price: Optional[float] = None) -> Optional[Dict]:
        """Emir gönder"""
        try:
            if not self.is_connected:
                logger.error("API not connected")
                return None
            
            # SendOrder için parametreler
            direction = "BUY" if side.upper() == "BUY" else "SELL"
            pricetype = "piyasa" if order_type.upper() == "MARKET" else "limit"
            
            # Emri gönder
            result = self.api.SendOrder(
                symbol=symbol,
                direction=direction,
                pricetype=pricetype,
                lot=str(int(quantity)),
                price=str(price) if price else "0",
                sms=False,
                email=False,
                subAccount=""
            )
            
            if result and result.get("success"):
                logger.info(f"Order placed successfully: {symbol} {side} {quantity}")
                
                # Trade'i logla
                log_trade(
                    symbol=symbol,
                    action=side,
                    quantity=quantity,
                    price=price or 0,
                    order_type=order_type
                )
                
                return result
            else:
                logger.error(f"Order failed: {result}")
                return None
                
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Emri iptal et"""
        try:
            if not self.is_connected:
                logger.error("API not connected")
                return False
            
            result = self.api.cancel_order(order_id)
            
            if result and result.get("status") == "success":
                logger.info(f"Order cancelled successfully: {order_id}")
                return True
            else:
                logger.error(f"Cancel order failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    def get_order_history(self, start_date: Optional[str] = None) -> Optional[List[Dict]]:
        """Emir geçmişini getir"""
        try:
            if not self.is_connected:
                logger.error("API not connected")
                return None
            
            orders = self.api.get_order_history(start_date=start_date)
            
            logger.info(f"Order history received: {len(orders)} orders")
            return orders
            
        except Exception as e:
            logger.error(f"Error getting order history: {str(e)}")
            return None
    
    def subscribe_market_data(self, symbols: List[str], callback):
        """Market data için WebSocket aboneliği"""
        try:
            if not self.socket:
                logger.error("WebSocket not connected")
                return False
            
            # Her sembol için abone ol
            for symbol in symbols:
                self.socket.subscribe_ticker(symbol, callback)
            
            logger.info(f"Subscribed to market data for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error subscribing to market data: {str(e)}")
            return False
    
    def update_all_market_data(self, csv_manager) -> Dict[str, bool]:
        """Tüm hisseler için market verilerini güncelle"""
        results = {}
        
        for symbol in self.symbols:
            try:
                # Her timeframe için veri çek
                for timeframe in self.settings["trading"]["timeframes"]:
                    # Bar sayısını timeframe'e göre ayarla
                    bar_counts = {
                        "15m": 500,
                        "1h": 300,
                        "4h": 200,
                        "1d": 100
                    }
                    
                    bar_count = bar_counts.get(timeframe, 100)
                    
                    # Veriyi çek
                    df = self.get_market_data(symbol, timeframe, bar_count)
                    
                    if df is not None:
                        # CSV'ye kaydet
                        csv_manager.save_raw_data(symbol, df, timeframe)
                        results[f"{symbol}_{timeframe}"] = True
                    else:
                        results[f"{symbol}_{timeframe}"] = False
                    
                    # Rate limit için bekle
                    time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error updating data for {symbol}: {str(e)}")
                results[symbol] = False
        
        # Özet
        success_count = sum(1 for v in results.values() if v)
        logger.info(f"Market data update completed: {success_count}/{len(results)} successful")
        
        return results
    
    def _start_session_refresh(self):
        """Session refresh thread'ini başlat"""
        if self.refresh_thread and self.refresh_thread.is_alive():
            return
        
        self.stop_refresh = False
        self.refresh_thread = threading.Thread(target=self._session_refresh_worker, daemon=True)
        self.refresh_thread.start()
        logger.info("Session refresh thread started")
    
    def _session_refresh_worker(self):
        """14 dakikada bir session yenile"""
        refresh_interval = 14 * 60  # 14 dakika
        
        while not self.stop_refresh and self.is_connected:
            try:
                time.sleep(refresh_interval)
                
                if self.is_connected and self.api:
                    result = self.api.SessionRefresh()
                    if result and result.get("success"):
                        logger.info("Session refreshed successfully")
                    else:
                        logger.warning("Session refresh failed, may need re-login")
                        self.is_connected = False
                        break
                        
            except Exception as e:
                logger.error(f"Error in session refresh: {e}")
                time.sleep(60)  # Hata durumunda 1 dakika bekle
    
    def disconnect(self):
        """Bağlantıları kapat"""
        try:
            # Session refresh'i durdur
            self.stop_refresh = True
            
            if self.socket:
                self.socket.disconnect()
                logger.info("WebSocket disconnected")
            
            if self.api:
                # Keep alive thread'i kapat
                if hasattr(self.api, 'keep_alive'):
                    self.api.keep_alive = False
                    logger.debug("Keep-alive thread stopped")
                
                # logout metodu yoksa sessizce geç
                if hasattr(self.api, 'logout'):
                    self.api.logout()
                    
                logger.info("API disconnected")
            
            self.is_connected = False
            
        except Exception as e:
            logger.error(f"Error disconnecting: {str(e)}")


if __name__ == "__main__":
    # Test
    wrapper = AlgoLabWrapper()
    
    # Bağlan
    if wrapper.connect():
        # Hesap bilgileri
        account = wrapper.get_account_info()
        print(f"Account: {account}")
        
        # Market data
        data = wrapper.get_market_data("THYAO", "1d", 10)
        if data is not None:
            print(f"Market data shape: {data.shape}")
            print(data.head())
        
        # Bağlantıyı kapat
        wrapper.disconnect()