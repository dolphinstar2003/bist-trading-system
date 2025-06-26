import os
import sys
from loguru import logger
from pathlib import Path
import json
from datetime import datetime


class LoggerConfig:
    """Trading sistemi için merkezi logger yapılandırması"""
    
    def __init__(self):
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Settings'den log ayarlarını yükle
        with open("settings.json", "r") as f:
            settings = json.load(f)
            self.log_config = settings.get("logging", {})
    
    def setup_logger(self):
        """Logger'ı yapılandır"""
        # Varsayılan logger'ı temizle
        logger.remove()
        
        # Console output
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level="DEBUG",  # Geçici olarak DEBUG'a çektik
            colorize=True
        )
        
        # File output - Genel log
        logger.add(
            self.log_dir / "trading_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=self.log_config.get("level", "INFO"),
            rotation="00:00",  # Her gece yarısı yeni dosya
            retention="30 days",  # 30 gün sakla
            compression="zip",  # Eski logları sıkıştır
            enqueue=True  # Thread-safe
        )
        
        # Error log - Sadece hatalar
        logger.add(
            self.log_dir / "errors_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level="ERROR",
            rotation="1 week",
            retention="3 months",
            compression="zip",
            enqueue=True
        )
        
        # Trade log - İşlem kayıtları
        logger.add(
            self.log_dir / "trades_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
            filter=lambda record: "trade" in record["extra"],
            rotation="1 day",
            retention="1 year",
            compression="zip",
            enqueue=True
        )
        
        # Performance log - Performans metrikleri
        logger.add(
            self.log_dir / "performance_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {message}",
            filter=lambda record: "performance" in record["extra"],
            rotation="1 week",
            retention="6 months",
            compression="zip",
            enqueue=True
        )
        
        return logger
    
    def get_logger(self, name: str):
        """Modül için özelleştirilmiş logger döndür"""
        return logger.bind(name=name)


# Global logger instance
config = LoggerConfig()
logger = config.setup_logger()

# Modül logger'ları için helper
def get_logger(name: str):
    """Modül için logger al"""
    return config.get_logger(name)


# Trade logger helper
def log_trade(symbol: str, action: str, quantity: float, price: float, **kwargs):
    """Trade işlemlerini logla"""
    trade_data = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "action": action,
        "quantity": quantity,
        "price": price,
        **kwargs
    }
    logger.bind(trade=True).info(f"TRADE: {json.dumps(trade_data)}")


# Performance logger helper
def log_performance(metric: str, value: float, **kwargs):
    """Performans metriklerini logla"""
    perf_data = {
        "timestamp": datetime.now().isoformat(),
        "metric": metric,
        "value": value,
        **kwargs
    }
    logger.bind(performance=True).info(f"PERFORMANCE: {json.dumps(perf_data)}")


if __name__ == "__main__":
    # Test
    test_logger = get_logger("test")
    test_logger.info("Logger yapılandırması başarılı")
    test_logger.debug("Debug mesajı")
    test_logger.warning("Uyarı mesajı")
    test_logger.error("Hata mesajı")
    
    # Trade log testi
    log_trade("THYAO", "BUY", 1000, 245.50, strategy="momentum")
    
    # Performance log testi
    log_performance("sharpe_ratio", 1.85, period="daily")