"""
BIST hisse senetleri listesi
"""

# Ana hisse senetleri
ASSETS = [
    'AEFES', 'AGESA', 'AKBNK', 'AKSEN', 'AKSA', 'ALGYO', 'ALKIM', 'ANSGR', 
    'AGHOL', 'ARCLK', 'ASELS', 'BIMAS', 'DOHOL', 'DOAS', 'ECILC', 'EGEEN', 
    'EKGYO', 'ENKAI', 'ERBOS', 'EREGL', 'FROTO', 'GARAN', 'GESAN', 'GUBRF', 
    'HALKB', 'HEKTS', 'INDES', 'IPEKE', 'ISCTR', 'ISDMR', 'ISFIN', 'ISGYO', 
    'IZMDC', 'KARSN', 'KARTN', 'KCHOL', 'KLRHO', 'KONTR', 'KONYA', 'KOZAA', 
    'KOZAL', 'KRDMD', 'MAVI', 'MPARK', 'ODAS', 'OTKAR', 'OYAKC', 'PETKM', 
    'PGSUS', 'QUAGR', 'SAHOL', 'SASA', 'SISE', 'SKBNK', 'SMRTG', 'SNGYO', 
    'TAVHL', 'THYAO', 'TKFEN'
]

# Alternatif liste (gerekirse kullanılabilir)
ASSETS_EXTENDED = ASSETS + [
    'AKENR', 'ALBRK', 'ALCTL', 'ALFAS', 'ALKA', 'ALMAD', 'ANELE', 'ANHYT',
    'ANSGR', 'ARASE', 'ARDYZ', 'ARENA', 'ARSAN', 'ASTOR', 'ATAGY', 'ATAKP',
    'ATATP', 'ATLAS', 'ATSYH', 'AVGYO', 'AVHOL', 'AVOD', 'AVTUR', 'AYCES',
    'AYEN', 'AYGAZ', 'AZTEK', 'BAGFS', 'BAKAB', 'BALAT', 'BANVT', 'BARMA',
    'BASCM', 'BASGZ', 'BAYRK', 'BEKES', 'BERA', 'BEYAZ', 'BIENY', 'BIGCH',
    'BINHO', 'BIOEN', 'BIZIM', 'BJKAS', 'BLCYT', 'BMSCH', 'BMSTL', 'BNTAS',
    'BOBET', 'BORLS', 'BORSK', 'BOSSA', 'BRISA', 'BRKVY', 'BRLSM', 'BRMEN'
]

# Endeks sembolleri (opsiyonel)
INDICES = [
    'XU100',  # BIST 100
    'XU030',  # BIST 30
    'XBANK',  # Bankacılık endeksi
    'XUSIN',  # Sanayi endeksi
    'XUTEK',  # Teknoloji endeksi
]

# Timeframe'ler
TIMEFRAMES = ['15m', '1h', '4h', '1d']

# API limitleri
API_LIMITS = {
    'max_bars_per_request': 250,
    'rate_limit_per_minute': 30,
    'max_concurrent_requests': 5
}