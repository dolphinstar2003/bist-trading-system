# AlgoLab API Python Örnekleri - Claude Tarafından Düzenlenmiş

Bu dokümantasyon, AlgoLab API'sinin Python ile kullanımını gösteren örnekleri içerir.

## İçindekiler
- Genel Bilgilendirme
- Etkileşim Talebi
- Standart Şartname
- Endpoints
- Kullanıcı Girişi: SMS Alma
- Kullanıcı Girişi: Oturum Açma
- Oturum Yenileme
- Sembol Bilgisi
- Alt Hesap Bilgileri
- Hisse Portföy Bilgisi
- Viop Özeti
- Hisse Özeti
- Hisse Günlük İşlemler
- Viop Portföy Bilgisi
- Viop Günlük İşlemler
- Emir Gönderim
- Nakit Bakiye
- Hesap Ekstresi

---

## Genel Bilgilendirme

**Versiyon:** 1.0.5

### Önemli Bilgiler:
- **APIKEY**: API işlemlerinin kimliğidir (örn: `API-04YW0b9Cb8S0MrgBw/Y4iPYi2hjIidW7qj4hrhBhwZg=`)
- **Username**: TC Kimlik No veya İnternet Bankacılığı Kullanıcı Adı
- **Password**: İnternet Bankacılığı Şifresi
- **SMS Kodu**: Telefonunuza gelen doğrulama kodu

### Veri Tipleri:
- **Canlı Veri**: Anlık veri (yoksa 15 dakika gecikmeli)
- **Derinlik Verisi**: 10 kademe alım-satım derinliği

## Etkileşim Talebi

### API Endpoints:
- REST API: `https://www.algolab.com.tr/api`
- WebSocket: `wss://www.algolab.com.tr/api/ws`

### API Doğrulama Headers:
- **APIKEY**: API anahtarınız
- **Authorization**: Login sonrası alınan hash
- **Checker**: SHA256 imza (APIKEY + URL + Body)

## Standart Şartname

### Kurallar:
- **Rate Limit**: 5 saniyede 1 istek (429 hatası verir)
- **Sadece Türkiye'den erişim**
- **Bir IP'den tek kullanıcı**
- **API açıkken Web/Mobil kullanılamaz**

---

## Kullanıcı Girişi: SMS Alma

**Endpoint:** `POST /api/LoginUser`

### Amaç:
İnternet bankacılığı bilgileriyle giriş yapıp SMS kodu almak.

### Önemli Notlar:
- Username ve Password AES ile şifrelenir
- AES Key = APIKEY'deki "-" den sonraki kısım
- IV = 16 byte sıfır (b'\0' * 16)
- Response'da token döner, SMS kodunu bekleyin

### Python Kodu:

```python
import requests, hashlib, json, inspect, time, base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

# Yapılandırma
apikey = "API-04YW0b9Cb8S0MrgBw/Y4iPYi2hjIidW7qj4hrhBhwZg="
username = "12345678901"  # TC No
password = "sifre123"     # İnternet bankacılığı şifresi

hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password):
        # API key'den code kısmını ayır
        try:
            self.api_code = api_key.split("-")[1]
        except:
            self.api_code = api_key
        
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.last_request = 0.0
        self.LOCK = False

    def LoginUser(self):
        """SMS gönderir, token döner"""
        try:
            # Username ve password'u şifrele
            u = self.encrypt(self.username)
            p = self.encrypt(self.password)
            
            payload = {"username": u, "password": p}
            endpoint = "/api/LoginUser"
            
            # login=True olduğu için sadece APIKEY header'ı gönderilir
            resp = self.post(endpoint, payload, login=True)
            return resp.json()
        except Exception as e:
            print(f"LoginUser() hatası: {e}")
    
    def encrypt(self, text):
        """AES-CBC şifreleme"""
        iv = b'\0' * 16  # 16 byte sıfır
        key = base64.b64decode(self.api_code.encode('utf-8'))
        cipher = AES.new(key, AES.MODE_CBC, iv)
        
        bytes_data = text.encode()
        padded_bytes = pad(bytes_data, 16)  # PKCS7 padding
        encrypted = cipher.encrypt(padded_bytes)
        
        return base64.b64encode(encrypted).decode("utf-8")
    
    def make_checker(self, endpoint, payload):
        """SHA256 imza oluştur"""
        body = json.dumps(payload).replace(' ', '') if payload else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()
    
    def _request(self, method, url, endpoint, payload, headers):
        """Rate limit korumalı HTTP request"""
        while self.LOCK:
            time.sleep(0.1)
        
        self.LOCK = True
        try:
            if method == "POST":
                # Rate limit: 5 saniye bekle
                t = time.time()
                diff = t - self.last_request
                if self.last_request > 0 and diff < 5.0:
                    time.sleep(5 - diff + 0.1)
                
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
                return response
        finally:
            self.LOCK = False
    
    def post(self, endpoint, payload, login=False):
        """POST request helper"""
        url = self.api_url
        
        if not login:
            # Normal istek: APIKEY, Checker, Authorization gerekli
            checker = self.make_checker(endpoint, payload)
            headers = {
                "APIKEY": self.api_key,
                "Checker": checker,
                "Authorization": self.hash  # Login sonrası dolu olacak
            }
        else:
            # Login isteği: Sadece APIKEY gerekli
            headers = {"APIKEY": self.api_key}
        
        return self._request("POST", url, endpoint, payload, headers)

# Kullanım
api = API(apikey, username, password)
login_response = api.LoginUser()

if login_response and login_response.get("success"):
    token = login_response["content"]["token"]
    print(f"SMS gönderildi. Token: {token[:20]}...")
    print("SMS kodunu bekleyin...")
else:
    print("Login başarısız:", login_response)
```

### Örnek Response:
```json
{
   "success": true,
   "message": "",
   "content": {
      "token": "Ys/WhU/D37vO71VIBumDRhZLmkcMlzyb3TKJVWxLlpb/4BByYLNfQ07dEe66P3Ab"
   }
}
```

---

## Kullanıcı Girişi: Oturum Açma

**Endpoint:** `POST /api/LoginUserControl`

### Amaç:
SMS kodu ile hash almak (15 dakika geçerli).

### Önemli Notlar:
- Token ve SMS kodu AES ile şifrelenir
- Response'da hash döner
- Bu hash tüm API çağrılarında Authorization header'ında kullanılır

### Python Kodu:

```python
class API():
    # ... önceki kodlar ...
    
    def __init__(self, api_key, username, password, token, sms):
        # Önceki init + token ve sms
        self.token = token
        self.sms_code = sms
        # ... diğer değişkenler ...
    
    def LoginUserControl(self):
        """SMS kodu ile hash al"""
        try:
            # Token ve SMS kodunu şifrele
            t = self.encrypt(self.token)
            s = self.encrypt(self.sms_code)
            
            payload = {'token': t, 'password': s}
            endpoint = "/api/LoginUserControl"
            
            # login=True çünkü henüz hash yok
            resp = self.post(endpoint, payload, login=True)
            return resp.json()
        except Exception as e:
            print(f"LoginUserControl() hatası: {e}")

# Kullanım (SMS geldiğinde)
token = "önceki_adımdan_gelen_token"
sms_code = "123456"  # Telefonunuza gelen kod

api = API(apikey, username, password, token, sms_code)
control_response = api.LoginUserControl()

if control_response and control_response.get("success"):
    hash_value = control_response["content"]["hash"]
    print(f"Giriş başarılı! Hash alındı.")
    print(f"Hash: {hash_value[:50]}...")
    
    # Hash'i sakla (15 dakika geçerli)
    api.hash = hash_value
else:
    print("SMS doğrulama başarısız:", control_response)
```

---

## Oturum Yenileme

**Endpoint:** `POST /api/SessionRefresh`

### Amaç:
Oturum süresini uzatmak (15 dakika daha).

### Önemli Notlar:
- Parametre almaz, boş {} gönderilir
- Authorization header'da hash olmalı
- Başarılıysa {"success": true} döner
- Hash geçersizse 401 hatası verir

### Python Kodu:

```python
class API():
    # ... önceki kodlar ...
    
    def SessionRefresh(self):
        """Oturum süresini uzat"""
        try:
            endpoint = "/api/SessionRefresh"
            payload = {}  # Boş payload
            
            # login=False çünkü Authorization header gerekli
            resp = self.post(endpoint, payload, login=False)
            return self.error_check(resp, "SessionRefresh")
        except Exception as e:
            print(f"SessionRefresh() hatası: {e}")
    
    def error_check(self, resp, func_name, silent=False):
        """HTTP response kontrol"""
        try:
            if resp.status_code == 200:
                return resp.json()
            
            if not silent:
                print(f"Error kodu: {resp.status_code}")
                print(resp.text)
            
            return False
        except:
            if not silent:
                print(f"{func_name}() JSON parse hatası")
                print(resp.text)
            return False

# Kullanım
api.hash = "onceden_alinan_hash_degeri"

# Oturumu yenile
refresh_result = api.SessionRefresh()

if refresh_result and refresh_result.get("success"):
    print("Oturum yenilendi, 15 dakika daha geçerli")
else:
    print("Oturum yenileme başarısız, yeniden login gerekli")
```

### Otomatik Yenileme:
```python
import threading
import time

def auto_refresh(api, interval=14*60):  # 14 dakikada bir
    """Arka planda otomatik session yenileme"""
    while True:
        time.sleep(interval)
        result = api.SessionRefresh()
        if result and result.get("success"):
            print(f"Session otomatik yenilendi: {time.strftime('%H:%M:%S')}")
        else:
            print("Session yenileme başarısız!")
            break

# Başlat
refresh_thread = threading.Thread(target=auto_refresh, args=(api,), daemon=True)
refresh_thread.start()
```

---

## Sembol Bilgisi

**Endpoint:** `POST /api/GetEquityInfo`

### Amaç:
Hisse senedi bilgilerini almak (fiyat, taban/tavan, vs).

### Önemli Alanlar:
- **lst**: Son fiyat
- **bid**: Alış fiyatı (en iyi)
- **ask**: Satış fiyatı (en iyi)
- **flr**: Taban fiyat
- **clg**: Tavan fiyat

### Python Kodu:

```python
class API():
    # ... önceki kodlar ...
    
    def GetEquityInfo(self, symbol):
        """Hisse bilgilerini al"""
        try:
            endpoint = "/api/GetEquityInfo"
            payload = {'symbol': symbol}
            
            resp = self.post(endpoint, payload)
            return self.error_check(resp, "GetEquityInfo")
        except Exception as e:
            print(f"GetEquityInfo() hatası: {e}")

# Kullanım
info = api.GetEquityInfo("THYAO")

if info and info.get("success"):
    content = info["content"]
    print(f"Sembol: {content['name']}")
    print(f"Son Fiyat: {content['lst']} TL")
    print(f"Alış: {content['bid']} - Satış: {content['ask']}")
    print(f"Taban: {content['flr']} - Tavan: {content['clg']}")
    print(f"Limit: {content.get('limit', 'N/A')}")
    
    # DataFrame olarak görüntüle
    import pandas as pd
    df = pd.DataFrame([content])
    print(df)
```

### Örnek Response:
```json
{
   "success": true,
   "message": "",
   "content": {
      "name": "THYAO",
      "flr": "238.70",
      "clg": "291.70",
      "ask": "265.50",
      "bid": "265.25",
      "lst": "265.25",
      "limit": "0.00",
      "min": "",
      "max": "",
      "step": ""
   }
}
```

### Tüm Hisseler İçin Döngü:
```python
symbols = ["THYAO", "GARAN", "AKBNK", "EREGL", "ASELS"]
results = []

for symbol in symbols:
    info = api.GetEquityInfo(symbol)
    if info and info.get("success"):
        data = info["content"]
        results.append({
            "Sembol": data["name"],
            "Son": float(data["lst"]),
            "Alış": float(data["bid"]),
            "Satış": float(data["ask"]),
            "Taban": float(data["flr"]),
            "Tavan": float(data["clg"])
        })
    
    time.sleep(5.1)  # Rate limit

# DataFrame olarak göster
df = pd.DataFrame(results)
print(df)
```

---

## Alt Hesap Bilgileri

**Endpoint:** `POST /api/GetSubAccounts`

### Amaç:
Kullanıcının alt hesaplarını listelemek.

### Önemli Notlar:
- Parametre almaz
- Array döner (birden fazla alt hesap olabilir)
- Her alt hesabın numarası ve limiti vardır

### Python Kodu:

```python
class API():
    # ... önceki kodlar ...
    
    def GetSubAccounts(self, silent=False):
        """Alt hesapları listele"""
        try:
            endpoint = "/api/GetSubAccounts"
            payload = {}  # Boş payload
            
            resp = self.post(endpoint, payload)
            return self.error_check(resp, "GetSubAccounts", silent)
        except Exception as e:
            print(f"GetSubAccounts() hatası: {e}")

# Kullanım
accounts = api.GetSubAccounts()

if accounts and accounts.get("success"):
    sub_accounts = accounts["content"]
    print(f"Toplam {len(sub_accounts)} alt hesap bulundu:\n")
    
    for acc in sub_accounts:
        print(f"Hesap No: {acc['number']}")
        print(f"\u0130şlem Limiti: {acc['tradeLimit']} TL")
        print("-" * 30)
    
    # DataFrame olarak
    df = pd.DataFrame(sub_accounts)
    print(df)
```

### Örnek Response:
```json
{
   "success": true,
   "message": "",
   "content": [
      {
         "number": "100",
         "tradeLimit": "1000.00"
      },
      {
         "number": "101",
         "tradeLimit": "2000.00"
      }
   ]
}
```

---

## Hisse Portföy Bilgisi

**Endpoint:** `POST /api/InstantPosition`

### Amaç:
Açık pozisyonları ve portföy detaylarını almak.

### Önemli Alanlar:
- **code**: Hisse kodu
- **totalstock**: Toplam adet
- **cost**: Birim maliyet
- **unitprice**: Güncel birim fiyat
- **profit**: Kar/Zarar
- **tlamaount**: TL tutarı

### Python Kodu:

```python
class API():
    # ... önceki kodlar ...
    
    def GetInstantPosition(self, sub_account=""):
        """Açık pozisyonları getir"""
        try:
            endpoint = "/api/InstantPosition"
            payload = {'Subaccount': sub_account}
            
            resp = self.post(endpoint, payload)
            return self.error_check(resp, "GetInstantPosition")
        except Exception as e:
            print(f"GetInstantPosition() hatası: {e}")

# Kullanım
positions = api.GetInstantPosition()

if positions and positions.get("success"):
    portfolio = positions["content"]
    print(f"\nToplam {len(portfolio)} pozisyon:\n")
    
    total_cost = 0
    total_value = 0
    
    for pos in portfolio:
        code = pos["code"]
        quantity = float(pos["totalstock"])
        cost = float(pos["cost"])
        current = float(pos["unitprice"])
        profit = float(pos["profit"])
        value = float(pos["tlamaount"])
        
        print(f"Hisse: {code}")
        print(f"  Adet: {quantity:,.0f}")
        print(f"  Maliyet: {cost:.2f} TL")
        print(f"  Güncel: {current:.2f} TL")
        print(f"  Kar/Zarar: {profit:,.2f} TL")
        print(f"  Değer: {value:,.2f} TL")
        print("-" * 40)
        
        total_cost += quantity * cost
        total_value += value
    
    print(f"\nToplam Maliyet: {total_cost:,.2f} TL")
    print(f"Toplam Değer: {total_value:,.2f} TL")
    print(f"Toplam Kar/Zarar: {total_value - total_cost:,.2f} TL")
    print(f"Getiri: {((total_value/total_cost - 1) * 100):.2f}%")
```

### Örnek Response:
```json
{
  "success": true,
  "message": "",
  "content": [
    {
      "maliyet": "245.50",
      "totalstock": "1000",
      "code": "THYAO",
      "profit": "19750.00",
      "cost": "245.50",
      "unitprice": "265.25",
      "totalamount": "",
      "tlamaount": "265250.00",
      "explanation": "THYAO",
      "type": "CH",
      "total": "0"
    }
  ]
}
```

### Portföy Analizi:
```python
# DataFrame olarak görüntüle
import pandas as pd

if positions and positions.get("success"):
    df = pd.DataFrame(positions["content"])
    
    # Sayısal kolonları dönüştür
    numeric_cols = ['totalstock', 'cost', 'unitprice', 'profit', 'tlamaount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Yeni kolonlar ekle
    df['getiri_yuzde'] = ((df['unitprice'] / df['cost'] - 1) * 100).round(2)
    df['agirlik'] = (df['tlamaount'] / df['tlamaount'].sum() * 100).round(2)
    
    # Sırala ve göster
    df_sorted = df.sort_values('tlamaount', ascending=False)
    print(df_sorted[['code', 'totalstock', 'cost', 'unitprice', 'profit', 'getiri_yuzde', 'agirlik']])
```

---

## Emir Gönderim

**Endpoint:** `POST /api/SendOrder`

### Amaç:
Alım/Satım emri göndermek.

### Önemli Notlar:
- **direction**: "BUY" veya "SELL"
- **pricetype**: "piyasa" veya "limit"
- **lot**: String olarak gönderilir ("100", "1000" gibi)
- Response'da referans numarası döner

### Python Kodu:

```python
class API():
    # ... önceki kodlar ...
    
    def SendOrder(self, symbol, direction, pricetype, price, lot, sms=False, email=False, subAccount=""):
        """Emir gönder"""
        try:
            endpoint = "/api/SendOrder"
            payload = {
                "symbol": symbol,
                "direction": direction,      # BUY veya SELL
                "pricetype": pricetype,      # piyasa veya limit
                "price": str(price),         # Limit emirde fiyat
                "lot": str(lot),             # Adet
                "sms": sms,
                "email": email,
                "subAccount": subAccount
            }
            
            resp = self.post(endpoint, payload)
            return resp.json() if resp else None
        except Exception as e:
            print(f"SendOrder() hatası: {e}")
            return None

# Kullanım Örnekleri

# 1. Piyasa Fiyatından Alım
result = api.SendOrder(
    symbol="THYAO",
    direction="BUY",
    pricetype="piyasa",
    price="0",          # Piyasa emirde 0
    lot="100",
    sms=True,           # SMS bildirim
    email=False
)

if result and result.get("success"):
    ref_no = result["content"]
    print(f"Emir gönderildi: {ref_no}")
    # Örnek: "Referans Numaranız: 001VEV;0000-2923NR-IET - HISSEOK"
    
    # Referans numarasını parse et
    order_id = ref_no.split(":")[1].split(";")[0].strip()
    print(f"Order ID: {order_id}")

# 2. Limit Emir
result = api.SendOrder(
    symbol="GARAN",
    direction="SELL",
    pricetype="limit",
    price="11.85",      # Limit fiyat
    lot="500",
    sms=False,
    email=False
)

# 3. Toplu Emir Gönderimi
orders = [
    {"symbol": "AKBNK", "direction": "BUY", "lot": 100, "price": 10.50},
    {"symbol": "EREGL", "direction": "BUY", "lot": 200, "price": 45.25},
    {"symbol": "SISE", "direction": "SELL", "lot": 150, "price": 28.90}
]

for order in orders:
    result = api.SendOrder(
        symbol=order["symbol"],
        direction=order["direction"],
        pricetype="limit",
        price=order["price"],
        lot=str(order["lot"]),
        sms=False,
        email=False
    )
    
    if result and result.get("success"):
        print(f"{order['symbol']} emri gönderildi: {result['content']}")
    else:
        print(f"{order['symbol']} emri başarısız: {result}")
    
    time.sleep(5.1)  # Rate limit
```

### Risk Yönetimi ile Emir:
```python
def send_order_with_risk_management(api, symbol, direction, lot, max_portfolio_percent=5):
    """Risk yönetimi ile emir gönder"""
    
    # Portföy kontrolü
    positions = api.GetInstantPosition()
    if not positions or not positions.get("success"):
        print("Portföy bilgisi alınamadı")
        return False
    
    # Toplam portföy değeri
    total_value = sum(float(p["tlamaount"]) for p in positions["content"])
    
    # Sembol bilgisi al
    info = api.GetEquityInfo(symbol)
    if not info or not info.get("success"):
        print(f"{symbol} bilgisi alınamadı")
        return False
    
    current_price = float(info["content"]["lst"])
    order_value = current_price * lot
    
    # Risk kontrolü
    if order_value > total_value * (max_portfolio_percent / 100):
        print(f"Risk limiti aşıldı! Max {max_portfolio_percent}% = {total_value * max_portfolio_percent / 100:.2f} TL")
        return False
    
    # Emri gönder
    return api.SendOrder(
        symbol=symbol,
        direction=direction,
        pricetype="piyasa",
        price="0",
        lot=str(lot)
    )
```

---

## Nakit Bakiye

**Endpoint:** `POST /api/CashFlow`

### Amaç:
T+0, T+1, T+2 nakit bakiyelerini görmek.

### Python Kodu:

```python
class API():
    # ... önceki kodlar ...
    
    def CashFlow(self, sub_account=""):
        """Nakit akışını getir"""
        try:
            endpoint = "/api/CashFlow"
            payload = {'Subaccount': sub_account}
            
            resp = self.post(endpoint, payload)
            return self.error_check(resp, "CashFlow")
        except Exception as e:
            print(f"CashFlow() hatası: {e}")

# Kullanım
cash = api.CashFlow()

if cash and cash.get("success"):
    content = cash["content"]
    
    t0 = float(content.get("t0", 0))
    t1 = float(content.get("t1", 0))
    t2 = float(content.get("t2", 0))
    
    print("Nakit Bakiye Durumu:")
    print(f"T+0 (Bugün): {t0:,.2f} TL")
    print(f"T+1 (Yarın): {t1:,.2f} TL")
    print(f"T+2 (2 gün sonra): {t2:,.2f} TL")
    
    # Kullanılabilir nakit
    available = min(t0, t1, t2)
    print(f"\nKullanılabilir Nakit: {available:,.2f} TL")
```

### Örnek Response:
```json
{
  "success": true,
  "message": "",
  "content": {
    "t0": "50000.00",
    "t1": "45000.00",
    "t2": "48000.00"
  }
}
```

---

## Hesap Ekstresi

**Endpoint:** `POST /api/AccountExtre`

### Amaç:
Belirli tarih aralığındaki hesap hareketlerini görmek.

### Önemli Notlar:
- Tarihler datetime formatında gönderilir
- Hem hisse hem VIOP ekstresi döner
- İşlem detaylarını içerir

### Python Kodu:

```python
from datetime import datetime, timedelta, timezone

class API():
    # ... önceki kodlar ...
    
    def AccountExtre(self, sub_account="", start_date=None, end_date=None):
        """Hesap ekstresini getir"""
        try:
            endpoint = "/api/AccountExtre"
            
            # Tarihler datetime ise string'e çevir
            payload = {
                'start': start_date.isoformat() if isinstance(start_date, datetime) else start_date,
                'end': end_date.isoformat() if isinstance(end_date, datetime) else end_date,
                'Subaccount': sub_account
            }
            
            resp = self.post(endpoint, payload)
            return self.error_check(resp, "AccountExtre")
        except Exception as e:
            print(f"AccountExtre() hatası: {e}")

# Kullanım Örnekleri

# 1. Bugünkü işlemler
today = datetime.now()
start_of_day = today.replace(hour=0, minute=0, second=0, microsecond=0)

extre = api.AccountExtre(
    start_date=start_of_day,
    end_date=today
)

if extre and extre.get("success"):
    content = extre["content"]
    
    # Hisse ekstresi
    if "accountextre" in content:
        hisse_extre = content["accountextre"]
        print(f"\nHisse İşlemleri ({len(hisse_extre)} adet):\n")
        
        for item in hisse_extre:
            date = item["transdate"]
            desc = item["explanation"]
            debit = float(item["debit"])
            credit = float(item["credit"])
            balance = float(item["balance"])
            
            print(f"Tarih: {date}")
            print(f"Açıklama: {desc}")
            if debit > 0:
                print(f"Borç: {debit:,.2f} TL")
            if credit > 0:
                print(f"Alacak: {credit:,.2f} TL")
            print(f"Bakiye: {balance:,.2f} TL")
            print("-" * 50)

# 2. Haftalık ekstre
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

extre = api.AccountExtre(
    start_date=start_date,
    end_date=end_date
)

# 3. Aylık ekstre analizi
def analyze_monthly_extre(api, month, year):
    """Aylık ekstre analizi"""
    from calendar import monthrange
    
    # Ayın ilk ve son günü
    first_day = datetime(year, month, 1)
    last_day = datetime(year, month, monthrange(year, month)[1], 23, 59, 59)
    
    extre = api.AccountExtre(
        start_date=first_day,
        end_date=last_day
    )
    
    if not extre or not extre.get("success"):
        print("Ekstre alınamadı")
        return
    
    hisse_extre = extre["content"].get("accountextre", [])
    
    # İşlemleri kategorize et
    alimlar = []
    satimlar = []
    temettular = []
    komisyonlar = []
    
    for item in hisse_extre:
        desc = item["explanation"].upper()
        debit = float(item["debit"])
        credit = float(item["credit"])
        
        if "ALIM" in desc or "BUY" in desc:
            alimlar.append(debit)
        elif "SATIM" in desc or "SELL" in desc:
            satimlar.append(credit)
        elif "TEMETTÜ" in desc:
            temettular.append(credit)
        elif "KOMİSYON" in desc:
            komisyonlar.append(debit)
    
    # Özet rapor
    print(f"\n{month}/{year} Aylık Özet:")
    print(f"Toplam Alım: {sum(alimlar):,.2f} TL ({len(alimlar)} işlem)")
    print(f"Toplam Satım: {sum(satimlar):,.2f} TL ({len(satimlar)} işlem)")
    print(f"Temettü Geliri: {sum(temettular):,.2f} TL")
    print(f"Komisyon Gideri: {sum(komisyonlar):,.2f} TL")
    print(f"Net Nakit Akışı: {sum(satimlar) + sum(temettular) - sum(alimlar) - sum(komisyonlar):,.2f} TL")

# Kullanım
analyze_monthly_extre(api, month=10, year=2024)
```

### DataFrame Analizi:
```python
import pandas as pd

# Ekstre verilerini DataFrame'e dönüştür
if extre and extre.get("success"):
    hisse_extre = extre["content"]["accountextre"]
    
    # DataFrame oluştur
    df = pd.DataFrame(hisse_extre)
    
    # Tarih kolonunu datetime'a çevir
    df['transdate'] = pd.to_datetime(df['transdate'], format='%d.%m.%Y %H:%M:%S')
    
    # Sayısal kolonları dönüştür
    numeric_cols = ['debit', 'credit', 'balance']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Tarih sıralaması
    df = df.sort_values('transdate')
    
    # Günlük özet
    daily_summary = df.groupby(df['transdate'].dt.date).agg({
        'debit': 'sum',
        'credit': 'sum',
        'balance': 'last'
    })
    
    print("\nGünlük Özet:")
    print(daily_summary)
    
    # İşlem tiplerine göre grupla
    df['islem_tipi'] = df['explanation'].apply(lambda x: 
        'Alım' if 'ALIM' in x.upper() else
        'Satım' if 'SATIM' in x.upper() else
        'Temettü' if 'TEMETTÜ' in x.upper() else
        'Diğer'
    )
    
    type_summary = df.groupby('islem_tipi').agg({
        'debit': 'sum',
        'credit': 'sum'
    })
    
    print("\nİşlem Tiplerine Göre Özet:")
    print(type_summary)
```

---

## Özet ve İpuçları

### Bağlantı Akışı:
1. `LoginUser()` - SMS gönderir, token alır
2. `LoginUserControl()` - SMS kodu ile hash alır
3. `SessionRefresh()` - Her 14 dakikada bir çağır

### Rate Limit:
- 5 saniyede 1 istek limiti var
- Kodda otomatik bekleme var

### Veri Tipleri:
- Tüm fiyatlar string olarak gelir
- Tarihler genelde "dd.mm.yyyy HH:MM:SS" formatında
- Başarı durumu her zaman "success" alanında

### Güvenlik:
- API key'i ve şifreyi .env dosyasında saklayın
- Hash'i 15 dakikadan fazla saklamayın
- Her zaman HTTPS kullanın

### Hata Yönetimi:
```python
def safe_api_call(func, *args, **kwargs):
    """Güvenli API çağrısı"""
    try:
        result = func(*args, **kwargs)
        if result and result.get("success"):
            return result["content"]
        else:
            print(f"API hatası: {result.get('message', 'Bilinmeyen hata')}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None

# Kullanım
data = safe_api_call(api.GetEquityInfo, "THYAO")
if data:
    print(f"Fiyat: {data['lst']}")
```

### Örnek Response:
```json
{
   "success": true,
   "message": "",
   "content": {
      "hash": "eyJhbGciOiJodHRwOi8vd3d3LnczLm9yZy8yMDAxLzA0L3htbGRzaWctbW9yZSNobWFjLXNoYTI1NiIsInR5cCI6IkpXVCJ9..."
   }
}
```
