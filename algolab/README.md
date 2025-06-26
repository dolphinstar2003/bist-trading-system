
# AlgoLab Python Wrapper

## Genel Bakış
Bu proje, AlgoLab API'si için bir Python wrapper'ıdır. `algolab.py` ve `algolab_socket.py` modülleri aracılığıyla AlgoLab API'sine erişim sağlar. Kullanıcıların AlgoLab platformundaki verilere programatik olarak erişmelerini ve işlemler yapmalarını kolaylaştırır.

## Kurulum
Projeyi kullanmak için, bu repoyu klonlayın ve gerekli bağımlılıkları kurun.

```
git clone [repo-url]
cd [repo-directory]
pip install -r requirements.txt
```

## Kullanım
API'yi kullanmak için, öncelikle `config.py` dosyasında gerekli yapılandırmaları yapın. Daha sonra `algolab.py` ve `algolab_socket.py` modüllerini projenizde import ederek kullanabilirsiniz.

Örnek kullanım:

```python
from algolab import AlgoLab
from algolab_socket import AlgoLabSocket

# API ile etkileşim
algolab = AlgoLab(api_key="your_api_key", username="your_username", password="your_password")
response = algolab.your_method()

# Soket ile etkileşim
socket = AlgoLabSocket(api_key="your_api_key", hash="your_hash")
socket.connect()
```

## Yapılandırma
`config.py` dosyası, API ve soket bağlantıları için temel yapılandırmaları içerir. API'nin hostname'i ve diğer sabitler bu dosyada tanımlanır.

## Bağımlılıklar
Bu wrapper'ın çalışması için gerekli bağımlılıklar `requirements.txt` dosyasında listelenmelidir.

## AlgoLab Link
https://algolab.com.tr/

## Lisans ve Yazar Bilgisi
Bu proje MIT altında yayınlanmıştır.
Atilla Yurtseven
---

Bu dokümantasyon, projenin temel kullanımını ve yapılandırmasını anlatmaktadır. Daha detaylı bilgi ve örnekler için, lütfen kod içindeki yorumları inceleyin.


Genel Bilgilendirmeler
Genel Bilgilendirme
Etkileşim Talebi
Standart Şartname
 Endpoints
Kullanıcı Girişi: SMS Alma
Kullanıcı Girişi: Oturum Açma
Oturum Yenileme
Sembol Bilgisi
Alt Hesap Bilgileri
Hisse Portföy Bilgisi
Viop Özeti
Hisse Özeti
Hisse Günlük İşlemler
Viop Portföy Bilgisi
Viop Günlük İşlemler
Emir Gönderim
Nakit Bakiye
Hesap Ekstresi



Genel Bilgilendirme
Versiyon: 1.0.5
Api Yetkilendirilmesi
API anahtarınızı Algolab üzerinden başvuru yapıp aldıktan sonra, Deniz Yatırım API ile bu dökümantasyonda yer alan yönlendirmelere göre uygulamanızı geliştirebilirsiniz. Her kullanıcı için bir adet API Anahtarı oluşturulmaktadır.

Başvurunun Onaylanması 
Başvurunuzun onaylanabilmesi için başvuru yaptığınız kısmın altında nasıl tamamlayacağınız belirtilen Uygunluk ve Yerindelik testleri en az Yüksek(4) olacak şekilde (Yerindelik testi sonunda yer alan Yatırım Danışmanlığı Çerçeve Sözleşmesi de onaylanmalı) tamamlanmalı ve Algoritmik İşlem Sözleşmesi onaylanmalıdır.
Canlı Veri Tipleri 
Anlık veri erişimi için iki adet izin mevcuttur. Bu izinler şu şekilde açıklanmaktadır:

Canlı Veri Sözleşmesi: Soketten gelen verilerin anlık olarak gelmesi için gerekli olan sözleşmedir. Eğer canlı veri yetkiniz bulunmuyorsa 15 dakika gecikmeli olarak belirli aralıklarla gelmektedir.

Derinlik Veri Sözleşmesi: Soketten gelen derinlik verilerine erişim sağlayan sözleşmedir. On kademe alım satım derinliği sağlanmaktadır.

API’ yi kullanırken lütfen aşağıdaki bilgileri unutmayın:

APIKEY: Rastgele bir algoritma tarafından oluşturulan API işlemlerinin kimliğidir.

Internet Bankacılığı Kullanıcı Adı/TCK Numarası: İnternet bankacılığına giriş yaparken sizin oluşturduğunuz kullanıcı adı veya internet bankacılığına giriş yaparken kullandığınız TCK numaranız.

Internet Bankacılığı Şifreniz: Sizin oluşturmuş olduğunuz internet bankacılığı şifreniz.

Sms Doğrulama Kodu: Sistemde kayıtlı telefon numaranıza gelen rastgele oluşturulmuş şifredir.


Etkileşim Talebi
Bu bölüm temel olarak erişime odaklanmaktadır:

Rest-API istekleri için aşağıdaki url ile erişim sağlanmaktadır.
   https://www.algolab.com.tr/api
Soket bağlantısı için aşağıdaki url ile erişim sağlanmaktadır.
   wss://www.algolab.com.tr/api/ws
API Doğrulaması
Bir istek başlatmak için aşağıdaki bilgiler gerekmektedir;
APIKEY: API Anahtarıdır.
Authorization: Kullanıcı girişi yapıldıktan sonraki isteklerde kullanılmaktadır.
Checker: Her isteğe özel imzadır. Her isteğe göre yeniden oluşturulur. APIKEY + RequestPath + QueryString veya Body(GET/POST Methoduna göre değişiklik göstermektedir.)
RequestPath: API yolu.
QueryString: İstek URL’ indeki sorgu dizesi (?’den sonraki istek parametresi soru işareti ile birlikte)
Body: İstek gövdesine karşılık gelen dize. Body genellikle POST isteğinde olur.
 

Örneğin Portföy bilgisini çekme;

APIKEY: APIKEY-04YW0b9Cb8S0MrgBw/Y4iPYi2hjIidW7qj4hrhBhwZg=

Yöntem: POST 

RequestPath: https://www.algolab.com.tr/api/api/Portfolio

QueryString: Yöntem POST olduğu için boş olarak girilir.

Body:

{
  "Subaccount": ""
}

 

JSON değer olduğu için String’ e dönüştürülerek Checker’ ı oluşturacak dizeye eklenir. Checker’ı oluşturacak dize aşağıdaki şekildedir.

"APIKEY04YW0b9Cb8S0MrgBw/Y4iPYi2hjIidW7qj4hrhBhwZg=https://www.algolab.com.tr/api/api/Portfolio{\"Subaccount\":\"\ "}"

Yukarıda string SHA256 hash algoritması ile şifrelenir. Şifrelemede oluşan String değer Checker parametresine yazılır.

Tüm istekler Https protokolüne dayalıdır ve Header (istek başlık) bilgilerindeki Content-Type (içerik türü) tamamını ‘application/json’ olarak ayarlanması gerekmektedir.

Başarı: HTTP durum kodu 200, başarılı bir yanıtı belirtir ve içerik içerebilir. Yanıt içerik içeriyorsa, ilgili dönüş içeriğinde görüntülenecektir. Başarılı dönen cevaplar aşağıdaki json model’ ine göre döner.

{
   "Success": bool, Başarılı bir istek ise true cevabı gelir
   "Message": string, Eğer başarısız bir istekse veya herhangi bir hata olursa hata mesajı döner
   "Content": object Her fonksiyona göre farklı model dönmektedir
}


Standart Şartname
Frekans Sınırlama Kuralları İstek çok sık olursa, sistem isteği otomatik olarak sınırlandırır ve http başlığında 429 çok fazla istek durum kodunu döndürür. Frekans limiti beş saniyede bir istektir.
Talep Formatı şu anda biçimlerde yalnızca iki istek yöntemi vardır: GET ve POST
GET: Parametreler sunucuya queryString yolu ile iletilir.
POST: Parametreler gövde json formatında gönderilerek sunucuya gönderilir.
Algolab API’sine yalnızca Türkiye içindeki sunuculardan erişim sağlanabilir. Yurt dışındaki sunucular üzerinden bağlantı kurulması mümkün değildir.
Bir IP adresi üzerinden aynı anda sadece bir kullanıcı Algolab API’sine bağlanabilir. Aynı IP adresinden birden fazla kullanıcı bağlantı kurmaya çalıştığında, yalnızca ilk kullanıcı bağlantı kurabilir.
Canlı veri erişim yetkisine sahip kullanıcılar, veri yayın kurallarına uymak zorundadır. Bu kuralları ihlal eden kullanıcılar, hukuki ve cezai sorumluluklarla karşı karşıya kalabilirler.
Algolab API’si üzerinden oturum açan bir kullanıcı, aynı anda Algolab Web veya mobil uygulaması üzerinden oturum açamaz. Bu durumda, API oturumu otomatik olarak sonlanır.
Algolab API’sini kullanarak geliştirilen üçüncü taraf yazılımlar, Algolab ile resmi bir anlaşmaya sahip değildir. Bu yazılımlar Algolab tarafından desteklenmez veya onaylanmaz.

Kullanıcı Girişi: SMS Alma
Internet Bankacılığı bilgileri ile giriş yapmanızı sağlar. İstek sonunda sistemde kayıtlı telefon numaranıza SMS doğrulama kodu gelir. Gelen SMS’ teki kod ile bir sonraki işlem gerçekleştirilecektir.

Http İsteği

POST /api/LoginUser
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan APIKEY
İstek Parametresi

Aşağıdaki parametreleri AES Algoritmasını kullanarak APIKEY içerisindeki “-” ‘den sonraki string değer ile şifrelemeniz gerekmektedir.

Örneğin:

APIKEY: APIKEY-04YW0b9Cb8S0MrgBw/Y4iPYi2hjIidW7qj4hrhBhwZg=

Yukarıdaki APIKEY’ e göre AES Algoritmasında kullanılacak key aşağıdaki şekildedir.

aes.Key: 04YW0b9Cb8S0MrgBw/Y4iPYi2hjIidW7qj4hrhBhwZg=

 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
Username	String	Kullanıcı Adı/ TCK Numarası
Password	String	İnternet Bankacılığı Şifresi
 

Örnek Request Body

{
   "Username": "YTZ1RF2Q04T/nZThi0JzUA==",
   "Password": "9LHZEiA2AhKsAtM4yOOrEw=="
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
token	String	SMS için gerekli token
 

 Örnek Response

{
   "success": true,
   "message": "",
   "content": {
      "token": "Ys/WhU/D37vO71VIBumDRhZLmkcMlzyb3TKJVWxLlpb/4BByYLNfQ07dEe66P3Ab"
   }
}

import requests, hashlib, json, inspect, time, base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

# Kullanıcı Bilgileri
apikey= "" # API Key'inizi Buraya Giriniz
username = "" # TC veya Denizbank Kullanıcı Adınızı Buraya Giriniz
password = "" # Denizbank İnternet Bankacılığı Şifrenizi Buraya Giriniz


# İstek gönderilecek host bilgileri
hostname = "www.algolab.com.tr" # İstek atılacak web sitesi
api_hostname = f"https://{hostname}" # HTTPS eklentisi
api_url = api_hostname + "/api" # API isteğinin atıdığı kısım

class API():
    def __init__(self, api_key, username, password):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
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
        try:
            f = inspect.stack()[0][3]
            u = self.encrypt(self.username)
            p = self.encrypt(self.password)
            payload = {"username": u, "password": p}
            endpoint = "/api/LoginUser"
            resp = self.post(endpoint=endpoint, payload=payload, login=True)
            return resp.json()
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
        
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
            
    def encrypt(self, text): 
        iv = b'\0' * 16
        key = base64.b64decode(self.api_code.encode('utf-8'))
        cipher = AES.new(key, AES.MODE_CBC, iv)
        bytes = text.encode()
        padded_bytes = pad(bytes, 16)
        r = cipher.encrypt(padded_bytes)
        return base64.b64encode(r).decode("utf-8")

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password)
    if login := algo.LoginUser():
        try:
            print(login)
        except Exception as e:
            print(f"Hata oluştu: {e}")


Kullanıcı Girişi: Oturum Açma
Kullanıcı girişi Sms alma metodunda alınan token ve sistemdeki kayıtlı telefonunuza gelen kod ile hash kodu almanızı sağlar.

Http İsteği: POST /api/LoginUserControl
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
token	String	SMS Alma metotundaki token
Password	String	SMS Kodu
Örnek Request Body

{
   "token": "Ys/WhU/D37vO71VIBumDRhZLmkcMlzyb3TKJVWxLlpb/4BByYLNfQ07dEe66P3Ab",
   "Password": "9LHZEiA2AhKsAtM4yOOrEw=="
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
Hash	String	Oturum süresi boyunca erişim sağlanacak oturum anahtarıdır.
 

Örnek Response:

{
   "success": true,
   "message": "",
   "content": {
      "hash": "eyJhbGciOiJodHRwOi8vd3d3LnczLm9yZy8yMDAxLzA0L3htbGRzaWctbW9yZSNobWFjLXNoYTI1NiIsInR5cCI6IkpX VCJ9.eyJBdXRob3JpemF0aW9uIjoiQXV0aG9yaXplZCIsIkN1c3RvbWVyTm8iOiIxMzQ1MTcyMCIsIk5ld3NsZXR0ZXIi OiJUcnVlIiwiSXNCbG9ja2VkIjoiRmFsc2UiLCJFbWFpbCI6IjEzNDUxNzIwIiwiVXNlcklkIjoiMTAxIiwiRGVuaXpiYW5rIjoi VHJ1ZSIsIm5iZiI6MTY1MzQ4NjMxMCwiZXhwIjoxNjUzNTcyNzEwfQ.8PtF5zNa24bSr3edBuqzpeWqbgxK2rLRXQReovoC2c"
   }
}

import requests, hashlib, json, inspect, time, base64
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad

# Kullanıcı Bilgileri
apikey="" # API Key'inizi Buraya Giriniz
username = "" # TC veya Denizbank Kullanıcı Adınızı Buraya Giriniz
password = "" # Denizbank İnternet Bankacılığı Şifrenizi Buraya Giriniz
token = "" # Login olduktan sonra aldığınız token
sms = "" # LoginUser'dan sonra kayıtlı cep telefonuna gelen SMS kodu

# İstek gönderilecek host bilgileri
hostname = "www.algolab.com.tr" # İstek atılacak web sitesi
api_hostname = f"https://{hostname}" # HTTPS eklentisi
api_url = api_hostname + "/api" # API isteğinin atıdığı kısım

class API():
    def __init__(self, api_key, username, password, token, sms):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        sms: Giriş yaptıktan sonra alınan sms
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.last_request = 0.0
        self.LOCK = False

    def LoginUserControl(self):
        try:
            self.sms_code = sms
            f = inspect.stack()[0][3]
            t = self.encrypt(self.token)
            s = self.encrypt(self.sms_code)
            payload = {'token': t, 'password': s}
            endpoint = "/api/LoginUserControl"
            resp = self.post(endpoint, payload=payload, login=True)
            return resp.json()
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
        
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
            
    def encrypt(self, text):
        iv = b'\0' * 16
        key = base64.b64decode(self.api_code.encode('utf-8'))
        cipher = AES.new(key, AES.MODE_CBC, iv)
        bytes = text.encode()
        padded_bytes = pad(bytes, 16)
        r = cipher.encrypt(padded_bytes)
        return base64.b64encode(r).decode("utf-8")

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, sms)
    if user_control := algo.LoginUserControl():
        try:
            print(user_control)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            
Oturum Yenileme
Oturum yenileme fonksiyonudur.

Http İsteği: POST /api/ SessionRefresh
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametresi herhangi bir parametre almamaktadır.

 

Sonuç Bool(true,false) değer döner.

import requests, hashlib, json, inspect, time, pandas as pd
apikey= ""
username = ""
password = ""
token = ""
hash = ""
hostname = "www.algolab.com.tr" 
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def SessionRefresh(self):
        """
        Oturum süresi uzatmak için atılan bir istektir.
        Cevap olarak Success: True veya eğer hash'iniz geçerliliğini yitirmişte 401 auth hatası olarak döner.
        """
        try:
            f = inspect.stack()[0][3]
            endpoint = "/api/SessionRefresh"
            payload = {}
            resp = self.post(endpoint, payload=payload)
            return self.error_check(resp, f)
        except Exception as e:
                print(f"{f}() fonsiyonunda hata oluştu: {e}")
                
    
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if session := algo.SessionRefresh():
        try:
            print(session)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            

Sembol Bilgisi
Sembol ile ilgili bilgileri getirir.

Http İsteği: POST /api/GetEquityInfo
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametreleri 

Parametre Adı	Parametre Tipi	Açıklama
symbol	String	Sembol Kodu
Örnek Request Body

{
   "symbol": "TSKB"
}

 

Sonuç Parametreleri 

Parametre Adı	Parametre Tipi	Açıklama
name	String	Sembol Adı
flr	String	Taban
clg	String	Tavan
ask	String	Alış Fiyatı
bid	String	Satış Fiyatı
lst	String	Son Fiyat
limit	String	İşlem Limiti
min	String	-
max	String	-
step	String	-
Örnek Response

{
   "success": true,
   "message": "",
   "content": {
      "name": "TSKB",
      "flr": "1.840",
      "clg": "2.240",
      "ask": "2.060",
      "bid": "2.050",
      "lst": "2.060",
      "limit": "0.00",
      "min": "",
      "max": "",
      "step": ""
   }
}


import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username = ""
password = ""
token = ""
hash = ""
sembol = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def GetEquityInfo(self, symbol):
        """
        Sembolle ilgili tavan taban yüksek düşük anlık fiyat gibi bilgileri çekebilirsiniz.
        :String symbol: Sembol Kodu Örn: ASELS
        """
        try:
            f = inspect.stack()[0][3]
            endpoint = "/api/GetEquityInfo"
            payload = {'symbol': symbol}
            resp = self.post(endpoint, payload=payload)
            return self.error_check(resp, f)
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
    
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if info := algo.GetEquityInfo(sembol):
        try:
            succ = info["success"]
            if succ:
                content = info["content"]
                df = pd.DataFrame(content,index=[0])
                print(df)
            else: print(info["message"]) 
        except Exception as e:
            print(f"Hata oluştu: {e}")
            

Alt Hesap Bilgileri
Kullanıcıya ait alt hesap bilgilerini getirir.

Http İsteği: POST /api/GetSubAccounts
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek parametresi bulunmamaktadır.

 

Sonuç Parametreleri 

Parametre Adı	Parametre Tipi	Açıklama
Number	String	Alt Hesap Numarası
TradeLimit	String	Alt Hesap İşlem Limiti
Örnek Response

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


import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username = ""
password = ""
token = ""
hash = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def GetSubAccounts(self, silent=False):
        try:
            f = inspect.stack()[0][3]
            end_point = "/api/GetSubAccounts"
            resp = self.post(end_point, {})
            return self.error_check(resp, f, silent=silent)
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
                
    
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request:= algo.GetSubAccounts():
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            
Hisse Portföy Bilgisi
Kullanıcıya ait anlık portföy bilgilerini getirir.

Http İsteği: POST /api/InstantPosition
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
Subaccount	String	
Alt Hesap Numarası (Boş olarak gönderilebilir,

boş olarak gönderilirse Aktif Hesap bilgilerini iletir.)

Örnek Request Body

{
  "Subaccount": ""
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
maliyet	String	
Menkul kıymetin alış fiyatı

totalstock	String	Menkul kıymetin Toplam Miktarı
code	String	Enstrüman ismi
profit	String	Menkul Kıymet Kar/Zararı
cost	String	Menkul kıymetin birim maliyeti
unitprice	String	Menkul kıymetin birim fiyatı
totalamount	String	Toplam satır bakiye TL değeri
tlamount	String	TL tutarı
explanation	String	Menkul kıymet Açıklaması
type	String	Overall kaleminin tipi
total	String	-
Örnek Response

{
  "success": true,
  "message": "",
  "content": [
    {
      "maliyet": "2.05",
      "totalstock": "1",
      "code": "TSKB",
      "profit": "0",
      "cost": "2.05",
      "unitprice": "2.05",
      "totalamount": "",
      "tlamaount": "2.05",
      "explanation": "TSKB",
      "type": "CH",
      "total": "0"
    }
  ]
}

import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username = ""
password = ""
token = ""
hash = ""
subAccount = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def GetInstantPosition(self, sub_account=""):
        """
        Yatırım Hesabınıza bağlı alt hesapları (101, 102 v.b.) ve limitlerini görüntüleyebilirsiniz.
        """
        try:
            f = inspect.stack()[0][3]
            end_point = "/api/InstantPosition"
            payload = {'Subaccount': sub_account}
            resp = self.post(end_point, payload)
            return self.error_check(resp, f)
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
                
    
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request:= algo.GetInstantPosition(subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            
Viop Özeti
Kullanıcıya ait Viop özet bilgilerini getirir.

Http İsteği: POST /api/ViopCollateralInfo
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
Subaccount	String	
Alt Hesap Numarası (Boş olarak gönderilebilir,

boş olarak gönderilirse Aktif Hesap bilgilerini iletir.)

Örnek Request Body

{
  "Subaccount": ""
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
sumcustody

String

Takastaki teminat

sumorganization

String

Kurum teminatı

sumcustodycalc

String

Çarpılmış hesaplanan kurum teminatı

sumorganizationcalc

String

Çarpılmış hesaplanan kurum teminatı

noncash

String

Takas nakit dışı teminat

noncashorg

String

Takas nakit dışı kurum teminatı

sumscanvalue

String

Scan değeri

sumspreadvalue

String

Yayılma maliyeti

sumshortoptionminimum

String

Kısa opt. Asgari teminatı

scanrisk

String

Scan riski

availablenetoption

String

Net opsiyon değeri

waitingpremium

String

Bekleyen ödenecek prim

deliverychange

String

Teslimat maliyeti

maintanancerequirements

String

Sürdürme teminatı

initialrequirements

String

Başlangıç teminatı

requiredcollateral

String

Bulunması gereken teminat

instantprofitloss

String

Anlık kar/zarar

freecoll

String

Çekilebilir teminat

usablecoll

String

Kullanılabilir teminat

risklevel

String

Risk seviyesi

custodymargincallamount

String

Margin call miktarı

Örnek Response

{
  "success": true,
  "message": "",
  "content": [
    {
      "sumcustody": "0",
      "sumorganization": "0",
      "sumcustodycalc": "0",
      "sumorganizationcalc": "0",
      "noncash": "0",
      "noncashorg": "0",
      "sumscanvalue": "0",
      "sumspreadvalue": "0",
      "sumshortoptionminimum": "0",
      "scanrisk": "0",
      "availablenetoption": "0",
      "waitingpremium": "0",
      "deliverychange": "0",
      "maintanancerequirements": "0",
      "initialrequirements": "0",
      "requiredcollateral": "0",
      "instantprofitloss": "0",
      "freecoll": "0",
      "usablecoll": "0",
      "risklevel": "0",
      "custodymargincallamount": "0"
    }
  ]
}

import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username = ""
password = ""
token = ""
hash = ""
subAccount = ""

hostname = "www.algolab.com.tr" 
api_hostname = f"https://{hostname}" 
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def ViopColleteralInfo(self, sub_account=""):
        try:
            f = inspect.stack()[0][3]
            end_point = "/api/ViopCollateralInfo"
            payload = {'Subaccount': sub_account}
            resp = self.post(end_point, payload)
            return self.error_check(resp, f)
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
                
    
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request:= algo.ViopColleteralInfo(subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")


Hisse Özeti
Kullanıcıya ait Hisse özet bilgilerini getirir.

Http İsteği: POST /api/RiskSimulation
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
Subaccount	String	
Alt Hesap Numarası (Boş olarak gönderilebilir,

boş olarak gönderilirse Aktif Hesap bilgilerini iletir.)

Örnek Request Body

{
  "Subaccount": ""
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
t0

String

T Nakit Bakiyesi

t1

String

T+1 Nakit Bakiyesi

t2

String

T+2 Nakit Bakiyesi

t0stock

String

-

t1stock

String

-

t2stock

String

-

t0equity

String

 T Hisse Portföy Değeri  

t1equity

String

T+1 Hisse Portföy Değeri 

t2equity

String

T+2 Hisse Portföy Değeri 

t0overall

String

T Overall Değeri Nakit Dahil

t1overall

String

T+1 Overall Değeri Nakit Dahil 

t2overall

String

T+2 Overall Değeri Nakit Dahil 

t0capitalrate

String

T Özkaynak Oranı  

t1capitalrate

String

T+1 Özkaynak Oranı 

t2capitalrate

String

T+2 Özkaynak Oranı 

netoverall

String

Nakit Hariç  Overall 

shortfalllimit

 String 

Açığa satış sözleşmesi olan müşteriler için kullanılabilir açığa satış bakiyesi 

credit0

String 

T Kredi Bakiyesi 

Örnek Response

{
  "success": true,
  "message":  "",
  "content": [
    {
      "t0": "0",
      "t1": "0",
      "t2": "0",
      "t0stock": "0",
      "t1stock": "0",
      "t2stock": "0",
      "t0equity": "0",
      "t1equity": "0",
      "t2equity": "0",
      "t0overall": "0",
      "t1overall": "0",
      "t2overall": "0",
      "t0capitalrate": "0",
      "t1capitalrate": "0",
      "t2capitalrate": "0",
      "netoverall": "0",
      "shortfalllimit": "0",
      "credit0": "0"
    }
  ]
}
            
import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username = ""
password = ""
token = ""
hash = ""
subAccount = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def RiskSimulation(self, sub_account=""):
        try:
            f = inspect.stack()[0][3]
            end_point = "/api/RiskSimulation"
            payload = {'Subaccount': sub_account}
            resp = self.post(end_point, payload)
            return self.error_check(resp, f)
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
                
    
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request:= algo.RiskSimulation(subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            

Hisse Günlük İşlemler
Kullanıcıya ait günlük işlem kayıtlarını getirir.

Http İsteği: POST /api/TodaysTransaction
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
Subaccount	String	
Alt Hesap Numarası (Boş olarak gönderilebilir,

boş olarak gönderilirse Aktif Hesap bilgilerini iletir.)

Örnek Request Body

{
  "Subaccount": ""
}

 

Sonuç Parametreleri 

Parametre Adı	Parametre Tipi	Açıklama
atpref	String	Referans Numarası
ticker	String	Hisse Senedi İsmi
buysell	String	İşlemin Cinsi (Alış, Satış)
ordersize	String	Emir Miktarı
remainingsize	String	Emrin BIST’ te henüz karşılanmamış
ve kalan bölümü bu sahada belirtilir.
price	String	Emrin ortalama gerçekleşme fiyatını
belirtir.
amount	String	Gerçekleşen kısım ile ilgili müşterinin hesabından çekilecek
veya hesabına yatırılacak meblağ bu
sahada bulunmaktadır.
transactiontime	String	Emrin giriş tarihini belirtir. Kısa tarih
formatındadır.
timetransaction	String	Emrin girildiği tarih uzun tarih
formatındadır.
valor	String	Emrin geçerliliğinin başladığı seans tarihini belirtir. Kısa tarih
formatındadır.
status	String	Emrin değiştirilebilme durumu bilgilerini içerir; Emir Silme,
İyileştirme ve valör iptali
işlemlerinin yapılıp yapılamayacağı bu bilgilerden anlaşılır. 5 haneden oluşan bir “string” değerdir. Her bir karakter “0” (“Mümkün Değil”) veya
“1” (“Mümkün”) olabilir. Soldan
itibaren birinci değer emrin silinip silinemeyeceğini belirtir. İkinci ve üçüncü değerler fiyat iyilestirme ve emir bölme işlemlerinin yapılıp yapılamayacağını belirtir. Sonraki değer ise emrin geçerlilik süresinin iptal edilip edilemeyeceğini belirtir.
En son değer emrin kötüleştirilip
kötüleştirilemiyeceğini belirtir.
waitingprice	String	Emrin bekleyen kısmının fiyatını belirtir. Emir fiyatı olarak bu alan kullanılmalıdır.
description	String	
Emir durumu bilgisini belirtir;

İletildi
Silindi
İyileştirme Talebi Alındı
İyileştirildi
Silme Talebi Alındı
İyileştirme Reddedildi
Emir Reddedildi
Silme Reddedildi
KIE Emri Silindi
KPY Emri Silindi
Gerçekleşti
Kısmi Gerçekleşti
transactionId	String	GTP’de tutulan referansı belirtir. İşlemler GTP’ye bu referans gönderilir. GTP emri bu id ile tanır. GTP’de unique olarak tutulur.
equityStatusDescription	String	
Ekranda emirleri gruplayabilmek amaçıyla gönderilen özel bir alandır.

WAITING: Bekleyen Emirler
DONE: Gerçekleşen Emirler
PARTIAL: Kısmi Gerçekleşen Emirler
IMPROVE_DEMAND: Emir iyileştirme talebi alındı
DELETE_DEMAND: Emir silme talebi alındı
DELETED: Gerçekleşmesi olmayan silinmiş emirler, borsadan red almış emirler.
shortfall	String	Açığa satış
timeinforce	String	Emrin geçerlilik süresini belirtir. Emir girişinde kullanılan değerler
geri dönülür.
fillunit	String	Gerçekleşme adet bilgisini verir.

Örnek Response

{
  "success": true,
  "message": "",
  "content": [
    {
      "atpref": "0013O2",
      "ticker": "TSKB",
      "buysell": "Alış",
      "ordersize": "1",
      "remainingsize": "0",
      "price": "2.050000",
      "amount": "2.050000",
      "transactiontime": "27.05.2022 00:00:00",
      "timetransaction": "27.05.2022 11:47:32",
      "valor": "27.05.2022",
      "status": "00000",
      "waitingprice": "2.050000",
      "description": "Gerçekleşti",
      "transactionId": "0000-291B5D-IET",
      "equityStatusDescription": "DONE",
      "shortfall": "0",
      "timeinforce": "",
      "fillunit": "1"
    }
  ]
}

import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username = ""
password = ""
token = ""
hash = ""
subAccount = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def GetTodaysTransaction(self, sub_account=""):
        """
        Günlük işlemlerinizi çekebilirsiniz.(Bekleyen gerçekleşen silinen v.b.)
        """
        try:
            f = inspect.stack()[0][3]
            end_point = "/api/TodaysTransaction"
            payload = {'Subaccount': sub_account}
            resp = self.post(end_point, payload)
            return self.error_check(resp, f)
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
                
    
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request:= algo.GetTodaysTransaction(subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            
 Viop Portföy Bilgisi
Müşterinin pozisyon ve kar zarar bilgilerini içeren overall bilgisini getirir.

Http İsteği: POST /api/ViopCustomerOverall
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
Subaccount	String	
Alt Hesap Numarası (Boş olarak gönderilebilir,

boş olarak gönderilirse Aktif Hesap bilgilerini iletir.)

Örnek Request Body

{
  "Subaccount": ""
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
contract	String	
Sözleşme Adı

contractname	String	Sözleşme Adı
shortlong	String	Uzun, Kısa
units	String	Adet
putcall	String	Put/Call Bilgisi
shortbalance	String	Kısa Adet
longbalance	String	Uzun Adet
openpositiontotal	String	Toplam Açık Pozisyon
exerciseunits	String	Fiziki Kullanım Miktarı
waitingexerciseunits	String	Fiziki Kullanım Bekletilen Miktar
unitnominal	String	Unit nominal
totalcost	String	Toplam Maliyet
profit	String	Opisyon Kullanımında elde edilecek kar/zarar
profitloss	String	Muhasebeleştirilmiş Kar/Zarar
dailyprofitloss	String	Adet * (uzlaşma - önceki uzlaşma) * birim nominal (pozisyon kar-zararı)
(Futures)
potprofit	String	Pozisyon maliyetine göre kar/zarar
fininstid	String	-
Örnek Response

{
  "success": true,
  "message": "",
  "content": [
    {
      "contract": "-",
      "contractname": "-",
      "shortlong": "-",
      "units": "-",
      "putcall": "-",
      "shortbalance": "-",
      "longbalance": "-",
      "openpositiontotal": "-",
      "exerciseunits": "-",
      "waitingexerciseunits": "-",
      "unitnominal": "-",
      "totalcost": "-",
      "profit": "-",
      "profitloss": "-",
      "dailyprofitloss": "-",
      "potprofit": "-",
      "fininstid": "-"
    }
  ]
}


import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username = ""
password = ""
token = ""
hash = ""
subAccount = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def GetViopCustomerOverall(self, sub_account=""):
        try:
            f = inspect.stack()[0][3]
            end_point = "/api/ViopCustomerOverall"
            payload = {'Subaccount': sub_account}
            resp = self.post(end_point, payload)
            return self.error_check(resp, f)
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
                
    
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request:= algo.GetViopCustomerOverall(subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            
Viop Günlük İşlemler
Müşterinin pozisyon ve kar zarar bilgilerini içeren overall bilgisini getirir.

Http İsteği: POST /api/ViopCustomerTransactions
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
Subaccount	String	
Alt Hesap Numarası (Boş olarak gönderilebilir,

boş olarak gönderilirse Aktif Hesap bilgilerini iletir.)

Örnek Request Body

{
  "Subaccount": ""
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
contract	String	Sözleşme adı
shortlong	String	Uzun kısa (Alış\Satış) sözleşme
bilgisi
units	String	Emir miktarı
leftunits	String	Kalan miktar
price	String	Emir fiyatı
transactionid	String	Emir ID’ si
transactiondate	String	Emir tarihi
transactionenddate	String	Emir bitiş tarihi
transactiondatetype	String	Emir gün tipi (DAY, DTD, SNS, vs.)
transactionunitnominal	String	Emir nominal değeri
transactionunit	String	Gerçekleşen adet
trexid	String	Emir referans değeri
description	String	Emir açıklaması (API’ de hataya düşmesi durumunda)
Gerçekleşti
Kısmen gerçekleşti
İletildi
Bekliyor
İptal
ordertime	String	Emir gerçekleşme zamanı
(dd.mm.yyyy hh:MM:SS)
validity	String	Emir gün tipi
fininstid	String	Alt sözleşme ID’ si
ordertype	String	Emir tipi
pricetype	String	Emir fiyat tipi açıklaması (Limitli,
Kalanı Pasife Yaz, vs.)
pricetypename	String	Emir fiyat kodu (LMT, KPY vs.)
info	String	Emir Durumu (Bekliyor, Gerçekleşti, İptal, Hata, Kısmi Gerçekleşti,
İletildi)
timeinforce	String	Emir Süresi
realizedunits	String	Emir gerçekleşen miktarı
priceaverage	String	Ortalama gerçekleşme fiyatı
transstatusname	String	Emir durum bilgisi
ACTIVE Aktif
IMPROVE_DEMAND İyileştirilen Emir
IMPROVE_ORDER İyileştirilecek Emir
READED_BY_DISC_ORDER Disket emir tarafından okunmuş kayıtlar.
CORRECT_DEMAND Update edilmek istenen kayıtlar.
CANCEL_DEMAND İptal edilmek istenen kayıtlar.
CANCEL_ORDER İptal edilen kayıtlar.
CORRECT_ORDER Update edilen kayıtlar.
API_ERROR API `den geri döndü
API_IMPROVE_ERROR Emir iyileştirmede API Hatası alındı
API_CANCEL_ERROR Emir iptalinde API Hatası alındı
PARTIAL_FILL Parçalı Gerçekleşme
FILLED Gerçekleşme
DONE Done For Day
STOPPED Durmuş
REJECTED Reddedildi
PENDING_NEW Onay bekliyor
CALCULATED Calculated
EXPIRED Süresi Dolmuş
ACCEPT_BIDDING Teklif Kabul Edilmiş
SUSPENDED Geçici Olarak Durmuş
Örnek Response

{
  "success": true,
  "message": "",
  "content": [
    {
      "contract": "-",
      "contractname": "-",
      "shortlong": "-",
      "units": "-",
      "putcall": "-",
      "shortbalance": "-",
      "longbalance": "-",
      "openpositiontotal": "-",
      "exerciseunits": "-",
      "waitingexerciseunits": "-",
      "unitnominal": "-",
      "totalcost": "-",
      "profit": "-",
      "profitloss": "-",
      "dailyprofitloss": "-",
      "potprofit": "-",
      "fininstid": "-"
    }
  ]
}

import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username = ""
password = ""
token = ""
hash = ""
subAccount = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def GetViopCustomerTransactions(self, sub_account=""):
        try:
            f = inspect.stack()[0][3]
            end_point = "/api/ViopCustomerTransactions"
            payload = {'Subaccount': sub_account}
            resp = self.post(end_point, payload)
            return self.error_check(resp, f)
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
                
    
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request:= algo.GetViopCustomerTransactions(subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            
Emir Gönderim
Alım/satım emrini iletir.

Http İsteği: POST /api/SendOrder
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
symbol	String	Sembol Kodu
direction	String	BUY/SELL
pricetype	String	Emir Tipi (piyasa/limit)
price	String	Emir tipi limit ise fiyat girilmelidir.(örneğin 1.98)
lot	String	Emir adedi
sms	Bool	Sms Gönderim
email	Bool	Email Gönderim
Subaccount	String	Alt Hesap Numarası (Boş gönderilebilir. Boş gönderilirse Aktif Hesap Bilgilerini Getirir.)
Örnek Request Body

{
  "symbol": "TSKB",
  "direction": "BUY",
  "pricetype": "limit",
  "price": "2.01",
  "lot": "1",
  "sms": true,
  "email": false,
  "Subaccount": ""
}

 

Emir doğru bir şekilde iletilmiş ise sistemden String olarak emir referans numarası dönmektedir. Aşağıdaki örnek response içinde yer alan işaretli numara ile emrinizi düzenleyebilir veya silebilirsiniz.

Örnek Response

{
  "success": true,
  "message": "",
  "content": "Referans Numaranız: 001VEV;0000-2923NR-IET - HISSEOK"
}

import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username = ""
password = ""
token = ""
hash = ""
subAccount = ""
sembol = ""
direction = ""
pricetype = ""
price = ""
lot = ""
sms = False
email = False
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"
class API():
    def __init__(self, api_key, username, password, token, hash):
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def SendOrder(self, symbol, direction, pricetype, price, lot, sms, email, subAccount):
        try:
            end_point = "/api/SendOrder"
            payload = {
                "symbol": symbol,
                "direction": direction,
                "pricetype": pricetype,
                "price": price,
                "lot": lot,
                "sms": sms,
                "email": email,
                "subAccount": subAccount
            }
            resp = self.post(end_point, payload)
            try:
                data = resp.json()
                return data
            except:
                f = inspect.stack()[0][3]
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                
                print(resp.text)
                
        except Exception as e:
            f = inspect.stack()[0][3]
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
       
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")
                print(resp.text)
            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request := algo.SendOrder(sembol,direction,pricetype,price,lot,sms,email,subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
Nakit Bakiye
T0, T+1, T+2 nakit bayileri getirir.

Http İsteği: POST /api/CashFlow
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
Subaccount	String	
Alt Hesap Numarası (Boş olarak gönderilebilir,

boş olarak gönderilirse Aktif Hesap bilgilerini iletir.)

Örnek Request Body

{
  "Subaccount": ""
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
t0	String	
T+0 Anındaki Nakit Bakiye

t1	String	T+1 Anındaki Nakit Bakiye
t2	String	T+2 Anındaki Nakit Bakiye
Örnek Response

{
  "success": true,
  "message": "Canceled",
  "content": {
    "t0": "",
    "t1": "",
    "t2": ""
  }
}

import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username =""
password = ""
token = ""
hash = ""
subAccount = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def CashFlow(self, sub_account=""):
        try:
            f = inspect.stack()[0][3]
            end_point = "/api/CashFlow"
            payload = {'Subaccount': sub_account}
            resp = self.post(end_point, payload)
            return self.error_check(resp, f)
        except Exception as e:
            print(f"{f}() fonsiyonunda hata oluştu: {e}")           
    
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request := algo.CashFlow(subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            
Hesap Ekstresi
Kullanıcıya ait ilgili tarihler arasındaki hesap ekstresini verir.

Http İsteği: POST /api/AccountExtre
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
start	DateTime	Başlangıç Tarihi
end	DateTime	Bitiş Tarihi
Subaccount	String	
Alt Hesap Numarası (Boş olarak gönderilebilir,

boş olarak gönderilirse Aktif Hesap bilgilerini iletir.)

Örnek Request Body

{
   "start": 2023-07-01 00:00:00,
   "end": 2023-07-31 00:00:00,
   "Subaccount": ""
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
accountextre	List<AccountExtre>	Hisse Ekstre
viopextre	List<ViopAccountStatement>	Viop Ekstre
 

Account Extre

Parametre Adı	Parametre Tipi	Açıklama
transdate	String	İşlemin muhasebe tarihi
explanation	String	İşlemin açıklaması
debit	String	İşlem ile ilgili borç miktarı
credit	String	İşlem ile ilgili alacak miktarı
balance	String	İşlem sonrasındaki hesabın bakiyesi
valuedate	String	İşlemin valör tarih ve saati
 

Viop Account Statement

Parametre Adı	Parametre Tipi	Açıklama
shortlong	String	Uzun kısa (Alış\Satış) sözleşme bilgisi
transactiondate	String	Emir zamanı
contract	String	İşlem yapılan sözleşme adı
credit	String	Alınan miktar
debit	String	Satılan miktar
units	String	Sözleşme adedi
price	String	Sözleşme fiyatı
balance	String	Hesap Bakiyesi
currency	String	Para birimi
Örnek Response

{
  "success": True,
  "message": "",
  "content": {
    "accountextre": [
      {
        "transdate": "01.01.0001 00:00:00",
        "explanation": "Devir",
        "debit": "0",
        "credit": "0",
        "balance": "0",
        "valuedate": "22.07.2024 00:00:00"
      }
    ],
    "viopextre": [
      {
        "shortlong": "-",
        "transactiondate": "-",
        "contract": "-",
        "credit": "-",
        "debit": "-",
        "units": "-",
        "price": "-",
        "balance": "Object reference not set to an instance of an object.",
        "currency": "-"
      }
    ]
  }
}

import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username =""
password = ""
token = ""
hash = ""
subAccount = ""
days = 1
end_date = datetime.now(timezone(timedelta(hours=3))) # Bugün
start_date = end_date - timedelta(days=days)#days kısmında belirtilen gün kadar öncesi
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def AccountExtre(self, sub_account="", start_date=None, end_date=None):
        try:
            f = inspect.stack()[0][3]
            end_point = "/api/AccountExtre"
            # datetime nesneleri isoformat() ile dönüştürülüyor
            payload = {
                'start': start_date.isoformat() if start_date else None,
                'end': end_date.isoformat() if end_date else None,
                'Subaccount': sub_account
            }
            resp = self.post(end_point, payload)
            return self.error_check(resp, f)
        except Exception as e:
            print(f"{f}() fonksiyonunda hata oluştu: {e}")
            
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request := algo.AccountExtre(subAccount,start_date,end_date):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            
VIOP Emri İptal Etme
VIOP Emri İptal Etme Gönderilen ve açık olan viop emrini iptal eder.

Http İsteği: POST /api/DeleteOrderViop
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
id	String	Emrin ID’ si
adet	String	İptal edilecek adet
Subaccount	String	Alt Hesap Numarası (Boş gönderilebilir. Boş gönderilirse Aktif Hesap Bilgilerini Getirir.)
Örnek Request Body

{
   "id": "001VEV",
   "adet": "1",
   "Subaccount": ""
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
message	String	Emrin iletimi hakkında bilgi verir.
duration	String	-
Örnek Response

{
   "success": true,
   "message": "Canceled",
   "content": {
      "message": "Canceled",
      "duration": "-"
   }
}

import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username =""
password = ""
token = ""
hash = ""
subAccount = ""
id = ""
adet = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"

class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def DeleteOrderViop(self, id, adet, subAccount):
        try:
            end_point = "/api/DeleteOrderViop"
            payload = {
                'id': id,
                'adet': adet,
                'subAccount': subAccount
            }
            resp = self.post(end_point, payload)
            try:
                data = resp.json()
                return data
            except:
                f = inspect.stack()[0][3]
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                
                print(resp.text)
                
        except Exception as e:
            f = inspect.stack()[0][3]
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
            
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request := algo.DeleteOrderViop(id,adet,subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            
Hisse Emri İptal Etme
Gönderilen ve açık olan hisse emrini iptal eder.

Http İsteği: POST /api/DeleteOrder
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

 İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
id	String	Emrin ID’ si
subAccount	String	Alt Hesap Numarası (Boş gönderilebilir. Boş gönderilirse Aktif Hesap Bilgilerini Getirir.)
Örnek Request Body

{
   "id": "001VEV",
   "subAccount": ""
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
message	String	Emrin iletimi hakkında bilgi verir.
duration	String	-
Örnek Response

{
   "success": true,
   "message": "Success",
   "content": {
      "message": "Success",
      "duration": "-"
   }
}

import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username =""
password = ""
token = ""
hash = ""
subAccount = ""
id = ""
adet = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"
class API():
    def __init__(self, api_key, username, password, token, hash):
        """
        api_key: API-KEY
        username: TC Kimlik No
        password: Hesap Şifreniz
        token: Giriş yaptıktan sonra alınan token
        hash: Giriş yaptıktan sonra alınan hash
        """
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def DeleteOrder(self, id, subAccount):
        try:
            end_point = "/api/DeleteOrder"
            payload = {
                'id': id,
                'subAccount': subAccount
            }
            resp = self.post(end_point, payload)
            try:
                data = resp.json()
                return data
            except:
                f = inspect.stack()[0][3]
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                
                print(resp.text)
                
        except Exception as e:
            f = inspect.stack()[0][3]
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
            
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request := algo.DeleteOrder(id,subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
            
Emir İyileştirme
Gönderilen ve açık olan emiri iyileştirir.

Http İsteği: POST /api/ModifyOrder
Http Headers Content-Type: application/json
APIKEY: Başvuru Sonucu Alınan API-KEY
APIKEY Authorization: Kullanıcı Girişi Oturum Alma işleminden dönen Hash değeri
Checker: Her istek için oluşturulan imzadır.
 

İstek Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
id	String	Emrin ID’ si
price	String	Düzeltilecek Fiyat
lot	String	Lot miktarı( Viop emri ise girilmelidir.)
viop	String	Emrin viop olduğunu belirtir(Viop ise true olmalıdır)
subAccount	String	Alt Hesap Numarası (Boş gönderilebilir. Boş gönderilirse Aktif Hesap Bilgilerini Getirir.)
Örnek Body

{
   "id": "001VEV",
   "price": "2.04",
   "lot": "0",
   "viop": false,
   "Subaccount": ""
}

 

Sonuç Parametreleri

Parametre Adı	Parametre Tipi	Açıklama
message	String	Emrin iletimi hakkında bilgi verir.
duration	String	-
Örnek Response

{
   "success": true,
   "message": "IYILESOK",
   "content": {
      "message": "IYILESOK",
      "duration": "-"
   }
}

import requests, hashlib, json, inspect, time, pandas as pd
apikey=""
username =""
password = ""
token = ""
hash = ""
id = ""
price = ""
lot = ""
viop = False
subAccount = ""
hostname = "www.algolab.com.tr"
api_hostname = f"https://{hostname}"
api_url = api_hostname + "/api"
class API():
    def __init__(self, api_key, username, password, token, hash):
        try:
            self.api_code = api_key.split("-")[1]
        except Exception:
            self.api_code = api_key
        self.api_key = "API-" + self.api_code
        self.username = username
        self.password = password
        self.api_hostname = api_hostname
        self.api_url = api_url
        self.headers = {"APIKEY": self.api_key}
        self.token = token
        self.hash = hash
        self.last_request = 0.0
        self.LOCK = False

    def ModifyOrder(self, id, price, lot, viop, subAccount):
        try:
            end_point = "/api/ModifyOrder"
            payload = {
                'id': id,
                'price': price,
                'lot': lot,
                'viop': viop,
                'subAccount': subAccount
            }
            resp = self.post(end_point, payload)
            try:
                data = resp.json()
                return data
            except:
                f = inspect.stack()[0][3]
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                
                print(resp.text)
                
        except Exception as e:
            f = inspect.stack()[0][3]
            print(f"{f}() fonsiyonunda hata oluştu: {e}")
            
            
    def post(self, endpoint, payload, login=False):
        url = self.api_url
        if not login:
            checker = self.make_checker(endpoint, payload)
            headers = {"APIKEY": self.api_key,
                       "Checker": checker,
                       "Authorization": self.hash
                       }
        else:
            headers = {"APIKEY": self.api_key}
        return self._request("POST", url, endpoint, payload=payload, headers=headers)
    
    def error_check(self, resp, f, silent=False):
        try:
            if resp.status_code == 200:
                return resp.json()
            if not silent:
                print(f"Error kodu: {resp.status_code}")

                print(resp.text)

            return False
        except Exception:
            if not silent:
                print(f"{f}() fonksiyonunda veri tipi hatasi. Veri, json formatindan farkli geldi:")
                print(resp.text)
            return False

    def make_checker(self, endpoint, payload):
        body = json.dumps(payload).replace(' ', '') if len(payload) > 0 else ""
        data = self.api_key + self.api_hostname + endpoint + body
        return hashlib.sha256(data.encode('utf-8')).hexdigest()

    def _request(self, method, url, endpoint, payload, headers):
        while self.LOCK:
            time.sleep(0.1)
        self.LOCK = True
        try:
            response = ""
            if method == "POST":
                t = time.time()
                diff = t - self.last_request
                wait_for = self.last_request > 0.0 and diff < 5.0 # son işlemden geçen süre 5 saniyeden küçükse bekle
                if wait_for:
                    time.sleep(5 - diff + 0.1)
                response = requests.post(url + endpoint, json=payload, headers=headers)
                self.last_request = time.time()
        finally:
            self.LOCK = False
        return response

if __name__ == "__main__":
    algo = API(apikey, username, password, token, hash)
    if request := algo.ModifyOrder(id,price,lot,viop,subAccount):
        try:
            print(request)
        except Exception as e:
            print(f"Hata oluştu: {e}")
            
Websocket Protokolü (WSS)
Canlı veya gecikmeli olarak veri görüntülemenizi sağlar.

 

HeartBeat Atma

Aşağıdaki şekilde Json String gönderilir. Websocket bağlantısının devam edebilmesi için düzenli aralıklarla HeartBeat gönderimi yapılması gerekmektedir.

Type = H olmalıdır 
Token değeri = Authorization
Örnek Request Body

{
  "Type": "H",
  "Token": "eyJhbGciOiJodHRwOi8vd3d3LnczLm9yZy8yMDAxLzA0L3htbGRzaWctbW9yZSNobWFjLXNoYTI1 NiIsInR5 cCI6IkpXVCJ9.eyJBdXRob3JpemF0aW9uIjoiQXV0aG9yaXplZCIsIkN1c3RvbWVyTm8iOiIxMzQ1MTcyMCIsIk5ld3NsZXR0 ZXIiOiJU cnVlIiwiSXNCbG9ja2VkIjoiRmFsc2UiLCJFbWFpbCI6IjEzNDUxNzIwIiwiVXNlcklkIjoiMTAxIiwiRGVuaXpiYW5rIjoiVHJ1ZSIsI m5iZiI6MTY1MzkyMDg2NiwiZXhwIjoxNjU0MDA3MjY2fQ.kzkSYQOnkA9Qn8qTiV_Fq8IvqXKsQ3m-QuMv6Kjqkdw"
}

 

T Paketi Abone Olma

Aşağıdaki şekilde Json String gönderilir.

Type = T olmalıdır
Token değeri = Authorization
Symbols = List()
Symbols yerine T paketinde gelmesi istenilen semboller liste şeklinde yazılır. Bütün Sembollerin gelmesi için "ALL" yazılması gerekmektedir.

Örnek Request Body

{
  "Type": "T",
  "Token": "eyJhbGciOiJodHRwOi8vd3d3LnczLm9yZy8yMDAxLzA0L3htbGRzaWctbW9yZSNobWFjLXNoYTI1 NiIsInR5 cCI6IkpXVCJ9.eyJBdXRob3JpemF0aW9uIjoiQXV0aG9yaXplZCIsIkN1c3RvbWVyTm8iOiIxMzQ1MTcyMCIsIk5ld3NsZXR0 ZXIiOiJU cnVlIiwiSXNCbG9ja2VkIjoiRmFsc2UiLCJFbWFpbCI6IjEzNDUxNzIwIiwiVXNlcklkIjoiMTAxIiwiRGVuaXpiYW5rIjoiVHJ1ZSIsI m5iZiI6MTY1MzkyMDg2NiwiZXhwIjoxNjU0MDA3MjY2fQ.kzkSYQOnkA9Qn8qTiV_Fq8IvqXKsQ3mQuMv6Kjqkdw",
  "Symbols": [
    "GARAN",
    "TSKB"
  ]
}

 

D Paketi Abone Olma

Aşağıdaki şekilde Json String gönderilir.

Type = D olmalıdır
Token değeri = Authorization
Symbols = List()
Symbols yerine D paketinde gelmesi istenilen semboller liste şeklinde yazılır. Bütün Sembollerin gelmesi için "ALL" yazılması gerekmektedir.

Örnek Request Body

{
  "Type": "D",
  "Token": "eyJhbGciOiJodHRwOi8vd3d3LnczLm9yZy8yMDAxLzA0L3htbGRzaWctbW9yZSNobWFjLXNoYTI1 NiIsInR5 cCI6IkpXVCJ9.eyJBdXRob3JpemF0aW9uIjoiQXV0aG9yaXplZCIsIkN1c3RvbWVyTm8iOiIxMzQ1MTcyMCIsIk5ld3NsZXR0 ZXIiOiJU cnVlIiwiSXNCbG9ja2VkIjoiRmFsc2UiLCJFbWFpbCI6IjEzNDUxNzIwIiwiVXNlcklkIjoiMTAxIiwiRGVuaXpiYW5rIjoiVHJ1ZSIsI m5iZiI6MTY1MzkyMDg2NiwiZXhwIjoxNjU0MDA3MjY2fQ.kzkSYQOnkA9Qn8qTiV_Fq8IvqXKsQ3mQuMv6Kjqkdw",
  "Symbols": [
    "GARAN"
  ]
}

 

Örnek "T" Tipindeki Response

{

   "Type":"T",

   "Content":{

      "Symbol":"PEHOL",

      "Market":"IMKBH",

      "Price":2.48,

      "Change":-0.27,

      "Ask":2.48,

      "Bid":0.0,

      "Date":"2024-09-25T14:46:35+03:00",

      "ChangePercentage":-9.82,

      "High":2.48,

      "Low":2.48,

      "TradeQuantity":100.0,

      "Direction":"B",

      "RefPrice":2.75,

      "BalancePrice":0.0,

      "BalanceAmount":0.0,

      "Buying":"Midas",

      "Selling":"Seker"

   }

}

 

Örnek "D" Tipindeki Response

{
   "Type":"D",
   "Content":{
      "Symbol":"ESEN",
      "Market":"IMKBH",
      "Direction":"B",
      "Row":0,
      "Quantity":8348,
      "Price":20.04,
      "OrderCount":128,
      "Date":"2024-09-25T14:51:18+03:00"
   }
}

Örnek "O" Tipindeki Response

{

   "Type":"O",

   "Content":{

      "Id":"fe52938525764d6c81aa305444e5937f",

      "Date":"2024-09-25T15:30:25.61",

      "Direction":1,

      "Symbol":"TSKB",

      "Lot":1.0,

      "PriceType":0,

      "Price":11.78,

      "Comment":"Referans Numaranız: FO7XEA;20240925FO7XEA - HISSEOK ",

      "Status":2,

      "Channel":"DenizYatirimHcp",

      "ExecutedLot":1.0,

      "ExecutedPrice":11.77

   }

}


import hashlib, json, subprocess, ssl, socket
import pandas as pd
from websocket import create_connection, WebSocketTimeoutException

class ConnectionTimedOutException(Exception):
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return self.message

class AlgoLabSocket():
    def __init__(self, api_key, hash):
        """
        :String api_key: API_KEY
        :String hash: LoginUser'dan dönen Hash kodu
        :String type: T: Tick Paketi (Fiyat), D: Depth Paketi (Derinlik), O: Emir Statüsü
        :Obj type: callback: Soketin veriyi göndereceği fonksiyon
        """
        self.connected = False
        self.df = pd.DataFrame(columns=["Date", "Hisse", "Yon", "Fiyat", "Lot", "Deger", "Usd", "Alici", "Satici"])
        self.ws = None
        self.api_key = api_key
        self.hash = hash
        self.data = self.api_key + api_hostname + "/ws"
        self.checker = hashlib.sha256(self.data.encode('utf-8')).hexdigest()
        self.headers = {
            "APIKEY": self.api_key,
            "Authorization": self.hash,
            "Checker": self.checker
        }

    def load_ciphers(self):
        output = subprocess.run(["openssl", "ciphers"], capture_output=True).stdout
        output_str = output.decode("utf-8")
        ciphers = output_str.strip().split("\n")
        return ciphers[0]

    def close(self):
        self.connected = False
        self.ws = None

    def connect(self):
        print("Socket bağlantisi kuruluyor...")
        context = ssl.create_default_context()
        context.set_ciphers("DEFAULT")
        try:
            sock = socket.create_connection((hostname, 443))
            ssock = context.wrap_socket(sock, server_hostname=hostname)
            self.ws = create_connection(socket_url, socket=ssock, header=self.headers)
            self.connected = True
        except Exception as e:
            self.close()
            print(f"Socket Hatasi: {e}")
            return False
        if self.connected:
            print("Socket bağlantisi başarili.")
        return self.connected

    def recv(self):
        try:
            data = self.ws.recv()
        except WebSocketTimeoutException:
            data = ""
        except Exception as e:
            print("Recv Error:", e)
            data = None
            self.close()
        return data
    def send(self, d):
        """
        :param d: Dict
        """
        try:
            data = {"token": self.hash}
            for s in d:
                data[s] = d[s]
            resp = self.ws.send(json.dumps(data))
        except Exception as e:
            print("Send Error:", e)
            resp = None
            self.close()
        return resp

if __name__ == "__main__":

    api_key = "" # Login için kullanılan API-KEY 
    hash = ""    # Login olduktan sonra alınan hash bilgisi

    data = {"Type": "T", "Symbols": ["ALL"]}
    # "T" ve "D"  tipinde verilere abone olunabilir. "O" tipindeki veriler otomatik olarak gelmektedir. 
    # "T" anlık gerçekleşen her işlem için , "D" ise 20 kademe alım satım kademelerindeki değişimler ile veri gönderimi sağlamaktadır.
    # ALL değeri tüm sembolleri getirir liste içerisinde sadece abone olmak istediğiniz sembolleri de yazabilirsiniz. Örneğin:  "Symbols": ["ASELS","THYAO","TUPRS"]
   
    # URLS
    hostname = "www.algolab.com.tr"
    api_hostname = f"https://{hostname}"
    api_url = api_hostname + "/api"
    socket_url = f"wss://{hostname}/api/ws"

    soket = AlgoLabSocket(api_key, hash) 
    soket.connect()
    
    if soket.connected:
        soket.send(data)

    while soket.connected:
        data = soket.recv()
        if data:
            try:
                msg = json.loads(data)
                print(msg)
            except:
                print("error")
                soket.close()
                break






