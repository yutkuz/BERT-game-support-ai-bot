# Support Bot v1

Support Bot v1, Turkce oyun destek mesajlarini 24 destek kategorisinden birine siniflandiran BERT tabanli bir destek otomasyon projesidir. Istege bagli Gemma rewrite katmani, kullanicidan gelen bozuk veya kisa mesajlari anlamini degistirmeden daha temiz bir metne donusturerek kategori tahminine yardimci olur.

## Kisa Ozet

- Ana model: `ytu-ce-cosmos/turkish-base-bert-uncased`
- Model tipi: BERT sequence classification
- Tahmin edilen kategori sayisi: 24
- Egitim verisi: `data_v6.xlsx`
- Egitim satiri: 6928
- Orijinal mesaj kolonu: `kullanici_mesaji`
- Rewrite kolonu: `rewrite`
- Label kolonu: `kategori`
- Macro F1: `0.8230`
- Accuracy: `0.8385`
- Model dosyasi: `artifacts/tr_bert_uncased_epoch9/bert_model/model.safetensors`
- Model agirlik boyutu: yaklasik `422 MB`
- Tokenizer/config dahil inference klasoru: yaklasik `423 MB`
- Rewrite modeli: `gemma4:e4b` via Ollama

## Kullanilan Diller ve Teknolojiler

| Alan | Teknoloji |
| --- | --- |
| Ana programlama dili | Python 3.10+ |
| Makine ogrenmesi | PyTorch, Hugging Face Transformers |
| NLP modeli | `ytu-ce-cosmos/turkish-base-bert-uncased` |
| Egitim altyapisi | Transformers `Trainer`, ozel `WeightedTrainer`, scikit-learn metrikleri |
| Veri isleme | pandas, NumPy, openpyxl |
| Frontend | React 18, JSX, HTML, CSS |
| Web sunucusu | Python `http.server.ThreadingHTTPServer` |
| API formati | JSON tabanli HTTP endpointleri |
| Rewrite/LLM | Ollama + `gemma4:e4b` |
| Dosya formatlari | XLSX, CSV, JSON, safetensors |

Projede Node.js build adimi yoktur. React arayuzu CDN uzerinden yuklenir ve `web/` klasorundeki statik dosyalar Python sunucusu tarafindan servis edilir.

## Amac

Bu proje destek ekibine gelen kullanici mesajlarini otomatik siniflandirmak icin hazirlanmistir. Her mesaj icin:

- tahmin edilen kategori,
- guven skoru,
- ilk 3 kategori adayi,
- otomatik cevap taslagi,
- gerekirse insan kontrolu onerisi

uretilir.

Web arayuzu su islemleri destekler:

- Manuel mesaj testi
- Satir satir toplu test
- CSV/XLSX dosyasindan rastgele mesaj cekme
- Rewrite acik/kapali test etme
- Yanlis kategori veya cevap icin feedback kaydetme
- Serbest kategori girisi ile yeni kategori feedback'i toplama
- Toplu sonuclari CSV/XLSX olarak disa aktarma

## Kategoriler

Model su 24 kategoriyi tahmin eder:

```text
arkadas_adina_ban_itirazi
ban_ceza_itirazi
bilgi_sorusu
davet_odulu
giris_hesap_erisim
hediye_bonus_talebi
hesap_islemleri
hesap_silme_kapatma
hile_itirazi
iade_tazmin_talebi
kullanici_adi_degistirme
odeme_cip_yukleme
olumlu_geri_bildirim
oyun_adalet_puan
oyun_nasil_oynanir_yardim
oyuncu_sikayet_raporlama
ozel_direkt_mesaj
ozellik_degisim_talebi
profil_fotografi_degistirme
reklam_odulu
ses_sorunu
sosyal_hesap_giris_gecis
teknik_uygulama_sorunu
toplu_sikayet
```

## Rewrite Neden Var?

Destek mesajlari genellikle kisa, yazim hatali veya kullanici agziyla duzensiz gelir. `m`, `mr`, `cip`, `ban`, `msj`, `tlf` gibi domain kisaltmalari modelin anlami yakalamasini zorlastirabilir.

Rewrite katmani:

- yazim hatalarini ve devrik ifadeleri duzeltir,
- oyun/support domainindeki kisaltmalari baglama uygun yorumlar,
- kullanici niyetini daha acik hale getirir,
- BERT siniflandirmasina daha temiz ikinci bir sinyal verir.

Rewrite modeli kategori tahmini yapmaz ve kullaniciya cevap uretmez; sadece mesaj metnini normalize eder.

## Verisetinde Neden Rewrite Kolonu Var?

`data_v6.xlsx` icinde hem orijinal kullanici mesaji hem de rewrite edilmis metin tutulur:

- `kullanici_mesaji`: kullanicinin orijinal mesaji
- `rewrite`: ayni mesajin normalize edilmis hali
- `kategori`: dogru sinif etiketi

Orijinal metin modelin gercek kullanici dilini ogrenmesini saglar. Rewrite metni ise ayni niyetin daha temiz bicimini gostererek ozellikle yazim hatasi, kisaltma ve dusuk kaliteli mesajlarda modeli guclendirir.

Tahmin sirasinda rewrite aciksa model hem orijinal mesaji hem rewrite edilmis mesaji degerlendirir. Rewrite kapaliysa yalnizca orijinal mesaj uzerinden tahmin yapilir.

## Proje Yapisi

```text
.
|-- artifacts/
|   `-- tr_bert_uncased_epoch9/
|       |-- metadata.json
|       |-- label_mapping.json
|       `-- bert_model/
|           |-- config.json
|           |-- tokenizer.json
|           |-- tokenizer_config.json
|           `-- model.safetensors
|-- docs/
|-- scripts/
|-- src/support_bot/
|-- supportbot_web/
|-- web/
|-- data_v6.xlsx
|-- gemma_rewrite_prompt.md
|-- server.py
|-- start.bat
`-- requirements.txt
```

## Kurulum

Python 3.10 veya ustu onerilir.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Rewrite kullanilacaksa Ollama ve Gemma modeli gerekir:

```bash
ollama pull gemma4:e4b
```

## Model Dosyasini Indirme

Model agirlik dosyasi Google Drive uzerinden indirilir:

```text
https://drive.google.com/file/d/1LYxtNAmDWiaADGymPMLfIZElo4_GPhxE/view?usp=sharing
```

Indirdikten sonra dosyayi su konuma koyun:

```text
artifacts/tr_bert_uncased_epoch9/bert_model/model.safetensors
```

Dosya adi tam olarak `model.safetensors` olmalidir.

## Calistirma

Windows'ta:

```bat
start.bat
```

Manuel:

```bash
python server.py
```

Varsayilan adres:

```text
http://127.0.0.1:8009
```

Ortam degiskenleri:

```bash
SUPPORT_BOT_PORT=8009
SUPPORT_BOT_ARTIFACT_DIR=artifacts/tr_bert_uncased_epoch9
GEMMA_REWRITE_ENABLED=1
GEMMA_REWRITE_MODEL=gemma4:e4b
GEMMA_REWRITE_URL=http://127.0.0.1:11434
```

## Egitim

Tam egitim ornegi:

```bash
python scripts/train_model.py train ^
  --dataset-path data_v6.xlsx ^
  --artifact-dir artifacts/tr_bert_uncased_epoch9 ^
  --bert-model-name ytu-ce-cosmos/turkish-base-bert-uncased ^
  --epochs 9 ^
  --batch-size 8 ^
  --max-length 104 ^
  --learning-rate 1.33e-05 ^
  --weight-decay 0.00110172 ^
  --warmup-ratio 0.0158 ^
  --label-smoothing 0.1319 ^
  --rewrite-boost 1.0 ^
  --test-size 0.15 ^
  --random-state 42 ^
  --device cuda ^
  --fp16 ^
  --no-auto-resume
```

CPU ile hizli test:

```bash
python scripts/train_model.py train --dataset-path data_v6.xlsx --artifact-dir artifacts/test_cpu --device cpu --sample-per-class 5 --epochs 1 --no-save-checkpoints
```

Daha fazla detay icin [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md) dosyasina bakin.

