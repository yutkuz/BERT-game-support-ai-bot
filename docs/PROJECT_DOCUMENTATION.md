# Support Bot v1 Proje Dokumantasyonu

## 1. Model Nedir?

Projede kullanilan ana model `ytu-ce-cosmos/turkish-base-bert-uncased` tabanli bir BERT siniflandirma modelidir. Hugging Face `AutoModelForSequenceClassification` ile 24 sinifli kategori tahmini icin fine-tune edilmistir.

Egitim ciktisi `metadata.json`, `label_mapping.json` ve `bert_model/` klasorunde saklanir.

## 2. Modelin Amaci

Amac, online oyun/support alanindaki kullanici mesajlarini dogru destek kategorisine yonlendirmektir. Model tek bir kullanici mesaji alir ve en olasi kategoriyi, guven skorunu, ilk 3 kategori adayini, gerekirse insan kontrolu onerisini ve kategoriye bagli otomatik cevap taslagini uretir.

## 3. Kullanilan Diller ve Teknolojiler

### Programlama Dilleri

| Dil | Kullanim |
| --- | --- |
| Python | Egitim, inference, web server, API handler, veri isleme |
| JavaScript / JSX | React tabanli web arayuzu |
| HTML | Web arayuzunun giris sayfasi |
| CSS | Arayuz tasarimi ve responsive layout |
| Batch | Windows icin `start.bat` baslatma scripti |

### Backend ve Uygulama Katmani

| Bilesen | Aciklama |
| --- | --- |
| `server.py` | Uygulamayi `ThreadingHTTPServer` ile baslatir |
| `supportbot_web/http_handler.py` | Statik dosyalari ve JSON API endpointlerini yonetir |
| `supportbot_web/predictor_service.py` | BERT modelini lazy-load eder ve tahmin uretir |
| `supportbot_web/rewriter_service.py` | Ollama uzerinden Gemma rewrite istegi gonderir |
| `supportbot_web/feedback_dataset.py` | Duzeltme kayitlarini Excel feedback dosyasina yazar |

### Makine Ogrenmesi ve NLP

| Teknoloji | Kullanim |
| --- | --- |
| PyTorch | Model egitimi ve inference calisma zamani |
| Hugging Face Transformers | BERT tokenizer, sequence classification modeli ve Trainer altyapisi |
| `ytu-ce-cosmos/turkish-base-bert-uncased` | Turkce BERT base model |
| scikit-learn | Train/validation split, accuracy, macro F1, classification report |
| NumPy | Olasilik hesaplari ve sinif skorlarinin islenmesi |
| safetensors | Egitilmis model agirligini saklama |

### Veri ve Dosya Isleme

| Teknoloji | Kullanim |
| --- | --- |
| pandas | XLSX/CSV okuma, veri hazirlama, feedback export |
| openpyxl | Excel dosyalarini okuma/yazma |
| JSON | Metadata, label mapping ve API payload formatlari |
| CSV/XLSX | Egitim verisi, feedback verisi ve toplu test ciktilari |

### Frontend

| Teknoloji | Kullanim |
| --- | --- |
| React 18 | Manuel/toplu test arayuzu ve sonuc tablosu |
| Babel standalone | JSX dosyasini tarayicida calistirma |
| SheetJS/XLSX | Toplu sonuclari XLSX olarak export etme |
| CSS Grid/Flexbox | Giris ve sonuc panellerinin responsive duzeni |

### Harici Servis

| Servis | Kullanim |
| --- | --- |
| Ollama | Lokal LLM runtime |
| `gemma4:e4b` | Mesaj rewrite/normalizasyon modeli |

Bu mimari sade tutulmustur: frontend build adimi gerektirmez, tek Python server hem API hem statik arayuz servis eder.

## 4. Egitim Veriseti

Egitim dosyasi: `data_v6.xlsx`

| Alan | Deger |
| --- | --- |
| Orijinal mesaj kolonu | `kullanici_mesaji` |
| Rewrite kolonu | `rewrite` |
| Label kolonu | `kategori` |
| Satir sayisi | `6928` |
| Kategori sayisi | `24` |
| Validation orani | `0.15` |
| Random state | `42` |

Kod egitimden once veriyi normalize eder, bos mesajlari atar, label degerlerini normalize edip id'ye cevirir ve stratified train/validation split uygular.

## 5. Egitim Kodlamasi

Ana egitim dosyasi:

```text
src/support_bot/training.py
```

Komut sarmalayicisi:

```text
scripts/train_model.py
```

Egitim akisi:

1. Dataset `pandas` ile okunur.
2. Mesaj, rewrite ve label kolonlari bulunur.
3. Label'lar normalize edilir.
4. `train_test_split(..., stratify=label_id)` ile validation ayrilir.
5. Orijinal metin ve rewrite metni egitim ornegine donusturulur.
6. BERT tokenizer ile metinler `max_length` degerine gore encode edilir.
7. `WeightedTrainer`, label smoothing ve sample weight destekli loss hesaplar.
8. Model epoch/step bazli egitilir.
9. Validation uzerinde accuracy ve macro F1 hesaplanir.
10. Model, tokenizer, metadata ve tahmin ciktilari artifact klasorune yazilir.

Ana hiperparametreler:

| Parametre | Deger |
| --- | --- |
| Epoch | `9` |
| Batch size | `8` |
| Max length | `104` |
| Learning rate | `1.33e-05` |
| Weight decay | `0.00110172` |
| Warmup ratio | `0.0158` |
| Label smoothing | `0.1319` |
| Rewrite boost | `1.0` |
| Device | `cuda` |
| Mixed precision | `fp16` |

## 6. Metrikler

| Metrik | Deger |
| --- | --- |
| Accuracy | `0.8384615385` |
| Macro F1 | `0.8230398204` |
| Weighted F1 | `0.8371003060` |

Macro F1 kategori dagilimi dengeli olmadiginda daha anlamli bir metriktir, cunku her kategoriyi esit agirlikta degerlendirir.

## 7. Model Boyutu

| Parca | Boyut |
| --- | --- |
| `model.safetensors` | yaklasik `422 MB` |
| `bert_model/` tokenizer/config dahil | yaklasik `423 MB` |
| tum artifact klasoru checkpointlerle | yaklasik `4.13 GB` |

Model agirlik dosyasi Google Drive uzerinden indirilir ve `artifacts/tr_bert_uncased_epoch9/bert_model/` klasorune koyulur.

## 8. Rewrite Mimarisi

Rewrite servisi:

```text
supportbot_web/rewriter_service.py
```

Prompt:

```text
gemma_rewrite_prompt.md
```

Varsayilan model:

```text
gemma4:e4b
```

Rewrite, Ollama `/api/chat` endpoint'i uzerinden calisir. Mesaji daha temiz ve anlami korunmus bir metne donusturur. Rewrite sonucunda bos donus yapilirsa sistem orijinal mesajla tahmine devam eder.

## 9. Inference Akisi

1. Kullanici web arayuzunden mesaj girer.
2. Rewrite aciksa mesaj Gemma'ya gonderilir.
3. BERT once orijinal mesaj icin olasilik uretir.
4. Rewrite varsa BERT rewrite metni icin de olasilik uretir.
5. Olasiliklar `rewrite_boost` ile birlestirilir.
6. En yuksek skorlu kategori tahmin olarak doner.
7. Guven dusukse insan kontrolu onerilir.

## 10. Web/API Yapisi

Giris noktasi:

```text
server.py
```

HTTP handler:

```text
supportbot_web/http_handler.py
```

Frontend:

```text
web/index.html
web/app.jsx
web/styles.css
```

API endpointleri:

| Endpoint | Aciklama |
| --- | --- |
| `GET /api/status` | Model, label, rewrite ve feedback durumu |
| `POST /api/predict` | Tekli/toplu kategori tahmini |
| `POST /api/cancel` | Devam eden toplu tahmini iptal |
| `POST /api/sample-upload` | CSV/XLSX dosyasindan rastgele mesaj cek |
| `POST /api/corrections` | Yanlis tahmin duzeltmesini kaydet |

## 11. Paketler ve Sistem Gereksinimleri

Python paketleri `requirements.txt` icindedir:

```text
pandas
openpyxl
scikit-learn
numpy
torch
transformers
accelerate
fastapi
uvicorn[standard]
pydantic
python-multipart
```

Calistirma icin minimum:

- Python 3.10+
- 8 GB RAM onerilir
- CPU ile inference mumkundur, GPU daha hizlidir
- Rewrite icin Ollama + `gemma4:e4b`

Egitim icin onerilen:

- NVIDIA GPU
- 8 GB+ VRAM
- 16 GB+ sistem RAM
- CUDA destekli PyTorch
- Checkpointlerle birlikte 8-10 GB bos disk alani

