from flask import Flask, request, render_template, send_from_directory
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor
import torch
from PIL import Image
import os

# Flask uygulaması
app = Flask(__name__)

# Yüklenen dosyaları saklamak için "uploads" klasörünü oluştur
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model ve işlemci yükleme
model_adi = "google/matcha-chartqa"
model = Pix2StructForConditionalGeneration.from_pretrained(model_adi)
islemci = Pix2StructProcessor.from_pretrained(model_adi)
cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(cihaz)

# Çıktıyı filtreleme fonksiyonu
def ciktiyi_filtrele(cikti):
    return cikti.replace("<0x0A>", "")

# Grafik Soru-Cevap fonksiyonu
def grafik_soru_cevap(resim, soru):
    girdiler = islemci(images=resim, text=soru, return_tensors="pt").to(cihaz)
    tahminler = model.generate(**girdiler, max_new_tokens=512)
    return ciktiyi_filtrele(islemci.decode(tahminler[0], skip_special_tokens=True))

# Ana sayfa rotası
@app.route('/', methods=['GET', 'POST'])
def ana_sayfa():
    cevap = None
    yuklenen_resim = None
    if request.method == 'POST':
        # Kullanıcıdan dosya al
        if 'image' not in request.files:
            return "Dosya kısmı yok"
        dosya = request.files['image']
        if dosya.filename == '':
            return "Seçili dosya yok"

        # Dosyayı kaydet
        resim_yolu = os.path.join(app.config['UPLOAD_FOLDER'], dosya.filename)
        dosya.save(resim_yolu)

        # Resmi aç ve soruyu işle
        soru = request.form['question']
        resim = Image.open(resim_yolu)
        cevap = grafik_soru_cevap(resim, soru)

        # Yüklenen resmi sayfada göstermek için resim yolunu sakla
        yuklenen_resim = dosya.filename

    return render_template('index.html', answer=cevap, uploaded_image=yuklenen_resim)

# Yüklenen dosyaları statik olarak sunmak için rota
@app.route('/uploads/<filename>')
def yuklenen_dosya(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
