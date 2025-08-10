# 🖼️ PyQt5 Photo Editor — ebs Edition

Bu proje, **PyQt5** kullanılarak geliştirilmiş, çok özellikli bir masaüstü fotoğraf düzenleme uygulamasıdır.  
JavaScript tabanlı web fotoğraf editörünün tüm özellikleri **Python** sürümüne taşınmıştır.  
**Pillow**, **OpenCV** ve **NumPy** ile güçlü filtreler ve düzenleme araçları sunar.
Web Sürümüne <a href="https://ebubekirbastama.com.tr/ebsrenkanalizindirmeotomasyonu/index.html">buradan</a> ulaşabilirsiniz 
---

## ✨ Özellikler

- **Çoklu Resim Yükleme**
  - Dosya seç veya sürükle & bırak ile ekle
  - Her resim için bağımsız düzenleme ve geri alma geçmişi
- **Ayarlar (Sliders)**
  - Parlaklık
  - Kontrast
  - Doygunluk
  - Beyaz Dengesi (Kelvin)
  - Gölgeler
  - Parlak Alanlar
- **Efekt Butonları**
  - 90° Döndür
  - Yatay Çevir
  - Keskinleştir
  - Turuncu / Kırmızı / Mavi ton
  - Beyazlatma
  - Clarity / Dehaze
  - Vignette
  - Noise Reduction (Gürültü Azaltma)
  - Portre Modu (Yüz algılama + Blur)
  - Otomatik İyileştir
  - Geri Al / Sıfırla
  - Tekli veya Toplu Dışa Aktarım
- **Histogram**
  - Luma + RGB grafiği (canlı güncellenir)
- **Responsive Arayüz**
  - Splitter ile ayarlanabilir panel genişlikleri

---

## 📦 Bağımlılıklar

Aşağıdaki kütüphaneleri yükleyin:

```bash
pip install PyQt5 pillow numpy opencv-python matplotlib
```

---

## 🚀 Çalıştırma

```bash
python photo_editor_pyqt5.py
```

---

## 📂 Proje Yapısı

```
pyqt5-photo-editor-ebs/
│
├── photo_editor_pyqt5.py   # Ana uygulama dosyası
├── README.md               # Bu dosya
└── örnekler/               # Örnek resimler (isteğe bağlı)
```

---

## 📸 Ekran Görüntüsü



---

## 📜 Lisans

MIT Lisansı altında yayınlanmıştır.  
Telif hakkı © 2025 ebubekir bastama

---

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Yeni bir branch oluşturun (`git checkout -b feature-yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'i push edin (`git push origin feature-yeni-ozellik`)
5. Pull Request oluşturun

---

💡 **Not:** Bu uygulama, profesyonel fotoğrafçılara yönelik **gelişmiş düzenleme** seçenekleri sunar ve tamamen **tek dosya** üzerinde çalışır.
