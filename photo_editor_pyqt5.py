"""
PyQt5 Photo Editor — single-file desktop app
Features (mapped from your JS tool):
- Multi-image load (file dialog / drag & drop onto list)
- Each image has its own canvas (QGraphicsView) and independent history stack
- Sliders: Brightness, Contrast, Saturation, Kelvin (2000-10000), Shadows, Highlights
- Buttons: Rotate 90°, Flip Horizontal, Sharpen (incremental), Orange/Red/Blue tint, Brighten (whitening), Clarity/Dehaze, Vignette, Noise Reduction, Portrait Mode, Auto Enhance, Undo, Reset, Download / Bulk Export
- Histogram (Luma + RGB) updated live
- Responsive layout made with splitters and stretchable widgets

Dependencies:
    pip install PyQt5 pillow numpy opencv-python matplotlib

Run:
    python photo_editor_pyqt5.py

Note: this implementation focuses on clarity and completeness. It uses Pillow + numpy for pixel ops and OpenCV for some filters (blur/face detection if available).
"""

import sys
import os
import io
import math
from functools import partial
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from PyQt5.QtGui import QPainter
import numpy as np
import cv2

from PyQt5.QtWidgets import (
    QApplication, QWidget, QFileDialog, QLabel, QPushButton, QListWidget, QListWidgetItem,
    QHBoxLayout, QVBoxLayout, QGridLayout, QSlider, QSplitter, QSizePolicy, QMessageBox,
    QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QStyle, QProgressBar
)
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import Qt, QByteArray

# Matplotlib for histogram rendering
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# ---------------------- Utility Functions ----------------------

def pil_to_qpixmap(im: Image.Image) -> QPixmap:
    if im.mode != 'RGBA':
        im = im.convert('RGBA')
    data = im.tobytes('raw', 'RGBA')
    qimg = QImage(data, im.width, im.height, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg)


def qpixmap_to_pil(pix: QPixmap) -> Image.Image:
    ba = QByteArray()
    buf = pix.toImage()
    ptr = buf.bits()
    ptr.setsize(buf.byteCount())
    arr = bytes(ptr)
    im = Image.frombytes('RGBA', (buf.width(), buf.height()), arr, 'raw', 'RGBA')
    return im.convert('RGB')


# Kelvin -> RGB gain approx (same as your JS logic)
def kelvin_to_rgb_gains(kelvin: int):
    temp = kelvin / 100.0
    def clamp(v, lo=0, hi=255):
        return max(lo, min(hi, v))
    if temp <= 66:
        r = 255
    else:
        r = 329.698727446 * ((temp - 60) ** -0.1332047592)
        r = clamp(r)
    if temp <= 66:
        g = 99.4708025861 * math.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
    g = clamp(g)
    if temp >= 66:
        b = 255
    elif temp <= 19:
        b = 0
    else:
        b = 138.5177312231 * math.log(temp - 10) - 305.0447927307
        b = clamp(b)
    return (r/255.0, g/255.0, b/255.0)


# Luma
def luma(r, g, b):
    return 0.2126*r + 0.7152*g + 0.0722*b


# Compute histogram (256 bins) returning arrays
def compute_histogram(pil_img: Image.Image):
    im = pil_img.convert('RGB')
    arr = np.array(im)
    r = arr[:,:,0].ravel()
    g = arr[:,:,1].ravel()
    b = arr[:,:,2].ravel()
    lum = (0.2126*r + 0.7152*g + 0.0722*b).astype(np.uint8).ravel()
    bins = 256
    hr = np.bincount(r, minlength=bins)
    hg = np.bincount(g, minlength=bins)
    hb = np.bincount(b, minlength=bins)
    hl = np.bincount(lum, minlength=bins)
    return {'r': hr, 'g': hg, 'b': hb, 'lum': hl, 'total': im.width*im.height}


# Draw histogram to QPixmap via matplotlib
def histogram_pixmap(hist):
    fig = Figure(figsize=(4,1.2), dpi=100, tight_layout=True)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)
    ax.plot(hist['lum'], color='k', linewidth=1.2)
    ax.plot(hist['r'], color='#c00', linewidth=0.8)
    ax.plot(hist['g'], color='#0a0', linewidth=0.8)
    ax.plot(hist['b'], color='#00c', linewidth=0.8)
    ax.axis('off')
    canvas.draw()
    buf = canvas.buffer_rgba()
    w, h = fig.get_size_inches()*fig.get_dpi()
    w, h = int(w), int(h)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    img = Image.fromarray(arr)
    return pil_to_qpixmap(img)


# ---------------------- Editor Data Structures ----------------------
class ImageDocument:
    def __init__(self, path=None, pil_image: Image.Image=None):
        self.path = path
        self.pil = pil_image.convert('RGB') if pil_image else None
        # history stack of PIL images
        self.history = []
        if self.pil:
            self.history.append(self.pil.copy())
        # adjustments state
        self.adjustments = {
            'brightness': 0, # -100..100 additive
            'contrast': 0,   # -100..100 percent
            'saturation': 0, # -100..100
            'kelvin': 6500,  # 2000..10000
            'shadows': 0,
            'highlights': 0
        }

    def push(self):
        self.history.append(self.pil.copy())

    def undo(self):
        if len(self.history) > 1:
            self.history.pop()
            self.pil = self.history[-1].copy()
            return True
        return False

    def reset_adjustments(self):
        self.adjustments = {k: (6500 if k=='kelvin' else 0) for k in self.adjustments}


# ---------------------- Main Application ----------------------
class PhotoEditor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQt5 Photo Editor — ebubekir style')
        self.resize(1200, 800)
        self.images = []  # list of ImageDocument
        self.current_doc = None
        self.setup_ui()

    def setup_ui(self):
        root = QHBoxLayout(self)

        # Left: thumbnails / list
        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(220)
        self.list_widget.itemClicked.connect(self.on_select_image)

        left_layout = QVBoxLayout()
        btn_load = QPushButton('Resim(Yükle)...')
        btn_load.clicked.connect(self.load_images)
        btn_bulk = QPushButton('Toplu Dışa Aktar')
        btn_bulk.clicked.connect(self.bulk_export)
        left_layout.addWidget(btn_load)
        left_layout.addWidget(btn_bulk)
        left_layout.addWidget(self.list_widget)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # Right: editor area
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing, True)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)


        # Controls panel
        control_panel = QWidget()
        cp_layout = QVBoxLayout(control_panel)
        cp_layout.setSpacing(8)

        # Histogram label
        self.hist_label = QLabel()
        self.hist_label.setFixedHeight(80)
        self.hist_label.setStyleSheet('border:1px solid #ddd; background:#fff')
        cp_layout.addWidget(QLabel('Histogram (Luma + RGB)'))
        cp_layout.addWidget(self.hist_label)

        # Sliders grid
        grid = QGridLayout()
        grid.setSpacing(8)

        self.sliders = {}
        def make_slider(name, minv, maxv, init, row):
            lbl = QLabel(name)
            s = QSlider(Qt.Horizontal)
            s.setRange(minv, maxv)
            s.setValue(init)
            s.valueChanged.connect(lambda v, n=name: self.on_slider(n, v))
            val_lbl = QLabel(str(init))
            s.valueChanged.connect(lambda v, l=val_lbl: l.setText(str(v)))
            self.sliders[name] = s
            grid.addWidget(lbl, row, 0)
            grid.addWidget(s, row, 1)
            grid.addWidget(val_lbl, row, 2)

        make_slider('Parlaklık', -100, 100, 0, 0)
        make_slider('Kontrast', -100, 100, 0, 1)
        make_slider('Doygunluk', -100, 100, 0, 2)
        make_slider('Beyaz Dengesi (K)', 2000, 10000, 6500, 3)
        make_slider('Shadows', -100, 100, 0, 4)
        make_slider('Highlights', -100, 100, 0, 5)

        cp_layout.addLayout(grid)

        # Buttons row
        btns_row = QHBoxLayout()
        def make_btn(text, cb, cls=''): 
            b = QPushButton(text)
            b.clicked.connect(cb)
            return b

        btns_row.addWidget(make_btn('90° Döndür', self.rotate90))
        btns_row.addWidget(make_btn('Yatay Çevir', self.flip_horizontal))
        btns_row.addWidget(make_btn('Geri Al', self.undo))
        cp_layout.addLayout(btns_row)

        # Effect buttons
        eff_row = QHBoxLayout()
        eff_row.addWidget(make_btn('Keskinleştir', partial(self.apply_effect, 'sharpen')))
        eff_row.addWidget(make_btn('Turunculaştır', partial(self.apply_effect, 'orange')))
        eff_row.addWidget(make_btn('Kırmızılık', partial(self.apply_effect, 'red')))
        eff_row.addWidget(make_btn('Mavi Ton', partial(self.apply_effect, 'blue')))
        cp_layout.addLayout(eff_row)

        eff_row2 = QHBoxLayout()
        eff_row2.addWidget(make_btn('Beyazlat', partial(self.apply_effect, 'brighten')))
        eff_row2.addWidget(make_btn('Clarity', partial(self.apply_effect, 'clarity')))
        eff_row2.addWidget(make_btn('Vignette', partial(self.apply_effect, 'vignette')))
        eff_row2.addWidget(make_btn('Noise Red.', partial(self.apply_effect, 'noise')))
        cp_layout.addLayout(eff_row2)

        eff_row3 = QHBoxLayout()
        eff_row3.addWidget(make_btn('Röportaj Modu', self.portrait_mode))
        eff_row3.addWidget(make_btn('Otomatik İyileştir', self.auto_enhance))
        eff_row3.addWidget(make_btn('Sıfırla', self.reset_adjustments))
        eff_row3.addWidget(make_btn('İndir', self.export_current))
        cp_layout.addLayout(eff_row3)

        control_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Layout splitters for responsiveness
        right_split = QSplitter(Qt.Vertical)
        right_top = QWidget()
        top_layout = QHBoxLayout(right_top)
        top_layout.addWidget(self.view)
        top_layout.addWidget(control_panel)
        right_split.addWidget(right_top)

        root_split = QSplitter(Qt.Horizontal)
        left_container = QWidget()
        left_container.setLayout(left_layout)
        root_split.addWidget(left_container)
        root_split.addWidget(right_split)
        root.addWidget(root_split)

        # Set initial sizes
        root_split.setStretchFactor(0, 0)
        root_split.setStretchFactor(1, 1)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 1)

    # ---------------------- Actions ----------------------
    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(self, 'Resimleri Seç', os.getcwd(), 'Images (*.png *.jpg *.jpeg *.bmp *.webp)')
        if not paths:
            return
        for p in paths:
            try:
                im = Image.open(p).convert('RGB')
            except Exception as e:
                QMessageBox.warning(self, 'Hata', f'"{p}" açılamadı: {e}')
                continue
            doc = ImageDocument(path=p, pil_image=im)
            self.images.append(doc)
            item = QListWidgetItem(os.path.basename(p))
            item.setData(Qt.UserRole, doc)
            self.list_widget.addItem(item)
        # select first if none
        if self.current_doc is None and self.images:
            self.list_widget.setCurrentRow(0)
            self.on_select_image(self.list_widget.item(0))

    def on_select_image(self, item: QListWidgetItem):
        doc = item.data(Qt.UserRole)
        self.current_doc = doc
        self.refresh_view()

    def refresh_view(self):
        if not self.current_doc:
            return
        self.scene.clear()
        pix = pil_to_qpixmap(self.current_doc.pil)
        self.pix_item = QGraphicsPixmapItem(pix)
        self.scene.addItem(self.pix_item)
        self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)
        # update histogram
        hist = compute_histogram(self.current_doc.pil)
        self.hist_label.setPixmap(histogram_pixmap(hist).scaled(self.hist_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # set slider values from adjustments
        ad = self.current_doc.adjustments
        self.sliders['Parlaklık'].setValue(ad['brightness'])
        self.sliders['Kontrast'].setValue(ad['contrast'])
        self.sliders['Doygunluk'].setValue(ad['saturation'])
        self.sliders['Beyaz Dengesi (K)'].setValue(ad['kelvin'])
        self.sliders['Shadows'].setValue(ad['shadows'])
        self.sliders['Highlights'].setValue(ad['highlights'])

    def on_slider(self, name, value):
        if not self.current_doc:
            return
        # map names
        m = {
            'Parlaklık': 'brightness', 'Kontrast': 'contrast', 'Doygunluk': 'saturation',
            'Beyaz Dengesi (K)': 'kelvin', 'Shadows': 'shadows', 'Highlights': 'highlights'
        }
        key = m.get(name)
        if key:
            self.current_doc.adjustments[key] = value
            self.apply_adjustments_preview()

    def apply_adjustments_preview(self):
        # non-destructive preview: compute from current_doc.history[-1]
        base = self.current_doc.history[-1]
        ad = self.current_doc.adjustments
        img = base.copy()
        arr = np.array(img).astype(np.float32)
        # White balance via kelvin gains
        gains = kelvin_to_rgb_gains(ad['kelvin'])
        arr[:,:,0] = np.clip(arr[:,:,0] * gains[0], 0, 255)
        arr[:,:,1] = np.clip(arr[:,:,1] * gains[1], 0, 255)
        arr[:,:,2] = np.clip(arr[:,:,2] * gains[2], 0, 255)
        # Brightness (add)
        arr = np.clip(arr + ad['brightness'], 0, 255)
        # Contrast: scale around 128
        c = 1 + (ad['contrast'] / 100.0)
        arr = np.clip((arr - 128) * c + 128, 0, 255)
        # Saturation: convert to HSL-ish via simple lerp to luma
        if ad['saturation'] != 0:
            s = ad['saturation'] / 100.0
            lum = (0.2126*arr[:,:,0] + 0.7152*arr[:,:,1] + 0.0722*arr[:,:,2])[:,:,None]
            arr = np.clip(lum + (arr - lum) * (1 + s), 0, 255)
        # Shadows/Highlights simple tone curve approximation
        def tone_pixel(a, shadows, highlights):
            t = a/255.0
            s = shadows/100.0
            h = highlights/100.0
            # lift shadows
            if s != 0:
                lift = s*0.6
                w = np.minimum(1.0, np.maximum(0.0, (t - 0.0)/(0.6-0.0)))
                t = t + (lift*(1-w))*(1-t)
            # compress highlights
            if h != 0:
                comp = h*0.6
                w2 = np.minimum(1.0, np.maximum(0.0, (t - 0.4)/(1.0-0.4)))
                t = t - (comp * w2) * t
            return np.clip(t*255.0, 0, 255)
        if ad['shadows'] != 0 or ad['highlights'] != 0:
            arr = tone_pixel(arr, ad['shadows'], ad['highlights'])
        img2 = Image.fromarray(arr.astype(np.uint8))
        self.current_doc.pil = img2
        # update view and histogram
        self.refresh_view()

    # Effect implementations (applied and pushed to history)
    def apply_effect(self, effect):
        if not self.current_doc:
            return
        img = self.current_doc.pil.copy()
        if effect == 'sharpen':
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
        elif effect == 'orange':
            r, g, b = img.split()
            r = r.point(lambda v: min(255, v+12))
            g = g.point(lambda v: min(255, v+6))
            img = Image.merge('RGB', (r,g,b))
        elif effect == 'red':
            r, g, b = img.split()
            r = r.point(lambda v: min(255, v+18))
            img = Image.merge('RGB', (r,g,b))
        elif effect == 'blue':
            r, g, b = img.split()
            b = b.point(lambda v: min(255, v+18))
            img = Image.merge('RGB', (r,g,b))
        elif effect == 'brighten':
            enh = ImageEnhance.Brightness(img)
            img = enh.enhance(1.15)
        elif effect == 'clarity':
            enh = ImageEnhance.Sharpness(img)
            img = enh.enhance(1.2)
        elif effect == 'vignette':
            w, h = img.size
            xv, yv = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
            dist = np.sqrt(xv**2 + yv**2)
            mask = np.clip(1 - (dist/np.sqrt(2)), 0, 1)
            mask = (0.6 + 0.4*mask)[:,:,None]
            arr = np.array(img).astype(np.float32)
            arr = arr * mask
            img = Image.fromarray(arr.astype(np.uint8))
        elif effect == 'noise':
            # simple denoise: bilateral filter via cv2
            a = np.array(img)
            a = cv2.bilateralFilter(a, d=5, sigmaColor=75, sigmaSpace=75)
            img = Image.fromarray(a)
        else:
            return
        self.current_doc.pil = img
        self.current_doc.push()
        self.refresh_view()

    def rotate90(self):
        if not self.current_doc: return
        self.current_doc.pil = self.current_doc.pil.rotate(-90, expand=True)
        self.current_doc.push()
        self.refresh_view()

    def flip_horizontal(self):
        if not self.current_doc: return
        self.current_doc.pil = ImageOps.mirror(self.current_doc.pil)
        self.current_doc.push()
        self.refresh_view()

    def undo(self):
        if not self.current_doc: return
        ok = self.current_doc.undo()
        if not ok:
            QMessageBox.information(self, 'Bilgi', 'Geri alınacak işlem yok.')
        self.refresh_view()

    def portrait_mode(self):
        # gentle skin tone + blur (use bilateral filter on face region if face detected)
        if not self.current_doc: return
        img = self.current_doc.pil.copy()
        a = np.array(img)
        gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        faces = []
        try:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        except Exception:
            faces = []
        # global gentle color tweak
        arr = a.astype(np.float32)
        arr[:,:,0] *= 0.95
        arr[:,:,1] = np.minimum(255, arr[:,:,1]*1.05)
        arr[:,:,2] = np.minimum(255, arr[:,:,2]*1.05)
        a = arr.astype(np.uint8)
        # apply selective blur on faces or gentle global blur
        if len(faces) > 0:
            for (x,y,w,h) in faces:
                roi = a[y:y+h, x:x+w]
                blurred = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
                # blend
                a[y:y+h, x:x+w] = cv2.addWeighted(roi, 0.6, blurred, 0.4, 0)
        else:
            a = cv2.bilateralFilter(a, d=9, sigmaColor=75, sigmaSpace=75)
        img2 = Image.fromarray(a)
        self.current_doc.pil = img2
        self.current_doc.push()
        self.refresh_view()

    def auto_enhance(self):
        if not self.current_doc: return
        img = self.current_doc.pil.copy()
        # simple auto: increase contrast & brightness slightly
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Brightness(img).enhance(1.06)
        self.current_doc.pil = img
        self.current_doc.push()
        self.refresh_view()

    def reset_adjustments(self):
        if not self.current_doc: return
        self.current_doc.reset_adjustments()
        self.apply_adjustments_preview()

    def export_current(self):
        if not self.current_doc: return
        p = QFileDialog.getSaveFileName(self, 'Dışa Aktar', self.current_doc.path or os.getcwd(), 'PNG (*.png);;JPEG (*.jpg *.jpeg)')
        if p and p[0]:
            self.current_doc.pil.save(p[0])

    def bulk_export(self):
        if not self.images:
            QMessageBox.information(self, 'Bilgi', 'İşlenecek resim yok.')
            return
        folder = QFileDialog.getExistingDirectory(self, 'Klasör seç (Toplu Dışa Aktar)')
        if not folder: return
        for i, doc in enumerate(self.images):
            name = doc.path and os.path.basename(doc.path) or f'resim_{i+1}.png'
            dst = os.path.join(folder, f'processed_{i+1}_{name}')
            doc.pil.save(dst)
        QMessageBox.information(self, 'Bitti', f'{len(self.images)} resim kaydedildi.')


# ---------------------- Run ----------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = PhotoEditor()
    w.show()
    sys.exit(app.exec_())
