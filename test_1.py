import sys
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QDoubleSpinBox, QSpinBox, QPushButton, QLabel,
    QGroupBox, QCheckBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def dbfs(x, full_scale=1.0, eps=1e-20):
    return 20 * np.log10(np.maximum(np.abs(x) / full_scale, eps))


def adc_model(x, bits=10, vfs=1.0, nonlinearity=(0.0, 0.0), dither_rms=0.0):
    """
    Basit ADC modeli:
      - doğrusal olmayanlık: y = x + a2*x^2 + a3*x^3
      - kırpma: [-vfs, +vfs]
      - kuantalama: uniform
      - dither_rms: (Volt) RMS gaussian noise, quantization öncesi eklenir
    """
    a2, a3 = nonlinearity

    if dither_rms > 0:
        x = x + dither_rms * np.random.randn(*x.shape)

    y = x + a2 * (x**2) + a3 * (x**3)
    y = np.clip(y, -vfs, vfs)

    levels = 2**bits
    lsb = (2 * vfs) / levels
    yq = np.round((y + vfs) / lsb) * lsb - vfs

    return yq, lsb


def fft_spectrum(x, fs, window="hann"):
    n = len(x)
    if window == "hann":
        w = np.hanning(n)
    elif window == "blackman":
        w = np.blackman(n)
    else:
        w = np.ones(n)

    xw = x * w
    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, d=1/fs)

    cg = np.sum(w) / n  # coherent gain
    mag = np.abs(X) / (n * cg)

    # tek taraflı genlik düzeltmesi
    if n % 2 == 0:
        mag[1:-1] *= 2
    else:
        mag[1:] *= 2

    return f, mag


def pick_peak_near(f, mag, f0, tol_hz=3000):
    idx = np.where((f > f0 - tol_hz) & (f < f0 + tol_hz))[0]
    if len(idx) == 0:
        return None, None
    k = idx[np.argmax(mag[idx])]
    return f[k], mag[k]


class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure()
        self.ax = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt6 ADC Örnekleme & Harmonik Benzetimi")

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # Sol: Kontroller
        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        params_box = QGroupBox("Parametreler")
        form = QFormLayout(params_box)

        self.fs = QDoubleSpinBox()
        self.fs.setRange(1e3, 200e6)
        self.fs.setDecimals(0)
        self.fs.setSingleStep(1000)
        self.fs.setValue(2_000_000)

        self.n = QSpinBox()
        self.n.setRange(2**10, 2**20)
        self.n.setSingleStep(1024)
        self.n.setValue(262_144)

        self.fin = QDoubleSpinBox()
        self.fin.setRange(1.0, 100e6)
        self.fin.setDecimals(0)
        self.fin.setSingleStep(1000)
        self.fin.setValue(123_456)

        self.amp = QDoubleSpinBox()
        self.amp.setRange(0.0, 1.2)
        self.amp.setDecimals(4)
        self.amp.setSingleStep(0.01)
        self.amp.setValue(0.9)

        self.vfs = QDoubleSpinBox()
        self.vfs.setRange(0.1, 10.0)
        self.vfs.setDecimals(3)
        self.vfs.setSingleStep(0.1)
        self.vfs.setValue(1.0)

        self.bits = QSpinBox()
        self.bits.setRange(2, 18)
        self.bits.setValue(10)

        self.a2 = QDoubleSpinBox()
        self.a2.setRange(-0.2, 0.2)
        self.a2.setDecimals(6)
        self.a2.setSingleStep(0.001)
        self.a2.setValue(0.02)

        self.a3 = QDoubleSpinBox()
        self.a3.setRange(-0.2, 0.2)
        self.a3.setDecimals(6)
        self.a3.setSingleStep(0.001)
        self.a3.setValue(0.01)

        self.dither = QDoubleSpinBox()
        self.dither.setRange(0.0, 0.05)
        self.dither.setDecimals(6)
        self.dither.setSingleStep(0.0005)
        self.dither.setValue(0.0)

        self.xmax_khz = QDoubleSpinBox()
        self.xmax_khz.setRange(1.0, 50000.0)
        self.xmax_khz.setDecimals(1)
        self.xmax_khz.setSingleStep(50.0)
        self.xmax_khz.setValue(600.0)

        self.show_ideal = QCheckBox("İdeal giriş spektrumu da çiz")
        self.show_ideal.setChecked(True)

        self.window_blackman = QCheckBox("Blackman pencere kullan")
        self.window_blackman.setChecked(False)

        form.addRow("fs (Hz)", self.fs)
        form.addRow("N (FFT noktası)", self.n)
        form.addRow("fin (Hz)", self.fin)
        form.addRow("amp (0..1)", self.amp)
        form.addRow("Vfs (tepe)", self.vfs)
        form.addRow("bits", self.bits)
        form.addRow("a2 (x² kats.)", self.a2)
        form.addRow("a3 (x³ kats.)", self.a3)
        form.addRow("dither RMS (V)", self.dither)
        form.addRow("x-ekseni max (kHz)", self.xmax_khz)

        controls_layout.addWidget(params_box)
        controls_layout.addWidget(self.show_ideal)
        controls_layout.addWidget(self.window_blackman)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Çalıştır / Güncelle")
        self.btn_run.clicked.connect(self.update_plot)
        btn_row.addWidget(self.btn_run)

        self.btn_quant_only = QPushButton("Sadece Kuantalama")
        self.btn_quant_only.clicked.connect(self.set_quant_only)
        btn_row.addWidget(self.btn_quant_only)

        self.btn_nl_on = QPushButton("Nonlinearity Aç")
        self.btn_nl_on.clicked.connect(self.set_nl_default)
        btn_row.addWidget(self.btn_nl_on)

        controls_layout.addLayout(btn_row)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        controls_layout.addWidget(self.info)

        # Sağ: Grafik
        self.canvas = MplCanvas()

        layout.addWidget(controls, 0)
        layout.addWidget(self.canvas, 1)

        self.update_plot()

    def set_quant_only(self):
        self.a2.setValue(0.0)
        self.a3.setValue(0.0)
        self.update_plot()

    def set_nl_default(self):
        self.a2.setValue(0.02)
        self.a3.setValue(0.01)
        self.update_plot()

    def update_plot(self):
        fs = float(self.fs.value())
        n = int(self.n.value())
        fin = float(self.fin.value())
        amp = float(self.amp.value())
        vfs = float(self.vfs.value())
        bits = int(self.bits.value())
        a2 = float(self.a2.value())
        a3 = float(self.a3.value())
        dither_rms = float(self.dither.value())
        xmax_khz = float(self.xmax_khz.value())

        # pencere seç
        window = "blackman" if self.window_blackman.isChecked() else "hann"

        # sinyal
        t = np.arange(n) / fs
        x = amp * np.sin(2 * np.pi * fin * t)

        # ADC
        x_adc, lsb = adc_model(
            x, bits=bits, vfs=vfs,
            nonlinearity=(a2, a3),
            dither_rms=dither_rms
        )

        f, mag_adc = fft_spectrum(x_adc, fs, window=window)

        # dBFS için kaba referans: full-scale sinüs genliği ~ vfs/2 ölçeği ile göreli kıyas
        mag_adc_db = dbfs(mag_adc, full_scale=vfs/2)

        self.canvas.ax.clear()
        if self.show_ideal.isChecked():
            _, mag_in = fft_spectrum(x, fs, window=window)
            mag_in_db = dbfs(mag_in, full_scale=vfs/2)
            self.canvas.ax.plot(f/1e3, mag_in_db, label="Giriş (ideal örneklenmiş)")

        self.canvas.ax.plot(
            f/1e3, mag_adc_db,
            label=f"ADC (bits={bits}, a2={a2:.4g}, a3={a3:.4g}, dither={dither_rms:g}V)"
        )

        self.canvas.ax.set_xlim(0, xmax_khz)
        self.canvas.ax.set_ylim(-140, 10)
        self.canvas.ax.grid(True)
        self.canvas.ax.set_xlabel("Frekans (kHz)")
        self.canvas.ax.set_ylabel("Genlik (dBFS ~ göreli)")
        self.canvas.ax.set_title("ADC örnekleme/kuantalama + doğrusal olmayanlık → harmonikler")
        self.canvas.ax.legend(loc="lower right")
        self.canvas.draw()

        # harmonik raporu (1..6)
        lines = [f"LSB ≈ {lsb:.3e} V | Pencere: {window}"]
        for k in range(1, 7):
            fk = k * fin
            fk_alias = fk % fs
            if fk_alias > fs/2:
                fk_alias = fs - fk_alias

            fpk, mpk = pick_peak_near(f, mag_adc, fk_alias, tol_hz=3000)
            if fpk is None:
                continue
            # lineer genlik dB (referanssız) sadece kıyas için:
            lines.append(f"{k}. harmonik (alias ~ {fk_alias:.1f} Hz): tepe {fpk:.1f} Hz, |X|={20*np.log10(mpk+1e-20):.1f} dB")

        self.info.setText("\n".join(lines))


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.resize(1200, 700)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
