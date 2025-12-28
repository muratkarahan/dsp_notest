import sys
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFormLayout, QGroupBox, QDoubleSpinBox, QSpinBox, QPushButton, QLabel
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# -----------------------------
# DSP helpers
# -----------------------------
def rfft_spectrum(x, fs, window="hann"):
    n = len(x)
    if window == "hann":
        w = np.hanning(n)
    elif window == "blackman":
        w = np.blackman(n)
    else:
        w = np.ones(n)

    xw = x * w
    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, 1/fs)

    cg = np.sum(w) / n
    mag = np.abs(X) / (n * cg)
    if n % 2 == 0:
        mag[1:-1] *= 2
    else:
        mag[1:] *= 2
    return f, mag

def db(x, eps=1e-20):
    return 20*np.log10(np.maximum(np.abs(x), eps))

def lowpass_iir(x, fs, fc):
    """Basit 1. derece IIR LPF"""
    if fc <= 0:
        return np.zeros_like(x)
    alpha = 1.0 - np.exp(-2*np.pi*fc/fs)
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = y[i-1] + alpha*(x[i] - y[i-1])
    return y


# -----------------------------
# NCO + mixing
# -----------------------------
def nco_phase(fs, n, fout):
    """32-bit phase accumulator benzeri (float ile)"""
    t = np.arange(n)/fs
    return 2*np.pi*fout*t

def real_mixer(x, phi):
    return x * np.cos(phi)

def complex_mixer(x, phi):
    I = x * np.cos(phi)
    Q = -x * np.sin(phi)
    return I, Q


# -----------------------------
# Matplotlib canvas
# -----------------------------
class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(figsize=(8, 6))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)


# -----------------------------
# Main window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NCO + Mixing + Image Rejection (PyQt6)")

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # ---- Controls ----
        ctrl = QWidget()
        ctrl_l = QVBoxLayout(ctrl)
        ctrl_l.setAlignment(Qt.AlignmentFlag.AlignTop)

        box = QGroupBox("Parametreler")
        form = QFormLayout(box)

        self.fs = QDoubleSpinBox(); self.fs.setRange(1e3, 5e6); self.fs.setDecimals(0); self.fs.setValue(500_000)
        self.n = QSpinBox(); self.n.setRange(1024, 262144); self.n.setSingleStep(1024); self.n.setValue(32768)
        self.fin = QDoubleSpinBox(); self.fin.setRange(1.0, 200e3); self.fin.setDecimals(1); self.fin.setValue(120_000)
        self.flo = QDoubleSpinBox(); self.flo.setRange(1.0, 200e3); self.flo.setDecimals(1); self.flo.setValue(100_000)
        self.amp = QDoubleSpinBox(); self.amp.setRange(0.0, 2.0); self.amp.setDecimals(3); self.amp.setValue(1.0)
        self.fc = QDoubleSpinBox(); self.fc.setRange(1.0, 200e3); self.fc.setDecimals(1); self.fc.setValue(20_000)
        self.xmax = QDoubleSpinBox(); self.xmax.setRange(1.0, 300e3); self.xmax.setDecimals(1); self.xmax.setValue(200_000)

        form.addRow("fs (Hz)", self.fs)
        form.addRow("N", self.n)
        form.addRow("fin (Hz)", self.fin)
        form.addRow("fLO / NCO (Hz)", self.flo)
        form.addRow("Amplitude", self.amp)
        form.addRow("LPF fc (Hz)", self.fc)
        form.addRow("FFT x-max (Hz)", self.xmax)

        ctrl_l.addWidget(box)

        self.btn = QPushButton("Çalıştır / Güncelle")
        self.btn.clicked.connect(self.update_plot)
        ctrl_l.addWidget(self.btn)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        ctrl_l.addWidget(self.info)

        # ---- Plot ----
        self.canvas = MplCanvas()

        layout.addWidget(ctrl, 0)
        layout.addWidget(self.canvas, 1)

        self.resize(1200, 700)
        self.update_plot()

    def update_plot(self):
        fs = float(self.fs.value())
        n = int(self.n.value())
        fin = float(self.fin.value())
        flo = float(self.flo.value())
        amp = float(self.amp.value())
        fc = float(self.fc.value())
        xmax = float(self.xmax.value())

        # --- Input signal ---
        t = np.arange(n)/fs
        x = amp * np.cos(2*np.pi*fin*t)

        # --- NCO ---
        phi = nco_phase(fs, n, flo)

        # --- Real mixing (image remains) ---
        y_real = real_mixer(x, phi)
        y_real_lpf = lowpass_iir(y_real, fs, fc)

        # --- Complex mixing (image rejected) ---
        I, Q = complex_mixer(x, phi)
        I_f = lowpass_iir(I, fs, fc)
        Q_f = lowpass_iir(Q, fs, fc)
        z = I_f + 1j*Q_f

        # --- Spectra ---
        f_in, mag_in = rfft_spectrum(x, fs)
        f_r, mag_r = rfft_spectrum(y_real_lpf, fs)
        # complex spectrum (use FFT of complex)
        Z = np.fft.fft(z*np.hanning(n))
        fz = np.fft.fftfreq(n, 1/fs)
        mag_z = np.abs(Z)/n

        # --- Plot ---
        self.canvas.ax.clear()
        self.canvas.ax.plot(f_in, db(mag_in), label="Input (real)")
        self.canvas.ax.plot(f_r, db(mag_r), label="Real mixer + LPF (image var)")
        self.canvas.ax.plot(fz, db(mag_z), label="Complex mixer + LPF (image yok)", alpha=0.9)

        self.canvas.ax.set_xlim(-xmax, xmax)
        self.canvas.ax.set_ylim(-140, 20)
        self.canvas.ax.grid(True)
        self.canvas.ax.set_xlabel("Frequency (Hz)")
        self.canvas.ax.set_ylabel("Magnitude (dB)")
        self.canvas.ax.set_title("NCO Mixing ve Image Bastırma")
        self.canvas.ax.legend(loc="lower right")
        self.canvas.draw()

        # --- Info ---
        self.info.setText(
            "\n".join([
                f"fin = {fin:.1f} Hz, fLO = {flo:.1f} Hz",
                "Real mixer: iki yan bant üst üste → image kalır",
                "Complex mixer: tek yan bant → image bastırılır",
                "LPF fc'yi daralt → bastırma artar"
            ])
        )


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
