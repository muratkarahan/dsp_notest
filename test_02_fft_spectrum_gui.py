import sys
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QGroupBox, QDoubleSpinBox, QSpinBox, QCheckBox, QPushButton, QLabel
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# -----------------------------
# FFT helpers
# -----------------------------
def rfft_spectrum(x: np.ndarray, fs: float, window="hann"):
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

    # single-sided amplitude correction
    if n % 2 == 0:
        mag[1:-1] *= 2
    else:
        mag[1:] *= 2

    return f, mag


def db(x, eps=1e-20):
    return 20 * np.log10(np.maximum(np.abs(x), eps))


def dbfs(mag, vfs):
    # dBFS-like reference for single-sided amplitude plot
    ref = vfs / 2
    return db(mag / ref)


def clip(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)


# -----------------------------
# ADC non-ideal blocks
# -----------------------------
def build_smooth_inl_function(inl_lsb, lsb, seed=1, terms=4):
    """
    Smooth static INL error ε(x) as a low-frequency function of x_norm in [-1,1].
    Peak is scaled to ±inl_lsb*lsb.
    """
    rng = np.random.default_rng(seed)
    freqs = rng.integers(1, 6, size=terms)
    phases = rng.uniform(0, 2*np.pi, size=terms)
    amps = rng.normal(0.0, 1.0, size=terms)

    def eps(x_norm):
        u = (x_norm + 1) * np.pi  # [0, 2π]
        e = np.zeros_like(x_norm, dtype=float)
        for k in range(terms):
            e += amps[k] * np.sin(freqs[k] * u + phases[k])
        peak = np.max(np.abs(e)) + 1e-12
        e = e / peak
        return e * (inl_lsb * lsb)

    return eps


def build_dnl_thresholds(n_bits, vfs, dnl_lsb, seed=2):
    """
    Transition thresholds perturbed to emulate DNL.
    thresholds length = 2^N + 1; enforced monotonic.
    """
    rng = np.random.default_rng(seed)
    levels = 2**n_bits
    lsb = (2 * vfs) / levels

    t_ideal = -vfs + np.arange(levels + 1) * lsb
    delta = rng.uniform(-dnl_lsb * lsb, dnl_lsb * lsb, size=levels + 1)
    delta[0] = 0.0
    delta[-1] = 0.0

    t = t_ideal + delta
    t = np.maximum.accumulate(t)
    t[0] = -vfs
    t[-1] = +vfs
    return t, lsb


def quantize_with_thresholds(x, thresholds):
    codes = np.searchsorted(thresholds, x, side="right") - 1
    return np.clip(codes, 0, len(thresholds) - 2)


def codes_to_voltage(codes, n_bits, vfs):
    levels = 2**n_bits
    lsb = (2 * vfs) / levels
    # mid-tread mapping
    return -vfs + (codes + 0.5) * lsb


def thd_to_a3(thd_db, A, vfs):
    """
    Fit cubic term a3 so that 3rd harmonic amplitude ratio ~ THD (rough).
    For y = x + a3 x^3, x=A sin(wt):
      HD3 ratio ≈ |a3|*A^2/4  (amplitude ratio, dBc)
      => a3 ≈ 4 * 10^(THD/20) / A^2
    """
    A = min(A, 0.98 * vfs)
    ratio = 10 ** (thd_db / 20)  # THD is negative (dBc)
    return 4 * ratio / (A * A + 1e-18)


def simulate_adc(
    fin=20_000.0,
    amp=0.9,
    phase=0.0,

    fs=2_000_000.0,
    n=262144,
    jitter_rms_s=30e-12,

    # S/H settling
    rs=1_000.0,
    csh=5e-12,
    ts_acq=0.5e-6,

    n_bits=12,
    vfs=1.0,

    snr_db=68.0,
    thd_db=-72.0,
    inl_lsb=3.0,
    dnl_lsb=2.0,

    vref_gain_rms=50e-6,
    dither_rms_v=0.0,

    seed=123
):
    rng = np.random.default_rng(seed)

    levels = 2**n_bits
    lsb = (2 * vfs) / levels

    t = np.arange(n) / fs
    tj = rng.normal(0.0, jitter_rms_s, size=n) if jitter_rms_s > 0 else 0.0

    # sampled input with jitter
    x = amp * np.sin(2 * np.pi * fin * (t + tj) + phase)

    if dither_rms_v > 0:
        x = x + rng.normal(0.0, dither_rms_v, size=n)

    # S/H settling (first-order memory)
    tau = max(rs * csh, 1e-18)
    alpha = 1.0 - np.exp(-ts_acq / tau)
    xh = np.empty_like(x)
    xh[0] = x[0]
    for i in range(1, n):
        xh[i] = xh[i - 1] + alpha * (x[i] - xh[i - 1])

    # reference noise as gain modulation
    if vref_gain_rms > 0:
        g = 1.0 + rng.normal(0.0, vref_gain_rms, size=n)
    else:
        g = 1.0

    y = g * xh

    # polynomial (cubic) from THD target
    a3 = thd_to_a3(thd_db=thd_db, A=amp, vfs=vfs)
    y = y + a3 * (y**3)

    # INL (smooth static error)
    if inl_lsb > 0:
        eps = build_smooth_inl_function(inl_lsb=inl_lsb, lsb=lsb, seed=seed + 10, terms=4)
        y = y + eps(clip(y / vfs, -1.0, 1.0))

    # add noise to match SNR (approx)
    if snr_db is not None:
        signal_rms = amp / np.sqrt(2)
        noise_rms = signal_rms / (10 ** (snr_db / 20))
        y = y + rng.normal(0.0, noise_rms, size=n)

    # saturation
    y = clip(y, -vfs, +vfs)

    # DNL thresholds + quantization
    if dnl_lsb > 0:
        thresholds, _ = build_dnl_thresholds(n_bits, vfs, dnl_lsb, seed=seed + 20)
        codes = quantize_with_thresholds(y, thresholds)
        yq = codes_to_voltage(codes, n_bits, vfs)
    else:
        codes = np.floor((y + vfs) / lsb).astype(int)
        codes = np.clip(codes, 0, levels - 1)
        yq = codes_to_voltage(codes, n_bits, vfs)

    return t, x, y, yq, codes, lsb, a3, alpha


def harmonic_report(f, mag, fin, fs, nh=6, tol_hz=1500):
    lines = []
    for h in range(1, nh + 1):
        fh = (h * fin) % fs
        if fh > fs / 2:
            fh = fs - fh

        idx = np.where((f > fh - tol_hz) & (f < fh + tol_hz))[0]
        if len(idx) == 0:
            continue
        pk = idx[np.argmax(mag[idx])]
        lines.append(f"{h}x: alias~{fh:9.1f} Hz, peak@{f[pk]:9.1f} Hz, |X|={db(mag[pk]):6.1f} dB")
    return "\n".join(lines)


class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure()
        self.ax = fig.add_subplot(111)
        super().__init__(fig)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STM32F401RE ADC Tam Model (PyQt6)")

        root = QWidget()
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        # -------- Controls (left) --------
        left = QWidget()
        left_l = QVBoxLayout(left)
        left_l.setAlignment(Qt.AlignmentFlag.AlignTop)

        box = QGroupBox("STM32F401RE-benzeri Parametreler")
        form = QFormLayout(box)

        self.fs = QDoubleSpinBox(); self.fs.setRange(1e3, 50e6); self.fs.setDecimals(0); self.fs.setValue(2_000_000)
        self.n = QSpinBox(); self.n.setRange(2**10, 2**20); self.n.setSingleStep(1024); self.n.setValue(262_144)
        self.fin = QDoubleSpinBox(); self.fin.setRange(1.0, 10e6); self.fin.setDecimals(0); self.fin.setValue(20_000)
        self.amp = QDoubleSpinBox(); self.amp.setRange(0.0, 1.2); self.amp.setDecimals(4); self.amp.setValue(0.9)
        self.vfs = QDoubleSpinBox(); self.vfs.setRange(0.1, 5.0); self.vfs.setDecimals(3); self.vfs.setValue(1.0)
        self.bits = QSpinBox(); self.bits.setRange(2, 16); self.bits.setValue(12)

        self.snr = QDoubleSpinBox(); self.snr.setRange(0.0, 120.0); self.snr.setDecimals(1); self.snr.setValue(68.0)
        self.thd = QDoubleSpinBox(); self.thd.setRange(-140.0, 0.0); self.thd.setDecimals(1); self.thd.setValue(-72.0)
        self.inl = QDoubleSpinBox(); self.inl.setRange(0.0, 10.0); self.inl.setDecimals(2); self.inl.setValue(3.0)
        self.dnl = QDoubleSpinBox(); self.dnl.setRange(0.0, 10.0); self.dnl.setDecimals(2); self.dnl.setValue(2.0)

        self.jitter_ps = QDoubleSpinBox(); self.jitter_ps.setRange(0.0, 5000.0); self.jitter_ps.setDecimals(1); self.jitter_ps.setValue(30.0)
        self.rs = QDoubleSpinBox(); self.rs.setRange(0.0, 200_000.0); self.rs.setDecimals(1); self.rs.setValue(1000.0)
        self.csh_pf = QDoubleSpinBox(); self.csh_pf.setRange(0.1, 200.0); self.csh_pf.setDecimals(2); self.csh_pf.setValue(5.0)
        self.tsacq_us = QDoubleSpinBox(); self.tsacq_us.setRange(0.01, 1000.0); self.tsacq_us.setDecimals(3); self.tsacq_us.setValue(0.5)

        self.vref_ppm = QDoubleSpinBox(); self.vref_ppm.setRange(0.0, 5000.0); self.vref_ppm.setDecimals(1); self.vref_ppm.setValue(50.0)
        self.dither = QDoubleSpinBox(); self.dither.setRange(0.0, 0.1); self.dither.setDecimals(6); self.dither.setValue(0.0)

        self.xmax_khz = QDoubleSpinBox(); self.xmax_khz.setRange(1.0, 50_000.0); self.xmax_khz.setDecimals(1); self.xmax_khz.setValue(200.0)

        self.win_blackman = QCheckBox("Blackman pencere")
        self.win_blackman.setChecked(False)
        self.show_ideal = QCheckBox("İdeal giriş spektrumu çiz")
        self.show_ideal.setChecked(True)

        form.addRow("fs (Hz)", self.fs)
        form.addRow("N (FFT)", self.n)
        form.addRow("fin (Hz)", self.fin)
        form.addRow("amp (Vpeak)", self.amp)
        form.addRow("Vfs (±V)", self.vfs)
        form.addRow("bits", self.bits)

        form.addRow("SNR (dB)", self.snr)
        form.addRow("THD (dBc)", self.thd)
        form.addRow("INL (±LSB)", self.inl)
        form.addRow("DNL (±LSB)", self.dnl)

        form.addRow("jitter (ps RMS)", self.jitter_ps)
        form.addRow("Rs (Ω)", self.rs)
        form.addRow("Csh (pF)", self.csh_pf)
        form.addRow("Tacq (µs)", self.tsacq_us)

        form.addRow("Vref noise (ppm RMS)", self.vref_ppm)
        form.addRow("dither RMS (V)", self.dither)
        form.addRow("x-axis max (kHz)", self.xmax_khz)

        left_l.addWidget(box)
        left_l.addWidget(self.win_blackman)
        left_l.addWidget(self.show_ideal)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("Çalıştır / Güncelle")
        self.btn_run.clicked.connect(self.update_plot)
        btn_row.addWidget(self.btn_run)

        self.btn_stm32_typ = QPushButton("STM32F401 Typik")
        self.btn_stm32_typ.clicked.connect(self.set_stm32_typ)
        btn_row.addWidget(self.btn_stm32_typ)

        left_l.addLayout(btn_row)

        self.info = QLabel("")
        self.info.setWordWrap(True)
        left_l.addWidget(self.info)

        # -------- Plot (right) --------
        self.canvas = MplCanvas()
        layout.addWidget(left, 0)
        layout.addWidget(self.canvas, 1)

        self.resize(1250, 720)
        self.update_plot()

    def set_stm32_typ(self):
        # Typical-ish starting point
        self.bits.setValue(12)
        self.snr.setValue(68.0)
        self.thd.setValue(-72.0)
        self.inl.setValue(3.0)
        self.dnl.setValue(2.0)
        self.jitter_ps.setValue(30.0)
        self.rs.setValue(1000.0)
        self.csh_pf.setValue(5.0)
        self.tsacq_us.setValue(0.5)
        self.vref_ppm.setValue(50.0)
        self.update_plot()

    def update_plot(self):
        fs = float(self.fs.value())
        n = int(self.n.value())
        fin = float(self.fin.value())
        amp = float(self.amp.value())
        vfs = float(self.vfs.value())
        bits = int(self.bits.value())

        snr_db = float(self.snr.value())
        thd_db = float(self.thd.value())
        inl_lsb = float(self.inl.value())
        dnl_lsb = float(self.dnl.value())

        jitter_rms_s = float(self.jitter_ps.value()) * 1e-12
        rs = float(self.rs.value())
        csh = float(self.csh_pf.value()) * 1e-12
        ts_acq = float(self.tsacq_us.value()) * 1e-6

        vref_gain_rms = float(self.vref_ppm.value()) * 1e-6
        dither_rms_v = float(self.dither.value())
        xmax_khz = float(self.xmax_khz.value())

        window = "blackman" if self.win_blackman.isChecked() else "hann"

        # run sim
        t, x_in, y_analog, y_adc, codes, lsb, a3, alpha = simulate_adc(
            fin=fin, amp=amp, fs=fs, n=n,
            jitter_rms_s=jitter_rms_s,
            rs=rs, csh=csh, ts_acq=ts_acq,
            n_bits=bits, vfs=vfs,
            snr_db=snr_db, thd_db=thd_db,
            inl_lsb=inl_lsb, dnl_lsb=dnl_lsb,
            vref_gain_rms=vref_gain_rms,
            dither_rms_v=dither_rms_v,
            seed=123
        )

        f, mag_out = rfft_spectrum(y_adc, fs, window=window)
        out_db = dbfs(mag_out, vfs)

        self.canvas.ax.clear()

        if self.show_ideal.isChecked():
            _, mag_in = rfft_spectrum(x_in, fs, window=window)
            in_db = dbfs(mag_in, vfs)
            self.canvas.ax.plot(f / 1e3, in_db, label="Input (ideal sampled)")

        self.canvas.ax.plot(f / 1e3, out_db, label="Output (ADC model)", alpha=0.9)

        self.canvas.ax.set_xlim(0, xmax_khz)
        self.canvas.ax.set_ylim(-140, 10)
        self.canvas.ax.grid(True)
        self.canvas.ax.set_xlabel("Frequency (kHz)")
        self.canvas.ax.set_ylabel("Amplitude (dBFS ~ relative)")
        self.canvas.ax.set_title("STM32F401RE-like ADC: S/H + INL + DNL + THD + noise + jitter")
        self.canvas.ax.legend(loc="lower right")
        self.canvas.draw()

        # info
        rep = harmonic_report(f, mag_out, fin=fin, fs=fs, nh=6, tol_hz=1500)
        self.info.setText(
            "\n".join([
                f"LSB ≈ {lsb:.3e} V (model units)",
                f"THD→a3 fit: a3 ≈ {a3:.3e}",
                f"S/H settling alpha ≈ {alpha:.4f} (Tacq, Rs, Csh)",
                f"Window: {window}",
                "",
                "Harmonic peaks (approx):",
                rep if rep else "(no peaks found in search windows)"
            ])
        )


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
