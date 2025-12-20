import sys
import numpy as np

import matplotlib
# Sende backend zaten qtagg görünüyor; bunu açıkça set edelim
matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
from PyQt6.QtWidgets import QApplication
from scipy import signal

# ======================
# Parametreler
# ======================
Fs = 200_000          # 200 kS/s
# Süre (s) ve örnek sayısı: N = int(Fs * T)

T = 1.0               # saniye cinsinden süre (varsayılan 1 s)
N = int(Fs * T)

f0 = 20_000           # 20 kHz tone
Nbit = 12
Vref = 3.3
R = 50.0

# --- İki harmonik çarpımı için parametreler ---
# add_harmonic_product: True ise aşağıdaki iki harmonik sinüsün çarpımı sinyale eklenir
# harm_f1, harm_f2: harmonik frekansları (Hz)
# harm_alpha: çarpımın ölçeği (amplitude)
add_harmonic_product = True
harm_f1 = 12_000.0
harm_f2 = 18_000.0
harm_alpha = 0.2
# Qt event loop garanti (VS Code/terminal kombinasyonlarında iyi gelir)
app = QApplication.instance() or QApplication([])

print("Python:", sys.executable)
print("Matplotlib backend:", matplotlib.get_backend())

# ======================
# Zaman ekseni
# ======================
# ADC örnekleme zamanı
t = np.arange(N) / Fs

# Analog (gerçek/sürekli) sinyal için daha yüksek örnekleme
Fs_analog = int(Fs * 10)   # analog sinyal 10x daha yüksek örnekli
decim = Fs_analog // Fs
N_analog = int(Fs_analog * T)
t_analog = np.arange(N_analog) / Fs_analog

# ======================
# Analog sinyal (Volt) - yüksek örneklemli (continuous-like)
# ======================
A = 0.5               # 0.5 Vpeak
noise_rms = 2e-3      # 2 mV RMS noise (analog-domain)

v_continuous = (
    A * np.sin(2*np.pi*f0*t_analog) +
    (harm_alpha * np.sin(2*np.pi*harm_f1*t_analog) * np.sin(2*np.pi*harm_f2*t_analog)
     if add_harmonic_product else 0.0) +
    noise_rms * np.random.randn(N_analog)
)

# ADC'nin göreceği örnekler: analog sinyalin decimate edilmiş hali (ADC rate)
v_analog = v_continuous[::decim]

# --- Plot A: Analog signal (pre-ADC, first 5 ms) ---
fig_analog = plt.figure()
# view lengths for ADC-rate and high-rate signals
n_view_adc = int(0.005 * Fs)
n_view_analog = int(0.005 * Fs_analog)
plt.plot(t_analog[:n_view_analog] * 1e3, v_continuous[:n_view_analog], linestyle='-', color='tab:blue')
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.title("Analog Signal (high-rate, first 5 ms)")
plt.grid(True)
try:
    fig_analog.canvas.manager.window.raise_()
    fig_analog.canvas.manager.window.activateWindow()
except Exception:
    pass

# ======================
# ADC quantization
# ======================
lsb = Vref / (2**Nbit)
x = np.round(v_analog / lsb).astype(np.int16)
x = np.clip(x, -(2**(Nbit-1)), 2**(Nbit-1)-1)

# --- Plot 0: Clipped ADC counts (first 5 ms) ---
fig0 = plt.figure()
n_view = int(0.005 * Fs)
plt.plot(t[:n_view] * 1e3, x[:n_view], marker='o', markersize=3, linestyle='-')
plt.xlabel("Time (ms)")
plt.ylabel("ADC code (counts)")
plt.title("ADC Codes (clipped, first 5 ms)")
plt.grid(True)
try:
    fig0.canvas.manager.window.raise_()
    fig0.canvas.manager.window.activateWindow()
except Exception:
    pass

# Counts → Volts, DC remove
v = (x - np.mean(x)) * lsb

# --- Plot Overlay: Analog (red) and ADC samples (first 5 ms) ---
fig_overlay = plt.figure()
n_view_adc = int(0.005 * Fs)
n_view_analog = int(0.005 * Fs_analog)
plt.plot(t_analog[:n_view_analog] * 1e3, v_continuous[:n_view_analog], color='red', linestyle='-', label='Analog (pre-ADC)')
plt.plot(t[:n_view_adc] * 1e3, v[:n_view_adc], marker='o', markersize=4, linestyle='None', label='ADC samples')
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.title("Analog (red) vs ADC samples (first 5 ms)")
plt.legend()
plt.grid(True)
try:
    fig_overlay.canvas.manager.window.raise_()
    fig_overlay.canvas.manager.window.activateWindow()
except Exception:
    pass

# Welch PSD (V^2/Hz)
f, Pvv = signal.welch(
    v,
    fs=Fs,
    window='hann',
    nperseg=4096,
    noverlap=2048,
    scaling='density'
)

# PSD → W/Hz → dBm/Hz
Ppp = Pvv / R
psd_dbm_hz = 10*np.log10(Ppp / 1e-3)

# Channel Power (10–30 kHz)
f1, f2 = 10e3, 30e3
mask = (f >= f1) & (f <= f2)
df = f[1] - f[0]
Pch = np.sum(Ppp[mask]) * df
Pch_dbm = 10*np.log10(Pch / 1e-3)

# --- Plot 1: Time domain (first 5 ms) ---
fig1 = plt.figure()
n_view = int(0.005 * Fs)
plt.plot(t[:n_view] * 1e3, v[:n_view], marker='o', markersize=3, linestyle='-')
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.title("ADC Samples (Time Domain, first 5 ms)")
plt.grid(True)

# Pencereyi öne getir
try:
    fig1.canvas.manager.window.raise_()
    fig1.canvas.manager.window.activateWindow()
except Exception:
    pass

# --- Plot 2: PSD ---
fig2 = plt.figure()
plt.plot(f / 1e3, psd_dbm_hz)
plt.xlabel("Frequency (kHz)")
plt.ylabel("PSD (dBm/Hz)")
plt.title(f"Welch PSD (Channel Power 10–30 kHz: {Pch_dbm:.2f} dBm)")
plt.grid(True)

# Mark channel band edges
plt.axvline(f1 / 1e3)
plt.axvline(f2 / 1e3)

# Pencereyi öne getir
try:
    fig2.canvas.manager.window.raise_()
    fig2.canvas.manager.window.activateWindow()
except Exception:
    pass

plt.show(block=True)
