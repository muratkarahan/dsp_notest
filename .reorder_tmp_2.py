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
N = 200_000           # 1 saniye veri
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
t = np.arange(N) / Fs

# ======================
# Analog sinyal (Volt)
# ======================
A = 0.5               # 0.5 Vpeak
noise_rms = 2e-3      # 2 mV RMS noise

v_analog = (
    A * np.sin(2*np.pi*f0*t) +
    (harm_alpha * np.sin(2*np.pi*harm_f1*t) * np.sin(2*np.pi*harm_f2*t)
     if add_harmonic_product else 0.0) +
    noise_rms * np.random.randn(N)
)

# --- Plot A: Analog signal (pre-ADC, first 5 ms) ---
fig_analog = plt.figure()
n_view = int(0.005 * Fs)
plt.plot(t[:n_view] * 1e3, v_analog[:n_view], marker='o', markersize=3, linestyle='-')
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (V)")
plt.title("Analog Signal (pre-ADC, first 5 ms)")
plt.grid(True)
try:
    fig_analog.canvas.manager.window.raise_()
    fig_analog.canvas.manager.window.activateWindow()
except Exception:
    pass
