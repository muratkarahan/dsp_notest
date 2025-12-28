import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

Fs = 1e6
t = np.arange(0, 2e-3, 1/Fs)

# baseband
xbb = np.cos(2*np.pi*5e3*t)

# IF taşıma
fIF = 50e3
xif = xbb * np.cos(2*np.pi*fIF*t)

# Hilbert
xa = hilbert(xif)

# FFT'ler
def fft_db(x):
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(len(x), 1/Fs))
    return f, 20*np.log10(np.abs(X)+1e-12)

f1, Xbb = fft_db(xbb)
f2, Xif = fft_db(xif)
f3, Xa  = fft_db(xa)

plt.figure(figsize=(10,6))
plt.plot(f1/1e3, Xbb, label="Baseband")
plt.plot(f2/1e3, Xif, label="IF (50 kHz)")
plt.plot(f3/1e3, Xa,  label="Hilbert sonrası")
plt.xlim(-100,100)
plt.legend()
plt.xlabel("Frequency (kHz)")
plt.ylabel("Magnitude (dB)")
plt.grid()
plt.show()
