import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

# 1. Parametreler (Parameters)
Fs = 1000            # Örnekleme Frekansı (Hz)
T = 4.0              # Toplam Süre (s) (-2 ile +2 arası)
t = np.linspace(-T/2, T/2, int(Fs*T), endpoint=False)

# 2. Sinc Sinyali (Sinc Signal)
# NumPy'ın sinc fonksiyonu normalize edilmiştir: sinc(x) = sin(pi*x) / (pi*x)
# Genişliği ayarlamak için zamanı ölçekleyelim.
# Bandwidth (B) parametresi ile oynayabiliriz.
B = 10.0             # Bant genişliği faktörü
x = B * t
y = np.sinc(x)       # sinc(B*t)

# 3. Frekans Analizi (FFT)
N = len(y)
yf = fft(y)
xf = fftfreq(N, 1/Fs)
yf_shifted = fftshift(yf)
xf_shifted = fftshift(xf)

# Genlik Spektrumu (Magnitude Spectrum)
magnitude = np.abs(yf_shifted) / N

# 4. Görselleştirme (Visualization)
fig = plt.figure(figsize=(14, 8))

# --- Zaman Bölgesi (Time Domain) ---
ax1 = plt.subplot(2, 1, 1)
ax1.plot(t, y, 'b-', linewidth=2)
ax1.set_title(f"Sinc Sinyali (Zaman Bölgesi)\n$y(t) = \mathrm{{sinc}}({B}t)$")
ax1.set_xlabel("Zaman (s)")
ax1.set_ylabel("Genlik")
ax1.grid(True, alpha=0.3)
ax1.axhline(0, color='black', linewidth=1)
ax1.axvline(0, color='black', linewidth=1)

# --- Frekans Bölgesi (Frequency Domain) ---
ax2 = plt.subplot(2, 1, 2)
ax2.plot(xf_shifted, magnitude, 'r-', linewidth=2)
ax2.set_title("Frekans Spektrumu (FFT)\n(Sinc'in Fourier Dönüşümü Dikdörtgendir)")
ax2.set_xlabel("Frekans (Hz)")
ax2.set_ylabel("Genlik")
ax2.set_xlim(-B*2, B*2) # İlgili frekans aralığına odaklan
ax2.grid(True, alpha=0.3)
ax2.fill_between(xf_shifted, magnitude, color='red', alpha=0.1)

# Açıklama
info_text = """
Sinc Fonksiyonu:
sinc(x) = sin(πx) / (πx)

Özellikler:
• Zaman bölgesinde sonsuza uzanır ama genliği azalır.
• Fourier dönüşümü (Frekans spektrumu) ideal bir
  DİKDÖRTGEN (Rectangular) fonksiyondur.
• Bu nedenle ideal Alçak Geçiren Filtre (LPF)
  darbe cevabı bir Sinc fonksiyonudur.
"""
fig.text(0.02, 0.02, info_text, fontsize=11, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.show()
