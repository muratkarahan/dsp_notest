import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Örnekleme frekansı: 100 kHz
Fs = 100e3
t = np.arange(0, 0.01, 1/Fs)  # 10 ms

# Sinyal: 10 kHz kosinüs
fc = 10e3
signal = np.cos(2*np.pi*fc*t)

# Hilbert dönüşümü - Analitik sinyal
z = hilbert(signal)
amplitude_env = np.abs(z)  # Zarf (envelope)
phase = np.unwrap(np.angle(z))  # Faz
inst_freq = np.diff(phase) / (2*np.pi*(t[1]-t[0]))  # Anlık frekans

# FFT
X = np.fft.fftshift(np.fft.fft(signal))
f = np.fft.fftshift(np.fft.fftfreq(len(signal), 1/Fs))
X_magnitude = np.abs(X)
X_magnitude_db = 20*np.log10(X_magnitude + 1e-12)
X_phase = np.angle(X)

# FFT tabanlı analitik sinyal (Hilbert'e alternatif yöntem)
# Negatif frekansları sıfırla, pozitif frekansları 2 ile çarp
X_analytic = np.fft.fft(signal)
X_analytic[len(X_analytic)//2+1:] = 0  # Negatif frekansları sıfırla
X_analytic[1:len(X_analytic)//2] *= 2  # Pozitif frekansları 2 ile çarp
z_fft = np.fft.ifft(X_analytic)
phase_fft = np.unwrap(np.angle(z_fft))

# Teorik faz (10 kHz kosinüs için, unwrap'siz)
phase_theoretical = (2*np.pi*fc*t) % (2*np.pi)

# Faz farklarını hesapla
phase_diff_hilbert_fft = phase - phase_fft
phase_diff_hilbert_theory = np.unwrap(np.angle(np.exp(1j*phase) / np.exp(1j*(2*np.pi*fc*t))))

# Çizim - 2x4 grid
fig = plt.figure(figsize=(24, 10))
ax1 = plt.subplot(2, 4, 1)
ax2 = plt.subplot(2, 4, 2)
ax3 = plt.subplot(2, 4, 3)
ax4 = plt.subplot(2, 4, 4)
ax5 = plt.subplot(2, 4, 5)
ax6 = plt.subplot(2, 4, 6)
ax7 = plt.subplot(2, 4, 7)
ax8 = plt.subplot(2, 4, 8)

# 1. Orijinal Sinyal
ax1.plot(t*1e3, signal, 'b-', linewidth=2)
ax1.set_xlabel("Zaman (ms)", fontsize=12)
ax1.set_ylabel("Genlik", fontsize=12)
ax1.set_title("ORİJİNAL SİNYAL (10 kHz Kosinüs)", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Hilbert - Analitik Sinyal (Gerçel ve Sanal)
ax2.plot(t*1e3, signal, 'b-', linewidth=2, alpha=0.7, label='Gerçel (Orijinal)')
ax2.plot(t*1e3, z.imag, 'r--', linewidth=2, alpha=0.7, label='Sanal (Hilbert)')
ax2.plot(t*1e3, amplitude_env, 'g-', linewidth=3, alpha=0.8, label='Zarf (Envelope)')
ax2.plot(t*1e3, -amplitude_env, 'g-', linewidth=3, alpha=0.8)
ax2.set_xlabel("Zaman (ms)", fontsize=12)
ax2.set_ylabel("Genlik", fontsize=12)
ax2.set_title("HİLBERT - ANALİTİK SİNYAL", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=11, loc='upper right')

# 3. Hilbert vs FFT - Faz Karşılaştırması (Zaman Bölgesi)
ax3.plot(t*1e3, phase, 'b-', linewidth=3, alpha=0.8, label='Hilbert Fazı')
ax3.plot(t*1e3, phase_fft, 'r--', linewidth=3, alpha=0.7, label='FFT Tabanlı Faz')
ax3.set_xlabel("Zaman (ms)", fontsize=12, fontweight='bold')
ax3.set_ylabel("Faz (radyan)", fontsize=12, fontweight='bold')
ax3.set_title("FAZ KARŞILAŞTIRMASI: Hilbert vs FFT\n(Her İkisi de Aynı)", fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11, loc='best')

# 4. Faz Farkı (Hilbert - FFT)
ax4.plot(t*1e3, phase_diff_hilbert_fft*180/np.pi, 'purple', linewidth=2)
ax4.set_xlabel("Zaman (ms)", fontsize=12, fontweight='bold')
ax4.set_ylabel("Faz Farkı (derece)", fontsize=12, fontweight='bold')
ax4.set_title("FAZ FARKI: Hilbert - FFT\n(Numerik Hata)", fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5)

# 5. FFT - Genlik Spektrumu
ax5.plot(f/1e3, X_magnitude, 'b-', linewidth=2)
ax5.set_xlim([-20, 20])
ax5.set_xlabel("Frekans (kHz)", fontsize=12)
ax5.set_ylabel("Genlik", fontsize=12)
ax5.set_title("FFT - GENLİK SPEKTRUMU", fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. FFT - Genlik (dB)
ax6.plot(f/1e3, X_magnitude_db, 'b-', linewidth=2)
ax6.set_xlim([-20, 20])
ax6.set_xlabel("Frekans (kHz)", fontsize=12)
ax6.set_ylabel("Genlik (dB)", fontsize=12)
ax6.set_title("FFT - GENLİK SPEKTRUMU (dB)", fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 7. FFT - Faz Spektrumu
mask = X_magnitude_db > (0.5 * np.max(X_magnitude_db))
ax7.scatter(f[mask]/1e3, X_phase[mask], s=100, c='red', alpha=0.8, marker='o', edgecolors='darkred', linewidths=2)
ax7.set_xlim([-20, 20])
ax7.set_ylim([-np.pi, np.pi])
ax7.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax7.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax7.set_xlabel("Frekans (kHz)", fontsize=12)
ax7.set_ylabel("Faz (radyan)", fontsize=12)
ax7.set_title("FFT - FAZ SPEKTRUMU", fontsize=13, fontweight='bold')
ax7.grid(True, alpha=0.3)
ax7.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 8. Anlık Frekans (Hilbert)
ax8.plot(t[1:]*1e3, inst_freq/1e3, 'b-', linewidth=2)
ax8.axhline(y=fc/1e3, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Teorik: {fc/1e3:.0f} kHz')
ax8.set_xlabel("Zaman (ms)", fontsize=12)
ax8.set_ylabel("Frekans (kHz)", fontsize=12)
ax8.set_title("ANLIK FREKANS (Hilbert)", fontsize=13, fontweight='bold')
ax8.set_ylim([9, 11])
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=11)

plt.tight_layout()
plt.show()
