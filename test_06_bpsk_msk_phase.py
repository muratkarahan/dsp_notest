import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Örnekleme frekansı: 100 kHz
Fs = 100e3
t = np.arange(0, 10, 1/Fs)

# Taşıyıcı: 10 kHz
fc = 10e3
carrier = np.cos(2*np.pi*fc*t)

# Pulse bilgisi: 1 kHz kare dalga
f_pulse = 1e3
pulse = np.sign(np.sin(2*np.pi*f_pulse*t))

# BPSK modülasyonu
signal_bpsk = pulse * carrier

# MSK modülasyonu
# MSK için modülasyon indeksi h = 0.5
h = 0.5
delta_f = h * f_pulse / 2  # Frekans sapması

# Faz hesapla (sürekli faz)
phase = np.zeros_like(t)
for i in range(1, len(t)):
    dt = t[i] - t[i-1]
    # Pulse değerine göre frekans sapması
    if pulse[i] > 0:
        phase[i] = phase[i-1] + 2*np.pi*(fc + delta_f)*dt
    else:
        phase[i] = phase[i-1] + 2*np.pi*(fc - delta_f)*dt

# MSK sinyali
signal_msk = np.cos(phase)

# FFT - BPSK
X_bpsk = np.fft.fftshift(np.fft.fft(signal_bpsk))
f = np.fft.fftshift(np.fft.fftfreq(len(signal_bpsk), 1/Fs))
X_bpsk_db = 20*np.log10(np.abs(X_bpsk)+1e-12)
X_bpsk_phase_fft = np.angle(X_bpsk)  # FFT fazı (frekansa göre)

# BPSK - Analitik sinyal ile faz hesapla (zaman bölgesi)
z_bpsk = hilbert(signal_bpsk)
X_bpsk_phase_time = np.unwrap(np.angle(z_bpsk))

# FFT - MSK
X_msk = np.fft.fftshift(np.fft.fft(signal_msk))
X_msk_db = 20*np.log10(np.abs(X_msk)+1e-12)
X_msk_phase_fft = np.angle(X_msk)  # FFT fazı (frekansa göre)

# MSK - Analitik sinyal ile faz hesapla (zaman bölgesi)
z_msk = hilbert(signal_msk)
X_msk_phase_time = np.unwrap(np.angle(z_msk))

# Çizim
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20,10))

# BPSK - Genlik
ax1.plot(f/1e3, X_bpsk_db)
ax1.set_xlim(-20, 20)
ax1.set_xlabel("Frekans (kHz)")
ax1.set_ylabel("Genlik (dB)")
ax1.set_title("BPSK - Genlik Spektrumu")
ax1.grid()

# BPSK - Faz (frekans bölgesi)
mask_bpsk = X_bpsk_db > (0.6 * np.max(X_bpsk_db))
ax2.scatter(f[mask_bpsk]/1e3, X_bpsk_phase_fft[mask_bpsk], s=50, c='red', alpha=0.8, marker='o', edgecolors='darkred', linewidths=2)
ax2.set_xlim(-20, 20)
ax2.set_ylim([-np.pi, np.pi])
ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax2.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax2.set_xlabel("Frekans (kHz)", fontsize=12)
ax2.set_ylabel("Faz (radyan)", fontsize=12)
ax2.set_title("BPSK - FAZ SPEKTRUMU (Frekansa Göre)", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# BPSK vs MSK - Hilbert Faz Karşılaştırması (zaman bölgesi)
ax3.plot(t[:200]*1e3, X_bpsk_phase_time[:200], 'r-', linewidth=3, marker='o', markersize=3, markevery=10, label='BPSK - Keskin Geçişler', alpha=0.8)
ax3.plot(t[:200]*1e3, X_msk_phase_time[:200], 'b-', linewidth=3, marker='s', markersize=3, markevery=10, label='MSK - Yumuşak Geçişler', alpha=0.8)
ax3.set_xlabel("Zaman (ms)", fontsize=12)
ax3.set_ylabel("Faz (radyan)", fontsize=12)
ax3.set_title("HILBERT FAZ KARŞILAŞTIRMASI (BPSK vs MSK)", fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11, loc='best')

# BPSK vs MSK - Sinyal Karşılaştırması (zaman bölgesi)
ax4.plot(t[:200]*1e3, signal_bpsk[:200], 'r-', linewidth=2, alpha=0.7, label='BPSK - Keskin Geçişler')
ax4.plot(t[:200]*1e3, signal_msk[:200], 'b-', linewidth=2, alpha=0.7, label='MSK - Yumuşak Geçişler')
ax4.set_xlabel("Zaman (ms)", fontsize=12)
ax4.set_ylabel("Genlik", fontsize=12)
ax4.set_title("BPSK vs MSK - SİNYAL KARŞILAŞTIRMASI (Zamana Göre)", fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=11, loc='best')

# MSK - Faz (frekans bölgesi)
mask_msk = X_msk_db > (0.6 * np.max(X_msk_db))
ax5.scatter(f[mask_msk]/1e3, X_msk_phase_fft[mask_msk], s=50, c='blue', alpha=0.8, marker='s', edgecolors='darkblue', linewidths=2)
ax5.set_xlim(-20, 20)
ax5.set_ylim([-np.pi, np.pi])
ax5.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax5.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax5.set_xlabel("Frekans (kHz)", fontsize=12)
ax5.set_ylabel("Faz (radyan)", fontsize=12)
ax5.set_title("MSK - FAZ SPEKTRUMU (Frekansa Göre)", fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# MSK - Hilbert Faz (zaman bölgesi)
ax6.plot(t[:200]*1e3, X_msk_phase_time[:200], 'b-', linewidth=3, marker='s', markersize=3, markevery=10)
ax6.set_xlabel("Zaman (ms)", fontsize=12)
ax6.set_ylabel("Faz (radyan)", fontsize=12)
ax6.set_title("MSK - HILBERT FAZ (Zamana Göre)", fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
