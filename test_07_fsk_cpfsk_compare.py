import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Örnekleme frekansı: 100 kHz
Fs = 100e3
t = np.arange(0, 1, 1/Fs)  # 1 sn

# Frekanslar
f1 = 9e3   # Bit 0 için frekans
f2 = 11e3  # Bit 1 için frekans
fc = 10e3  # Merkez frekans

# Pulse bilgisi: 1 kHz kare dalga
f_pulse = 1e3
pulse = np.sign(np.sin(2*np.pi*f_pulse*t))
bits = (pulse + 1) / 2  # 0 ve 1'e çevir

# FSK modülasyonu (faz süreksizliği VAR - bit değişiminde faz sıfırlanır)
phase_fsk = 0
prev_bit = bits[0]
signal_fsk = np.zeros_like(t)
for i in range(len(t)):
    dt = t[i] - t[i-1] if i > 0 else 0
    
    # Bit değiştiğinde fazı sıfırla (FSK'nin temel özelliği)
    if bits[i] != prev_bit:
        phase_fsk = 0
    prev_bit = bits[i]
    
    # Frekansa göre faz artışı
    if bits[i] == 1:
        phase_fsk += 2*np.pi*f2*dt
    else:
        phase_fsk += 2*np.pi*f1*dt
    
    signal_fsk[i] = np.cos(phase_fsk)

# CPFSK modülasyonu (sürekli faz)
# Modülasyon indeksi h = 0.5 (MSK benzeri)
h = 0.5
delta_f = (f2 - f1) / 2  # Frekans sapması

# Faz hesapla (sürekli faz)
phase_cpfsk = np.zeros_like(t)
for i in range(1, len(t)):
    dt = t[i] - t[i-1]
    # Bit değerine göre frekans
    if bits[i] == 1:
        freq = fc + delta_f
    else:
        freq = fc - delta_f
    phase_cpfsk[i] = phase_cpfsk[i-1] + 2*np.pi*freq*dt

# CPFSK sinyali
signal_cpfsk = np.cos(phase_cpfsk)

# FFT - FSK
X_fsk = np.fft.fftshift(np.fft.fft(signal_fsk))
f = np.fft.fftshift(np.fft.fftfreq(len(signal_fsk), 1/Fs))
X_fsk_db = 20*np.log10(np.abs(X_fsk)+1e-12)
X_fsk_phase_fft = np.angle(X_fsk)  # FFT fazı (frekansa göre)

# FSK - Analitik sinyal ile faz hesapla (zaman bölgesi)
z_fsk = hilbert(signal_fsk)
X_fsk_phase_time = np.unwrap(np.angle(z_fsk))  # Unwrap VAR - sürekli faz görünsün

# FFT - CPFSK
X_cpfsk = np.fft.fftshift(np.fft.fft(signal_cpfsk))
X_cpfsk_db = 20*np.log10(np.abs(X_cpfsk)+1e-12)
X_cpfsk_phase_fft = np.angle(X_cpfsk)  # FFT fazı (frekansa göre)

# CPFSK - Analitik sinyal ile faz hesapla (zaman bölgesi)
z_cpfsk = hilbert(signal_cpfsk)
X_cpfsk_phase_time = np.unwrap(np.angle(z_cpfsk))  # Unwrap VAR - sürekli faz

# Anlık frekans hesapla (faz türevinden)
# df/dt = (1/2π) * dφ/dt
dt = t[1] - t[0]
fsk_inst_freq = np.diff(X_fsk_phase_time) / (2*np.pi*dt)  # Hz cinsinden
cpfsk_inst_freq = np.diff(X_cpfsk_phase_time) / (2*np.pi*dt)  # Hz cinsinden

# Çizim
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20,10))

# FSK - Genlik
ax1.plot(f/1e3, X_fsk_db)
ax1.set_xlim(-20, 20)
ax1.set_xlabel("Frekans (kHz)")
ax1.set_ylabel("Genlik (dB)")
ax1.set_title("FSK - Genlik Spektrumu")
ax1.grid()

# FSK - Faz (frekans bölgesi)
mask_fsk = X_fsk_db > (0.6 * np.max(X_fsk_db))
ax2.scatter(f[mask_fsk]/1e3, X_fsk_phase_fft[mask_fsk], s=50, c='red', alpha=0.8, marker='o', edgecolors='darkred', linewidths=2)
ax2.set_xlim(-20, 20)
ax2.set_ylim([-np.pi, np.pi])
ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax2.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax2.set_xlabel("Frekans (kHz)", fontsize=12)
ax2.set_ylabel("Faz (radyan)", fontsize=12)
ax2.set_title("FSK - FAZ SPEKTRUMU (Frekansa Göre)", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# FSK vs CPFSK - Anlık Frekans Karşılaştırması (faz türevinden)
t_diff = t[1:]  # diff bir eleman azaltır
ax3.plot(t_diff*1e3, fsk_inst_freq/1e3, 'r-', linewidth=2, alpha=0.7, label='FSK - Ani Frekans Değişimleri')
ax3.plot(t_diff*1e3, cpfsk_inst_freq/1e3, 'b-', linewidth=2, alpha=0.7, label='CPFSK - Yumuşak Frekans Değişimleri')
ax3.axhline(y=f1/1e3, color='gray', linestyle=':', linewidth=2, alpha=0.5, label=f'f1={f1/1e3:.0f} kHz')
ax3.axhline(y=f2/1e3, color='gray', linestyle=':', linewidth=2, alpha=0.5, label=f'f2={f2/1e3:.0f} kHz')
ax3.set_xlabel("Zaman (ms)", fontsize=13, fontweight='bold')
ax3.set_ylabel("Anlık Frekans (kHz)", fontsize=13, fontweight='bold')
ax3.set_title("ANLIK FREKANS KARŞILAŞTIRMASI\n(FSK: KESİN SIÇRAMALAR, CPFSK: YUMUŞAK GEÇİŞLER)", fontsize=14, fontweight='bold')
ax3.set_ylim([8, 12])
ax3.grid(True, alpha=0.4, linewidth=1.5)
ax3.legend(fontsize=10, loc='best', framealpha=0.9)

# FSK vs CPFSK - Sinyal Karşılaştırması (zaman bölgesi)
ax4.plot(t*1e3, signal_fsk, 'r-', linewidth=1.5, alpha=0.7, label='FSK - Keskin Geçişler')
ax4.plot(t*1e3, signal_cpfsk, 'b-', linewidth=1.5, alpha=0.6, label='CPFSK - Yumuşak Geçişler')
ax4.set_xlabel("Zaman (ms)", fontsize=12)
ax4.set_ylabel("Genlik", fontsize=12)
ax4.set_title("FSK vs CPFSK - SİNYAL KARŞILAŞTIRMASI (Zamana Göre)", fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=11, loc='best')

# CPFSK - Faz (frekans bölgesi)
mask_cpfsk = X_cpfsk_db > (0.6 * np.max(X_cpfsk_db))
ax5.scatter(f[mask_cpfsk]/1e3, X_cpfsk_phase_fft[mask_cpfsk], s=50, c='blue', alpha=0.8, marker='s', edgecolors='darkblue', linewidths=2)
ax5.set_xlim(-20, 20)
ax5.set_ylim([-np.pi, np.pi])
ax5.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax5.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax5.set_xlabel("Frekans (kHz)", fontsize=12)
ax5.set_ylabel("Faz (radyan)", fontsize=12)
ax5.set_title("CPFSK - FAZ SPEKTRUMU (Frekansa Göre)", fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# CPFSK - Genlik
ax6.plot(f/1e3, X_cpfsk_db)
ax6.set_xlim(-20, 20)
ax6.set_xlabel("Frekans (kHz)")
ax6.set_ylabel("Genlik (dB)")
ax6.set_title("CPFSK - Genlik Spektrumu")
ax6.grid()

plt.tight_layout()
plt.show()
