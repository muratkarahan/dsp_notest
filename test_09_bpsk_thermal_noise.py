import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Örnekleme frekansı: 40 kHz
Fs = 40e3
t = np.arange(0, 1, 1/Fs)  # 1 saniye

# Taşıyıcı: 10 kHz kosinüs
fc = 10e3
carrier = np.cos(2*np.pi*fc*t)

# Veri sinyali: 1 kHz kosinüs
fd = 1e3
data = np.cos(2*np.pi*fd*t)

# BPSK modülasyonu: data * carrier
# Kosinüs +1 ile -1 arasında değiştiği için fazı 0 ve π arasında değiştirir
signal_bpsk_raw = data * carrier

# İletilen sinyal: Peak-to-peak 1V olacak şekilde normalize et
Vpp = 1.0  # Peak-to-peak voltaj (V)
signal_bpsk = signal_bpsk_raw * (Vpp / (np.max(signal_bpsk_raw) - np.min(signal_bpsk_raw)))

# Taşıyıcı güç hesabı (1 ohm yük için)
# Sinyal RMS = Vpp / (2 * sqrt(2)) = 1 / (2.828) = 0.3536 V
# Güç = V_RMS^2 / R = 0.3536^2 / 1 = 0.125 W
carrier_power = np.mean(signal_bpsk**2)  # Watt (1 ohm için)
carrier_power_dbm = 10*np.log10(carrier_power * 1000)  # dBm

# Alıcı tarafı - Termal gürültü (fiziksel sıcaklığa dayalı)
# Boltzmann sabiti
k_boltzmann = 1.38e-23  # J/K (Watt·second/Kelvin)

# Alıcı sıcaklığı
T_celsius = 30  # derece Celsius
T_kelvin = 273.15 + T_celsius  # Kelvin

# Bant genişliği: Örnekleme frekansının yarısı (Nyquist)
bandwidth_Hz = Fs / 2  # Hz

# Termal gürültü spektral yoğunluğu (Johnson-Nyquist gürültüsü)
N0 = k_boltzmann * T_kelvin  # W/Hz
N0_dBm_Hz = 10*np.log10(N0 * 1000)  # dBm/Hz

# Toplam gürültü gücü
noise_power = N0 * bandwidth_Hz  # Watt
noise_power_dbm = 10*np.log10(noise_power * 1000)  # dBm

# Gürültü sinyali ekle
noise = np.sqrt(noise_power) * np.random.randn(len(signal_bpsk))
signal_noisy = signal_bpsk + noise

# C/N oranı (dB-Hz)
CN_ratio_dBHz = carrier_power_dbm - N0_dBm_Hz  # dB-Hz

# Klasik C/N (toplam güçler)
CN_ratio = carrier_power / noise_power
CN_ratio_dB = 10*np.log10(CN_ratio)
SNR_dB = CN_ratio_dB  # Beyaz gürültü için SNR = C/N
noise_power_dbm = 10*np.log10(noise_power * 1000)  # dBm

print(f"\n=== İLETKEN SİNYAL ÖZELLİKLERİ ===")
print(f"Peak-to-Peak Voltaj: {Vpp} V")
print(f"RMS Voltaj: {np.sqrt(carrier_power):.4f} V")
print(f"Taşıyıcı Gücü (C): {carrier_power:.6f} W = {carrier_power_dbm:.2f} dBm")
print(f"\n=== ALICI VE GÜRÜLTÜ ÖZELLİKLERİ ===")
print(f"Alıcı Sıcaklığı: {T_celsius}°C = {T_kelvin:.2f} K")
print(f"Boltzmann Sabiti (k): {k_boltzmann:.2e} W/K/Hz")
print(f"Bant Genişliği: {bandwidth_Hz/1e3:.1f} kHz = {bandwidth_Hz:.0f} Hz")
print(f"\n--- Termal Gürültü Hesabı (Johnson-Nyquist) ---")
print(f"N0 = k × T = {k_boltzmann:.2e} × {T_kelvin:.2f}")
print(f"N0 = {N0:.2e} W/Hz = {N0_dBm_Hz:.2f} dBm/Hz")
print(f"Toplam Gürültü = N0 × BW = {N0:.2e} × {bandwidth_Hz:.0f}")
print(f"Toplam Gürültü (N): {noise_power:.2e} W = {noise_power_dbm:.2f} dBm")
print(f"\n=== C/N ORANLARI ===")
print(f"C/N (Hz başına) = C - N0 = {carrier_power_dbm:.2f} - ({N0_dBm_Hz:.2f})")
print(f"C/N (Hz başına): {CN_ratio_dBHz:.2f} dB-Hz")
print(f"C/N (toplam) = C/N = {carrier_power_dbm:.2f} - {noise_power_dbm:.2f}")
print(f"C/N (toplam): {CN_ratio_dB:.2f} dB")
print(f"SNR: {SNR_dB:.2f} dB")
print(f"=============================\n")

# Hilbert dönüşümü - Temiz sinyal
z = hilbert(signal_bpsk)
amplitude_env = np.abs(z)
phase = np.unwrap(np.angle(z))
inst_freq = np.diff(phase) / (2*np.pi*(t[1]-t[0]))

# Hilbert dönüşümü - Gürültülü sinyal
z_noisy = hilbert(signal_noisy)
amplitude_env_noisy = np.abs(z_noisy)
phase_noisy = np.unwrap(np.angle(z_noisy))
inst_freq_noisy = np.diff(phase_noisy) / (2*np.pi*(t[1]-t[0]))

# FFT - Temiz sinyal
X = np.fft.fftshift(np.fft.fft(signal_bpsk))
f = np.fft.fftshift(np.fft.fftfreq(len(signal_bpsk), 1/Fs))
X_magnitude = np.abs(X)
X_magnitude_db = 20*np.log10(X_magnitude + 1e-12)
X_phase = np.angle(X)

# FFT - Gürültülü sinyal
X_noisy = np.fft.fftshift(np.fft.fft(signal_noisy))
X_noisy_magnitude_db = 20*np.log10(np.abs(X_noisy) + 1e-12)
X_noisy_phase = np.angle(X_noisy)

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

# 1. Veri Sinyali (1 kHz)
n_samples = 200
ax1.plot(t[:n_samples]*1e3, data[:n_samples], 'b-', linewidth=2)
ax1.set_xlabel("Zaman (ms)", fontsize=12)
ax1.set_ylabel("Genlik", fontsize=12)
ax1.set_title("VERİ SİNYALİ (1 kHz Kosinüs)", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. Taşıyıcı (10 kHz)
ax2.plot(t[:n_samples]*1e3, carrier[:n_samples], 'r-', linewidth=2)
ax2.set_xlabel("Zaman (ms)", fontsize=12)
ax2.set_ylabel("Genlik", fontsize=12)
ax2.set_title("TAŞIYICI (10 kHz Kosinüs)", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. BPSK - Temiz vs Gürültülü Sinyal
ax3.plot(t[:n_samples]*1e3, signal_bpsk[:n_samples], 'g-', linewidth=2, alpha=0.8, label='Temiz Sinyal')
ax3.plot(t[:n_samples]*1e3, signal_noisy[:n_samples], 'r-', linewidth=1, alpha=0.6, label=f'Gürültülü (C/N={CN_ratio_dBHz:.1f} dB-Hz)')
ax3.set_xlabel("Zaman (ms)", fontsize=12)
ax3.set_ylabel("Voltaj (V)", fontsize=12)
ax3.set_title(f"BPSK - TEMİZ vs GÜRÜLTÜLÜ\nVpp=1V, C={carrier_power_dbm:.1f} dBm, C/N={CN_ratio_dBHz:.1f} dB-Hz", fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

# 4. Hilbert - Zarf Karşılaştırması
ax4.plot(t[:n_samples]*1e3, amplitude_env[:n_samples], 'g-', linewidth=3, alpha=0.8, label='Temiz - Zarf')
ax4.plot(t[:n_samples]*1e3, amplitude_env_noisy[:n_samples], 'r-', linewidth=2, alpha=0.7, label='Gürültülü - Zarf')
ax4.plot(t[:n_samples]*1e3, np.abs(data[:n_samples]), 'b--', linewidth=2, alpha=0.5, label='Orijinal Veri Zarfi')
ax4.set_xlabel("Zaman (ms)", fontsize=12)
ax4.set_ylabel("Genlik", fontsize=12)
ax4.set_title("HİLBERT - ZARF KARŞILAŞTIRMASI", fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10, loc='upper right')

# 5. FFT - Genlik Spektrumu Karşılaştırması
ax5.plot(f/1e3, X_magnitude_db, 'g-', linewidth=2, alpha=0.8, label='Temiz Sinyal')
ax5.plot(f/1e3, X_noisy_magnitude_db, 'r-', linewidth=1.5, alpha=0.6, label=f'Gürültülü (C/N={CN_ratio_dBHz:.1f} dB-Hz)')
ax5.set_xlim([-20, 20])
ax5.set_xlabel("Frekans (kHz)", fontsize=12)
ax5.set_ylabel("Genlik (dB)", fontsize=12)
ax5.set_title(f"FFT - GENLİK SPEKTRUMU\nC={carrier_power_dbm:.1f} dBm, N0={N0_dBm_Hz:.1f} dBm/Hz, BW={bandwidth_Hz/1e3:.1f} kHz", fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.legend(fontsize=11)

# 6. FFT - Faz Spektrumu
mask = X_magnitude_db > (0.6 * np.max(X_magnitude_db))
ax6.scatter(f[mask]/1e3, X_phase[mask], s=100, c='red', alpha=0.8, marker='o', edgecolors='darkred', linewidths=2)
ax6.set_xlim([-20, 20])
ax6.set_ylim([-np.pi, np.pi])
ax6.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax6.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
ax6.set_xlabel("Frekans (kHz)", fontsize=12)
ax6.set_ylabel("Faz (radyan)", fontsize=12)
ax6.set_title("FFT - FAZ SPEKTRUMU", fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3)
ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# 7. Hilbert - Faz (Zaman)
ax7.plot(t[:n_samples]*1e3, phase[:n_samples], 'b-', linewidth=2)
ax7.set_xlabel("Zaman (ms)", fontsize=12)
ax7.set_ylabel("Faz (radyan)", fontsize=12)
ax7.set_title("HİLBERT - FAZ (Zamana Göre)", fontsize=13, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 8. Anlık Frekans Karşılaştırması
ax8.plot(t[1:n_samples]*1e3, inst_freq[:n_samples-1]/1e3, 'g-', linewidth=2, alpha=0.8, label='Temiz Sinyal')
ax8.plot(t[1:n_samples]*1e3, inst_freq_noisy[:n_samples-1]/1e3, 'r-', linewidth=1.5, alpha=0.6, label='Gürültülü Sinyal')
ax8.axhline(y=fc/1e3, color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Taşıyıcı: {fc/1e3:.0f} kHz')
ax8.set_xlabel("Zaman (ms)", fontsize=12)
ax8.set_ylabel("Frekans (kHz)", fontsize=12)
ax8.set_title("ANLIK FREKANS KARŞILAŞTIRMASI", fontsize=13, fontweight='bold')
ax8.set_ylim([8, 12])
ax8.grid(True, alpha=0.3)
ax8.legend(fontsize=10)

plt.tight_layout()
plt.show()
