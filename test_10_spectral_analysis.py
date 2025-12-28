import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import dct
try:
    import pywt
    _HAS_PYWT = True
except ImportError:
    pywt = None
    _HAS_PYWT = False

# Örnekleme frekansı: 40 kHz
Fs = 40e3
t = np.arange(0, 1, 1/Fs)  # 1 saniye

# Taşıyıcı: 10 kHz kosinüs
fc = 10e3
carrier = np.cos(2*np.pi*fc*t)

# Veri sinyali: 1 kHz kosinüs
fd = 1e3
data = np.cos(2*np.pi*fd*t)

# BPSK modülasyonu
signal_bpsk = data * carrier

# DFT (Discrete Fourier Transform) - Tek FFT ile tüm sinyalin frekans analizi
X_dft = np.fft.fftshift(np.fft.fft(signal_bpsk))
f_dft = np.fft.fftshift(np.fft.fftfreq(len(signal_bpsk), 1/Fs))
X_dft_magnitude_db = 20*np.log10(np.abs(X_dft) + 1e-12)

# DCT (Discrete Cosine Transform) - Frekans analizi (sadece kosinüs baz fonksiyonları)
X_dct = dct(signal_bpsk, type=2, norm='ortho')
f_dct = np.arange(len(X_dct)) * Fs / (2 * len(X_dct))  # DCT frekans ekseni
X_dct_magnitude_db = 20*np.log10(np.abs(X_dct) + 1e-12)

# Wavelet Transform (CWT - Continuous Wavelet Transform)
# Morlet wavelet kullanıyoruz (zaman-frekans lokalizasyonu iyi)
if _HAS_PYWT:
    scales = np.arange(1, 128)  # Ölçek aralığı
    coefficients, frequencies = pywt.cwt(signal_bpsk, scales, 'morl', 1/Fs)
    coefficients_db = 20*np.log10(np.abs(coefficients) + 1e-12)
else:
    scales = None
    coefficients = None
    frequencies = None
    coefficients_db = None

# STFT (Short-Time Fourier Transform) - Zaman-frekans analizi
# Farklı pencere boyutları
nperseg_small = 256   # Küçük pencere: İyi zaman çözünürlüğü, kötü frekans çözünürlüğü
nperseg_medium = 1024  # Orta pencere: Dengeli
nperseg_large = 4096   # Büyük pencere: İyi frekans çözünürlüğü, kötü zaman çözünürlüğü

# STFT hesaplamaları
f_stft_small, t_stft_small, Zxx_small = signal.stft(signal_bpsk, Fs, nperseg=nperseg_small, noverlap=nperseg_small//2)
f_stft_medium, t_stft_medium, Zxx_medium = signal.stft(signal_bpsk, Fs, nperseg=nperseg_medium, noverlap=nperseg_medium//2)
f_stft_large, t_stft_large, Zxx_large = signal.stft(signal_bpsk, Fs, nperseg=nperseg_large, noverlap=nperseg_large//2)

# Magnitude'yi dB'ye çevir
Zxx_small_db = 20*np.log10(np.abs(Zxx_small) + 1e-12)
Zxx_medium_db = 20*np.log10(np.abs(Zxx_medium) + 1e-12)
Zxx_large_db = 20*np.log10(np.abs(Zxx_large) + 1e-12)

# Çizim - 2x4 grid
fig = plt.figure(figsize=(28, 12))

# 1. Orijinal Sinyal (Zaman)
ax1 = plt.subplot(2, 4, 1)
n_samples = 400
ax1.plot(t[:n_samples]*1e3, signal_bpsk[:n_samples], 'b-', linewidth=1.5)
ax1.set_xlabel("Zaman (ms)", fontsize=12)
ax1.set_ylabel("Genlik", fontsize=12)
ax1.set_title("BPSK MODÜLELİ SİNYAL\n(10 kHz taşıyıcı, 1 kHz veri)", fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2. DFT - Frekans Spektrumu (Tüm Sinyal)
ax2 = plt.subplot(2, 4, 2)
ax2.plot(f_dft/1e3, X_dft_magnitude_db, 'b-', linewidth=2)
ax2.set_xlim([-20, 20])
ax2.set_xlabel("Frekans (kHz)", fontsize=12)
ax2.set_ylabel("Genlik (dB)", fontsize=12)
ax2.set_title("DFT - FREKANS SPEKTRUMU\n(Tüm Sinyal, Zaman Çözünürlüğü YOK)", fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. DCT - Kosinüs Spektrumu
ax3 = plt.subplot(2, 4, 3)
ax3.plot(f_dct/1e3, X_dct_magnitude_db, 'g-', linewidth=2)
ax3.set_xlim([0, 20])
ax3.set_xlabel("Frekans (kHz)", fontsize=12)
ax3.set_ylabel("Genlik (dB)", fontsize=12)
ax3.set_title("DCT - KOSİNÜS SPEKTRUMU\n(Sadece Gerçel, Simetrik)", fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 4. Wavelet Transform - Scalogram (pywt yoksa bilgilendirme)
ax4 = plt.subplot(2, 4, 4)
if _HAS_PYWT:
    pcm4 = ax4.pcolormesh(t, frequencies/1e3, coefficients_db, shading='gouraud', cmap='jet')
    ax4.set_ylim([0, 20])
    ax4.set_xlabel("Zaman (s)", fontsize=12)
    ax4.set_ylabel("Frekans (kHz)", fontsize=12)
    ax4.set_title("WAVELET (CWT - Morlet)\nZaman-Frekans-Ölçek Analizi", fontsize=13, fontweight='bold')
    fig.colorbar(pcm4, ax=ax4, label='Genlik (dB)')
else:
    ax4.axis('off')
    ax4.text(0.05, 0.5, (
        "Wavelet (CWT) için 'pywt' paketine ihtiyaç var.\n"
        "Yüklemek için:\n\n"
        "pip install pywt"
    ), fontsize=12, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.6))

# 5. STFT - Küçük Pencere (İyi Zaman Çözünürlüğü)
ax5 = plt.subplot(2, 4, 5)
pcm5 = ax5.pcolormesh(t_stft_small, f_stft_small/1e3, Zxx_small_db, shading='gouraud', cmap='viridis')
ax5.set_ylim([-20, 20])
ax5.set_xlabel("Zaman (s)", fontsize=12)
ax5.set_ylabel("Frekans (kHz)", fontsize=12)
ax5.set_title(f"STFT - KÜÇÜK PENCERE (nperseg={nperseg_small})\nİyi Zaman - Kötü Frekans Çözünürlüğü", fontsize=13, fontweight='bold')
fig.colorbar(pcm5, ax=ax5, label='Genlik (dB)')

# 6. STFT - Orta Pencere (Dengeli)
ax6 = plt.subplot(2, 4, 6)
pcm6 = ax6.pcolormesh(t_stft_medium, f_stft_medium/1e3, Zxx_medium_db, shading='gouraud', cmap='viridis')
ax6.set_ylim([-20, 20])
ax6.set_xlabel("Zaman (s)", fontsize=12)
ax6.set_ylabel("Frekans (kHz)", fontsize=12)
ax6.set_title(f"STFT - ORTA PENCERE (nperseg={nperseg_medium})\nDengeli Çözünürlük", fontsize=13, fontweight='bold')
fig.colorbar(pcm6, ax=ax6, label='Genlik (dB)')

# 7. STFT - Büyük Pencere (İyi Frekans Çözünürlüğü)
ax7 = plt.subplot(2, 4, 7)
pcm7 = ax7.pcolormesh(t_stft_large, f_stft_large/1e3, Zxx_large_db, shading='gouraud', cmap='viridis')
ax7.set_ylim([-20, 20])
ax7.set_xlabel("Zaman (s)", fontsize=12)
ax7.set_ylabel("Frekans (kHz)", fontsize=12)
ax7.set_title(f"STFT - BÜYÜK PENCERE (nperseg={nperseg_large})\nİyi Frekans - Kötü Zaman Çözünürlüğü", fontsize=13, fontweight='bold')
fig.colorbar(pcm7, ax=ax7, label='Genlik (dB)')

# 8. Karşılaştırma Tablosu
ax8 = plt.subplot(2, 4, 8)
ax8.axis('off')
comparison_text = f"""
SPEKTRAL ANALİZ YÖNTEMLERİ KARŞILAŞTIRMASI

DFT (Discrete Fourier Transform):
• Tüm sinyalin frekans analizi (kompleks)
• ZAMAN bilgisi kaybolur
• Pozitif ve negatif frekanslar
• Frekans çözünürlüğü: {Fs/len(signal_bpsk):.2f} Hz

DCT (Discrete Cosine Transform):
• Sadece kosinüs baz fonksiyonları (gerçel)
• Simetrik sinyaller için verimli
• JPEG, MP3 sıkıştırmasında kullanılır
• Sadece pozitif frekanslar

WAVELET (CWT - Morlet):
• Zaman-frekans-ölçek analizi
• Değişken çözünürlük (multi-resolution)
• Geçici olaylar için mükemmel
• Medikal sinyal, sismik analiz

STFT (Short-Time Fourier Transform):
• Zaman-frekans analizi (pencereli FFT)
• Sabit pencere boyutu
• Heisenberg belirsizlik ilkesi
• Konuşma, müzik analizi

Pencere: Küçük({nperseg_small}), Orta({nperseg_medium}), Büyük({nperseg_large})
"""
ax8.text(0.05, 0.5, comparison_text, fontsize=10, family='monospace', 
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()
plt.show()

print(f"\n=== DFT vs STFT ANALİZİ ===")
print(f"Sinyal Süresi: {len(t)/Fs:.2f} s")
print(f"Örnekleme Frekansı: {Fs/1e3:.1f} kHz")
print(f"\nDFT:")
print(f"  Frekans Çözünürlüğü: {Fs/len(signal_bpsk):.2f} Hz")
print(f"  Toplam FFT Noktası: {len(signal_bpsk)}")
print(f"\nSTFT Pencere Boyutları:")
print(f"  Küçük ({nperseg_small}): Δf = {Fs/nperseg_small:.1f} Hz, Δt = {nperseg_small/Fs*1000:.2f} ms")
print(f"  Orta ({nperseg_medium}): Δf = {Fs/nperseg_medium:.1f} Hz, Δt = {nperseg_medium/Fs*1000:.2f} ms")
print(f"  Büyük ({nperseg_large}): Δf = {Fs/nperseg_large:.1f} Hz, Δt = {nperseg_large/Fs*1000:.2f} ms")
print(f"========================\n")
if not _HAS_PYWT:
    print("Uyarı: Wavelet (CWT) için pywt bulunamadı. 'pip install pywt' ile yükleyebilirsiniz.")
