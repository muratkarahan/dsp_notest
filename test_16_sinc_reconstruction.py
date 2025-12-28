import numpy as np
import matplotlib.pyplot as plt

# 1. Parametreler (Parameters)
f_sig = 10000        # Sinyal Frekansı: 10 kHz
Fs = 300000          # Örnekleme Frekansı: 300 kHz
T_duration = 0.0004  # Toplam Süre: 0.4 ms (4 periyot)

# 2. Analog Sinyal (Yüksek Çözünürlüklü)
# Çizimlerin pürüzsüz olması için çok yüksek örnekleme kullanıyoruz
Fs_analog = 10000000 # 10 MHz
t_analog = np.arange(0, T_duration, 1/Fs_analog)
x_analog = np.cos(2 * np.pi * f_sig * t_analog)

# 3. Örnekleme (Sampling)
Ts = 1 / Fs
t_samples = np.arange(0, T_duration, Ts)
x_samples = np.cos(2 * np.pi * f_sig * t_samples)

# 4. Sinc İnterpolasyonu (Reconstruction)
# x_r(t) = sum( x[n] * sinc( (t - n*Ts) / Ts ) )
x_reconstructed = np.zeros_like(t_analog)

for i, (ts, xs) in enumerate(zip(t_samples, x_samples)):
    # Her örnek için Sinc katkısı
    # NumPy sinc: sin(pi*x)/(pi*x)
    sinc_pulse = xs * np.sinc((t_analog - ts) / Ts)
    x_reconstructed += sinc_pulse

# 5. Görselleştirme (Visualization)
# "Grafikleri ayrı plotlar ile çiz" isteği üzerine subplots kullanıyoruz
fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)

# --- Grafik 1: Orijinal Analog Sinyal ---
axes[0].plot(t_analog * 1e6, x_analog, 'b-', linewidth=2)
axes[0].set_title(f"1. Orijinal Analog Sinyal (10 kHz Cosine)\n(Sürekli Zaman)")
axes[0].set_xlabel("Zaman (µs)")
axes[0].set_ylabel("Genlik")
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(0, T_duration*1e6)

# --- Grafik 2: Örneklenmiş Sinyal ---
axes[1].stem(t_samples * 1e6, x_samples, linefmt='r-', markerfmt='ro', basefmt=' ')
# Referans için silik analog sinyal
axes[1].plot(t_analog * 1e6, x_analog, 'b--', alpha=0.2)
axes[1].set_title(f"2. Örneklenmiş Sinyal\nFs = {Fs/1000:.0f} kHz (Ts = {Ts*1e6:.2f} µs)")
axes[1].set_xlabel("Zaman (µs)")
axes[1].set_ylabel("Genlik")
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(0, T_duration*1e6)

# --- Grafik 3: Sinc ile Yeniden Oluşturma ---
axes[2].plot(t_analog * 1e6, x_reconstructed, 'g-', linewidth=2, label='Reconstructed (Sinc Sum)')
axes[2].plot(t_analog * 1e6, x_analog, 'b--', linewidth=1, alpha=0.5, label='Original Reference')
axes[2].set_title("3. Sinc İnterpolasyonu ile Yeniden Oluşturulmuş Sinyal")
axes[2].set_xlabel("Zaman (µs)")
axes[2].set_ylabel("Genlik")
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)
axes[2].set_xlim(0, T_duration*1e6)

# Açıklama
info_text = f"""
Parametreler:
• Sinyal: 10 kHz Cosine
• Örnekleme (Fs): 300 kHz
• Periyot Başına Örnek: {Fs/f_sig:.0f}

Sonuç:
300 kHz örnekleme, Nyquist sınırının
(20 kHz) çok üzerindedir.
Sinc interpolasyonu sinyali mükemmel
şekilde geri oluşturur.
"""
fig.text(0.02, 0.02, info_text, fontsize=11, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.show()
