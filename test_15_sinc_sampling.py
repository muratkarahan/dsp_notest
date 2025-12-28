import numpy as np
import matplotlib.pyplot as plt

# 1. Parametreler (Parameters)
T_total = 1.0        # Toplam süre (s)
Fs_analog = 1000     # Analog sinyali simüle etmek için yüksek örnekleme (Hz)
Fs_sample = 20       # Örnekleme Frekansı (Hz) (Nyquist'e dikkat!)
Ts = 1 / Fs_sample   # Örnekleme Periyodu

# 2. Analog Sinyal Oluşturma (Simulated Analog Signal)
t_analog = np.linspace(0, T_total, int(Fs_analog * T_total), endpoint=False)

# Örnek Sinyal: İki sinüsün toplamı
# f1 = 2 Hz, f2 = 5 Hz (Maksimum frekans 5 Hz, Nyquist için Fs > 10 Hz olmalı. Biz 20 Hz seçtik.)
def analog_func(t):
    return 1.0 * np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t)

x_analog = analog_func(t_analog)

# 3. Örnekleme (Sampling)
# Fs_sample frekansında örnekler alıyoruz
t_samples = np.arange(0, T_total, Ts)
x_samples = analog_func(t_samples)

# 4. Sinc İnterpolasyonu ile Yeniden Oluşturma (Reconstruction)
# Whittaker-Shannon İnterpolasyon Formülü:
# xr(t) = sum( x[n] * sinc( (t - n*Ts) / Ts ) )

x_reconstructed = np.zeros_like(t_analog)
sinc_components = []

# Her bir örnek için bir Sinc fonksiyonu oluşturup topluyoruz
for i, (t_s, x_s) in enumerate(zip(t_samples, x_samples)):
    # Sinc fonksiyonunu zamanda kaydır (t - t_s) ve ölçekle (x_s)
    # NumPy sinc tanımı: sinc(x) = sin(pi*x)/(pi*x)
    # Formüldeki argüman: (t - t_s) / Ts
    
    sinc_pulse = x_s * np.sinc((t_analog - t_s) / Ts)
    
    # Toplama ekle
    x_reconstructed += sinc_pulse
    
    # Görselleştirme için sakla (sadece ilk birkaç tanesini veya hepsini)
    sinc_components.append(sinc_pulse)

# 5. Görselleştirme
fig = plt.figure(figsize=(14, 8))

# --- Ana Grafik ---
ax = plt.subplot(1, 1, 1)

# 1. Orijinal Analog Sinyal
ax.plot(t_analog, x_analog, 'k-', linewidth=1.5, alpha=0.3, label='Orijinal Analog Sinyal')

# 2. Örnekler (Samples)
ax.stem(t_samples, x_samples, linefmt='b-', markerfmt='bo', basefmt=' ', label='Örnekler (Samples)')

# 3. Tekil Sinc Bileşenleri (Individual Sinc Pulses)
# Kalabalık olmaması için alpha düşük tutulur
for i, pulse in enumerate(sinc_components):
    # Sadece genliği belirgin olanları veya hepsini çizdirebiliriz
    ax.plot(t_analog, pulse, 'g--', linewidth=1, alpha=0.3)

# 4. Yeniden Oluşturulmuş Sinyal (Reconstructed)
ax.plot(t_analog, x_reconstructed, 'r-', linewidth=2, label='Sinc Toplamı (Reconstructed)')

ax.set_title(f"Örnekleme ve Sinc İnterpolasyonu ile Yeniden Oluşturma\nFs = {Fs_sample} Hz")
ax.set_xlabel("Zaman (s)")
ax.set_ylabel("Genlik")
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

# Sadece bir Sinc darbesini vurgulamak için açıklama (Opsiyonel)
# İlk örneğin Sinc darbesini işaret edelim
if len(t_samples) > 0:
    ax.annotate('Tekil Sinc Darbesi', xy=(t_samples[1], x_samples[1]), xytext=(t_samples[1]+0.1, x_samples[1]+0.5),
                arrowprops=dict(facecolor='green', shrink=0.05, alpha=0.5), color='green')

# Açıklama Kutusu
info_text = """
Whittaker-Shannon İnterpolasyonu:
Her örnek nokta (mavi nokta), bir Sinc fonksiyonunun
(yeşil kesikli çizgi) merkezini ve genliğini belirler.

Bu Sinc fonksiyonlarının toplamı (kırmızı çizgi),
orijinal analog sinyali (siyah) kusursuz şekilde
yeniden oluşturur (Eğer Nyquist kriteri sağlanıyorsa).

x_r(t) = Σ x[n] * sinc((t - nT)/T)
"""
fig.text(0.02, 0.02, info_text, fontsize=11, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.show()
