import numpy as np
import matplotlib.pyplot as plt

# 1. Parametreler (Parameters)
Fs = 100000          # Örnekleme Frekansı (Hz)
bit_rate = 1000      # Bit Hızı (bps)
T = 1 / bit_rate     # Bit Süresi (s)
h = 0.5              # Modülasyon İndeksi
num_bits = 24        # Gösterilecek bit sayısı

# 2. Veri (Data)
np.random.seed(42)
bits = np.random.choice([1, -1], num_bits)

# 3. Zaman Ekseni (Time Axis)
# Her bit için zaman vektörü
t_bit = np.linspace(0, T, int(Fs*T), endpoint=False)
# Toplam zaman vektörü
t_total = np.linspace(0, num_bits*T, int(Fs*T)*num_bits, endpoint=False)

# 4. Half-sine Pulse (Frekans Darbesi)
# Formül: g(t) = sin(pi * t / (2*T))  for 0 <= t <= T
g_t = np.sin(np.pi * t_bit / (2*T))

# 5. Anlık Frekans (Instantaneous Frequency)
# Formül: fi(t) = (h / (2*T)) * bk * g(t)
# Not: Bu formül SFSK (Sinusoidal Frequency Shift Keying) benzeri bir yapı oluşturur.
# Standart MSK'de frekans darbesi dikdörtgendir (Rectangular).
# Ancak görseldeki formülü uyguluyoruz.

freq_deviation_scale = h / (2*T)
fi_total = np.array([])

for b in bits:
    # Her bit için frekans darbesi: bk * g(t)
    fi_chunk = freq_deviation_scale * b * g_t
    fi_total = np.concatenate((fi_total, fi_chunk))

# 6. Faz (Phase)
# Frekansın integrali: phi(t) = 2 * pi * integral(fi(t))
phase_total = 2 * np.pi * np.cumsum(fi_total) / Fs

# 7. Görselleştirme
fig = plt.figure(figsize=(14, 10))

# --- A) Tekil Half-sine Pulse ---
ax1 = plt.subplot(2, 2, 1)
ax1.plot(t_bit*1000, g_t, 'b-', linewidth=2)
ax1.set_title(r"Half-sine Pulse $g(t) = \sin(\frac{\pi t}{2T})$")
ax1.set_xlabel("Zaman (ms)")
ax1.set_ylabel("Genlik")
ax1.grid(True, alpha=0.3)
ax1.fill_between(t_bit*1000, g_t, alpha=0.1, color='blue')

# --- B) Bit Dizisi ---
ax2 = plt.subplot(2, 2, 2)
ax2.step(np.arange(num_bits), bits, where='post', color='k', linewidth=2)
ax2.set_title("Bit Dizisi ($b_k$)")
ax2.set_xlabel("Bit İndeksi")
ax2.set_ylabel("Değer ($\pm 1$)")
ax2.set_ylim(-1.5, 1.5)
ax2.set_yticks([-1, 1])
ax2.grid(True, alpha=0.3)
for i in range(num_bits):
    ax2.text(i+0.5, bits[i]*1.2, str(bits[i]), ha='center', color='red')

# --- C) Anlık Frekans (Instantaneous Frequency) ---
ax3 = plt.subplot(2, 1, 2)
ax3.plot(t_total*1000, fi_total, 'r-', linewidth=2)
ax3.set_title(r"Anlık Frekans $f_i(t) = \frac{h}{2T} b_k g(t)$")
ax3.set_xlabel("Zaman (ms)")
ax3.set_ylabel("Frekans Sapması (Hz)")
ax3.grid(True, alpha=0.3)

# Bit sınırlarını çiz
for i in range(num_bits + 1):
    ax3.axvline(i*T*1000, color='gray', linestyle='--', alpha=0.5)

# Açıklama Kutusu
info_text = f"""
Görseldeki Formüller:
• Bit Süresi (T): {T*1000:.1f} ms
• Modülasyon İndeksi (h): {h}
• Frekans Ölçeği (h/2T): {freq_deviation_scale:.1f} Hz

Bu darbe şekli (Half-sine), frekans geçişlerini
yumuşatarak spektral yan bantları bastırır.
Standart MSK'de dikdörtgen (Rectangular) darbe
kullanılırken, bu şekil SFSK (Sinusoidal FSK)
olarak da bilinir.
"""
fig.text(0.02, 0.02, info_text, fontsize=11, family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.show()
