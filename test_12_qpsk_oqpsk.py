import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# 1. Parametreler (Parameters)
Fs = 100000          # Örnekleme Frekansı (Hz)
sym_rate = 1000      # Sembol Hızı (Symbol Rate)
T_sym = 1/sym_rate   # Sembol Süresi
sps = int(Fs*T_sym)  # Sembol başına örnek (Samples per Symbol)
num_symbols = 20     # Sembol Sayısı

# 2. Veri Üretimi (Data Generation)
np.random.seed(42)
# Rastgele semboller (-1, +1)
bits_I = 2 * np.random.randint(0, 2, num_symbols) - 1
bits_Q = 2 * np.random.randint(0, 2, num_symbols) - 1

# 3. Sinyal Hazırlığı (Signal Prep)
# Upsampling (Araya sıfır ekleme)
# Darbe şekillendirme filtresi uygulayabilmek için önce dürtü (impulse) dizisi oluşturuyoruz.
I_upsampled = np.zeros(num_symbols * sps)
Q_upsampled_qpsk = np.zeros(num_symbols * sps)
Q_upsampled_oqpsk = np.zeros(num_symbols * sps)

# İğne darbelerini (impulses) yerleştir
# QPSK: I ve Q aynı anda (her sembol başında)
# OQPSK: Q, yarım sembol (sps/2) gecikmeli
for i in range(num_symbols):
    idx = i * sps
    I_upsampled[idx] = bits_I[i]
    Q_upsampled_qpsk[idx] = bits_Q[i]
    
    # OQPSK Offset: Yarım sembol kaydır
    idx_offset = i * sps + sps // 2
    if idx_offset < len(Q_upsampled_oqpsk):
        Q_upsampled_oqpsk[idx_offset] = bits_Q[i]

# 4. Darbe Şekillendirme (Pulse Shaping)
# Geçişleri yumuşatmak ve yörüngeyi (trajectory) görmek için filtre
# Basit bir Gaussian veya Blackman penceresi kullanalım
filter_len = 4 * sps
t_filter = np.linspace(-2, 2, filter_len)
# Gaussian benzeri bir filtre (Root Raised Cosine simülasyonu için yumuşak geçiş)
pulse_shape = np.exp(-t_filter**2 * 2) 
pulse_shape /= np.sum(pulse_shape) # Normalize et

# Konvolüsyon (Filtreleme)
I_shaped = convolve(I_upsampled, pulse_shape, mode='same')
Q_qpsk_shaped = convolve(Q_upsampled_qpsk, pulse_shape, mode='same')
Q_oqpsk_shaped = convolve(Q_upsampled_oqpsk, pulse_shape, mode='same')

# Genlik normalizasyonu (Görsel düzgünlük için)
I_shaped /= np.max(np.abs(I_shaped))
Q_qpsk_shaped /= np.max(np.abs(Q_qpsk_shaped))
Q_oqpsk_shaped /= np.max(np.abs(Q_oqpsk_shaped))

# Karmaşık Sinyaller (Complex Signals)
sig_qpsk = I_shaped + 1j * Q_qpsk_shaped
sig_oqpsk = I_shaped + 1j * Q_oqpsk_shaped

# Zaman ekseni
t = np.arange(len(sig_qpsk)) / Fs

# 5. Görselleştirme (Visualization)
fig = plt.figure(figsize=(18, 10))

# --- QPSK Zaman Diyagramı ---
ax1 = plt.subplot(2, 3, 1)
ax1.plot(t*1000, I_shaped, 'b-', label='I (In-phase)')
ax1.plot(t*1000, Q_qpsk_shaped, 'r--', label='Q (Quadrature)')
ax1.set_title("QPSK Zaman Diyagramı\n(I ve Q Eş Zamanlı)")
ax1.set_xlabel("Zaman (ms)")
ax1.set_ylabel("Genlik")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# --- OQPSK Zaman Diyagramı ---
ax2 = plt.subplot(2, 3, 2)
ax2.plot(t*1000, I_shaped, 'b-', label='I (In-phase)')
ax2.plot(t*1000, Q_oqpsk_shaped, 'r--', label='Q (Offset)')
ax2.set_title("Offset QPSK (OQPSK) Zaman Diyagramı\n(Q, T/2 Gecikmeli)")
ax2.set_xlabel("Zaman (ms)")
ax2.set_ylabel("Genlik")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# --- Faz Karşılaştırması ---
ax3 = plt.subplot(2, 3, 3)
phase_qpsk = np.angle(sig_qpsk)
phase_oqpsk = np.angle(sig_oqpsk)
ax3.plot(t*1000, phase_qpsk, 'b-', alpha=0.6, label='QPSK Faz')
ax3.plot(t*1000, phase_oqpsk, 'r-', alpha=0.6, label='OQPSK Faz')
ax3.set_title("Faz Değişimleri")
ax3.set_xlabel("Zaman (ms)")
ax3.set_ylabel("Faz (radyan)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- QPSK Yörüngesi (Trajectory) ---
ax4 = plt.subplot(2, 3, 4)
ax4.plot(np.real(sig_qpsk), np.imag(sig_qpsk), 'b-', alpha=0.7)
ax4.scatter(bits_I, bits_Q, color='black', zorder=5, label='Semboller')
ax4.set_title("QPSK Yörüngesi (Trajectory)\n(Sıfır Geçişi VAR!)")
ax4.set_xlabel("I (In-phase)")
ax4.set_ylabel("Q (Quadrature)")
ax4.axhline(0, color='black', alpha=0.3)
ax4.axvline(0, color='black', alpha=0.3)
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)
ax4.grid(True, alpha=0.3)
# Sıfır noktasını işaretle
circle = plt.Circle((0, 0), 0.1, color='red', fill=False, linestyle='--', linewidth=2)
ax4.add_artist(circle)
ax4.text(0.1, 0.1, "Sıfır Geçişi", color='red', fontsize=9)

# --- OQPSK Yörüngesi (Trajectory) ---
ax5 = plt.subplot(2, 3, 5)
ax5.plot(np.real(sig_oqpsk), np.imag(sig_oqpsk), 'r-', alpha=0.7)
ax5.scatter(bits_I, bits_Q, color='black', zorder=5, label='Semboller')
ax5.set_title("OQPSK Yörüngesi (Trajectory)\n(Sıfır Geçişi YOK)")
ax5.set_xlabel("I (In-phase)")
ax5.set_ylabel("Q (Quadrature)")
ax5.axhline(0, color='black', alpha=0.3)
ax5.axvline(0, color='black', alpha=0.3)
ax5.set_xlim(-1.5, 1.5)
ax5.set_ylim(-1.5, 1.5)
ax5.grid(True, alpha=0.3)

# --- Açıklama ---
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
info_text = """
QPSK vs Offset QPSK (OQPSK)

QPSK:
• I ve Q kanalları aynı anda değişir.
• Eğer I ve Q aynı anda işaret değiştirirse
  (örn: 1+j -> -1-j), sinyal orijinden (0,0) geçer.
• Bu "Sıfır Geçişi" (Zero Crossing), RF yükselteçlerde
  genlik değişimine (AM) neden olur.
• Lineer olmayan yükselteçlerde spektral
  yayılmaya (regrowth) yol açar.

OQPSK:
• Q kanalı, I kanalına göre yarım sembol
  süresi (T/2) kadar geciktirilir.
• I ve Q asla aynı anda değişmez.
• Faz değişimleri maksimum ±90° olur.
• Sinyal asla orijinden (0,0) geçmez.
• Genlik dalgalanması daha azdır.
• Güç yükselteçleri (PA) için daha verimlidir.
"""
ax6.text(0, 0.5, info_text, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.show()
