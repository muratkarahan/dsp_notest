import numpy as np
import matplotlib.pyplot as plt

# 1. Parametreler (Parameters)
N = 64              # Alt taşıyıcı sayısı (FFT boyutu)
Ncp = 16            # Cyclic Prefix (CP) uzunluğu
num_symbols = 3     # OFDM sembol sayısı
mod_order = 4       # QPSK (4-QAM)
num_bits_per_symbol = int(np.log2(mod_order)) # 2 bit/sembol

# 2. Veri Üretimi (Data Generation)
np.random.seed(42)
total_subcarriers = N * num_symbols
total_bits = total_subcarriers * num_bits_per_symbol

# Rastgele Bit Dizisi (0 veya 1)
tx_bits = np.random.randint(0, 2, total_bits)

# 3. Modülasyon (QPSK Mapping)
# Bitleri sembollere dönüştür (2'şerli grupla)
# 00 -> 0, 01 -> 1, 10 -> 2, 11 -> 3 (Decimal)
tx_bits_reshaped = tx_bits.reshape(-1, num_bits_per_symbol)
# Bitleri decimal değere çevir: [b0, b1] -> b0*2 + b1
symbol_indices = tx_bits_reshaped.dot(1 << np.arange(num_bits_per_symbol)[::-1])

# QPSK Haritalama Tablosu (Gray Coding tercih edilir ama basitlik için düz)
# 0 (00) -> 1+j
# 1 (01) -> -1+j
# 2 (10) -> -1-j
# 3 (11) -> 1-j
constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
qpsk_symbols = constellation[symbol_indices]

# Veriyi OFDM sembollerine böl (Serial to Parallel)
ofdm_data_freq = qpsk_symbols.reshape(num_symbols, N)

# 4. IFFT (Frekans -> Zaman)
ofdm_time_domain = np.fft.ifft(ofdm_data_freq, axis=1)

# 5. Cyclic Prefix Ekleme (Add CP)
cp = ofdm_time_domain[:, -Ncp:]
ofdm_with_cp = np.hstack([cp, ofdm_time_domain])

# 6. Seri Hale Getirme (Parallel to Serial) - TX Sinyali
tx_signal = ofdm_with_cp.flatten()

# --- KANAL (Channel) ---
# Şimdilik gürültüsüz (Noiseless) ve sönümlemesiz (No fading)
rx_signal = tx_signal

# 7. ALICI (Receiver)
# Seri -> Paralel
rx_parallel = rx_signal.reshape(num_symbols, N + Ncp)

# CP Kaldırma (Remove CP)
rx_no_cp = rx_parallel[:, Ncp:]

# FFT (Zaman -> Frekans)
rx_data_freq = np.fft.fft(rx_no_cp, axis=1)

# Paralel -> Seri
rx_symbols = rx_data_freq.flatten()

# 8. Demodülasyon (Demapping)
# En yakın komşu kararı (Minimum Distance)
# QPSK için basitçe quadrant kontrolü yapılabilir
# Reel > 0, Imag > 0 -> 0 (00)
# Reel < 0, Imag > 0 -> 1 (01)
# Reel < 0, Imag < 0 -> 2 (10)
# Reel > 0, Imag < 0 -> 3 (11)

rx_indices = np.zeros(len(rx_symbols), dtype=int)
# Bu basit mantık yukarıdaki constellation tanımına uymalı:
# 0: 1+j (R>0, I>0)
# 1: -1+j (R<0, I>0)
# 2: -1-j (R<0, I<0)
# 3: 1-j (R>0, I<0)

for i, sym in enumerate(rx_symbols):
    if sym.real >= 0 and sym.imag >= 0:
        rx_indices[i] = 0
    elif sym.real < 0 and sym.imag >= 0:
        rx_indices[i] = 1
    elif sym.real < 0 and sym.imag < 0:
        rx_indices[i] = 2
    else:
        rx_indices[i] = 3

# Sembol İndekslerini Bitlere Çevir
# Decimal -> Binary
rx_bits = np.zeros(total_bits, dtype=int)
for i, idx in enumerate(rx_indices):
    # 2 bitlik değer
    b0 = (idx >> 1) & 1
    b1 = idx & 1
    rx_bits[2*i] = b0
    rx_bits[2*i+1] = b1

# Hata Kontrolü
bit_errors = np.sum(tx_bits != rx_bits)

# 9. Görselleştirme
fig = plt.figure(figsize=(16, 12))

# --- A) Zaman Bölgesi Sinyali ---
ax1 = plt.subplot(3, 1, 1)
t = np.arange(len(tx_signal))
ax1.plot(t, np.real(tx_signal), 'b-', linewidth=1.5, label='Reel')
ax1.plot(t, np.imag(tx_signal), 'r--', linewidth=1, alpha=0.5, label='Sanal')
# CP Bölgeleri
symbol_len = N + Ncp
for i in range(num_symbols):
    start_idx = i * symbol_len
    ax1.axvspan(start_idx, start_idx + Ncp, color='yellow', alpha=0.3)
    if i == 0: ax1.text(start_idx + Ncp/2, np.max(np.real(tx_signal))*0.8, "CP", ha='center', fontsize=8)
    ax1.axvline(start_idx + symbol_len, color='k', linestyle=':', alpha=0.3)

ax1.set_title(f"OFDM Zaman Sinyali (N={N}, CP={Ncp})")
ax1.set_ylabel("Genlik")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, len(tx_signal))

# --- B) Bit Akışları (İlk 50 bit) ---
ax2 = plt.subplot(3, 1, 2)
n_bits_show = 50
ax2.step(np.arange(n_bits_show), tx_bits[:n_bits_show] + 1.5, 'b-', where='mid', label='TX Bits (Modüle Edilen)')
ax2.step(np.arange(n_bits_show), rx_bits[:n_bits_show], 'g--', where='mid', label='RX Bits (Demodüle Edilen)')
# Hata varsa işaretle
error_idx = np.where(tx_bits[:n_bits_show] != rx_bits[:n_bits_show])[0]
if len(error_idx) > 0:
    ax2.plot(error_idx, rx_bits[error_idx], 'rx', markersize=10, label='Hata')

ax2.set_yticks([0, 1, 1.5, 2.5])
ax2.set_yticklabels(['0', '1', '0', '1'])
ax2.text(-2, 0.5, "RX", fontsize=12, fontweight='bold', color='green')
ax2.text(-2, 2.0, "TX", fontsize=12, fontweight='bold', color='blue')
ax2.set_title(f"Bit Akışı Karşılaştırması (İlk {n_bits_show} bit) - Toplam Hata: {bit_errors}")
ax2.set_xlabel("Bit İndeksi")
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-0.5, 3)

# --- C) Spektrum ve Takımyıldızı (Yan Yana) ---
# Alt kısmı ikiye böl
ax3 = plt.subplot(3, 2, 5)
f_axis = np.fft.fftshift(np.fft.fftfreq(len(tx_signal)))
spectrum = 20 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(tx_signal))))
ax3.plot(f_axis, spectrum, 'k-', linewidth=1)
ax3.set_title("OFDM Spektrumu")
ax3.set_xlabel("Normalize Frekans")
ax3.set_ylabel("dB")
ax3.grid(True, alpha=0.3)

ax4 = plt.subplot(3, 2, 6)
ax4.scatter(np.real(rx_symbols), np.imag(rx_symbols), c='blue', marker='x', label='Alınan Semboller')
# Orijinal noktaları da göster
ax4.scatter(np.real(constellation), np.imag(constellation), c='red', marker='o', alpha=0.5, label='Referans')
ax4.set_title("RX Takımyıldızı (Constellation)")
ax4.set_xlabel("I")
ax4.set_ylabel("Q")
ax4.grid(True, alpha=0.3)
ax4.legend()
ax4.set_xlim(-1.5, 1.5)
ax4.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.show()
