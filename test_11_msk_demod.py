import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# 1. Parametreler (Parameters)
Fs = 40000          # Sampling Frequency (Hz)
bit_rate = 1000     # Bit Rate (bps)
fc = 10000          # Carrier Frequency (Hz)
num_bits = 20       # Bit Count
h = 0.5             # Modulation Index (MSK için 0.5)

# 2. Data Generation (Veri Üretimi)
np.random.seed(42)
data_bits = np.random.randint(0, 2, num_bits)

# 3. MSK Signal Generation
T = 1 / bit_rate    # Bit Duration
samples_per_bit = int(Fs * T)
t = np.arange(0, num_bits * T, 1/Fs)

# Frequency Deviation: h = 2 * delta_f * T  => delta_f = h / (2*T)
delta_f = h * bit_rate / 2

# Basit MSK üretimi (Phase Accumulation yöntemi)
phase_acc = 0
signal_msk = np.zeros(len(t))
dt = 1/Fs

for i in range(len(t)):
    # Hangi bit? (Current Bit)
    bit_idx = int(t[i] * bit_rate)
    if bit_idx >= num_bits: bit_idx = num_bits - 1
    
    bit = data_bits[bit_idx]
    # Instantaneous Frequency
    f_inst = fc + (delta_f if bit == 1 else -delta_f)
    
    # Phase Update
    phase_acc += 2 * np.pi * f_inst * dt
    signal_msk[i] = np.cos(phase_acc)

# 4. Demodulation (Phase Based)
# Analytic Signal ve Phase eldesi
analytic_signal = hilbert(signal_msk)
received_phase = np.unwrap(np.angle(analytic_signal))

# Her bit süresince Phase Change hesabı
decoded_bits = np.zeros(num_bits, dtype=int)
sampled_phases = [] # (t_start, t_end, phase_diff)

for i in range(num_bits):
    # Bit sınırları (sample cinsinden)
    idx_start = i * samples_per_bit
    idx_end = (i + 1) * samples_per_bit
    if idx_end > len(received_phase): idx_end = len(received_phase) - 1
    
    # Start ve End Phase değerleri
    phi_start = received_phase[idx_start]
    phi_end = received_phase[idx_end-1]
    
    phase_diff = phi_end - phi_start
    
    # Measured Time Interval
    dt_measured = t[idx_end-1] - t[idx_start]
    
    # Carrier kaynaklı beklenen Phase Change
    carrier_phase_change = 2 * np.pi * fc * dt_measured
    
    # Baseband Phase Difference (Carrier Removal)
    baseband_phase_diff = phase_diff - carrier_phase_change
    
    sampled_phases.append((t[idx_start], t[idx_end-1], baseband_phase_diff))
    
    # Decision: Pozitif -> 1, Negatif -> 0
    if baseband_phase_diff > 0:
        decoded_bits[i] = 1
    else:
        decoded_bits[i] = 0

# Error Calculation
bit_errors = np.sum(data_bits != decoded_bits)
ber = bit_errors / num_bits

print(f"MSK Simulation tamamlandı.")
print(f"Bit Rate: {bit_rate} bps, Carrier Freq (Fc): {fc} Hz")
print(f"Bit Error Rate (BER): {ber:.2%} ({bit_errors}/{num_bits} error)")
print("Bit başına Phase Difference değerleri (ilk 10):")
for i in range(min(10, num_bits)):
    print(f"Bit {i}: {data_bits[i]} -> dPhi: {sampled_phases[i][2]:.3f} rad")

# 5. Visualization
"""
Görselleştirme: 2×2
"""

fig = plt.figure(figsize=(16, 10))

# 1) MSK Signal (Time Domain)
ax1 = plt.subplot(2, 2, 1)
n_plot = min(len(t), int(5e-3 * Fs))
ax1.plot(t[:n_plot]*1e3, signal_msk[:n_plot], 'r-', linewidth=1.5)
ax1.set_xlabel("Time (ms)")
ax1.set_ylabel("Amplitude")
ax1.set_title(f"MSK Signal (fc={fc/1e3:.0f} kHz, Δf={delta_f:.0f} Hz)")
ax1.grid(True, alpha=0.3)

# 2) Bit başına Phase Difference
ax2 = plt.subplot(2, 2, 2)
bit_centers = np.arange(num_bits) + 0.5
phase_diffs = [sp[2] for sp in sampled_phases]
ax2.bar(bit_centers, phase_diffs, width=0.8, alpha=0.8,
        color=["blue" if pd > 0 else "red" for pd in phase_diffs])
ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
ax2.set_xlabel("Bit Index")
ax2.set_ylabel("Δφ (radians)")
ax2.set_title("Bit başına Baseband Phase Difference (Carrier Removed)")
ax2.grid(True, axis='y', alpha=0.3)

# 3) Original vs Decoded Bits
ax3 = plt.subplot(2, 2, 3)
ax3.step(range(num_bits), data_bits, 'b--', linewidth=2, where='post', label='Original')
ax3.step(range(num_bits), decoded_bits, 'g-', linewidth=3, where='post', label='Decode')
error_indices = np.where(data_bits != decoded_bits)[0]
if len(error_indices) > 0:
    ax3.plot(error_indices, decoded_bits[error_indices], 'ro', markersize=8, label='Error')
ax3.set_xlabel("Bit Index")
ax3.set_ylabel("Bit Value")
ax3.set_title(f"Decoding Result (BER={ber:.2%})")
ax3.set_ylim([-0.5, 1.5])
ax3.set_yticks([0, 1])
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4) Summary
ax4 = plt.subplot(2, 2, 4)
ax4.axis('off')
summary = (
    f"MSK Phase-Based Decoding\n"
    f"Bits: {num_bits}, Bit Rate: {bit_rate/1e3:.1f} kbps\n"
    f"Carrier Freq: {fc/1e3:.0f} kHz, Δf: {delta_f:.0f} Hz\n"
    f"BER: {ber:.2%} ({bit_errors}/{num_bits})\n"
    f"Expected Δφ: ±{2*np.pi*delta_f/bit_rate:.3f} rad"
)
ax4.text(0.05, 0.5, summary, fontsize=11, family='monospace',
         verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

plt.tight_layout()
plt.show()
