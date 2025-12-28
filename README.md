# DSP (Sayısal Sinyal İşleme) Çalışma Dosyaları

Bu proje, çeşitli DSP kavramlarını, modülasyon tekniklerini ve spektral analiz yöntemlerini görselleştirmek için oluşturulmuş Python betiklerini içerir.

## Dosya Listesi ve Açıklamalar

### Temel Sinyal İşleme ve GUI
*   **`test_00_signal_gen_qt.py`**: PyQt6 tabanlı sinyal üreteci arayüzü.
*   **`test_01_adc_model_gui.py`**: ADC (Analog-Dijital Dönüştürücü) modellemesi (doğrusal olmayanlık, kırpma, kuantalama) ve GUI.
*   **`test_02_fft_spectrum_gui.py`**: Gerçek zamanlı FFT spektrum analizi arayüzü.
*   **`test_03_iir_lpf_gui.py`**: IIR Alçak Geçiren Filtre (Low Pass Filter) tasarımı ve görselleştirmesi.
*   **`test_04_signal_gen_qt_v2.py`**: Sinyal üreteci arayüzünün geliştirilmiş versiyonu.

### Modülasyon Teknikleri
*   **`test_06_bpsk_msk_phase.py`**: BPSK ve MSK modülasyonlarının faz davranışlarının frekans domeninde karşılaştırılması.
*   **`test_07_fsk_cpfsk_compare.py`**: FSK (Frekans Kaydırmalı Anahtarlama) ve CPFSK (Sürekli Fazlı FSK) karşılaştırması. Anlık frekans ve faz sürekliliği analizi.
*   **`test_09_bpsk_thermal_noise.py`**: Termal gürültü altında BPSK modülasyonu. C/N (Carrier-to-Noise) oranı ve dB/Hz hesaplamaları.
*   **`test_11_msk_demod.py`**: MSK (Minimum Shift Keying) modülasyonu ve faz bazlı demodülasyon simülasyonu. BER (Bit Hata Oranı) hesabı.
*   **`test_12_qpsk_oqpsk.py`**: QPSK ve Offset QPSK (OQPSK) karşılaştırması. Yörünge (trajectory) ve sıfır geçişi analizleri.
*   **`test_13_msk_halfsine.py`**: MSK modülasyonunda kullanılan "Half-sine" frekans darbesinin ve anlık frekans değişiminin görselleştirilmesi.
*   **`test_17_ofdm_cp.py`**: OFDM (Orthogonal Frequency Division Multiplexing) simülasyonu. Cyclic Prefix (CP) ekleme, QPSK haritalama, IFFT/FFT işlemleri ve bit hata kontrolü.

### Spektral Analiz ve Dönüşümler
*   **`test_05_hilbert_mixing.py`**: Hilbert dönüşümü ve sinyal karıştırma (mixing) işlemleri.
*   **`test_08_hilbert_fft_phase.py`**: Hilbert dönüşümü ile elde edilen fazın FFT fazı ile karşılaştırılması.
*   **`test_10_spectral_analysis.py`**: Farklı spektral analiz yöntemlerinin (DFT, STFT, DCT, Wavelet/CWT) karşılaştırılması.

### Örnekleme Teorisi (Sampling Theory)
*   **`test_14_sinc.py`**: Sinc fonksiyonunun zaman ve frekans (dikdörtgen spektrum) domeninde incelenmesi.
*   **`test_15_sinc_sampling.py`**: Whittaker-Shannon interpolasyon formülü ile örneklenmiş sinyalin Sinc fonksiyonları kullanılarak yeniden oluşturulması.
*   **`test_16_sinc_reconstruction.py`**: Yüksek frekanslı örnekleme ve Sinc interpolasyonu ile sinyal inşası örneği.

## Gereksinimler

Bu betikleri çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyacınız vardır:

```bash
pip install numpy matplotlib scipy PyQt6 PyWavelets
```

## Kullanım

Herhangi bir testi çalıştırmak için terminalde ilgili dosya adını girin:

```bash
python test_17_ofdm_cp.py
```
