"""Generate sample_clean.wav and sample_noisy.wav for a quick demo (no upload needed)."""
from signal_processing import make_demo_signal, add_white_noise, save_wav

sr, clean = make_demo_signal(sr=16000, duration=2.0)
noisy, _ = add_white_noise(clean, snr_db=5)

save_wav("sample_clean.wav", sr, clean)
save_wav("sample_noisy.wav", sr, noisy)
print("Wrote sample_clean.wav and sample_noisy.wav (16 kHz, 2 s, 440+880+1320 Hz)")
