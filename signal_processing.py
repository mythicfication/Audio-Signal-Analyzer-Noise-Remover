"""
Audio Signal Analyzer & Noise Remover — core DSP functions
Signal & Systems project | Concepts: Fourier Transform, FFT, Filters

All functions are pure NumPy / SciPy — no UI code here, so they can be
imported in a notebook, a script, or the Streamlit app.
"""
import numpy as np
from scipy import signal
from scipy.io import wavfile


# ---------------------------------------------------------------- load
def load_audio(path_or_file):
    """
    Load an audio file as mono float32 in [-1, 1].
    Tries librosa (handles MP3/WAV/OGG) first, falls back to scipy (WAV only).
    Returns (sample_rate, samples).
    """
    try:
        import librosa
        y, sr = librosa.load(path_or_file, sr=None, mono=True)
        return sr, y.astype(np.float32)
    except Exception:
        # Fallback: WAV via scipy
        sr, data = wavfile.read(path_or_file)
        if data.ndim > 1:                      # stereo -> mono
            data = data.mean(axis=1)
        if data.dtype == np.int16:
            y = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            y = data.astype(np.float32) / 2147483648.0
        else:
            y = data.astype(np.float32)
            peak = np.max(np.abs(y)) or 1.0
            y = y / peak
        return sr, y


def save_wav(path, sr, y):
    """Save float samples in [-1, 1] as 16-bit WAV."""
    y16 = np.clip(y, -1.0, 1.0)
    y16 = (y16 * 32767).astype(np.int16)
    wavfile.write(path, sr, y16)


# ---------------------------------------------------------------- FFT
def compute_fft(y, sr):
    """
    FFT of a real signal (rfft — only positive frequencies).
    Returns (freqs_Hz, magnitude_dB).
    Magnitude is normalised by N so different lengths are comparable.
    """
    n = len(y)
    window = np.hanning(n)               # reduces spectral leakage
    spectrum = np.fft.rfft(y * window)
    freqs = np.fft.rfftfreq(n, d=1 / sr)
    magnitude = np.abs(spectrum) / (n / 2)
    magnitude_db = 20 * np.log10(magnitude + 1e-12)
    return freqs, magnitude_db


def dominant_frequencies(y, sr, top_n=5):
    """Return the top-N peaks of the spectrum as [(freq_Hz, mag_dB), ...]."""
    freqs, mag_db = compute_fft(y, sr)
    peaks, _ = signal.find_peaks(mag_db, distance=5, prominence=3)
    peaks = peaks[mag_db[peaks] > -60]   # ignore the -170 dB numerical floor
    if len(peaks) == 0:
        return []
    order = peaks[np.argsort(mag_db[peaks])[::-1][:top_n]]
    return [(float(freqs[i]), float(mag_db[i])) for i in sorted(order)]


# ---------------------------------------------------------------- noise
def add_white_noise(y, snr_db):
    """
    Add white Gaussian noise at a target SNR (dB).
    SNR = 10*log10(P_signal / P_noise)
    Returns (noisy_signal, noise).
    """
    rng = np.random.default_rng(42)      # fixed seed -> reproducible demo
    p_signal = np.mean(y ** 2)
    p_noise = p_signal / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(p_noise), size=y.shape)
    return y + noise, noise


def add_hum(y, sr, freq=50.0, amplitude=0.15):
    """Add mains hum (50 Hz sine) — a classic narrowband interference."""
    t = np.arange(len(y)) / sr
    return y + amplitude * np.sin(2 * np.pi * freq * t)


# ---------------------------------------------------------------- filters
def butter_filter(y, sr, cutoff, filter_type="lowpass", order=4, highcut=None):
    """
    Butterworth filter applied with filtfilt (zero-phase, no delay).
    filter_type: 'lowpass' | 'highpass' | 'bandpass'
    cutoff: Hz (lowpass/highpass) or (low, high) for bandpass.
    """
    nyq = sr / 2
    if filter_type == "bandpass":
        low, high = cutoff if isinstance(cutoff, (tuple, list)) else (cutoff, highcut)
        wn = [low / nyq, high / nyq]
    else:
        wn = cutoff / nyq
    b, a = signal.butter(order, wn, btype=filter_type)
    return signal.filtfilt(b, a, y)


def fft_lowpass(y, sr, cutoff):
    """
    Ideal low-pass in the frequency domain (for comparison with Butterworth).
    Zero out every FFT bin above the cutoff, then inverse FFT.
    Shows clearly why FFT filtering causes ringing — the sharp cut in
    frequency is a sinc in time.
    """
    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1 / sr)
    spectrum[freqs > cutoff] = 0
    return np.fft.irfft(spectrum, n=len(y)).real.astype(np.float32)


# ---------------------------------------------------------------- metrics
def snr_db(clean, processed):
    """SNR of `processed` relative to `clean` (dB). Higher = cleaner."""
    noise = processed - clean
    p_signal = np.mean(clean ** 2)
    p_noise = np.mean(noise ** 2) + 1e-12
    return float(10 * np.log10(p_signal / p_noise))


def mse(a, b):
    return float(np.mean((a - b) ** 2))


def make_demo_signal(sr=16000, duration=2.0):
    """
    Clean demo signal: 440 Hz + 880 Hz + 1320 Hz tones.
    The 440 Hz tone is the 'wanted' signal; the higher tones show
    clearly in the spectrum and are removed by a low-pass filter.
    """
    t = np.arange(int(sr * duration)) / sr
    y = (0.6 * np.sin(2 * np.pi * 440 * t)
         + 0.4 * np.sin(2 * np.pi * 880 * t)
         + 0.2 * np.sin(2 * np.pi * 1320 * t))
    return sr, y.astype(np.float32)
