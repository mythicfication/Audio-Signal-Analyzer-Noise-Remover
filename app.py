"""
Audio Signal Analyzer & Noise Remover — Streamlit app
Signal & Systems project

Run:  pip install -r requirements.txt
      streamlit run app.py
"""
import io
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy.io import wavfile

from signal_processing import (
    load_audio, save_wav, compute_fft, dominant_frequencies,
    add_white_noise, add_hum, butter_filter, fft_lowpass,
    snr_db, mse, make_demo_signal,
)

st.set_page_config(page_title="Audio Signal Analyzer & Noise Remover",
                   page_icon="🎧", layout="wide")
st.title("🎧 Audio Signal Analyzer & Noise Remover")
st.caption("Signal & Systems  •  Fourier Transform  •  FFT  •  Low-pass / High-pass Filters")

# ------------------------------------------------------------ sidebar: input
st.sidebar.header("1 · Input signal")
source = st.sidebar.radio("Signal source",
                          ["Use demo tone (440 + 880 + 1320 Hz)", "Upload an audio file"])
if source.startswith("Upload"):
    uploaded = st.sidebar.file_uploader("Upload WAV / MP3", type=["wav", "mp3", "ogg", "flac"])
    if uploaded is None:
        st.info("👈 Upload an audio file, or switch to the demo tone to try it instantly.")
        st.stop()
    sr, y = load_audio(uploaded)
    st.sidebar.success(f"Loaded: {uploaded.name}  •  {sr} Hz  •  {len(y)/sr:.2f} s")
else:
    sr, y = make_demo_signal()
    st.sidebar.success(f"Demo tone  •  {sr} Hz  •  {len(y)/sr:.2f} s")

# Keep clips short so plots stay readable
MAX_SEC = 10
if len(y) / sr > MAX_SEC:
    y = y[: int(sr * MAX_SEC)]
    st.sidebar.warning(f"Trimmed to the first {MAX_SEC} s for display.")

# ------------------------------------------------------------ sidebar: noise
st.sidebar.header("2 · Add artificial noise")
noise_type = st.sidebar.selectbox("Noise type",
                                  ["White Gaussian noise", "White noise + 50 Hz hum", "No noise"])
snr_in = st.sidebar.slider("Noise level — target SNR (dB)", 0, 20, 5,
                           help="Lower = noisier. SNR = 10·log10(Psignal / Pnoise)")
if noise_type == "No noise":
    noisy = y.copy()
elif noise_type == "White noise + 50 Hz hum":
    noisy, _ = add_white_noise(y, snr_in)
    noisy = add_hum(noisy, sr)
else:
    noisy, _ = add_white_noise(y, snr_in)

# ------------------------------------------------------------ sidebar: filter
st.sidebar.header("3 · Remove it — filter")
filter_type = st.sidebar.selectbox("Filter", ["Low-pass", "High-pass", "Band-pass", "FFT ideal low-pass"])
order = st.sidebar.slider("Butterworth order", 1, 8, 4,
                          help="Higher order = sharper cut-off, more ringing.")
if filter_type == "Band-pass":
    lo, hi = st.sidebar.slider("Pass band (Hz)", 20, sr // 2, (300, 3000))
    filtered = butter_filter(noisy, sr, (lo, hi), "bandpass", order)
elif filter_type == "High-pass":
    cutoff = st.sidebar.slider("Cut-off frequency (Hz)", 20, sr // 2, 1000)
    filtered = butter_filter(noisy, sr, cutoff, "highpass", order)
elif filter_type == "FFT ideal low-pass":
    cutoff = st.sidebar.slider("Cut-off frequency (Hz)", 20, sr // 2, 1000)
    filtered = fft_lowpass(noisy, sr, cutoff)
else:
    cutoff = st.sidebar.slider("Cut-off frequency (Hz)", 20, sr // 2, 1000)
    filtered = butter_filter(noisy, sr, cutoff, "lowpass", order)

# ------------------------------------------------------------ metrics
snr_before = snr_db(y, noisy)
snr_after = snr_db(y, filtered)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sample rate", f"{sr} Hz")
c2.metric("SNR — noisy", f"{snr_before:.2f} dB")
c3.metric("SNR — filtered", f"{snr_after:.2f} dB")
c4.metric("Improvement", f"{snr_after - snr_before:+.2f} dB",
          delta=f"MSE {mse(y, filtered):.5f}")
st.progress(min(max((snr_after + 10) / 40, 0.0), 1.0))

# ------------------------------------------------------------ waveform
t = np.arange(len(y)) / sr

def plot_waveforms():
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, sig, title in [(axes[0], y, "Original — clean signal"),
                           (axes[1], noisy, f"Noisy signal (SNR {snr_before:.1f} dB)"),
                           (axes[2], filtered, f"Filtered signal ({filter_type})")]:
        ax.plot(t, sig, linewidth=0.8)
        ax.set_ylabel("Amplitude")
        ax.set_title(title, loc="left")
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig

def plot_spectra():
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    for ax, sig, title in [(axes[0], y, "Spectrum — original"),
                           (axes[1], noisy, "Spectrum — noisy (note the raised noise floor)"),
                           (axes[2], filtered, "Spectrum — filtered")]:
        freqs, mag = compute_fft(sig, sr)
        ax.plot(freqs, mag, linewidth=0.8)
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(title, loc="left")
        ax.set_xlim(0, sr / 2)
        ax.grid(alpha=0.3)
    axes[2].set_xlabel("Frequency (Hz)")
    fig.tight_layout()
    return fig

tab1, tab2, tab3 = st.tabs(["📈 Waveform (time domain)", "📊 Spectrum (frequency domain)", "🔊 Listen & compare"])
with tab1:
    st.pyplot(plot_waveforms())
    st.markdown("**Reading it:** the clean waveform is smooth and periodic. Noise adds jagged, "
                "random wiggles. After filtering, the waveform should look smooth again — "
                "close to the original.")
with tab2:
    st.pyplot(plot_spectra())
    peaks = dominant_frequencies(y, sr)
    if peaks:
        st.write("**Dominant frequencies in the original:** " +
                 ", ".join(f"{f:.0f} Hz ({m:.1f} dB)" for f, m in peaks))
    st.markdown("**Reading it:** each tone is a sharp peak. White noise lifts the whole "
                "floor. A low-pass filter cuts everything above the cut-off — that is why "
                "choosing the cut-off above your wanted tones (e.g. 1000 Hz for a 440 Hz tone) matters.")
with tab3:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.subheader("Original"); st.audio(y, sample_rate=sr)
    with col_b:
        st.subheader("Noisy"); st.audio(noisy, sample_rate=sr)
    with col_c:
        st.subheader("Filtered"); st.audio(filtered, sample_rate=sr)
    buf = io.BytesIO()
    wavfile.write(buf, sr, (np.clip(filtered, -1, 1) * 32767).astype(np.int16))
    st.download_button("⬇️ Download filtered audio (.wav)", buf.getvalue(),
                       file_name="filtered_audio.wav", mime="audio/wav")

# ------------------------------------------------------------ theory
with st.expander("📚 Theory — the Signal & Systems behind this app"):
    st.markdown(r"""
**1. Fourier Transform** — any signal $x(t)$ can be written as a sum of sinusoids:
$$X(f) = \int_{-\infty}^{\infty} x(t)\, e^{-j2\pi ft}\, dt$$
The spectrum $X(f)$ shows *which frequencies* are present and how strong each is.

**2. FFT (Fast Fourier Transform)** — the discrete, fast algorithm for the DFT:
$$X[k] = \sum_{n=0}^{N-1} x[n]\, e^{-j2\pi kn/N}$$
Used here (`numpy.fft.rfft`) to draw the spectrum. Complexity drops from $O(N^2)$ to $O(N \log N)$.

**3. Filters**
- **Low-pass:** passes $f < f_c$, blocks $f > f_c$ — removes high-frequency hiss.
- **High-pass:** passes $f > f_c$ — removes low rumble / hum.
- **Butterworth:** maximally flat in the passband. Order $n$ sets how sharp the transition is:
$$|H(f)| = \frac{1}{\sqrt{1 + (f/f_c)^{2n}}}$$
Applied with `filtfilt` (forward + backward) so there is no phase delay.

**4. SNR** — $ \mathrm{SNR_{dB}} = 10\log_{10}(P_{signal}/P_{noise}) $.
Adding noise lowers it; a good filter raises it back. Compare the two SNR numbers above.
""")

st.markdown("---")
st.caption("Tip for your demo: use the demo tone, set SNR to 5 dB, low-pass at 1000 Hz — "
           "the 880 Hz and 1320 Hz tones are cut, the 440 Hz tone survives, and SNR jumps up.")
