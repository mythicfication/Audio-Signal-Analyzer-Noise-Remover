# Audio Signal Analyzer & Noise Remover — Project Report

## 1. Aim
Build a tool that takes an audio signal, shows it in the time and frequency
domains, corrupts it with artificial noise, and recovers it with filters —
so the effect of each Signal & Systems concept is visible and audible.

## 2. Theory
### 2.1 Fourier Transform
$$X(f) = \int_{-\infty}^{\infty} x(t) e^{-j2\pi ft} dt$$
Decomposes a signal into sinusoids. The magnitude $|X(f)|$ is the spectrum.

### 2.2 FFT
The DFT $X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$ computed in
$O(N \log N)$ instead of $O(N^2)$. Implemented with `numpy.fft.rfft`
(real input → only positive frequencies). A Hann window is applied first
to reduce spectral leakage.

### 2.3 Filters
Butterworth magnitude response (maximally flat passband):
$$|H(f)| = \frac{1}{\sqrt{1 + (f/f_c)^{2n}}}$$
- **Low-pass:** $|H| \approx 1$ for $f < f_c$, $\approx 0$ for $f > f_c$.
- **High-pass:** the reverse.
- Applied with `filtfilt` (forward + backward pass) → zero phase shift,
  so the filtered waveform is not delayed relative to the original.
- An **ideal FFT low-pass** (zeroing bins above $f_c$) is included for
  comparison — it rings in the time domain (Gibbs phenomenon), which is
  why practical filters use a gradual roll-off.

### 2.4 SNR
$$\mathrm{SNR_{dB}} = 10 \log_{10}\frac{P_{signal}}{P_{noise}}$$
Measured before filtering (noisy vs clean) and after (filtered vs clean).
Improvement = SNR_after − SNR_before.

## 3. Method
1. Load audio (mono, normalised to [-1, 1]) or generate a demo tone
   (440 + 880 + 1320 Hz, 16 kHz, 2 s).
2. Plot waveform and FFT spectrum of the clean signal.
3. Add white Gaussian noise at a chosen SNR (5 dB default), optionally
   50 Hz mains hum. Re-plot — noise floor rises, waveform roughens.
4. Filter (Butterworth low/high/band-pass, or FFT ideal low-pass).
   Re-plot and re-measure SNR.
5. Listen to original / noisy / filtered and download the filtered WAV.

## 4. Results (fill in from the app)
Run the demo tone at SNR 5 dB, low-pass 1000 Hz, order 4, and record
SNR before / after and the dominant frequencies. Repeat for other
SNRs and cut-offs — see the table in README.md.

Expected pattern: low-pass at 1000 Hz keeps the 440 Hz tone, removes
880 / 1320 Hz and most white noise above 1000 Hz, so SNR improves by
several dB. High-pass at 1000 Hz does the opposite and SNR vs the
*original* (dominated by 440 Hz) gets worse — a good talking point:
a filter only helps if it keeps the wanted band.

## 5. Limitations & future work
- White noise is the simplest model; real noise is often coloured or
  non-stationary (adaptive / Wiener filtering would do better).
- Single-channel only; stereo is averaged to mono.
- Ideal FFT filtering rings; windowed / FIR designs (e.g. Parks-McClellan)
  would be a next step, as would spectral subtraction.

## 6. Viva Q&A (likely questions)
- **Why FFT and not DFT directly?** Speed: $O(N \log N)$ vs $O(N^2)$.
- **Why a Hann window?** A finite clip leaks energy into neighbouring
  bins; windowing tapers the edges and sharpens peaks.
- **Why filtfilt instead of lfilter?** `lfilter` delays the signal
  (phase shift); `filtfilt` runs forward and backward, cancelling it.
- **What does filter order do?** Higher order = sharper transition
  around $f_c$, at the cost of more ringing and computation.
- **Why does SNR improve but never reach infinity?** Some noise lies
  *inside* the passband and cannot be removed by frequency selection alone.
