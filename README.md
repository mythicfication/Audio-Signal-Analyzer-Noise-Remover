# Audio Signal Analyzer & Noise Remover

**Signal & Systems project** — upload an audio file, see its waveform and
frequency spectrum, add artificial noise, and remove it with filters.
Compare before / after by sight *and* by ear.

## Concepts used
- **Fourier Transform** — time domain → frequency domain
- **FFT** (`numpy.fft.rfft`) — fast spectrum computation, $O(N \log N)$
- **Low-pass / High-pass / Band-pass filters** — Butterworth, applied with
  `scipy.signal.filtfilt` (zero-phase), plus an ideal FFT low-pass for comparison
- **SNR** — $10\log_{10}(P_{signal}/P_{noise})$, before vs after filtering

## Files
| File | What it is |
|---|---|
| `app.py` | Streamlit UI — upload, noise, filter, plots, playback, download |
| `signal_processing.py` | All DSP: load, FFT, noise, Butterworth / FFT filters, SNR, MSE |
| `generate_sample.py` | Makes `sample_clean.wav` / `sample_noisy.wav` (440 + 880 + 1320 Hz) |
| `requirements.txt` | Dependencies |
| `PROJECT_REPORT.md` | Theory, method, results table, viva Q&A |

## Run it
```bash
cd audio-signal-analyzer
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

No upload needed to try it: choose **"Use demo tone"** in the sidebar.

## How to demo it (2 minutes)
1. Demo tone, noise SNR **5 dB** — waveform gets jagged, spectrum floor rises.
2. Filter **Low-pass, cut-off 1000 Hz, order 4** — the 880 Hz and 1320 Hz
   peaks disappear, the 440 Hz tone survives, SNR improves.
3. Play Original → Noisy → Filtered in the **Listen & compare** tab.
4. Try **High-pass at 1000 Hz** — the opposite: only the high tones survive.
5. Try **FFT ideal low-pass** — same cut-off, but note the ringing; that is
   why Butterworth is used in practice.

## Suggested experiment for the report
| Noise SNR in | Filter | Cut-off | SNR out | Improvement |
|---|---|---|---|---|
| 5 dB | Low-pass, order 4 | 1000 Hz | *fill from app* | *fill from app* |
| 0 dB | Low-pass, order 4 | 1000 Hz | *fill from app* | *fill from app* |
| 10 dB | Low-pass, order 4 | 1000 Hz | *fill from app* | *fill from app* |
| 5 dB | High-pass, order 4 | 1000 Hz | *fill from app* | *fill from app* |

## Notes
- WAV works without extra setup. MP3 needs `librosa` (in requirements) —
  if MP3 fails, convert to WAV first.
- Clips longer than 10 s are trimmed to the first 10 s for display.
- No secrets / keys in this project. Nothing to put in `.env`.
