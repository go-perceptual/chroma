# chroma

Chroma audio features in pure Go, no cgo: the front end of
[Chromaprint](https://github.com/acoustid/chromaprint), ported stage by stage
against that project's **own unit tests**.

```go
features, err := chroma.Features(pcm)   // mono int16 at 11025 Hz
// one []float64 of 12 bands per frame, normalised
```

A chroma vector says how much energy a moment of audio holds at each of the
twelve notes, with octaves folded together: a C is a C whether it is played
low or high. That is what makes it the right feature for "are these two files
the same recording?" — it survives re-encoding, volume changes and a different
master, where a waveform comparison sees none of those as the same.

## ⛔ Not an AcoustID fingerprint

Chromaprint turns these features into 32-bit sub-fingerprints with sixteen
trained classifiers. **That stage is not here**, so nothing this package
produces can be looked up against the AcoustID database. What it gives is the
feature sequence those classifiers run on, which is what near-duplicate
detection needs.

## The chain

| stage | |
|---|---|
| window | Hamming, scaled by 1/32767 |
| transform | `RFFT` from [go-fft](https://github.com/go-fft/fft) |
| spectrum | **energy** (`r² + i²`), not magnitude |
| chroma | bins folded onto 12 pitch classes, 28–3520 Hz |
| filter | 5-tap blur over time, `{0.25, 0.75, 1, 0.75, 0.25}` |
| normalise | unit length, or **zero** below 0.01 |

Geometry: 4096-sample frames at 11025 Hz, hopping by `FrameSize/3` ≈ 124 ms.

Each number earns its place. The top of the band stops well short of Nyquist
because that is where lossy codecs differ most, so including it would make the
feature depend on the encoder. Normalisation **zeroes** a quiet frame instead
of scaling it: dividing a nearly silent frame by its tiny norm would amplify
its noise into a confident-looking chord, making the loudest thing in the
fingerprint the quietest moment of the recording.

## ⚠ Two traps, both measured

**Energy, not magnitude.** The reference stores `r*r + i*i` and the bands are
sums of that. Taking a square root first changes every value and still looks
entirely reasonable.

**Concert A is the worst thing to test with.** The octave scale is based at
A0 = 27.5 Hz, so A sits exactly on a band edge. At this geometry the nearest
bin to 440 Hz is 163.46, and the energy lands in bin 163 (438.75 Hz, band 11)
and bin 164 (441.44 Hz, band 0) — opposite sides of the wrap-around. Nothing
is wrong: a boundary tone belongs to both neighbours. `TestAToneOnTheBand
BoundarySplits` keeps that written down, because reaching for 440 Hz is the
obvious move and it fails looking like a broken mapping.

## ⛔ Resampling is not done here

`Features` expects mono PCM **already at 11025 Hz**. Resampling belongs to the
decoder, which knows the source rate; doing it badly here would silently
change every feature. Audio at another rate produces a plausible result that
matches nothing.

## Verified against the reference

Chromaprint's own test values are reproduced as Go tests: six for the chroma
mapping — including the interpolated cases, which are what distinguish
truncating from rounding — and `Blur2`, `Blur3` and `Diff` for the filter,
with its exact float literals.

Beyond those: a steady tone does not wander between frames, an octave apart is
the same band end to end, volume does not change the fingerprint, silence
produces no pitch, and the frame count follows the hop less the frames the
filter swallows while filling.

## Licence

BSD-3-Clause.
