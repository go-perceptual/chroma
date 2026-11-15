package chroma

import (
	"errors"

	"github.com/go-fft/fft"
)

// The geometry Chromaprint uses, and the reason for each number.
const (
	// SampleRate is what audio must be resampled to. Low, because pitch
	// lives well below it and a smaller rate means fewer samples to transform.
	SampleRate = 11025
	// FrameSize is the window length: 4096 samples, about 370 ms, long
	// enough to resolve a bass note.
	FrameSize = 4096
	// Overlap is how much of each frame the next one repeats, leaving a hop
	// of FrameSize/3 ≈ 124 ms. Overlapping frames are what keep a note from
	// vanishing because it fell on a frame boundary.
	Overlap = FrameSize - FrameSize/3
	// Hop is the distance between consecutive frames.
	Hop = FrameSize - Overlap
	// MinFreq and MaxFreq bound the band that carries pitch. The top stops
	// well short of Nyquist on purpose: that is where lossy codecs differ
	// most, so including it would make the feature depend on the encoder.
	MinFreq = 28
	MaxFreq = 3520
)

// ErrTooShort is returned when the audio does not fill one window.
var ErrTooShort = errors.New("chroma: audio shorter than one frame")

// Features turns mono PCM into a sequence of normalised chroma vectors.
//
// ⛔ samples must already be mono at SampleRate. Resampling is not done here:
// it belongs to the decoder, which knows the source rate, and doing it badly
// here would silently change every feature. Feeding audio at another rate
// produces a plausible result that matches nothing.
//
// Each returned vector is a moment of music described only by the
// proportions of the twelve pitch classes — indifferent to volume, and to
// any re-encoding that preserves what a listener hears.
func Features(samples []int16) ([][]float64, error) {
	if len(samples) < FrameSize {
		return nil, ErrTooShort
	}

	// Hamming, scaled so int16 samples land in about [-1, 1], exactly as the
	// reference prepares it.
	window := fft.Hamming(FrameSize)
	for i := range window {
		window[i] /= 32767
	}

	c := New(MinFreq, MaxFreq, FrameSize, SampleRate)
	filter, err := NewFilter(DefaultCoefficients)
	if err != nil {
		return nil, err
	}

	frame := make([]float64, FrameSize)
	raw := make([]float64, Bands)
	var out [][]float64

	for start := 0; start+FrameSize <= len(samples); start += Hop {
		for i := 0; i < FrameSize; i++ {
			frame[i] = float64(samples[start+i]) * window[i]
		}

		spectrum := fft.RFFT(frame)

		// ⛔ ENERGY, not magnitude: the reference stores r*r + i*i and the
		// chroma bands are sums of that. Taking the square root here would
		// change every value and still look entirely reasonable.
		energy := make([]float64, len(spectrum))
		for i, v := range spectrum {
			re, im := real(v), imag(v)
			energy[i] = re*re + im*im
		}

		raw = c.Consume(energy, raw)
		filtered := filter.Consume(raw)
		if filtered == nil {
			continue // the filter is still filling
		}
		Normalize(filtered)
		out = append(out, filtered)
	}

	if out == nil {
		// Long enough for a frame, but not for the filter's window.
		return nil, ErrTooShort
	}
	return out, nil
}
