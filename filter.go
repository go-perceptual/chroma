package chroma

import (
	"errors"
	"fmt"
	"math"
)

// ringSize is the depth of the filter's history. The reference fixes it at 8,
// which bounds the longest filter it will accept.
const ringSize = 8

// Filter smooths a stream of chroma vectors over TIME, one tap per past
// frame.
//
// Smoothing in time is what makes the feature survive re-encoding: a lossy
// codec moves energy between neighbouring frames, and a filter that spans
// several frames sees the same shape either way. Chromaprint's own
// coefficients, DefaultCoefficients below, are a five-tap blur.
//
// ⛔ A Filter is STATEFUL and emits nothing until it has seen as many frames
// as it has taps: a 5-tap filter over 4 frames produces no output at all.
// That is correct and is the first thing to check when a short clip yields an
// empty fingerprint.
type Filter struct {
	coefficients []float64
	ring         [ringSize][]float64
	offset       int
	size         int
}

// DefaultCoefficients is Chromaprint's five-tap blur.
var DefaultCoefficients = []float64{0.25, 0.75, 1.0, 0.75, 0.25}

// ErrTooManyTaps is returned for a filter longer than the ring can hold.
var ErrTooManyTaps = errors.New("chroma: filter has more taps than the frame history can hold")

// NewFilter returns a filter with the given taps.
func NewFilter(coefficients []float64) (*Filter, error) {
	if len(coefficients) > ringSize {
		return nil, fmt.Errorf("%w: %d taps, %d frames", ErrTooManyTaps, len(coefficients), ringSize)
	}
	f := &Filter{
		coefficients: append([]float64(nil), coefficients...),
		size:         1,
	}
	for i := range f.ring {
		f.ring[i] = make([]float64, Bands)
	}
	return f, nil
}

// Consume takes one chroma vector and returns the filtered vector, or nil
// while the filter is still filling.
//
// The returned slice is reused on the next call: a caller that keeps it must
// copy it.
func (f *Filter) Consume(features []float64) []float64 {
	copy(f.ring[f.offset], features)
	f.offset = (f.offset + 1) % ringSize

	if f.size < len(f.coefficients) {
		f.size++
		return nil
	}

	// The oldest frame still inside the window.
	start := (f.offset + ringSize - len(f.coefficients)) % ringSize
	out := make([]float64, Bands)
	for i := 0; i < Bands; i++ {
		var sum float64
		for j, c := range f.coefficients {
			sum += f.ring[(start+j)%ringSize][i] * c
		}
		out[i] = sum
	}
	return out
}

// Reset forgets the history, as if no frame had been seen.
func (f *Filter) Reset() {
	f.size = 1
	f.offset = 0
	for i := range f.ring {
		for j := range f.ring[i] {
			f.ring[i][j] = 0
		}
	}
}

// NormalizeThreshold is the norm below which a vector is considered to carry
// no pitch at all.
//
// ⛔ Below it the vector is zeroed rather than scaled up. Dividing a nearly
// silent frame by its tiny norm would amplify whatever noise it holds into a
// confident-looking chord — the loudest thing in the fingerprint would be the
// quietest moment of the recording.
const NormalizeThreshold = 0.01

// Normalize scales a chroma vector to unit length in place, or zeroes it when
// it is too quiet to mean anything.
//
// After this a vector says only the PROPORTIONS of the twelve pitches, which
// is what makes the feature indifferent to volume: the same music mastered
// louder normalises to the same thing.
func Normalize(features []float64) {
	var squares float64
	for _, v := range features {
		squares += v * v
	}
	norm := 0.0
	if squares > 0 {
		norm = math.Sqrt(squares)
	}
	if norm < NormalizeThreshold {
		for i := range features {
			features[i] = 0
		}
		return
	}
	for i := range features {
		features[i] /= norm
	}
}
