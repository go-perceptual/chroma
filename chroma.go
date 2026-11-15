// Package chroma turns a spectrum into the twelve pitch classes.
//
// A chroma vector says how much energy a moment of audio holds at each of the
// twelve notes, with octaves folded together: a C is a C whether it is played
// low or high. That is what makes it the right feature for asking "are these
// two files the same recording?" — it survives re-encoding, volume changes
// and a different mix of the same master, while a waveform comparison sees
// none of those as the same.
//
// This is the front end of Chromaprint (github.com/acoustid/chromaprint),
// ported stage by stage against that project's OWN unit tests, which are
// reproduced here as Go tests with their original values.
//
// ⛔ It is NOT an AcoustID-compatible fingerprint. Chromaprint turns these
// features into 32-bit sub-fingerprints with sixteen trained classifiers;
// that stage is not here, so nothing produced by this package can be looked
// up against the AcoustID database. What it gives is the feature sequence
// those classifiers run on, which is what near-duplicate detection needs.
package chroma

import "math"

// Bands is the number of pitch classes: twelve semitones in an octave.
const Bands = 12

// baseFreq is A0, 27.5 Hz. The reference writes it as 440/16 — concert A
// dropped four octaves — and octave 0 starts there, so a frequency's position
// within its octave is what names the note.
const baseFreq = 440.0 / 16.0

// Chroma maps the bins of one spectrum onto the twelve pitch classes.
//
// It is built once for a given frame geometry and reused for every frame: the
// bin-to-note table depends only on the frame size and the sample rate, not
// on the audio.
type Chroma struct {
	minIndex int
	maxIndex int
	notes    []int8
	frac     []float64
	// Interpolate spreads each bin's energy between its own note and the
	// neighbour it leans toward, instead of giving all of it to the nearest.
	// Off by default, as in the reference.
	Interpolate bool
}

// New builds the mapping for a frame of frameSize samples taken at
// sampleRate, considering only frequencies between minFreq and maxFreq.
//
// Chromaprint's own values are 28 Hz and 3520 Hz at 11025 Hz: low enough for
// a bass line, high enough for the notes that carry melody, and deliberately
// short of the top of the spectrum, where lossy codecs differ most.
func New(minFreq, maxFreq, frameSize, sampleRate int) *Chroma {
	c := &Chroma{
		notes: make([]int8, frameSize),
		frac:  make([]float64, frameSize),
	}

	// ⛔ The low bound is at least 1: bin 0 is DC, the signal's mean, which
	// is not a pitch and would land on an arbitrary note.
	c.minIndex = max(1, freqToIndex(float64(minFreq), frameSize, sampleRate))
	// ⛔ And the high bound is at most frameSize/2, the Nyquist bin. Bins
	// above it are the mirror image of those below and would count the same
	// energy twice.
	c.maxIndex = min(frameSize/2, freqToIndex(float64(maxFreq), frameSize, sampleRate))

	for i := c.minIndex; i < c.maxIndex; i++ {
		freq := indexToFreq(i, frameSize, sampleRate)
		octave := freqToOctave(freq)
		note := Bands * (octave - math.Floor(octave))
		// Truncation, not rounding: the integer part names the note and the
		// remainder says how far into it the frequency sits, which is what
		// Interpolate uses.
		c.notes[i] = int8(note)
		c.frac[i] = note - math.Trunc(note)
	}
	return c
}

// Consume folds one spectrum into twelve pitch classes.
//
// spectrum holds the magnitudes of a frame's bins; only those inside the
// frequency range given to New are read. The result is written into out,
// which must have room for Bands values, and returned.
func (c *Chroma) Consume(spectrum []float64, out []float64) []float64 {
	if cap(out) < Bands {
		out = make([]float64, Bands)
	}
	out = out[:Bands]
	for i := range out {
		out[i] = 0
	}

	for i := c.minIndex; i < c.maxIndex && i < len(spectrum); i++ {
		note := int(c.notes[i])
		energy := spectrum[i]
		if !c.Interpolate {
			out[note] += energy
			continue
		}
		// Share the energy with whichever neighbour the frequency leans
		// toward, in proportion to how far it leans.
		note2, a := note, 1.0
		switch {
		case c.frac[i] < 0.5:
			note2 = (note + Bands - 1) % Bands
			a = 0.5 + c.frac[i]
		case c.frac[i] > 0.5:
			note2 = (note + 1) % Bands
			a = 1.5 - c.frac[i]
		}
		out[note] += energy * a
		out[note2] += energy * (1 - a)
	}
	return out
}

// freqToOctave is how many octaves above A0 a frequency sits. Its fractional
// part is the position within the octave, which is the note.
func freqToOctave(freq float64) float64 {
	return math.Log2(freq / baseFreq)
}

func indexToFreq(i, frameSize, sampleRate int) float64 {
	return float64(i) * float64(sampleRate) / float64(frameSize)
}

func freqToIndex(freq float64, frameSize, sampleRate int) int {
	return int(math.Round(float64(frameSize) * freq / float64(sampleRate)))
}
