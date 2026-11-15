package chroma

import (
	"math"
	"testing"
)

// These are Chromaprint's own unit tests, with its values, ported.
//
// ⛔ They are the reason this package can claim to BE the reference's front
// end rather than something like it. The geometry is the reference's too —
// Chroma(10, 510, 256, 1000) over a 128-bin spectrum — chosen so that single
// bins land on known notes, which is what makes a one-line expectation
// meaningful.
const (
	testMinFreq    = 10
	testMaxFreq    = 510
	testFrameSize  = 256
	testSampleRate = 1000
	testBins       = 128
)

func spike(bin int) []float64 {
	s := make([]float64, testBins)
	s[bin] = 1.0
	return s
}

func assertFeatures(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != Bands {
		t.Fatalf("got %d features, want %d", len(got), Bands)
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 0.0001 {
			t.Errorf("feature %d = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestNormalA(t *testing.T) {
	c := New(testMinFreq, testMaxFreq, testFrameSize, testSampleRate)
	got := c.Consume(spike(113), nil)
	assertFeatures(t, got, []float64{1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0})
}

func TestNormalGSharp(t *testing.T) {
	c := New(testMinFreq, testMaxFreq, testFrameSize, testSampleRate)
	got := c.Consume(spike(112), nil)
	assertFeatures(t, got, []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1})
}

func TestNormalB(t *testing.T) {
	c := New(testMinFreq, testMaxFreq, testFrameSize, testSampleRate)
	got := c.Consume(spike(64), nil) // 250 Hz
	assertFeatures(t, got, []float64{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0})
}

// The interpolated cases are the ones that would survive a wrong fractional
// part: without them, truncating where the reference truncates and rounding
// where it rounds look identical.
func TestInterpolatedB(t *testing.T) {
	c := New(testMinFreq, testMaxFreq, testFrameSize, testSampleRate)
	c.Interpolate = true
	got := c.Consume(spike(64), nil)
	assertFeatures(t, got, []float64{0, 0.286905, 0.713095, 0, 0, 0, 0, 0, 0, 0, 0, 0})
}

func TestInterpolatedA(t *testing.T) {
	c := New(testMinFreq, testMaxFreq, testFrameSize, testSampleRate)
	c.Interpolate = true
	got := c.Consume(spike(113), nil)
	assertFeatures(t, got, []float64{0.555242, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.444758})
}

func TestInterpolatedGSharp(t *testing.T) {
	c := New(testMinFreq, testMaxFreq, testFrameSize, testSampleRate)
	c.Interpolate = true
	got := c.Consume(spike(112), nil)
	assertFeatures(t, got, []float64{0.401354, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.598646})
}

// ---- properties beyond the reference's cases ----

// TestDCAndNyquistAreExcluded pins the two bounds that are easy to write off
// by one. Bin 0 is the signal's mean, which is not a pitch; bins above
// frameSize/2 mirror those below and would count the same energy twice.
func TestDCAndNyquistAreExcluded(t *testing.T) {
	c := New(0, testSampleRate, testFrameSize, testSampleRate)
	if c.minIndex < 1 {
		t.Errorf("minIndex = %d, want at least 1: bin 0 is DC", c.minIndex)
	}
	if c.maxIndex > testFrameSize/2 {
		t.Errorf("maxIndex = %d, want at most %d: above Nyquist the spectrum mirrors", c.maxIndex, testFrameSize/2)
	}

	// A spike at DC must not reach any note.
	s := make([]float64, testBins)
	s[0] = 1000
	got := c.Consume(s, nil)
	for i, v := range got {
		if v != 0 {
			t.Fatalf("DC energy landed on note %d (%v)", i, v)
		}
	}
}

// TestEnergyIsConserved: folding octaves moves energy between bands but must
// not create or destroy it, with or without interpolation.
func TestEnergyIsConserved(t *testing.T) {
	for _, interp := range []bool{false, true} {
		c := New(testMinFreq, testMaxFreq, testFrameSize, testSampleRate)
		c.Interpolate = interp

		s := make([]float64, testBins)
		var in float64
		for i := c.minIndex; i < c.maxIndex; i++ {
			s[i] = float64(i%7) + 1
			in += s[i]
		}
		got := c.Consume(s, nil)
		var out float64
		for _, v := range got {
			out += v
		}
		if math.Abs(in-out) > 1e-9 {
			t.Errorf("interpolate=%v: %v in, %v out", interp, in, out)
		}
	}
}

// TestOctavesFoldTogether is the property the whole feature exists for: the
// same note an octave apart is the same chroma band.
func TestOctavesFoldTogether(t *testing.T) {
	c := New(testMinFreq, testMaxFreq, testFrameSize, testSampleRate)

	// ⚠ 125 and 250 Hz, not 250 and 500: at this geometry 500 Hz is bin 128,
	// which is the Nyquist bin AND one past the end of a 128-bin spectrum.
	// The first version of this test asked for it and panicked — the octave
	// above the top of the range is not a case the feature has an answer for.
	lowBand := bandOf(t, c.Consume(spike(32), nil))  // 125 Hz
	highBand := bandOf(t, c.Consume(spike(64), nil)) // 250 Hz, one octave up

	if lowBand != highBand {
		t.Fatalf("125 Hz landed on band %d and 250 Hz on band %d; an octave apart is the same note", lowBand, highBand)
	}
}

// bandOf returns the single band holding energy, failing if there is not
// exactly one.
func bandOf(t *testing.T, features []float64) int {
	t.Helper()
	band := -1
	for i, v := range features {
		if v > 0 {
			if band >= 0 {
				t.Fatalf("energy in bands %d and %d; expected one", band, i)
			}
			band = i
		}
	}
	if band < 0 {
		t.Fatal("no band holds any energy")
	}
	return band
}

// TestConsumeReusesTheBuffer: the hot loop of any fingerprinter calls this
// once per frame, so it must not allocate when given room.
func TestConsumeReusesTheBuffer(t *testing.T) {
	c := New(testMinFreq, testMaxFreq, testFrameSize, testSampleRate)
	buf := make([]float64, Bands)
	got := c.Consume(spike(113), buf)
	if &got[0] != &buf[0] {
		t.Fatal("Consume allocated a new slice although one was supplied")
	}
	allocs := testing.AllocsPerRun(100, func() { c.Consume(spike(113), buf) })
	// spike() itself allocates once per call; the point is that Consume adds
	// nothing on top of it.
	if allocs > 1 {
		t.Fatalf("Consume allocates %v times per call", allocs)
	}
}
