package chroma

import (
	"math"
	"testing"
)

// row makes a 12-band vector with the first two bands set, which is the shape
// the reference's filter tests use.
func row(a, b float64) []float64 {
	v := make([]float64, Bands)
	v[0], v[1] = a, b
	return v
}

func consumeAll(t *testing.T, f *Filter, rows ...[]float64) [][]float64 {
	t.Helper()
	var out [][]float64
	for _, r := range rows {
		if got := f.Consume(r); got != nil {
			out = append(out, got)
		}
	}
	return out
}

// TestBlur2 is Chromaprint's own ChromaFilter.Blur2, with its values.
func TestBlur2(t *testing.T) {
	f, err := NewFilter([]float64{0.5, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	got := consumeAll(t, f, row(0, 5), row(1, 6), row(2, 7))

	if len(got) != 2 {
		t.Fatalf("got %d rows, want 2: three frames through a 2-tap filter", len(got))
	}
	want := [][2]float64{{0.5, 5.5}, {1.5, 6.5}}
	for i, w := range want {
		if math.Abs(got[i][0]-w[0]) > 1e-9 || math.Abs(got[i][1]-w[1]) > 1e-9 {
			t.Errorf("row %d = (%v, %v), want (%v, %v)", i, got[i][0], got[i][1], w[0], w[1])
		}
	}
}

// TestBlur3 is Chromaprint's own ChromaFilter.Blur3, with its values —
// including the ones it writes out to their last decimal.
func TestBlur3(t *testing.T) {
	f, err := NewFilter([]float64{0.5, 0.7, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	got := consumeAll(t, f, row(0, 5), row(1, 6), row(2, 7), row(3, 8))

	if len(got) != 2 {
		t.Fatalf("got %d rows, want 2: four frames through a 3-tap filter", len(got))
	}
	want := [][2]float64{
		{1.7, 10.199999999999999},
		{3.399999999999999, 11.899999999999999},
	}
	for i, w := range want {
		if math.Abs(got[i][0]-w[0]) > 1e-9 || math.Abs(got[i][1]-w[1]) > 1e-9 {
			t.Errorf("row %d = (%v, %v), want (%v, %v)", i, got[i][0], got[i][1], w[0], w[1])
		}
	}
}

// TestDiff is Chromaprint's ChromaFilter.Diff: a filter whose taps subtract.
// It proves nothing assumes the coefficients are a blur, or positive.
func TestDiff(t *testing.T) {
	f, err := NewFilter([]float64{1.0, -1.0})
	if err != nil {
		t.Fatal(err)
	}
	got := consumeAll(t, f, row(0, 5), row(1, 6), row(2, 7))

	if len(got) != 2 {
		t.Fatalf("got %d rows, want 2", len(got))
	}
	for i := range got {
		if math.Abs(got[i][0]-(-1)) > 1e-9 || math.Abs(got[i][1]-(-1)) > 1e-9 {
			t.Errorf("row %d = (%v, %v), want (-1, -1)", i, got[i][0], got[i][1])
		}
	}
}

// TestEmitsNothingUntilItHasEnoughFrames pins the behaviour most likely to be
// mistaken for a bug: a clip shorter than the filter produces NO output, not
// a partial one.
func TestEmitsNothingUntilItHasEnoughFrames(t *testing.T) {
	f, err := NewFilter(DefaultCoefficients) // five taps
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < len(DefaultCoefficients)-1; i++ {
		if got := f.Consume(row(float64(i), 0)); got != nil {
			t.Fatalf("frame %d produced output before the filter was full", i)
		}
	}
	if got := f.Consume(row(99, 0)); got == nil {
		t.Fatal("the fifth frame produced nothing; the filter should be full")
	}
}

func TestResetForgetsHistory(t *testing.T) {
	f, err := NewFilter([]float64{0.5, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	f.Consume(row(0, 5))
	f.Reset()
	if got := f.Consume(row(1, 6)); got != nil {
		t.Fatal("output after one frame: Reset did not forget the first")
	}
}

func TestTooManyTapsIsRefused(t *testing.T) {
	if _, err := NewFilter(make([]float64, ringSize+1)); err == nil {
		t.Fatalf("a filter with %d taps was accepted, though the history holds %d", ringSize+1, ringSize)
	}
	if _, err := NewFilter(make([]float64, ringSize)); err != nil {
		t.Fatalf("a filter with exactly %d taps was refused: %v", ringSize, err)
	}
}

// ---- normalisation ----

func TestNormalizeMakesUnitLength(t *testing.T) {
	v := row(3, 4) // norm 5
	Normalize(v)
	if math.Abs(v[0]-0.6) > 1e-9 || math.Abs(v[1]-0.8) > 1e-9 {
		t.Fatalf("got (%v, %v), want (0.6, 0.8)", v[0], v[1])
	}
	var sq float64
	for _, x := range v {
		sq += x * x
	}
	if math.Abs(sq-1) > 1e-9 {
		t.Fatalf("squared norm = %v, want 1", sq)
	}
}

// TestVolumeDoesNotChangeTheFeature is the property normalisation exists for:
// the same music mastered louder must normalise to the same vector.
func TestVolumeDoesNotChangeTheFeature(t *testing.T) {
	quiet, loud := row(0.3, 0.4), row(30, 40)
	Normalize(quiet)
	Normalize(loud)
	for i := range quiet {
		if math.Abs(quiet[i]-loud[i]) > 1e-9 {
			t.Fatalf("band %d: quiet %v, loud %v — volume changed the feature", i, quiet[i], loud[i])
		}
	}
}

// TestASilentFrameIsZeroedNotAmplified is the one that matters. Scaling a
// nearly silent frame up to unit length would turn whatever noise it holds
// into a confident chord, making the loudest thing in the fingerprint the
// quietest moment of the recording.
func TestASilentFrameIsZeroedNotAmplified(t *testing.T) {
	v := row(0.001, 0.002) // norm well under the threshold
	Normalize(v)
	for i, x := range v {
		if x != 0 {
			t.Fatalf("band %d = %v in a frame below the threshold; it should be zeroed", i, x)
		}
	}

	// The control: just above the threshold it is scaled, not zeroed.
	w := row(0.02, 0)
	Normalize(w)
	if w[0] == 0 {
		t.Fatal("a frame above the threshold was zeroed")
	}
	if math.Abs(w[0]-1) > 1e-9 {
		t.Fatalf("got %v, want 1 after scaling", w[0])
	}
}

func TestNormalizeOfAllZeroIsAllZero(t *testing.T) {
	v := make([]float64, Bands)
	Normalize(v)
	for i, x := range v {
		if x != 0 {
			t.Fatalf("band %d = %v, want 0", i, x)
		}
	}
}
