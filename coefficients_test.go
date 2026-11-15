package chroma

import (
	"errors"
	"testing"
)

// TestFeaturesRefusesAnImpossibleFilter covers the one path in Features that
// the default parameters cannot reach.
//
// ⛔ It is reachable, which is why it is tested rather than deleted:
// DefaultCoefficients is a package variable, so a caller can set a filter
// longer than the frame history and Features must refuse it instead of
// returning a fingerprint built from whatever the ring happened to hold.
func TestFeaturesRefusesAnImpossibleFilter(t *testing.T) {
	saved := DefaultCoefficients
	t.Cleanup(func() { DefaultCoefficients = saved })

	DefaultCoefficients = make([]float64, ringSize+1)
	_, err := Features(tone(midBand, 2.0, 0.5))
	if !errors.Is(err, ErrTooManyTaps) {
		t.Fatalf("Features with %d taps gave %v, want ErrTooManyTaps", ringSize+1, err)
	}
}

// TestErrTooManyTapsSaysTheNumbers: an error a reader cannot act on is half
// an error, so it names both the taps asked for and the history available.
func TestErrTooManyTapsSaysTheNumbers(t *testing.T) {
	_, err := NewFilter(make([]float64, ringSize+1))
	if err == nil {
		t.Fatal("no error")
	}
	msg := err.Error()
	for _, want := range []string{"9 taps", "8 frames"} {
		if !contains(msg, want) {
			t.Errorf("error %q does not mention %q", msg, want)
		}
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
