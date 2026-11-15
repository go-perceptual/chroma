package chroma

import (
	"math"
	"testing"
)

// tone renders a pure sine of the given frequency, seconds long, as int16 PCM
// at SampleRate.
func tone(freq float64, seconds float64, amplitude float64) []int16 {
	n := int(seconds * SampleRate)
	out := make([]int16, n)
	for i := range out {
		v := amplitude * math.Sin(2*math.Pi*freq*float64(i)/SampleRate)
		out[i] = int16(v * 32767)
	}
	return out
}

// bandOfVector returns the strongest band.
func strongest(v []float64) int {
	best, bi := -1.0, -1
	for i, x := range v {
		if x > best {
			best, bi = x, i
		}
	}
	return bi
}

// midBand is a frequency sitting in the MIDDLE of a chroma band rather than
// on its edge: 27.5 * 2^(4 + 3.5/12). Mid-band matters — see
// TestAToneOnTheBandBoundarySplits below for what happens otherwise.
const midBand = 538.68
const midBandOctaveUp = 2 * midBand

// TestAPureToneLandsOnOneBand is the end-to-end check that the whole chain —
// window, transform, energy, chroma, filter, normalise — is wired the right
// way round.
func TestAPureToneLandsOnOneBand(t *testing.T) {
	features, err := Features(tone(midBand, 2.0, 0.5))
	if err != nil {
		t.Fatal(err)
	}
	if len(features) == 0 {
		t.Fatal("no features from two seconds of audio")
	}
	want := strongest(features[0])
	for i, f := range features {
		if got := strongest(f); got != want {
			t.Fatalf("frame %d: strongest band is %d, frame 0 said %d; a steady tone must not wander", i, got, want)
		}
		// And the band must actually dominate, not merely lead.
		var rest float64
		for b, v := range f {
			if b != want {
				rest += v
			}
		}
		if f[want] < rest {
			t.Fatalf("frame %d: band %d holds %v against %v elsewhere; a pure tone should dominate", i, want, f[want], rest)
		}
	}
}

// TestAnOctaveApartIsTheSameFeature is the property the whole representation
// exists for, end to end this time.
func TestAnOctaveApartIsTheSameFeature(t *testing.T) {
	low, err := Features(tone(midBand, 2.0, 0.5))
	if err != nil {
		t.Fatal(err)
	}
	high, err := Features(tone(midBandOctaveUp, 2.0, 0.5))
	if err != nil {
		t.Fatal(err)
	}
	if strongest(low[0]) != strongest(high[0]) {
		t.Fatalf("%.0f Hz is band %d and %.0f Hz is band %d; an octave apart is the same note",
			midBand, strongest(low[0]), midBandOctaveUp, strongest(high[0]))
	}
}

// TestADifferentNoteIsADifferentFeature is the control: a chain that always
// answered the same band would pass both tests above. A fourth up is five
// semitones.
func TestADifferentNoteIsADifferentFeature(t *testing.T) {
	a, err := Features(tone(midBand, 2.0, 0.5))
	if err != nil {
		t.Fatal(err)
	}
	b, err := Features(tone(midBand*math.Pow(2, 5.0/12), 2.0, 0.5))
	if err != nil {
		t.Fatal(err)
	}
	if strongest(a[0]) == strongest(b[0]) {
		t.Fatalf("two notes a fourth apart both landed on band %d", strongest(a[0]))
	}
}

// TestAToneOnTheBandBoundarySplits records a real property of this feature,
// found by writing a test that assumed the opposite.
//
// ⚠ Concert A, 440 Hz, is the WORST probe for a chroma feature: the octave
// scale is based at A0 = 27.5 Hz, so A sits exactly on a band edge. At this
// geometry the nearest FFT bin is 163.46, and the energy lands in bin 163
// (438.75 Hz, band 11) and bin 164 (441.44 Hz, band 0) — on opposite sides of
// the wrap-around.
//
// Nothing is wrong: a boundary tone genuinely belongs to both neighbours. It
// is recorded because "test it with concert A" is the obvious thing to reach
// for, and it fails in a way that looks like a broken mapping.
func TestAToneOnTheBandBoundarySplits(t *testing.T) {
	features, err := Features(tone(440, 2.0, 0.5))
	if err != nil {
		t.Fatal(err)
	}
	f := features[0]
	// The two bands either side of the boundary must hold almost all of it.
	share := f[0] + f[Bands-1]
	var total float64
	for _, v := range f {
		total += v
	}
	if share < 0.9*total {
		t.Fatalf("bands 0 and %d hold %v of %v; a boundary tone should sit on both", Bands-1, share, total)
	}
	if f[0] == 0 || f[Bands-1] == 0 {
		t.Fatalf("the tone landed entirely on one side: band 0 = %v, band %d = %v", f[0], Bands-1, f[Bands-1])
	}
}

// TestVolumeDoesNotChangeTheFingerprint: the same tone recorded quietly and
// loudly must give the same features, which is what normalisation buys and
// what makes the feature survive a re-master.
func TestVolumeDoesNotChangeTheFingerprint(t *testing.T) {
	quiet, err := Features(tone(440, 2.0, 0.05))
	if err != nil {
		t.Fatal(err)
	}
	loud, err := Features(tone(440, 2.0, 0.9))
	if err != nil {
		t.Fatal(err)
	}
	if len(quiet) != len(loud) {
		t.Fatalf("%d frames quiet, %d loud", len(quiet), len(loud))
	}
	for i := range quiet {
		for b := range quiet[i] {
			if math.Abs(quiet[i][b]-loud[i][b]) > 0.02 {
				t.Fatalf("frame %d band %d: quiet %v, loud %v", i, b, quiet[i][b], loud[i][b])
			}
		}
	}
}

// TestSilenceProducesNoPitch: a silent recording must not invent a chord.
func TestSilenceProducesNoPitch(t *testing.T) {
	features, err := Features(make([]int16, SampleRate*2))
	if err != nil {
		t.Fatal(err)
	}
	for i, f := range features {
		for b, v := range f {
			if v != 0 {
				t.Fatalf("frame %d band %d = %v in silence", i, b, v)
			}
		}
	}
}

// TestFrameCountFollowsTheHop pins the geometry: one frame every Hop samples,
// less the frames the filter swallows while it fills.
func TestFrameCountFollowsTheHop(t *testing.T) {
	const seconds = 3
	features, err := Features(tone(440, seconds, 0.5))
	if err != nil {
		t.Fatal(err)
	}
	windows := (seconds*SampleRate-FrameSize)/Hop + 1
	want := windows - (len(DefaultCoefficients) - 1)
	if len(features) != want {
		t.Fatalf("got %d features, want %d (%d windows less %d swallowed while the filter fills)",
			len(features), want, windows, len(DefaultCoefficients)-1)
	}
}

func TestTooShortIsAnError(t *testing.T) {
	if _, err := Features(make([]int16, FrameSize-1)); err != ErrTooShort {
		t.Fatalf("one sample short of a frame gave %v, want ErrTooShort", err)
	}
	// Long enough for a window but not for the filter: still an error, not an
	// empty success.
	if _, err := Features(make([]int16, FrameSize+Hop)); err != ErrTooShort {
		t.Fatalf("too short for the filter gave %v, want ErrTooShort", err)
	}
}
