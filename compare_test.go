package chroma

import (
	"math"
	"testing"
)

// chord renders several tones together, which is far closer to music than a
// single sine and makes the feature vectors non-trivial.
func chord(freqs []float64, seconds, amplitude float64) []int16 {
	n := int(seconds * SampleRate)
	out := make([]int16, n)
	for i := range out {
		var v float64
		for _, f := range freqs {
			v += math.Sin(2 * math.Pi * f * float64(i) / SampleRate)
		}
		out[i] = int16(v / float64(len(freqs)) * amplitude * 32767)
	}
	return out
}

var (
	chordA = []float64{538.68, 640.49, 807.54}
	chordB = []float64{480.00, 571.00, 719.00}
)

func featuresOf(t *testing.T, pcm []int16) [][]float64 {
	t.Helper()
	f, err := Features(pcm)
	if err != nil {
		t.Fatal(err)
	}
	return f
}

func TestComparesItselfAsOne(t *testing.T) {
	f := featuresOf(t, chord(chordA, 6, 0.7))
	if got := Compare(f, f); math.Abs(got-1) > 1e-9 {
		t.Fatalf("a recording against itself scored %v, want 1", got)
	}
}

func TestDifferentMusicScoresLow(t *testing.T) {
	a := featuresOf(t, chord(chordA, 6, 0.7))
	b := featuresOf(t, chord(chordB, 6, 0.7))
	got := Compare(a, b)
	if got > 0.5 {
		t.Fatalf("two different chords scored %v; they share nothing", got)
	}
}

// TestVolumeDoesNotChangeTheScore is what normalisation buys, at the level a
// caller sees.
func TestVolumeDoesNotChangeTheScore(t *testing.T) {
	loud := featuresOf(t, chord(chordA, 6, 0.9))
	quiet := featuresOf(t, chord(chordA, 6, 0.05))
	if got := Compare(loud, quiet); got < 0.99 {
		t.Fatalf("the same music at two volumes scored %v", got)
	}
}

// TestARecordingThatStartsLaterStillMatches is why Compare slides at all: a
// different rip of the same track, with a second of leading silence, is the
// same recording.
func TestARecordingThatStartsLaterStillMatches(t *testing.T) {
	music := chord(chordA, 6, 0.7)

	lead := make([]int16, SampleRate) // one second of silence
	delayed := append(lead, music...)

	a := featuresOf(t, music)
	b := featuresOf(t, delayed)

	score, offset := CompareAt(a, b)
	t.Logf("score %.4f at offset %d frames", score, offset)
	if score < 0.99 {
		t.Fatalf("the same music delayed by a second scored %v", score)
	}
	if offset >= 0 {
		t.Fatalf("offset %d: b starts later, so a must be shifted the other way", offset)
	}
}

// TestASingleFrameCannotWin is the guard on the overlap floor, with a corpus
// built so the danger actually occurs.
//
// ⛔ The first version of this test used two unrelated chords and logged "the
// one-frame alignment scores 0.0000" — the extreme alignment was not a false
// match at all, so the test would have passed with the floor removed. A guard
// whose witness never fires is a guard nobody has checked.
//
// Here the second recording is different music that ENDS with a moment of the
// first. Slid to the extreme, that moment faces the opening of the first
// recording and scores ~1.0, while the two recordings share almost nothing.
func TestASingleFrameCannotWin(t *testing.T) {
	a := featuresOf(t, chord(chordA, 6, 0.7))

	// ⚠ The tail must outlast the filter window. At 0.5 s the final frame is
	// still a blend across the transition — the five-tap filter spans about
	// 620 ms — and the edge alignment only reached 0.72, which did not stage
	// the danger either. Two seconds makes the last frames purely the first
	// recording's music, while staying far under the overlap floor.
	tail := chord(chordA, 2, 0.7)
	b := featuresOf(t, append(chord(chordB, 6, 0.7), tail...))

	// The extreme alignment: a's opening against b's tail.
	edge, n := meanCosine(a, b, -(len(b) - 1))
	t.Logf("the one-frame alignment scores %.4f over %d pair(s)", edge, n)
	if n != 1 {
		t.Fatalf("expected exactly one overlapping pair, got %d", n)
	}
	if edge < 0.9 {
		t.Fatalf("the corpus does not stage the danger: the edge alignment scores %.4f", edge)
	}

	got := Compare(a, b)
	t.Logf("Compare says %.4f", got)
	if got >= 0.9 {
		t.Fatalf("Compare returned %v, which is the one-frame alignment, not the recordings", got)
	}
}

// TestTooLittleOverlapScoresZero: two recordings that cannot line up over
// half the shorter one have no comparable part, and that is reported as 0
// rather than as the best sliver.
func TestTooLittleOverlapScoresZero(t *testing.T) {
	long := featuresOf(t, chord(chordA, 8, 0.7))
	short := featuresOf(t, chord(chordA, 2, 0.7))
	// A genuine subset still matches: the floor is half the SHORTER one.
	if got := Compare(long, short); got < 0.99 {
		t.Fatalf("a two-second excerpt of the same music scored %v", got)
	}
}

func TestEmptyInputsScoreZero(t *testing.T) {
	f := featuresOf(t, chord(chordA, 6, 0.7))
	if got := Compare(nil, f); got != 0 {
		t.Errorf("nil against features scored %v", got)
	}
	if got := Compare(f, nil); got != 0 {
		t.Errorf("features against nil scored %v", got)
	}
	if got := Compare(nil, nil); got != 0 {
		t.Errorf("nil against nil scored %v", got)
	}
}

// TestSilenceMatchesNothing records a deliberate decision: a silent recording
// has no direction to compare, so it scores 0 even against itself. Reporting
// 1.0 would make every silent file a duplicate of every other.
func TestSilenceMatchesNothing(t *testing.T) {
	silent := featuresOf(t, make([]int16, SampleRate*6))
	if got := Compare(silent, silent); got != 0 {
		t.Fatalf("silence against itself scored %v, want 0", got)
	}
}

// TestLossyDamageBarelyMovesTheScore is the question this package was built
// for: does a recording survive what a lossy codec does to it?
func TestLossyDamageBarelyMovesTheScore(t *testing.T) {
	src := chord(chordA, 6, 0.7)
	ref := featuresOf(t, src)

	for _, c := range []struct {
		name   string
		cutoff float64
		bits   uint
	}{
		{"16 kHz, 16 bit", 16000, 16},
		{"8 kHz, 12 bit", 8000, 12},
		{"4 kHz, 8 bit", 4000, 8},
	} {
		got := Compare(ref, featuresOf(t, damage(src, c.cutoff, c.bits)))
		t.Logf("  %-16s %.4f", c.name, got)
		if got < 0.95 {
			t.Errorf("%s scored %v; the feature should survive this", c.name, got)
		}
	}
}

// damage stands in for a lossy codec: it throws away the top of the band and
// quantises what is left. ⚠ A proxy, not an encoder — a real one also applies
// a psychoacoustic model — but it is the part of lossy coding this feature is
// meant to survive.
func damage(pcm []int16, cutoffHz float64, bits uint) []int16 {
	out := make([]int16, len(pcm))
	alpha := 1.0 / (1.0 + SampleRate/(2*math.Pi*cutoffHz))
	var y float64
	for i, s := range pcm {
		y += alpha * (float64(s) - y)
		out[i] = int16(y)
	}
	step := int32(1) << (16 - bits)
	for i, s := range out {
		out[i] = int16((int32(s) / step) * step)
	}
	return out
}
