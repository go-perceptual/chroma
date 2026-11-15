package chroma

import "math"

// MinOverlap is the smallest share of the shorter recording that must line up
// before an alignment counts.
//
// ⛔ Without it the score is meaningless. Sliding two sequences past each
// other eventually leaves a single frame facing a single frame, and one frame
// matching one frame is a cosine of 1.0 — a perfect score built on a quarter
// of a second. The floor is what makes a high score mean "these recordings
// agree over most of their length".
const MinOverlap = 0.5

// Compare scores two feature sequences in [0,1]: 1.0 when they agree
// everywhere, 0.0 when they share nothing.
//
// One sequence is slid against the other and the best alignment wins, so a
// recording that starts a few seconds later — a different rip, a leading
// silence, a trimmed intro — still matches. Only alignments overlapping at
// least MinOverlap of the shorter sequence are considered.
//
// The score is the mean cosine of the aligned frames. Each frame is already
// unit length or zero, so the cosine is a dot product; a frame zeroed for
// being too quiet contributes nothing and is not counted, which keeps a long
// silence from either inflating or deflating the result.
//
// ⚠ It returns 0 when neither sequence has a usable frame, which is also what
// two recordings of silence give. Silence matches nothing, including itself,
// because there is nothing in it to match on.
func Compare(a, b [][]float64) float64 {
	best, _ := CompareAt(a, b)
	return best
}

// CompareAt is Compare plus the offset that produced the best score: the
// number of frames b must be shifted by to line up with a, negative when b
// starts first.
//
// The offset is worth having for more than diagnosis: a caller that knows two
// files are the same recording can use it to say WHERE they differ.
func CompareAt(a, b [][]float64) (score float64, offset int) {
	if len(a) == 0 || len(b) == 0 {
		return 0, 0
	}

	shorter := len(a)
	if len(b) < shorter {
		shorter = len(b)
	}
	// At least one pair, without a guard: the early return above leaves
	// shorter >= 1, and ceil(0.5*x) >= 1 for every x >= 1. A clamp here would
	// be unreachable code in the middle of the scoring rule, which a reader
	// would have to disprove before trusting the rest.
	need := int(math.Ceil(MinOverlap * float64(shorter)))

	bestScore, bestOffset := 0.0, 0
	found := false

	// offset is how far b is shifted right relative to a. The range covers
	// every alignment that can still meet the overlap floor.
	for off := -(len(b) - need); off <= len(a)-need; off++ {
		s, n := meanCosine(a, b, off)
		if n < need {
			continue
		}
		if !found || s > bestScore {
			bestScore, bestOffset, found = s, off, true
		}
	}
	if !found {
		return 0, 0
	}
	return bestScore, bestOffset
}

// meanCosine averages the cosine of the frames that line up at this offset,
// and reports how many pairs it averaged over.
//
// Pairs where either frame is silent are skipped rather than scored as 0: a
// silent frame has no direction, and calling that a disagreement would punish
// a recording for its quiet passages.
func meanCosine(a, b [][]float64, offset int) (float64, int) {
	var total float64
	var n int
	for i := range a {
		j := i - offset
		if j < 0 || j >= len(b) {
			continue
		}
		var dot float64
		var silentA, silentB = true, true
		for k := range a[i] {
			dot += a[i][k] * b[j][k]
			if a[i][k] != 0 {
				silentA = false
			}
			if b[j][k] != 0 {
				silentB = false
			}
		}
		if silentA || silentB {
			continue
		}
		// Features are unit length, so the dot product IS the cosine.
		total += dot
		n++
	}
	if n == 0 {
		return 0, 0
	}
	return total / float64(n), n
}
