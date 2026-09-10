// Ported from AMD FidelityFX SDK (MIT) — see THIRD_PARTY_NOTICES.md
//
// Geometry shared by the three scene-change-detection passes. The frame is
// split into a 3x3 grid of regions and each gets its own 256-bin luma
// histogram; comparing this frame's histograms against last frame's is what
// detects a cut. Regional rather than global so a cut that only redraws part
// of the frame still registers.

const SCD_HISTOGRAM_BINS: u32 = 256u;
const SCD_HISTOGRAMS_PER_DIM: i32 = 3;
const SCD_HISTOGRAM_COUNT: u32 = u32(SCD_HISTOGRAMS_PER_DIM * SCD_HISTOGRAMS_PER_DIM);

// Each histogram is compared against the previous frame three times, with the
// filtered curve shifted by -1, 0 and +1 bins; the lowest of the three
// decides, so a global brightness drift does not read as a cut.
const SCD_SHIFT_COUNT: u32 = 3u;

// Divergence is accumulated through integer atomics, so it is carried as fixed
// point at this scale.
const SCD_DIVERGENCE_FACTOR: f32 = 1000000.0;

const SCD_SCENE_CHANGE_THRESHOLD: f32 = 0.45;
