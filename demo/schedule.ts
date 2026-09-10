// What each display refresh puts on screen. Kept apart from the render loop and
// free of both GPU and DOM, because the interesting behaviour is a transition —
// what happens across the refresh where the toggle is flipped — and that is only
// checkable frame by frame.
//
// Displayed time runs in half real frames: real frame n is 2n, and the frame
// interpolated between n-1 and n is 2n-1. The schedule's whole job is to keep
// that non-decreasing.

export enum FrameGen {
  Off,
  On,
}

export enum Shown {
  RealCurrent,
  RealPrevious,
  Interpolated,
}

export interface Refresh {
  mode: FrameGen;
  // True on the refresh that rendered a new real frame.
  rendered: boolean;
  realFrame: number;
  // The first real frame whose interpolated frame is newer than anything
  // already on screen. See `enabledFrom` below.
  enabledFrom: number;
}

export function shown(refresh: Refresh): Shown {
  if (refresh.mode === FrameGen.Off || refresh.realFrame < refresh.enabledFrom) {
    return Shown.RealCurrent;
  }

  return refresh.rendered ? Shown.RealPrevious : Shown.Interpolated;
}

// Frame generation takes effect at the *next* real frame, never the current one.
//
// While it is off, the refresh that renders real frame n also presents it — so
// displayed time is already 2n. The frame available to interpolate at that
// moment is the one between n-1 and n, at 2n-1, which is older. Switching on
// and using it straight away would step the picture backwards, visibly, at the
// exact moment the viewer engaged the feature. Waiting for real frame n+1 means
// the first interpolated frame shown is at 2n+1, and the real frame n presented
// in between is a repeat rather than a step back.
//
// Also the starting value, for the same reason from the other end: real frame 0
// has no predecessor, so nothing has been interpolated yet.
export function enabledFrom(realFrame: number): number {
  return realFrame + 1;
}
