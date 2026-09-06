package dev.riszn.portal;

final class PortalState {
    enum Mode { NONE, ONE_HAND, TWO_HAND }

    final Mode mode;
    final float[] nodes;         // x0,y0,x1,y1... normalized display coordinates.
    final int[] anchors;         // node indices backed by real fingertip landmarks.
    final float[][] skeletons;   // one flattened 21-landmark skeleton per detected hand.
    final String[] handedness;   // MediaPipe handedness labels, same order as skeletons.
    final long producedAtNanos;
    final int hands;             // RAW number of hands MediaPipe detected, independent of portal validity.

    PortalState(
            Mode mode,
            float[] nodes,
            int[] anchors,
            float[][] skeletons,
            String[] handedness,
            long producedAtNanos,
            int hands) {
        this.mode = mode;
        this.nodes = nodes;
        this.anchors = anchors;
        this.skeletons = skeletons;
        this.handedness = handedness;
        this.producedAtNanos = producedAtNanos;
        this.hands = hands;
    }

    static PortalState none() {
        return invalid(0, new float[0][], new String[0]);
    }

    static PortalState invalid(int detectedHands, float[][] skeletons, String[] handedness) {
        return new PortalState(
                Mode.NONE,
                new float[0],
                new int[0],
                skeletons == null ? new float[0][] : skeletons,
                handedness == null ? new String[0] : handedness,
                System.nanoTime(),
                Math.max(0, detectedHands));
    }
}
