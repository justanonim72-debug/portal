package dev.riszn.portal;

final class PortalState {
    enum Mode { NONE, ONE_HAND, TWO_HAND }

    final Mode mode;
    final float[] nodes;      // x0,y0,x1,y1... in normalized DISPLAY coordinates.
    final int[] anchors;      // node indices that are real fingertip anchors.
    final long producedAtNanos;
    final int hands;

    PortalState(Mode mode, float[] nodes, int[] anchors, long producedAtNanos, int hands) {
        this.mode = mode;
        this.nodes = nodes;
        this.anchors = anchors;
        this.producedAtNanos = producedAtNanos;
        this.hands = hands;
    }

    static PortalState none() {
        return new PortalState(Mode.NONE, new float[0], new int[0], System.nanoTime(), 0);
    }
}
