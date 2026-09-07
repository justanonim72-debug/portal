package dev.riszn.portal;

import android.graphics.Matrix;

final class PortalState {
    enum Mode { NONE, ONE_HAND, TWO_HAND }

    final Mode mode;
    final float[] nodes;         // normalized coordinates in MediaPipe rotated/display space.
    final int[] anchors;
    final float[][] skeletons;
    final String[] handedness;
    final long producedAtNanos;
    final int hands;

    // Exact CameraX transform metadata for the ImageAnalysis frame that produced this result.
    // MediaPipe receives the analysis buffer plus analysisRotationDegrees. Renderer first undoes
    // that rotation, maps analysis pixels -> sensor via analysisToSensor, then sensor -> current
    // OverlayEffect frame via Frame.getSensorToBufferTransform(). This handles different crops /
    // aspect ratios between ImageAnalysis, Preview and VideoCapture without guessing.
    final Matrix analysisToSensor;
    final int analysisWidth;
    final int analysisHeight;
    final int analysisRotationDegrees;
    final boolean sourceMirrored;

    PortalState(
            Mode mode,
            float[] nodes,
            int[] anchors,
            float[][] skeletons,
            String[] handedness,
            long producedAtNanos,
            int hands,
            Matrix analysisToSensor,
            int analysisWidth,
            int analysisHeight,
            int analysisRotationDegrees,
            boolean sourceMirrored) {
        this.mode = mode;
        this.nodes = nodes == null ? new float[0] : nodes;
        this.anchors = anchors == null ? new int[0] : anchors;
        this.skeletons = skeletons == null ? new float[0][] : skeletons;
        this.handedness = handedness == null ? new String[0] : handedness;
        this.producedAtNanos = producedAtNanos;
        this.hands = Math.max(0, hands);
        this.analysisToSensor = analysisToSensor == null ? new Matrix() : new Matrix(analysisToSensor);
        this.analysisWidth = Math.max(1, analysisWidth);
        this.analysisHeight = Math.max(1, analysisHeight);
        this.analysisRotationDegrees = normalizeRotation(analysisRotationDegrees);
        this.sourceMirrored = sourceMirrored;
    }

    static PortalState none() {
        return invalid(0, new float[0][], new String[0], new Matrix(), 1, 1, 0, false);
    }

    static PortalState invalid(
            int detectedHands,
            float[][] skeletons,
            String[] handedness,
            Matrix analysisToSensor,
            int analysisWidth,
            int analysisHeight,
            int analysisRotationDegrees,
            boolean sourceMirrored) {
        return new PortalState(
                Mode.NONE,
                new float[0],
                new int[0],
                skeletons,
                handedness,
                System.nanoTime(),
                detectedHands,
                analysisToSensor,
                analysisWidth,
                analysisHeight,
                analysisRotationDegrees,
                sourceMirrored);
    }

    private static int normalizeRotation(int degrees) {
        int d = ((degrees % 360) + 360) % 360;
        if (d < 45 || d >= 315) return 0;
        if (d < 135) return 90;
        if (d < 225) return 180;
        return 270;
    }
}
