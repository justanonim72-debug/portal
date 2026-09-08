package dev.riszn.portal;

import android.graphics.Matrix;

/** Immutable publication from inference thread to rendering thread. Coordinates are analysis pixels. */
final class PortalState {
    final PortalInteraction.Phase phase;
    final float[] nodes;
    final float[][] skeletons;
    final int[] handIds, grabbedHandles;
    final int hands, analysisWidth, analysisHeight;
    final long producedAtNanos, analysisStartedAtNanos;
    final Matrix analysisToSensor;

    PortalState(PortalInteraction.Phase phase, float[] nodes, float[][] skeletons, int[] handIds,
            int[] grabbedHandles, int hands, long produced, long started, Matrix analysisToSensor,
            int width, int height) {
        this.phase = phase; this.nodes = nodes; this.skeletons = skeletons; this.handIds = handIds;
        this.grabbedHandles = grabbedHandles; this.hands = hands;
        producedAtNanos = produced; analysisStartedAtNanos = started;
        this.analysisToSensor = new Matrix(analysisToSensor);
        analysisWidth = width; analysisHeight = height;
    }
    boolean visible() { return nodes.length == 8; }
    static PortalState none() {
        return new PortalState(PortalInteraction.Phase.IDLE,new float[0],new float[0][],new int[0],
                new int[0],0,System.nanoTime(),0,new Matrix(),1,1);
    }
}
