package dev.riszn.portal;

import android.graphics.Matrix;

import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import java.util.ArrayList;
import java.util.List;

final class PortalGeometry {
    private static final int[] FINGER_TIPS = {8, 12, 16, 20};
    private static final int[] FINGER_DIPS = {7, 11, 15, 19};
    private static final int[] FINGER_PIPS = {6, 10, 14, 18};

    private static final float PINCH_ON = 0.34f;
    private static final float PINCH_OFF = 0.50f;
    private static final int TRIGGER_HOLD_FRAMES = 2;
    private static final int GRAB_HOLD_FRAMES = 2;
    private static final int GRAB_MISS_FRAMES = 3;
    private static final long NO_HAND_CLOSE_NS = 1_050_000_000L;

    private boolean active = false;
    private Vec center = new Vec(0.5f, 0.5f);
    private Vec axisU = new Vec(1f, 0f);
    private Vec axisV = new Vec(0f, 1f);
    private float halfWidth = 0.025f;
    private float halfHeight = 0.025f;
    private float seedHalf = 0.025f;

    private int triggerVotes = 0;
    private int triggerFingerSlot = -1;
    private Vec triggerLastCenter = null;

    private Grab grab = null;
    private int pendingHandle = -1;
    private int pendingVotes = 0;
    private Vec pendingTip = null;
    private long lastHandSeenNs = 0L;

    // Metadata of the latest ImageAnalysis frame. Geometry stays in MediaPipe's rotated/display
    // normalized space, but renderer needs this exact transform to place it on Preview/Video.
    private Matrix analysisToSensor = new Matrix();
    private int analysisWidth = 1;
    private int analysisHeight = 1;
    private int analysisRotation = 0;
    private boolean sourceMirrored = false;

    synchronized void reset() {
        active = false;
        triggerVotes = 0;
        triggerFingerSlot = -1;
        triggerLastCenter = null;
        grab = null;
        clearPendingGrab();
    }

    synchronized PortalState fromResult(
            HandLandmarkerResult result,
            boolean mirrorX,
            Matrix frameAnalysisToSensor,
            int frameWidth,
            int frameHeight,
            int frameRotation) {
        long now = System.nanoTime();
        analysisToSensor = frameAnalysisToSensor == null ? new Matrix() : new Matrix(frameAnalysisToSensor);
        analysisWidth = Math.max(1, frameWidth);
        analysisHeight = Math.max(1, frameHeight);
        analysisRotation = normalizeRotation(frameRotation);
        sourceMirrored = mirrorX;

        List<List<NormalizedLandmark>> hands = result.landmarks();
        int detected = hands == null ? 0 : Math.min(2, hands.size());

        float[][] skeletons = new float[detected][];
        String[] handedness = new String[detected];
        for (int i = 0; i < detected; i++) {
            skeletons[i] = flattenHand(hands.get(i), mirrorX);
            handedness[i] = handednessLabel(result, i);
        }

        if (detected > 0) lastHandSeenNs = now;

        if (!active) {
            updateTrigger(hands, detected, mirrorX);
            if (!active) return invalidState(detected, skeletons, handedness);
        }

        if (detected == 0) {
            if (lastHandSeenNs != 0L && now - lastHandSeenNs > NO_HAND_CLOSE_NS) {
                reset();
                return invalidState(0, skeletons, handedness);
            }
            return buildActiveState(0, skeletons, handedness);
        }

        updateStretch(hands, detected, mirrorX);
        return buildActiveState(detected, skeletons, handedness);
    }

    private PortalState invalidState(int detected, float[][] skeletons, String[] handedness) {
        return PortalState.invalid(
                detected,
                skeletons,
                handedness,
                analysisToSensor,
                analysisWidth,
                analysisHeight,
                analysisRotation,
                sourceMirrored);
    }

    private void updateTrigger(List<List<NormalizedLandmark>> hands, int detected, boolean mirrorX) {
        if (detected == 0 || hands == null) {
            triggerVotes = 0;
            triggerFingerSlot = -1;
            triggerLastCenter = null;
            return;
        }

        Pinch best = null;
        for (int h = 0; h < detected; h++) {
            Pinch p = bestPinch(hands.get(h), h, mirrorX);
            if (p != null && (best == null || p.ratio < best.ratio)) best = p;
        }

        if (best == null || best.ratio > PINCH_ON) {
            if (best == null || best.ratio > PINCH_OFF) {
                triggerVotes = 0;
                triggerFingerSlot = -1;
                triggerLastCenter = null;
            }
            return;
        }

        boolean sameGesture = triggerFingerSlot == best.fingerSlot &&
                triggerLastCenter != null && dist(triggerLastCenter, best.center) < 0.13f;
        if (sameGesture) triggerVotes++;
        else {
            triggerFingerSlot = best.fingerSlot;
            triggerVotes = 1;
        }
        triggerLastCenter = best.center;

        if (triggerVotes >= TRIGGER_HOLD_FRAMES) summon(best);
    }

    private Pinch bestPinch(List<NormalizedLandmark> h, int handIndex, boolean mirrorX) {
        if (h == null || h.size() < 21) return null;
        Vec thumb = p(h, 4, mirrorX);
        float scale = handScale(h, mirrorX);
        Pinch best = null;

        for (int slot = 0; slot < FINGER_TIPS.length; slot++) {
            Vec tip = p(h, FINGER_TIPS[slot], mirrorX);
            Vec dip = p(h, FINGER_DIPS[slot], mirrorX);
            Vec pip = p(h, FINGER_PIPS[slot], mirrorX);
            float ratio = dist(thumb, tip) / Math.max(scale, 1e-5f);
            float segment = Math.max(dist(tip, dip), dist(dip, pip));
            Vec fingerAxis = normalize(sub(tip, pip));
            Vec pinchCenter = mul(add(thumb, tip), 0.5f);
            Pinch candidate = new Pinch(handIndex, slot, ratio, pinchCenter, segment, scale, fingerAxis);
            if (best == null || candidate.ratio < best.ratio) best = candidate;
        }
        return best;
    }

    private void summon(Pinch pinch) {
        float diameter = Math.min(pinch.segmentLength * 0.88f, pinch.handScale * 0.22f);
        diameter = clamp(diameter, 0.018f, 0.060f);

        center = pinch.center;
        axisU = pinch.fingerAxis;
        if (length(axisU) < 1e-5f) axisU = new Vec(1f, 0f);
        axisV = perpendicular(axisU);
        halfWidth = diameter * 0.5f;
        halfHeight = diameter * 0.5f;
        seedHalf = diameter * 0.5f;

        active = true;
        grab = null;
        clearPendingGrab();
        triggerVotes = 0;
        triggerFingerSlot = -1;
        triggerLastCenter = null;
    }

    private void updateStretch(List<List<NormalizedLandmark>> hands, int detected, boolean mirrorX) {
        List<FingerPoint> candidates = dragCandidates(hands, detected, mirrorX);

        if (grab != null) {
            FingerPoint nearest = nearestFinger(candidates, grab.tip, 0.19f);
            if (nearest != null) {
                grab.tip = nearest.point;
                grab.misses = 0;
                applyHandleDrag(grab.handle, nearest.point);
            } else {
                grab.misses++;
                if (grab.misses >= GRAB_MISS_FRAMES) grab = null;
            }
            return;
        }

        HandleHit hit = nearestHandleHit(candidates);
        if (hit == null) {
            clearPendingGrab();
            return;
        }

        boolean samePending = pendingHandle == hit.handle && pendingTip != null &&
                dist(pendingTip, hit.point) < 0.09f;
        if (samePending) pendingVotes++;
        else {
            pendingHandle = hit.handle;
            pendingVotes = 1;
        }
        pendingTip = hit.point;

        if (pendingVotes >= GRAB_HOLD_FRAMES) {
            grab = new Grab(hit.handle, hit.point);
            clearPendingGrab();
        }
    }

    private List<FingerPoint> dragCandidates(
            List<List<NormalizedLandmark>> hands,
            int detected,
            boolean mirrorX) {
        List<FingerPoint> out = new ArrayList<>();
        for (int hand = 0; hand < detected; hand++) {
            List<NormalizedLandmark> h = hands.get(hand);
            if (h == null || h.size() < 21) continue;
            Vec thumb = p(h, 4, mirrorX);
            float scale = handScale(h, mirrorX);
            for (int slot = 0; slot < FINGER_TIPS.length; slot++) {
                Vec tip = p(h, FINGER_TIPS[slot], mirrorX);
                float pinchRatio = dist(thumb, tip) / Math.max(scale, 1e-5f);
                if (pinchRatio < PINCH_OFF) continue;
                out.add(new FingerPoint(tip));
            }
        }
        return out;
    }

    private HandleHit nearestHandleHit(List<FingerPoint> candidates) {
        if (candidates == null || candidates.isEmpty()) return null;
        Vec[] handles = handles();
        float captureRadius = clamp(Math.min(halfWidth, halfHeight) * 0.72f + 0.031f, 0.040f, 0.078f);

        HandleHit best = null;
        for (FingerPoint finger : candidates) {
            for (int handle = 0; handle < handles.length; handle++) {
                float d = dist(finger.point, handles[handle]);
                if (d <= captureRadius && (best == null || d < best.distance)) {
                    best = new HandleHit(handle, finger.point, d);
                }
            }
        }
        return best;
    }

    private FingerPoint nearestFinger(List<FingerPoint> candidates, Vec previous, float maxDistance) {
        FingerPoint best = null;
        float bestDistance = Float.MAX_VALUE;
        for (FingerPoint p : candidates) {
            float d = dist(previous, p.point);
            if (d < bestDistance && d <= maxDistance) {
                best = p;
                bestDistance = d;
            }
        }
        return best;
    }

    private void applyHandleDrag(int handle, Vec fingertip) {
        Vec[] h = handles();
        float minHalf = Math.max(seedHalf * 0.65f, 0.009f);
        float maxHalf = 0.43f;

        if (handle == 0 || handle == 2) {
            Vec fixed = h[handle == 0 ? 2 : 0];
            Vec delta = handle == 0 ? sub(fingertip, fixed) : sub(fixed, fingertip);
            float span = length(delta);
            if (span < minHalf * 2f) return;

            Vec newU = normalize(delta);
            Vec newV = perpendicular(newU);
            if (dot(newV, axisV) < 0f) newV = mul(newV, -1f);

            center = mul(add(fixed, fingertip), 0.5f);
            halfWidth = clamp(span * 0.5f, minHalf, maxHalf);
            axisU = newU;
            axisV = newV;
        } else {
            Vec fixed = h[handle == 1 ? 3 : 1];
            Vec delta = handle == 1 ? sub(fingertip, fixed) : sub(fixed, fingertip);
            float span = length(delta);
            if (span < minHalf * 2f) return;

            Vec newV = normalize(delta);
            Vec newU = new Vec(newV.y, -newV.x);
            if (dot(newU, axisU) < 0f) newU = mul(newU, -1f);

            center = mul(add(fixed, fingertip), 0.5f);
            halfHeight = clamp(span * 0.5f, minHalf, maxHalf);
            axisV = newV;
            axisU = newU;
        }

        center = new Vec(clamp(center.x, -0.20f, 1.20f), clamp(center.y, -0.20f, 1.20f));
    }

    private PortalState buildActiveState(int detected, float[][] skeletons, String[] handedness) {
        Vec[] nodes = ellipseNodes(16);
        PortalState.Mode mode = detected >= 2 ? PortalState.Mode.TWO_HAND : PortalState.Mode.ONE_HAND;
        return new PortalState(
                mode,
                flatten(nodes),
                new int[] {0, 4, 8, 12},
                skeletons,
                handedness,
                System.nanoTime(),
                detected,
                analysisToSensor,
                analysisWidth,
                analysisHeight,
                analysisRotation,
                sourceMirrored);
    }

    private Vec[] ellipseNodes(int count) {
        Vec[] out = new Vec[count];
        for (int i = 0; i < count; i++) {
            double angle = Math.PI * 2.0 * i / count;
            float c = (float) Math.cos(angle);
            float s = (float) Math.sin(angle);
            float ripple = 1f + 0.018f * (float) Math.sin(angle * 3.0);
            out[i] = add(center,
                    add(mul(axisU, halfWidth * c * ripple),
                            mul(axisV, halfHeight * s * ripple)));
        }
        return out;
    }

    private Vec[] handles() {
        return new Vec[] {
                add(center, mul(axisU, halfWidth)),
                add(center, mul(axisV, halfHeight)),
                add(center, mul(axisU, -halfWidth)),
                add(center, mul(axisV, -halfHeight))
        };
    }

    private void clearPendingGrab() {
        pendingHandle = -1;
        pendingVotes = 0;
        pendingTip = null;
    }

    private static float handScale(List<NormalizedLandmark> h, boolean mirrorX) {
        Vec wrist = p(h, 0, mirrorX);
        Vec middleMcp = p(h, 9, mirrorX);
        Vec indexMcp = p(h, 5, mirrorX);
        Vec pinkyMcp = p(h, 17, mirrorX);
        return Math.max(0.035f, (dist(wrist, middleMcp) + dist(indexMcp, pinkyMcp)) * 0.5f);
    }

    private static String handednessLabel(HandLandmarkerResult result, int index) {
        try {
            List<List<Category>> all = result.handedness();
            if (all == null || index >= all.size() || all.get(index).isEmpty()) return "?";
            Category c = all.get(index).get(0);
            return c.categoryName() + " " + Math.round(c.score() * 100f) + "%";
        } catch (Throwable ignored) {
            return "?";
        }
    }

    private static float[] flattenHand(List<NormalizedLandmark> h, boolean mirrorX) {
        int count = Math.min(21, h.size());
        float[] out = new float[count * 2];
        for (int i = 0; i < count; i++) {
            Vec v = p(h, i, mirrorX);
            out[i * 2] = v.x;
            out[i * 2 + 1] = v.y;
        }
        return out;
    }

    private static Vec p(List<NormalizedLandmark> h, int index, boolean mirrorX) {
        NormalizedLandmark l = h.get(index);
        float x = mirrorX ? 1f - l.x() : l.x();
        return new Vec(x, l.y());
    }

    private static float[] flatten(Vec[] nodes) {
        float[] out = new float[nodes.length * 2];
        for (int i = 0; i < nodes.length; i++) {
            out[i * 2] = nodes[i].x;
            out[i * 2 + 1] = nodes[i].y;
        }
        return out;
    }

    private static Vec add(Vec a, Vec b) { return new Vec(a.x + b.x, a.y + b.y); }
    private static Vec sub(Vec a, Vec b) { return new Vec(a.x - b.x, a.y - b.y); }
    private static Vec mul(Vec a, float s) { return new Vec(a.x * s, a.y * s); }
    private static float dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y; }
    private static float dist(Vec a, Vec b) { return length(sub(a, b)); }
    private static float length(Vec a) { return (float) Math.hypot(a.x, a.y); }
    private static Vec normalize(Vec a) {
        float m = length(a);
        return m < 1e-6f ? new Vec(1f, 0f) : new Vec(a.x / m, a.y / m);
    }
    private static Vec perpendicular(Vec a) { return new Vec(-a.y, a.x); }
    private static float clamp(float v, float lo, float hi) { return Math.max(lo, Math.min(hi, v)); }

    private static int normalizeRotation(int degrees) {
        int d = ((degrees % 360) + 360) % 360;
        if (d < 45 || d >= 315) return 0;
        if (d < 135) return 90;
        if (d < 225) return 180;
        return 270;
    }

    private static final class Vec {
        final float x, y;
        Vec(float x, float y) { this.x = x; this.y = y; }
    }

    private static final class Pinch {
        final int handIndex;
        final int fingerSlot;
        final float ratio;
        final Vec center;
        final float segmentLength;
        final float handScale;
        final Vec fingerAxis;

        Pinch(int handIndex, int fingerSlot, float ratio, Vec center, float segmentLength, float handScale, Vec fingerAxis) {
            this.handIndex = handIndex;
            this.fingerSlot = fingerSlot;
            this.ratio = ratio;
            this.center = center;
            this.segmentLength = segmentLength;
            this.handScale = handScale;
            this.fingerAxis = fingerAxis;
        }
    }

    private static final class FingerPoint {
        final Vec point;
        FingerPoint(Vec point) { this.point = point; }
    }

    private static final class HandleHit {
        final int handle;
        final Vec point;
        final float distance;
        HandleHit(int handle, Vec point, float distance) {
            this.handle = handle;
            this.point = point;
            this.distance = distance;
        }
    }

    private static final class Grab {
        final int handle;
        Vec tip;
        int misses = 0;
        Grab(int handle, Vec tip) {
            this.handle = handle;
            this.tip = tip;
        }
    }
}
