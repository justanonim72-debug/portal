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
    private static final int GRAB_COOLDOWN_AFTER_SUMMON = 4;
    private static final long NO_HAND_CLOSE_NS = 1_050_000_000L;

    // Portal geometry lives in an isotropic metric space: one unit means the same pixel distance
    // horizontally and vertically. MediaPipe normalized X/Y are NOT isotropic on portrait frames.
    // This is essential for finger-segment sizing and for a rectangle to actually look rectangular.
    private Vec[] corners = null;
    private boolean active = false;

    private int triggerVotes = 0;
    private int triggerFingerSlot = -1;
    private Vec triggerLastCenter = null;

    private Grab grab = null;
    private int pendingHandle = -1;
    private int pendingFingerSlot = -1;
    private int pendingVotes = 0;
    private Vec pendingTip = null;
    private int grabCooldownFrames = 0;
    private long lastHandSeenNs = 0L;

    private Matrix analysisToSensor = new Matrix();
    private int analysisWidth = 1;
    private int analysisHeight = 1;
    private int analysisRotation = 0;
    private boolean sourceMirrored = false;

    synchronized void reset() {
        active = false;
        corners = null;
        triggerVotes = 0;
        triggerFingerSlot = -1;
        triggerLastCenter = null;
        grab = null;
        grabCooldownFrames = 0;
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
            // Skeleton stays in MediaPipe normalized display space for the renderer transform.
            skeletons[i] = flattenHandNormalized(hands.get(i), mirrorX);
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
        Vec thumb = pMetric(h, 4, mirrorX);
        float scale = handScaleMetric(h, mirrorX);
        Pinch best = null;

        for (int slot = 0; slot < FINGER_TIPS.length; slot++) {
            Vec tip = pMetric(h, FINGER_TIPS[slot], mirrorX);
            Vec dip = pMetric(h, FINGER_DIPS[slot], mirrorX);
            Vec pip = pMetric(h, FINGER_PIPS[slot], mirrorX);
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
        // Maximum dimension stays below one measured finger segment.
        float maxDimension = Math.min(pinch.segmentLength * 0.88f, pinch.handScale * 0.22f);
        maxDimension = clamp(maxDimension, 0.015f, 0.055f);
        float width = maxDimension;
        float height = maxDimension * 0.58f;

        Vec u = pinch.fingerAxis;
        if (length(u) < 1e-5f) u = new Vec(1f, 0f);
        u = normalize(u);
        Vec v = perpendicular(u);
        float hw = width * 0.5f;
        float hh = height * 0.5f;
        Vec c = pinch.center;

        corners = new Vec[] {
                add(c, add(mul(u, -hw), mul(v, -hh))),
                add(c, add(mul(u,  hw), mul(v, -hh))),
                add(c, add(mul(u,  hw), mul(v,  hh))),
                add(c, add(mul(u, -hw), mul(v,  hh)))
        };

        active = true;
        grab = null;
        grabCooldownFrames = GRAB_COOLDOWN_AFTER_SUMMON;
        clearPendingGrab();
        triggerVotes = 0;
        triggerFingerSlot = -1;
        triggerLastCenter = null;
    }

    private void updateStretch(List<List<NormalizedLandmark>> hands, int detected, boolean mirrorX) {
        if (corners == null || corners.length != 4) return;
        List<FingerPoint> candidates = dragCandidates(hands, detected, mirrorX);

        if (grabCooldownFrames > 0) {
            grabCooldownFrames--;
            grab = null;
            clearPendingGrab();
            return;
        }

        if (grab != null) {
            FingerPoint match = matchingFinger(candidates, grab.fingerSlot, grab.tip, 0.17f);
            if (match != null) {
                Vec bounded = limitStep(grab.tip, match.point, 0.080f);
                Vec previous = grab.tip;
                grab.tip = bounded;
                grab.misses = 0;
                applyHandleDrag(grab.handle, previous, bounded);
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

        boolean samePending = pendingHandle == hit.handle &&
                pendingFingerSlot == hit.fingerSlot &&
                pendingTip != null && dist(pendingTip, hit.point) < 0.075f;
        if (samePending) pendingVotes++;
        else {
            pendingHandle = hit.handle;
            pendingFingerSlot = hit.fingerSlot;
            pendingVotes = 1;
        }
        pendingTip = hit.point;

        if (pendingVotes >= GRAB_HOLD_FRAMES) {
            grab = new Grab(hit.handle, hit.fingerSlot, hit.point);
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
            Vec thumb = pMetric(h, 4, mirrorX);
            float scale = handScaleMetric(h, mirrorX);
            for (int slot = 0; slot < FINGER_TIPS.length; slot++) {
                Vec tip = pMetric(h, FINGER_TIPS[slot], mirrorX);
                float pinchRatio = dist(thumb, tip) / Math.max(scale, 1e-5f);
                if (pinchRatio < PINCH_OFF) continue;
                out.add(new FingerPoint(slot, tip));
            }
        }
        return out;
    }

    private HandleHit nearestHandleHit(List<FingerPoint> candidates) {
        if (candidates == null || candidates.isEmpty()) return null;
        Vec[] handles = handles();
        float minEdge = Math.min(
                Math.min(dist(corners[0], corners[1]), dist(corners[1], corners[2])),
                Math.min(dist(corners[2], corners[3]), dist(corners[3], corners[0])));
        float captureRadius = clamp(minEdge * 0.62f + 0.018f, 0.030f, 0.070f);

        HandleHit best = null;
        for (FingerPoint finger : candidates) {
            for (int handle = 0; handle < handles.length; handle++) {
                float d = dist(finger.point, handles[handle]);
                if (d <= captureRadius && (best == null || d < best.distance)) {
                    best = new HandleHit(handle, finger.slot, finger.point, d);
                }
            }
        }
        return best;
    }

    private FingerPoint matchingFinger(List<FingerPoint> candidates, int fingerSlot, Vec previous, float maxDistance) {
        FingerPoint best = null;
        float bestDistance = Float.MAX_VALUE;
        for (FingerPoint p : candidates) {
            if (p.slot != fingerSlot) continue;
            float d = dist(previous, p.point);
            if (d < bestDistance && d <= maxDistance) {
                best = p;
                bestDistance = d;
            }
        }
        return best;
    }

    // Handles 0..3: corners. Handles 4..7: top/right/bottom/left edge midpoints.
    private void applyHandleDrag(int handle, Vec previousTip, Vec fingertip) {
        Vec[] candidate = cloneCorners(corners);

        if (handle >= 0 && handle < 4) {
            candidate[handle] = clampMetricPoint(fingertip);
        } else if (handle >= 4 && handle < 8) {
            int edge = handle - 4;
            int a = edge;
            int b = (edge + 1) % 4;
            Vec delta = sub(fingertip, previousTip);
            candidate[a] = clampMetricPoint(add(candidate[a], delta));
            candidate[b] = clampMetricPoint(add(candidate[b], delta));
        } else {
            return;
        }

        if (validQuad(candidate)) corners = candidate;
    }

    private PortalState buildActiveState(int detected, float[][] skeletons, String[] handedness) {
        if (corners == null || corners.length != 4) return invalidState(detected, skeletons, handedness);
        PortalState.Mode mode = detected >= 2 ? PortalState.Mode.TWO_HAND : PortalState.Mode.ONE_HAND;
        return new PortalState(
                mode,
                flattenCornersNormalized(corners),
                new int[] {0, 1, 2, 3},
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

    private Vec[] handles() {
        return new Vec[] {
                corners[0], corners[1], corners[2], corners[3],
                midpoint(corners[0], corners[1]),
                midpoint(corners[1], corners[2]),
                midpoint(corners[2], corners[3]),
                midpoint(corners[3], corners[0])
        };
    }

    private boolean validQuad(Vec[] q) {
        if (q == null || q.length != 4) return false;
        for (int i = 0; i < 4; i++) {
            if (dist(q[i], q[(i + 1) % 4]) < 0.007f) return false;
        }
        if (Math.abs(signedArea(q)) < 0.00010f) return false;

        float expectedSign = 0f;
        for (int i = 0; i < 4; i++) {
            Vec a = q[i];
            Vec b = q[(i + 1) % 4];
            Vec c = q[(i + 2) % 4];
            float turn = cross(sub(b, a), sub(c, b));
            if (Math.abs(turn) < 1e-6f) continue;
            float sign = Math.signum(turn);
            if (expectedSign == 0f) expectedSign = sign;
            else if (sign != expectedSign) return false;
        }
        return true;
    }

    private void clearPendingGrab() {
        pendingHandle = -1;
        pendingFingerSlot = -1;
        pendingVotes = 0;
        pendingTip = null;
    }

    // --- coordinate spaces ---------------------------------------------------------------

    private Vec pMetric(List<NormalizedLandmark> h, int index, boolean mirrorX) {
        return normalizedToMetric(pNormalized(h, index, mirrorX));
    }

    private static Vec pNormalized(List<NormalizedLandmark> h, int index, boolean mirrorX) {
        NormalizedLandmark l = h.get(index);
        float x = mirrorX ? 1f - l.x() : l.x();
        return new Vec(x, l.y());
    }

    private float handScaleMetric(List<NormalizedLandmark> h, boolean mirrorX) {
        Vec wrist = pMetric(h, 0, mirrorX);
        Vec middleMcp = pMetric(h, 9, mirrorX);
        Vec indexMcp = pMetric(h, 5, mirrorX);
        Vec pinkyMcp = pMetric(h, 17, mirrorX);
        return Math.max(0.030f, (dist(wrist, middleMcp) + dist(indexMcp, pinkyMcp)) * 0.5f);
    }

    private Vec normalizedToMetric(Vec n) {
        float rw = rotatedWidthPx();
        float rh = rotatedHeightPx();
        float base = Math.max(rw, rh);
        return new Vec(n.x * rw / base, n.y * rh / base);
    }

    private Vec metricToNormalized(Vec m) {
        float rw = rotatedWidthPx();
        float rh = rotatedHeightPx();
        float base = Math.max(rw, rh);
        return new Vec(m.x * base / rw, m.y * base / rh);
    }

    private float rotatedWidthPx() {
        return (analysisRotation == 90 || analysisRotation == 270) ? analysisHeight : analysisWidth;
    }

    private float rotatedHeightPx() {
        return (analysisRotation == 90 || analysisRotation == 270) ? analysisWidth : analysisHeight;
    }

    private Vec clampMetricPoint(Vec p) {
        float rw = rotatedWidthPx();
        float rh = rotatedHeightPx();
        float base = Math.max(rw, rh);
        float maxX = rw / base;
        float maxY = rh / base;
        float margin = 0.12f;
        return new Vec(clamp(p.x, -margin, maxX + margin), clamp(p.y, -margin, maxY + margin));
    }

    private float[] flattenCornersNormalized(Vec[] metricCorners) {
        float[] out = new float[metricCorners.length * 2];
        for (int i = 0; i < metricCorners.length; i++) {
            Vec n = metricToNormalized(metricCorners[i]);
            out[i * 2] = n.x;
            out[i * 2 + 1] = n.y;
        }
        return out;
    }

    private static float[] flattenHandNormalized(List<NormalizedLandmark> h, boolean mirrorX) {
        int count = Math.min(21, h.size());
        float[] out = new float[count * 2];
        for (int i = 0; i < count; i++) {
            Vec v = pNormalized(h, i, mirrorX);
            out[i * 2] = v.x;
            out[i * 2 + 1] = v.y;
        }
        return out;
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

    // --- math ---------------------------------------------------------------------------

    private static Vec[] cloneCorners(Vec[] source) {
        Vec[] out = new Vec[source.length];
        for (int i = 0; i < source.length; i++) out[i] = new Vec(source[i].x, source[i].y);
        return out;
    }

    private static Vec midpoint(Vec a, Vec b) { return mul(add(a, b), 0.5f); }
    private static Vec add(Vec a, Vec b) { return new Vec(a.x + b.x, a.y + b.y); }
    private static Vec sub(Vec a, Vec b) { return new Vec(a.x - b.x, a.y - b.y); }
    private static Vec mul(Vec a, float s) { return new Vec(a.x * s, a.y * s); }
    private static float cross(Vec a, Vec b) { return a.x * b.y - a.y * b.x; }
    private static float dist(Vec a, Vec b) { return length(sub(a, b)); }
    private static float length(Vec a) { return (float) Math.hypot(a.x, a.y); }
    private static Vec normalize(Vec a) {
        float m = length(a);
        return m < 1e-6f ? new Vec(1f, 0f) : new Vec(a.x / m, a.y / m);
    }
    private static Vec perpendicular(Vec a) { return new Vec(-a.y, a.x); }
    private static float clamp(float v, float lo, float hi) { return Math.max(lo, Math.min(hi, v)); }
    private static Vec limitStep(Vec from, Vec to, float maxDistance) {
        Vec delta = sub(to, from);
        float d = length(delta);
        if (d <= maxDistance || d < 1e-6f) return to;
        return add(from, mul(delta, maxDistance / d));
    }
    private static float signedArea(Vec[] q) {
        float a = 0f;
        for (int i = 0; i < q.length; i++) {
            Vec p0 = q[i];
            Vec p1 = q[(i + 1) % q.length];
            a += p0.x * p1.y - p1.x * p0.y;
        }
        return a * 0.5f;
    }

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
        final int slot;
        final Vec point;
        FingerPoint(int slot, Vec point) {
            this.slot = slot;
            this.point = point;
        }
    }

    private static final class HandleHit {
        final int handle;
        final int fingerSlot;
        final Vec point;
        final float distance;
        HandleHit(int handle, int fingerSlot, Vec point, float distance) {
            this.handle = handle;
            this.fingerSlot = fingerSlot;
            this.point = point;
            this.distance = distance;
        }
    }

    private static final class Grab {
        final int handle;
        final int fingerSlot;
        Vec tip;
        int misses = 0;
        Grab(int handle, int fingerSlot, Vec tip) {
            this.handle = handle;
            this.fingerSlot = fingerSlot;
            this.tip = tip;
        }
    }
}
