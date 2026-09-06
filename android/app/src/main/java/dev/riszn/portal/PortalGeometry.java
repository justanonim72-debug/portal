package dev.riszn.portal;

import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import java.util.List;

final class PortalGeometry {
    // Two-hand endpoint pairing can become numerically tied while hands rotate. Keep the
    // previous topology until the alternative is clearly better for several detections.
    private boolean crossPairing = false;
    private int pairingFlipVotes = 0;

    PortalState fromResult(HandLandmarkerResult result, boolean mirrorX) {
        List<List<NormalizedLandmark>> hands = result.landmarks();
        int detected = hands == null ? 0 : Math.min(2, hands.size());
        if (detected == 0) return PortalState.none();

        float[][] skeletons = new float[detected][];
        String[] handedness = new String[detected];
        for (int i = 0; i < detected; i++) {
            skeletons[i] = flattenHand(hands.get(i), mirrorX);
            handedness[i] = handednessLabel(result, i);
        }

        // IMPORTANT: detected hand count and portal mode are separate. If MediaPipe sees two
        // hands but their four control points cannot yet make a valid portal, DO NOT silently
        // fall back to one-hand mode. That was the main source of mode confusion in V1 native.
        if (detected >= 2) {
            PortalState two = twoHands(hands.get(0), hands.get(1), mirrorX, skeletons, handedness);
            return two != null ? two : PortalState.invalid(2, skeletons, handedness);
        }

        PortalState one = oneHand(hands.get(0), mirrorX, skeletons, handedness);
        return one != null ? one : PortalState.invalid(1, skeletons, handedness);
    }

    private PortalState oneHand(
            List<NormalizedLandmark> h,
            boolean mirrorX,
            float[][] skeletons,
            String[] handedness) {
        if (h.size() < 21) return null;

        // Only these two points control pose, scale and rotation.
        Vec thumb = p(h, 4, mirrorX);
        Vec index = p(h, 8, mirrorX);
        float gap = dist(thumb, index);
        if (gap < 0.028f) return null;

        Vec center = mul(add(thumb, index), 0.5f);
        Vec u = normalize(sub(index, thumb));
        Vec n = new Vec(-u.y, u.x);

        // Other fingers are SECONDARY only. They can make the rift a little fuller or thinner,
        // never translate it, rotate it, or choose which side it opens toward.
        Vec middle = p(h, 12, mirrorX);
        Vec ring = p(h, 16, mirrorX);
        Vec pinky = p(h, 20, mirrorX);
        float secondarySpread = (dist(middle, ring) + dist(ring, pinky)) * 0.5f;
        float secondaryRatio = clamp(secondarySpread / Math.max(gap, 1e-5f), 0f, 1.5f);

        float halfWidth = clamp(gap * (0.50f + secondaryRatio * 0.07f), 0.030f, 0.175f);
        float quarter = gap * 0.28f;

        // Symmetric 8-node lens. thumb/index are opposite REAL endpoints. Because the membrane
        // is symmetric around their line, there is no synthetic "opening direction" that can
        // randomly flip. This makes one-hand control behave like holding a rift between two tips.
        Vec[] nodes = new Vec[] {
                thumb,
                add(add(center, mul(u, -quarter)), mul(n, -halfWidth * 0.82f)),
                add(center, mul(n, -halfWidth)),
                add(add(center, mul(u, quarter)), mul(n, -halfWidth * 0.82f)),
                index,
                add(add(center, mul(u, quarter)), mul(n, halfWidth * 0.82f)),
                add(center, mul(n, halfWidth)),
                add(add(center, mul(u, -quarter)), mul(n, halfWidth * 0.82f))
        };

        return new PortalState(
                PortalState.Mode.ONE_HAND,
                flatten(nodes),
                new int[] {0, 4},
                skeletons,
                handedness,
                System.nanoTime(),
                1
        );
    }

    private PortalState twoHands(
            List<NormalizedLandmark> a,
            List<NormalizedLandmark> b,
            boolean mirrorX,
            float[][] skeletons,
            String[] handedness) {
        if (a.size() < 21 || b.size() < 21) return null;

        Vec aThumb = p(a, 4, mirrorX);
        Vec aIndex = p(a, 8, mirrorX);
        Vec bThumb = p(b, 4, mirrorX);
        Vec bIndex = p(b, 8, mirrorX);

        if (dist(aThumb, aIndex) < 0.022f || dist(bThumb, bIndex) < 0.022f) return null;

        // There are only two legal ways to connect the two thumb-index side segments.
        // Choose the shorter topology, but use hysteresis so topology doesn't flip on a tie.
        float straightCost = dist(aThumb, bThumb) + dist(aIndex, bIndex);
        float crossCost = dist(aThumb, bIndex) + dist(aIndex, bThumb);
        boolean rawCross = crossCost < straightCost;
        float advantage = Math.abs(straightCost - crossCost);
        if (rawCross != crossPairing && advantage > 0.035f) {
            pairingFlipVotes++;
            if (pairingFlipVotes >= 3) {
                crossPairing = rawCross;
                pairingFlipVotes = 0;
            }
        } else {
            pairingFlipVotes = 0;
        }

        Vec b0 = crossPairing ? bIndex : bThumb;
        Vec b1 = crossPairing ? bThumb : bIndex;
        Vec[] anchors = new Vec[] {aThumb, b0, b1, aIndex};

        float polygonArea = area(anchors);
        if (polygonArea < 0.0014f) return null;

        Vec center = avg(anchors);
        float bulge = clamp((float) Math.sqrt(polygonArea) * 0.085f, 0.004f, 0.020f);
        Vec[] nodes = new Vec[8];
        for (int i = 0; i < 4; i++) {
            Vec p0 = anchors[i];
            Vec p1 = anchors[(i + 1) % 4];
            Vec midpoint = mul(add(p0, p1), 0.5f);
            Vec outward = normalize(sub(midpoint, center));
            nodes[i * 2] = p0;
            nodes[i * 2 + 1] = add(midpoint, mul(outward, bulge));
        }

        return new PortalState(
                PortalState.Mode.TWO_HAND,
                flatten(nodes),
                new int[] {0, 2, 4, 6},
                skeletons,
                handedness,
                System.nanoTime(),
                2
        );
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

    private static float area(Vec[] p) {
        float a = 0f;
        for (int i = 0; i < p.length; i++) {
            Vec x = p[i];
            Vec y = p[(i + 1) % p.length];
            a += x.x * y.y - y.x * x.y;
        }
        return Math.abs(a) * 0.5f;
    }

    private static Vec avg(Vec... pts) {
        float x = 0f, y = 0f;
        for (Vec p : pts) { x += p.x; y += p.y; }
        return new Vec(x / pts.length, y / pts.length);
    }

    private static Vec add(Vec a, Vec b) { return new Vec(a.x + b.x, a.y + b.y); }
    private static Vec sub(Vec a, Vec b) { return new Vec(a.x - b.x, a.y - b.y); }
    private static Vec mul(Vec a, float s) { return new Vec(a.x * s, a.y * s); }
    private static float dist(Vec a, Vec b) { return (float) Math.hypot(a.x - b.x, a.y - b.y); }
    private static Vec normalize(Vec a) {
        float m = (float) Math.hypot(a.x, a.y);
        return m < 1e-6f ? new Vec(1f, 0f) : new Vec(a.x / m, a.y / m);
    }
    private static float clamp(float v, float lo, float hi) { return Math.max(lo, Math.min(hi, v)); }

    private static final class Vec {
        final float x, y;
        Vec(float x, float y) { this.x = x; this.y = y; }
    }
}
