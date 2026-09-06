package dev.riszn.portal;

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

final class PortalGeometry {
    private int oneHandSide = 0;
    private int sideFlipVotes = 0;

    PortalState fromResult(HandLandmarkerResult result, boolean mirrorX) {
        List<List<NormalizedLandmark>> hands = result.landmarks();
        if (hands == null || hands.isEmpty()) return PortalState.none();
        if (hands.size() >= 2) {
            PortalState two = twoHands(hands.get(0), hands.get(1), mirrorX);
            if (two != null) return two;
        }
        PortalState one = oneHand(hands.get(0), mirrorX);
        return one != null ? one : PortalState.none();
    }

    private PortalState oneHand(List<NormalizedLandmark> h, boolean mirrorX) {
        if (h.size() < 21) return null;

        Vec thumb = p(h, 4, mirrorX);
        Vec index = p(h, 8, mirrorX);
        Vec thumbIp = p(h, 3, mirrorX);
        Vec indexPip = p(h, 6, mirrorX);
        Vec middleTip = p(h, 12, mirrorX);
        Vec ringTip = p(h, 16, mirrorX);
        Vec pinkyTip = p(h, 20, mirrorX);
        Vec middleRoot = p(h, 9, mirrorX);
        Vec ringRoot = p(h, 13, mirrorX);
        Vec pinkyRoot = p(h, 17, mirrorX);

        float gap = dist(thumb, index);
        if (gap < 0.035f) return null;

        Vec center = mul(add(thumb, index), 0.5f);
        Vec u = normalize(sub(index, thumb));
        Vec baseN = new Vec(-u.y, u.x);

        // Primary opening direction comes from the two controlling fingers themselves:
        // fingertip midpoint vs their proximal joints. Other fingers do NOT move anchors.
        Vec supportMid = mul(add(thumbIp, indexPip), 0.5f);
        float signed = dot(sub(center, supportMid), baseN);
        int rawSide = signed >= 0f ? 1 : -1;
        if (Math.abs(signed) < 0.012f && oneHandSide != 0) rawSide = oneHandSide;
        if (oneHandSide == 0) {
            oneHandSide = rawSide;
        } else if (rawSide != oneHandSide && Math.abs(signed) > 0.018f) {
            sideFlipVotes++;
            if (sideFlipVotes >= 4) {
                oneHandSide = rawSide;
                sideFlipVotes = 0;
            }
        } else {
            sideFlipVotes = 0;
        }
        Vec n = mul(baseN, oneHandSide == 0 ? rawSide : oneHandSide);

        // Thumb-index distance controls the portal scale. Secondary fingers only add a small bend.
        float depth = clamp(gap * 1.35f, 0.10f, 0.38f);
        Vec secTips = avg(middleTip, ringTip, pinkyTip);
        Vec secRoots = avg(middleRoot, ringRoot, pinkyRoot);
        float secondaryExtension = dot(sub(secTips, secRoots), n);
        float bend = clamp(secondaryExtension * 0.13f, -0.018f, 0.028f);
        float spread = dist(middleTip, pinkyTip);
        float farHalf = clamp(gap * 0.46f + spread * 0.09f, gap * 0.40f, gap * 0.60f);
        float skew = clamp(dot(sub(secTips, center), u) * 0.08f, -gap * 0.07f, gap * 0.07f);

        Vec farCenter = add(add(center, mul(n, depth + bend)), mul(u, skew));

        // Organic 8-node membrane. Node 0/2 are the actual fingertip anchors and never synthesized.
        Vec[] nodes = new Vec[] {
                thumb,
                add(lerp(thumb, index, 0.5f), mul(n, -gap * 0.045f)),
                index,
                add(add(index, mul(n, depth * 0.46f)), mul(u, gap * 0.07f)),
                add(farCenter, mul(u, farHalf)),
                add(farCenter, mul(n, depth * 0.075f)),
                add(farCenter, mul(u, -farHalf)),
                add(add(thumb, mul(n, depth * 0.46f)), mul(u, -gap * 0.07f))
        };

        return new PortalState(
                PortalState.Mode.ONE_HAND,
                flatten(nodes),
                new int[] {0, 2},
                System.nanoTime(),
                1
        );
    }

    private PortalState twoHands(List<NormalizedLandmark> a, List<NormalizedLandmark> b, boolean mirrorX) {
        if (a.size() < 21 || b.size() < 21) return null;

        List<Vec> anchors = new ArrayList<>(4);
        anchors.add(p(a, 4, mirrorX));
        anchors.add(p(a, 8, mirrorX));
        anchors.add(p(b, 4, mirrorX));
        anchors.add(p(b, 8, mirrorX));

        Vec center = avg(anchors.get(0), anchors.get(1), anchors.get(2), anchors.get(3));
        anchors.sort(Comparator.comparingDouble(v -> Math.atan2(v.y - center.y, v.x - center.x)));

        float polygonArea = area(anchors);
        if (polygonArea < 0.0020f) return null;

        float bulge = clamp((float) Math.sqrt(polygonArea) * 0.055f, 0.004f, 0.025f);
        Vec[] nodes = new Vec[8];
        for (int i = 0; i < 4; i++) {
            Vec p0 = anchors.get(i);
            Vec p1 = anchors.get((i + 1) % 4);
            Vec midpoint = mul(add(p0, p1), 0.5f);
            Vec outward = normalize(sub(midpoint, center));
            nodes[i * 2] = p0;
            nodes[i * 2 + 1] = add(midpoint, mul(outward, bulge));
        }

        return new PortalState(
                PortalState.Mode.TWO_HAND,
                flatten(nodes),
                new int[] {0, 2, 4, 6},
                System.nanoTime(),
                2
        );
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

    private static float area(List<Vec> p) {
        float a = 0f;
        for (int i = 0; i < p.size(); i++) {
            Vec x = p.get(i);
            Vec y = p.get((i + 1) % p.size());
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
    private static float dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y; }
    private static float dist(Vec a, Vec b) { return (float) Math.hypot(a.x - b.x, a.y - b.y); }
    private static Vec normalize(Vec a) {
        float n = (float) Math.hypot(a.x, a.y);
        return n < 1e-6f ? new Vec(1f, 0f) : new Vec(a.x / n, a.y / n);
    }
    private static Vec lerp(Vec a, Vec b, float t) { return new Vec(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t); }
    private static float clamp(float v, float lo, float hi) { return Math.max(lo, Math.min(hi, v)); }

    private static final class Vec {
        final float x, y;
        Vec(float x, float y) { this.x = x; this.y = y; }
    }
}
