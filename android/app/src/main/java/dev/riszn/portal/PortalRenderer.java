package dev.riszn.portal;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.LinearGradient;
import android.graphics.Paint;
import android.graphics.Path;
import android.graphics.PointF;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.Shader;

import androidx.camera.effects.Frame;

import java.util.concurrent.atomic.AtomicReference;

final class PortalRenderer {
    // Keep effects readable, but visually much closer to the reference: translucent filtered panel,
    // thin bright perimeter, minimal decoration. No giant round glow / particle halo.
    private static final Style[] STYLES = new Style[] {
            new Style("THERMAL HOLO", 0x3600CFFF, 0x30FF4FCB, 0xFFE8FCFF, 0xFF55E8FF, true, false),
            new Style("VOID NEON",    0x351A083A, 0x342C7FFF, 0xFFE6E9FF, 0xFF9B76FF, true, false),
            new Style("XRAY GLITCH", 0x30216078, 0x2B54BFFF, 0xFFF5FCFF, 0xFF82D3FF, true, false),
            new Style("HOLOGRAM+",    0x3000C9E8, 0x2B39E7D0, 0xFFE9FFFF, 0xFF63E8FF, true, true),
            new Style("RAW SHIELD",   0x1100D7FF, 0x0D00D7FF, 0xFFF0FFFF, 0xFF61E8FF, false, false)
    };

    private static final int[][] HAND_CONNECTIONS = new int[][] {
            {0,1},{1,5},{5,9},{9,13},{13,17},{17,0},
            {1,2},{2,3},{3,4},
            {5,6},{6,7},{7,8},
            {9,10},{10,11},{11,12},
            {13,14},{14,15},{15,16},
            {17,18},{18,19},{19,20}
    };

    private static final long SKELETON_STALE_NS = 300_000_000L;

    private final AtomicReference<PortalState> stateRef;
    private float[] current = null;
    private float[] target = null;
    private long lastTargetStateNs = -1L;
    private float visibility = 0f;
    private long lastDrawNs = System.nanoTime();
    private int styleIndex = 0;
    private volatile boolean debug = false;
    private boolean overlayWasDrawn = false;

    private final float[][] skeletonTarget = new float[2][];
    private final float[][] skeletonCurrent = new float[2][];
    private final long[] skeletonSeenNs = new long[] {0L, 0L};
    private long lastSkeletonStateNs = -1L;

    private final Paint fillPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint strokePaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint particlePaint = new Paint(Paint.ANTI_ALIAS_FLAG);

    PortalRenderer(AtomicReference<PortalState> stateRef) {
        this.stateRef = stateRef;
        fillPaint.setStyle(Paint.Style.FILL);
        strokePaint.setStyle(Paint.Style.STROKE);
        strokePaint.setStrokeCap(Paint.Cap.ROUND);
        strokePaint.setStrokeJoin(Paint.Join.ROUND);
        particlePaint.setStyle(Paint.Style.FILL);
    }

    synchronized String nextStyle() {
        styleIndex = (styleIndex + 1) % STYLES.length;
        return STYLES[styleIndex].name;
    }

    synchronized String styleName() { return STYLES[styleIndex].name; }

    void setDebug(boolean enabled) {
        debug = enabled;
        if (!enabled) {
            skeletonTarget[0] = skeletonTarget[1] = null;
            skeletonCurrent[0] = skeletonCurrent[1] = null;
            skeletonSeenNs[0] = skeletonSeenNs[1] = 0L;
        }
    }

    void resetTrackingVisuals() {
        current = null;
        target = null;
        visibility = 0f;
        lastTargetStateNs = -1L;
        lastSkeletonStateNs = -1L;
        skeletonTarget[0] = skeletonTarget[1] = null;
        skeletonCurrent[0] = skeletonCurrent[1] = null;
        skeletonSeenNs[0] = skeletonSeenNs[1] = 0L;
    }

    boolean draw(Frame frame) {
        PortalState raw = stateRef.get();
        long now = System.nanoTime();
        float dt = Math.min(0.05f, Math.max(1f / 240f, (now - lastDrawNs) / 1_000_000_000f));
        lastDrawNs = now;

        boolean freshPortal = raw != null && raw.mode != PortalState.Mode.NONE &&
                raw.nodes != null && raw.nodes.length >= 8 &&
                (now - raw.producedAtNanos) < 320_000_000L;

        float targetVisibility = freshPortal ? 1f : 0f;
        float visibilityRate = freshPortal ? 20f : 10f;
        visibility += (targetVisibility - visibility) * (1f - (float) Math.exp(-visibilityRate * dt));
        if (!freshPortal && visibility < 0.008f) visibility = 0f;

        if (debug && raw != null && raw.producedAtNanos != lastSkeletonStateNs) {
            acceptSkeletonTargets(raw, now);
            lastSkeletonStateNs = raw.producedAtNanos;
        }
        if (debug) smoothSkeletons(dt, now);

        boolean hasDebug = debug && hasFreshSkeleton(now);
        boolean hasPortal = visibility > 0.008f && (freshPortal || current != null);
        boolean needCanvas = hasDebug || hasPortal || overlayWasDrawn;
        if (!needCanvas) return true;

        Canvas canvas = frame.getOverlayCanvas();
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        overlayWasDrawn = hasDebug || hasPortal;

        if (hasDebug && raw != null) drawSkeletons(canvas, frame, raw, now);

        if (!hasPortal || raw == null || raw.mode == PortalState.Mode.NONE || raw.nodes == null || raw.nodes.length < 8) {
            if (!freshPortal && visibility == 0f) {
                current = null;
                target = null;
                lastTargetStateNs = -1L;
            }
            return true;
        }

        if (target == null || target.length != raw.nodes.length || current == null || current.length != raw.nodes.length) {
            target = raw.nodes.clone();
            current = raw.nodes.clone();
            lastTargetStateNs = raw.producedAtNanos;
        } else if (raw.producedAtNanos != lastTargetStateNs) {
            for (int i = 0; i < raw.nodes.length / 2; i++) {
                int j = i * 2;
                float baseX = current[j];
                float baseY = current[j + 1];
                float dx = raw.nodes[j] - baseX;
                float dy = raw.nodes[j + 1] - baseY;
                float d = (float) Math.hypot(dx, dy);
                float maxJump = 0.15f;
                if (d > maxJump) {
                    float s = maxJump / d;
                    dx *= s;
                    dy *= s;
                }
                target[j] = baseX + dx;
                target[j + 1] = baseY + dy;
            }
            lastTargetStateNs = raw.producedAtNanos;
        }

        // All four corners are direct manipulation anchors, so keep them responsive. Small motion
        // gets enough damping to stop shimmer; deliberate pulls catch up quickly.
        for (int i = 0; i < current.length / 2; i++) {
            int j = i * 2;
            float dx = target[j] - current[j];
            float dy = target[j + 1] - current[j + 1];
            float d = (float) Math.hypot(dx, dy);
            float rate = d > 0.075f ? 70f : (d > 0.022f ? 48f : 28f);
            float a = 1f - (float) Math.exp(-rate * dt);
            current[j] += dx * a;
            current[j + 1] += dy * a;
        }

        PointF[] pts = new PointF[current.length / 2];
        for (int i = 0; i < pts.length; i++) {
            pts[i] = toBuffer(frame, raw, current[i * 2], current[i * 2 + 1]);
        }

        // Four-corner reference panel: almost straight sides with only tiny rounded corners.
        Path path = referenceQuadPath(pts);
        Style style = STYLES[styleIndex];
        Rect crop = frame.getCropRect();

        fillPaint.setShader(new LinearGradient(
                crop.left, crop.top,
                crop.right, crop.bottom,
                withAlpha(style.fillA, visibility),
                withAlpha(style.fillB, visibility),
                Shader.TileMode.CLAMP));
        canvas.drawPath(path, fillPaint);
        fillPaint.setShader(null);

        if (style.grid || style.scan) {
            canvas.save();
            canvas.clipPath(path);
            if (style.grid) drawGrid(canvas, crop, style, now);
            if (style.scan) drawScan(canvas, crop, style, now);
            canvas.restore();
        }

        // Thin layered perimeter like the first reference video, not a thick circular aura.
        strokePaint.setColor(withAlpha(style.accent, visibility * 0.18f));
        strokePaint.setStrokeWidth(8f);
        canvas.drawPath(path, strokePaint);
        strokePaint.setColor(withAlpha(style.edge, visibility * 0.98f));
        strokePaint.setStrokeWidth(2.0f);
        canvas.drawPath(path, strokePaint);
        strokePaint.setColor(withAlpha(style.accent, visibility * 0.62f));
        strokePaint.setStrokeWidth(0.9f);
        canvas.drawPath(path, strokePaint);

        drawEnergyNodes(canvas, pts, raw, style, now);
        return true;
    }

    private void acceptSkeletonTargets(PortalState raw, long now) {
        if (raw.skeletons == null || raw.skeletons.length == 0) return;

        float[] a = validSkeleton(raw.skeletons[0]);
        float[] b = raw.skeletons.length > 1 ? validSkeleton(raw.skeletons[1]) : null;

        if (a != null && b != null) {
            if (skeletonCurrent[0] != null && skeletonCurrent[1] != null) {
                float straight = wristDistance(a, skeletonCurrent[0]) + wristDistance(b, skeletonCurrent[1]);
                float crossed = wristDistance(a, skeletonCurrent[1]) + wristDistance(b, skeletonCurrent[0]);
                if (crossed < straight) {
                    setSkeletonTarget(0, b, now);
                    setSkeletonTarget(1, a, now);
                } else {
                    setSkeletonTarget(0, a, now);
                    setSkeletonTarget(1, b, now);
                }
            } else if (a[0] <= b[0]) {
                setSkeletonTarget(0, a, now);
                setSkeletonTarget(1, b, now);
            } else {
                setSkeletonTarget(0, b, now);
                setSkeletonTarget(1, a, now);
            }
            return;
        }

        float[] only = a != null ? a : b;
        if (only == null) return;
        int slot;
        if (skeletonCurrent[0] == null && skeletonCurrent[1] == null) slot = 0;
        else if (skeletonCurrent[0] == null) slot = 0;
        else if (skeletonCurrent[1] == null) slot = 1;
        else slot = wristDistance(only, skeletonCurrent[0]) <= wristDistance(only, skeletonCurrent[1]) ? 0 : 1;
        setSkeletonTarget(slot, only, now);
    }

    private void setSkeletonTarget(int slot, float[] incoming, long now) {
        skeletonTarget[slot] = incoming.clone();
        skeletonSeenNs[slot] = now;
        if (skeletonCurrent[slot] == null || skeletonCurrent[slot].length != incoming.length) {
            skeletonCurrent[slot] = incoming.clone();
        }
    }

    private void smoothSkeletons(float dt, long now) {
        for (int slot = 0; slot < 2; slot++) {
            float[] cur = skeletonCurrent[slot];
            float[] tgt = skeletonTarget[slot];
            if (cur == null || tgt == null || cur.length != tgt.length) continue;
            if (now - skeletonSeenNs[slot] > SKELETON_STALE_NS) continue;

            for (int i = 0; i < cur.length / 2; i++) {
                int j = i * 2;
                float dx = tgt[j] - cur[j];
                float dy = tgt[j + 1] - cur[j + 1];
                float d = (float) Math.hypot(dx, dy);
                boolean tip = i == 4 || i == 8 || i == 12 || i == 16 || i == 20;
                float rate;
                if (tip) rate = d > 0.055f ? 58f : (d > 0.018f ? 40f : 24f);
                else rate = d > 0.055f ? 46f : (d > 0.018f ? 32f : 20f);
                float alpha = 1f - (float) Math.exp(-rate * dt);
                cur[j] += dx * alpha;
                cur[j + 1] += dy * alpha;
            }
        }
    }

    private boolean hasFreshSkeleton(long now) {
        for (int slot = 0; slot < 2; slot++) {
            if (skeletonCurrent[slot] != null && now - skeletonSeenNs[slot] <= SKELETON_STALE_NS) return true;
        }
        return false;
    }

    private void drawSkeletons(Canvas canvas, Frame frame, PortalState raw, long now) {
        for (int hand = 0; hand < 2; hand++) {
            float[] flat = skeletonCurrent[hand];
            if (flat == null || flat.length < 42 || now - skeletonSeenNs[hand] > SKELETON_STALE_NS) continue;

            PointF[] p = new PointF[21];
            for (int i = 0; i < 21; i++) {
                p[i] = toBuffer(frame, raw, flat[i * 2], flat[i * 2 + 1]);
            }

            int lineColor = hand == 0 ? 0xFF59E9FF : 0xFFFF63D8;
            strokePaint.setColor(withAlpha(lineColor, 0.90f));
            strokePaint.setStrokeWidth(3.0f);
            for (int[] edge : HAND_CONNECTIONS) {
                PointF p0 = p[edge[0]];
                PointF p1 = p[edge[1]];
                canvas.drawLine(p0.x, p0.y, p1.x, p1.y, strokePaint);
            }

            for (int i = 0; i < p.length; i++) {
                boolean controlTip = i == 4 || i == 8 || i == 12 || i == 16 || i == 20;
                particlePaint.setColor(controlTip ? 0xFFFFFF55 : withAlpha(lineColor, 0.94f));
                canvas.drawCircle(p[i].x, p[i].y, controlTip ? 5.8f : 2.8f, particlePaint);
            }
        }
    }

    private void drawGrid(Canvas c, Rect crop, Style style, long now) {
        strokePaint.setStrokeWidth(0.8f);
        strokePaint.setColor(withAlpha(style.accent, visibility * 0.09f));
        float step = Math.max(32f, Math.min(crop.width(), crop.height()) / 12f);
        float drift = ((now / 1_000_000L) % 1400L) / 1400f * step;
        for (float x = crop.left - step + drift; x < crop.right + step; x += step) {
            c.drawLine(x, crop.top, x, crop.bottom, strokePaint);
        }
        for (float y = crop.top - step + drift; y < crop.bottom + step; y += step) {
            c.drawLine(crop.left, y, crop.right, y, strokePaint);
        }
    }

    private void drawScan(Canvas c, Rect crop, Style style, long now) {
        float travel = ((now / 1_000_000L) % 1650L) / 1650f;
        float y = crop.top + travel * crop.height();
        strokePaint.setColor(withAlpha(style.edge, visibility * 0.20f));
        strokePaint.setStrokeWidth(2.0f);
        c.drawLine(crop.left, y, crop.right, y, strokePaint);
        strokePaint.setColor(withAlpha(style.accent, visibility * 0.07f));
        strokePaint.setStrokeWidth(8f);
        c.drawLine(crop.left, y, crop.right, y, strokePaint);
    }

    private void drawEnergyNodes(Canvas c, PointF[] pts, PortalState raw, Style style, long now) {
        if (pts.length < 4) return;
        float pulse = 0.5f + 0.5f * (float) Math.sin(now / 150_000_000.0);
        for (int anchor : raw.anchors) {
            if (anchor < 0 || anchor >= pts.length) continue;
            PointF p = pts[anchor];
            particlePaint.setColor(withAlpha(style.edge, visibility * 0.95f));
            c.drawCircle(p.x, p.y, 2.4f + pulse * 0.7f, particlePaint);
            particlePaint.setColor(withAlpha(style.accent, visibility * 0.10f));
            c.drawCircle(p.x, p.y, 6.0f + pulse * 1.5f, particlePaint);
        }
    }

    /**
     * Reference-style four-sided portal. Sides stay straight. Corners only get a very small radius
     * so a stretched rectangle/trapezoid never turns into the oval/leaf shape from old builds.
     */
    private static Path referenceQuadPath(PointF[] p) {
        Path path = new Path();
        if (p == null || p.length < 4) return path;
        if (p.length != 4) {
            path.moveTo(p[0].x, p[0].y);
            for (int i = 1; i < p.length; i++) path.lineTo(p[i].x, p[i].y);
            path.close();
            return path;
        }

        float[] radius = new float[4];
        for (int i = 0; i < 4; i++) {
            PointF prev = p[(i + 3) % 4];
            PointF cur = p[i];
            PointF next = p[(i + 1) % 4];
            float a = distance(cur, prev);
            float b = distance(cur, next);
            radius[i] = Math.min(10f, Math.min(a, b) * 0.075f);
        }

        PointF start = toward(p[0], p[1], radius[0]);
        path.moveTo(start.x, start.y);
        for (int step = 1; step <= 4; step++) {
            int i = step % 4;
            PointF prev = p[(i + 3) % 4];
            PointF cur = p[i];
            PointF next = p[(i + 1) % 4];
            PointF approach = toward(cur, prev, radius[i]);
            PointF depart = toward(cur, next, radius[i]);
            path.lineTo(approach.x, approach.y);
            path.quadTo(cur.x, cur.y, depart.x, depart.y);
        }
        path.close();
        return path;
    }

    private static PointF toward(PointF from, PointF to, float distance) {
        float dx = to.x - from.x;
        float dy = to.y - from.y;
        float d = (float) Math.hypot(dx, dy);
        if (d < 1e-4f || distance <= 0f) return new PointF(from.x, from.y);
        float t = Math.min(1f, distance / d);
        return new PointF(from.x + dx * t, from.y + dy * t);
    }

    private static float distance(PointF a, PointF b) {
        return (float) Math.hypot(a.x - b.x, a.y - b.y);
    }

    /**
     * Exact cross-use-case mapping:
     * MediaPipe rotated/display normalized -> undo app mirror -> undo MediaPipe rotation ->
     * ImageAnalysis buffer pixels -> camera sensor -> OverlayEffect buffer pixels.
     */
    private static PointF toBuffer(Frame frame, PortalState raw, float displayX, float displayY) {
        float x = displayX;
        float y = displayY;
        if (raw.sourceMirrored) x = 1f - x;

        float ax;
        float ay;
        switch (raw.analysisRotationDegrees) {
            case 90:
                ax = y;
                ay = 1f - x;
                break;
            case 180:
                ax = 1f - x;
                ay = 1f - y;
                break;
            case 270:
                ax = 1f - y;
                ay = x;
                break;
            default:
                ax = x;
                ay = y;
                break;
        }

        float[] point = new float[] {
                ax * raw.analysisWidth,
                ay * raw.analysisHeight
        };
        raw.analysisToSensor.mapPoints(point);
        frame.getSensorToBufferTransform().mapPoints(point);
        return new PointF(point[0], point[1]);
    }

    private static float[] validSkeleton(float[] in) {
        return in != null && in.length >= 42 ? in : null;
    }

    private static float wristDistance(float[] a, float[] b) {
        if (a == null || b == null || a.length < 2 || b.length < 2) return Float.MAX_VALUE / 4f;
        return (float) Math.hypot(a[0] - b[0], a[1] - b[1]);
    }

    private static int withAlpha(int color, float alpha) {
        int a = Math.max(0, Math.min(255, Math.round(Color.alpha(color) * alpha)));
        return (color & 0x00FFFFFF) | (a << 24);
    }

    private static final class Style {
        final String name;
        final int fillA;
        final int fillB;
        final int edge;
        final int accent;
        final boolean scan;
        final boolean grid;

        Style(String name, int fillA, int fillB, int edge, int accent, boolean scan, boolean grid) {
            this.name = name;
            this.fillA = fillA;
            this.fillB = fillB;
            this.edge = edge;
            this.accent = accent;
            this.scan = scan;
            this.grid = grid;
        }
    }
}
