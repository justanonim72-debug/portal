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
    private static final Style[] STYLES = new Style[] {
            new Style("THERMAL HOLO", 0x5A00D9FF, 0x42FF55CC, 0xFFB8F7FF, 0xFF4FE5FF, true, true),
            new Style("VOID NEON",    0x531E073F, 0x4D00A8FF, 0xFFB789FF, 0xFF38E8FF, true, false),
            new Style("XRAY GLITCH", 0x462A7FA8, 0x3649E7FF, 0xFFF0FCFF, 0xFF82CFFF, true, false),
            new Style("HOLOGRAM+",    0x3D00C9E8, 0x353BFFD0, 0xFFC8FFFF, 0xFF63E8FF, true, true),
            new Style("RAW SHIELD",   0x1800D7FF, 0x1200D7FF, 0xFFE7FFFF, 0xFF61E8FF, false, false)
    };

    // MediaPipe's canonical 21-point hand topology.
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

    // Separate target/current arrays let the 12-20 Hz detector feed a skeleton that is rendered
    // on every CameraX preview frame. Assignment is nearest-wrist based so two hands do not swap
    // visual identities every time MediaPipe changes result ordering.
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
                raw.nodes != null && raw.nodes.length >= 6 &&
                (now - raw.producedAtNanos) < 320_000_000L;

        float targetVisibility = freshPortal ? 1f : 0f;
        float visibilityRate = freshPortal ? 18f : 9f;
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

        // getOverlayCanvas() performs synchronization work. Do not call it on completely idle
        // frames; CameraX explicitly recommends only locking the canvas when an overlay is needed.
        if (!needCanvas) return true;

        Canvas canvas = frame.getOverlayCanvas();
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);
        overlayWasDrawn = hasDebug || hasPortal;

        if (hasDebug) drawSkeletons(canvas, frame, now);

        if (!hasPortal || raw == null || raw.mode == PortalState.Mode.NONE || raw.nodes == null || raw.nodes.length < 6) {
            if (!freshPortal && visibility == 0f) {
                current = null;
                target = null;
                lastTargetStateNs = -1L;
            }
            return true;
        }

        // Portal target acceptance is detector-rate; interpolation below is preview-rate.
        // Mode (one/two detected hands) no longer resets geometry because the new portal is a
        // persistent object and a second hand is merely another manipulation tool.
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
                float maxJump = isAnchor(raw, i) ? 0.16f : 0.13f;
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

        for (int i = 0; i < current.length / 2; i++) {
            int j = i * 2;
            float dx = target[j] - current[j];
            float dy = target[j + 1] - current[j + 1];
            float d = (float) Math.hypot(dx, dy);
            float rate = d > 0.08f ? 62f : (d > 0.025f ? 42f : 26f);
            float a = 1f - (float) Math.exp(-rate * dt);
            current[j] += dx * a;
            current[j + 1] += dy * a;
        }

        PointF[] pts = new PointF[current.length / 2];
        for (int i = 0; i < pts.length; i++) {
            pts[i] = toBuffer(frame, current[i * 2], current[i * 2 + 1]);
        }

        Path path = smoothClosedPath(pts);
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

        strokePaint.setColor(withAlpha(style.edge, visibility * 0.15f));
        strokePaint.setStrokeWidth(16f);
        canvas.drawPath(path, strokePaint);
        strokePaint.setColor(withAlpha(style.accent, visibility * 0.28f));
        strokePaint.setStrokeWidth(7f);
        canvas.drawPath(path, strokePaint);
        strokePaint.setColor(withAlpha(style.edge, visibility * 0.98f));
        strokePaint.setStrokeWidth(2.2f);
        canvas.drawPath(path, strokePaint);

        drawEnergyNodes(canvas, pts, raw, style, now);
        drawParticles(canvas, pts, style, now);
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
            } else {
                // Stable left-to-right ordering for initial assignment.
                if (a[0] <= b[0]) {
                    setSkeletonTarget(0, a, now);
                    setSkeletonTarget(1, b, now);
                } else {
                    setSkeletonTarget(0, b, now);
                    setSkeletonTarget(1, a, now);
                }
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
                // Fingertips respond faster; tiny motion is damped more heavily than deliberate motion.
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

    private void drawSkeletons(Canvas canvas, Frame frame, long now) {
        for (int hand = 0; hand < 2; hand++) {
            float[] flat = skeletonCurrent[hand];
            if (flat == null || flat.length < 42 || now - skeletonSeenNs[hand] > SKELETON_STALE_NS) continue;

            PointF[] p = new PointF[21];
            for (int i = 0; i < 21; i++) {
                p[i] = toBuffer(frame, flat[i * 2], flat[i * 2 + 1]);
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
        strokePaint.setStrokeWidth(1f);
        strokePaint.setColor(withAlpha(style.accent, visibility * 0.13f));
        float step = Math.max(28f, Math.min(crop.width(), crop.height()) / 14f);
        float drift = ((now / 1_000_000L) % 1200L) / 1200f * step;
        for (float x = crop.left - step + drift; x < crop.right + step; x += step) {
            c.drawLine(x, crop.top, x, crop.bottom, strokePaint);
        }
        for (float y = crop.top - step + drift; y < crop.bottom + step; y += step) {
            c.drawLine(crop.left, y, crop.right, y, strokePaint);
        }
    }

    private void drawScan(Canvas c, Rect crop, Style style, long now) {
        float travel = ((now / 1_000_000L) % 1500L) / 1500f;
        float y = crop.top + travel * crop.height();
        strokePaint.setColor(withAlpha(style.edge, visibility * 0.30f));
        strokePaint.setStrokeWidth(3.2f);
        c.drawLine(crop.left, y, crop.right, y, strokePaint);
        strokePaint.setColor(withAlpha(style.accent, visibility * 0.11f));
        strokePaint.setStrokeWidth(12f);
        c.drawLine(crop.left, y, crop.right, y, strokePaint);
    }

    private void drawEnergyNodes(Canvas c, PointF[] pts, PortalState raw, Style style, long now) {
        float pulse = 0.5f + 0.5f * (float) Math.sin(now / 120_000_000.0);
        particlePaint.setColor(withAlpha(style.edge, visibility * 0.96f));
        for (int anchor : raw.anchors) {
            if (anchor < 0 || anchor >= pts.length) continue;
            PointF p = pts[anchor];
            c.drawCircle(p.x, p.y, 4.2f + pulse * 1.8f, particlePaint);
            particlePaint.setColor(withAlpha(style.accent, visibility * 0.16f));
            c.drawCircle(p.x, p.y, 11f + pulse * 3f, particlePaint);
            particlePaint.setColor(withAlpha(style.edge, visibility * 0.96f));
        }
    }

    private void drawParticles(Canvas c, PointF[] pts, Style style, long now) {
        if (pts.length < 2) return;
        particlePaint.setColor(withAlpha(style.accent, visibility * 0.62f));
        long tick = now / 48_000_000L;
        for (int k = 0; k < Math.min(8, pts.length); k++) {
            int edge = (int) ((tick + k * 3L) % pts.length);
            PointF a = pts[edge];
            PointF b = pts[(edge + 1) % pts.length];
            float t = ((tick * 17L + k * 29L) % 100L) / 100f;
            float x = a.x + (b.x - a.x) * t;
            float y = a.y + (b.y - a.y) * t;
            c.drawCircle(x, y, 1.1f + (k % 3) * 0.35f, particlePaint);
        }
    }

    private static Path smoothClosedPath(PointF[] p) {
        Path path = new Path();
        if (p.length < 3) return path;
        path.moveTo(p[0].x, p[0].y);
        final float tension = 0.62f;
        for (int i = 0; i < p.length; i++) {
            PointF p0 = p[(i - 1 + p.length) % p.length];
            PointF p1 = p[i];
            PointF p2 = p[(i + 1) % p.length];
            PointF p3 = p[(i + 2) % p.length];
            float c1x = p1.x + (p2.x - p0.x) * tension / 6f;
            float c1y = p1.y + (p2.y - p0.y) * tension / 6f;
            float c2x = p2.x - (p3.x - p1.x) * tension / 6f;
            float c2y = p2.y - (p3.y - p1.y) * tension / 6f;
            path.cubicTo(c1x, c1y, c2x, c2y, p2.x, p2.y);
        }
        path.close();
        return path;
    }

    /**
     * Converts a normalized coordinate in MediaPipe's ROTATED/DISPLAY-oriented image space back
     * into the OverlayEffect canvas' PRE-ROTATION full-buffer space.
     *
     * Important: the normalized landmark belongs to the full analysis image. cropRect is only the
     * region CameraX later exposes after crop. Mapping 0..1 into cropRect (the old implementation)
     * shrinks/translates every landmark and is why the debug skeleton did not sit on the hand.
     */
    private static PointF toBuffer(Frame frame, float displayX, float displayY) {
        float x = displayX;
        float y = displayY;

        // PortalGeometry stores front-camera points in final mirrored/display orientation. Undo
        // that mirror first; CameraX will apply the real output mirror again after rotation.
        if (frame.isMirroring()) x = 1f - x;

        float bx;
        float by;
        switch (frame.getRotationDegrees()) {
            case 90:
                // Inverse of clockwise 90°: display(x,y) = (1-bufferY, bufferX)
                bx = y;
                by = 1f - x;
                break;
            case 180:
                bx = 1f - x;
                by = 1f - y;
                break;
            case 270:
                bx = 1f - y;
                by = x;
                break;
            default:
                bx = x;
                by = y;
                break;
        }

        bx = clamp01(bx);
        by = clamp01(by);
        return new PointF(
                bx * frame.getSize().getWidth(),
                by * frame.getSize().getHeight());
    }

    private static float[] validSkeleton(float[] in) {
        return in != null && in.length >= 42 ? in : null;
    }

    private static float wristDistance(float[] a, float[] b) {
        if (a == null || b == null || a.length < 2 || b.length < 2) return Float.MAX_VALUE / 4f;
        return (float) Math.hypot(a[0] - b[0], a[1] - b[1]);
    }

    private static boolean isAnchor(PortalState s, int nodeIndex) {
        if (s == null || s.anchors == null) return false;
        for (int i : s.anchors) if (i == nodeIndex) return true;
        return false;
    }

    private static float clamp01(float v) {
        return Math.max(0f, Math.min(1f, v));
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
