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
            new Style("VOID NEON",   0x531E073F, 0x4D00A8FF, 0xFFB789FF, 0xFF38E8FF, true, false),
            new Style("XRAY GLITCH", 0x462A7FA8, 0x3649E7FF, 0xFFF0FCFF, 0xFF82CFFF, true, false),
            new Style("HOLOGRAM+",    0x3D00C9E8, 0x353BFFD0, 0xFFC8FFFF, 0xFF63E8FF, true, true),
            new Style("RAW SHIELD",   0x1800D7FF, 0x1200D7FF, 0xFFE7FFFF, 0xFF61E8FF, false, false)
    };

    private final AtomicReference<PortalState> stateRef;
    private float[] current = null;
    private PortalState.Mode currentMode = PortalState.Mode.NONE;
    private float visibility = 0f;
    private long lastDrawNs = System.nanoTime();
    private int styleIndex = 0;
    private volatile boolean debug = false;

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
    void setDebug(boolean enabled) { debug = enabled; }

    boolean draw(Frame frame) {
        Canvas canvas = frame.getOverlayCanvas();
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR);

        PortalState raw = stateRef.get();
        long now = System.nanoTime();
        float dt = Math.min(0.05f, Math.max(1f / 240f, (now - lastDrawNs) / 1_000_000_000f));
        lastDrawNs = now;

        boolean fresh = raw != null && raw.mode != PortalState.Mode.NONE &&
                (now - raw.producedAtNanos) < 260_000_000L;
        float targetVisibility = fresh ? 1f : 0f;
        float visibilityRate = fresh ? 14f : 7f;
        visibility += (targetVisibility - visibility) * (1f - (float) Math.exp(-visibilityRate * dt));

        if (raw == null || raw.nodes.length < 6 || visibility < 0.01f) return true;
        if (current == null || current.length != raw.nodes.length || currentMode != raw.mode) {
            current = raw.nodes.clone();
            currentMode = raw.mode;
        } else if (fresh) {
            for (int i = 0; i < current.length / 2; i++) {
                int j = i * 2;
                float dx = raw.nodes[j] - current[j];
                float dy = raw.nodes[j + 1] - current[j + 1];
                float d = (float) Math.hypot(dx, dy);
                boolean anchor = isAnchor(raw, i);
                float rate;
                if (anchor) rate = d > 0.08f ? 72f : (d > 0.025f ? 52f : 34f);
                else rate = d > 0.08f ? 48f : (d > 0.025f ? 32f : 20f);
                float a = 1f - (float) Math.exp(-rate * dt);
                current[j] += dx * a;
                current[j + 1] += dy * a;
            }
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

        // Multi-pass border gives a bright energy edge without expensive blur/shadow filters.
        strokePaint.setColor(withAlpha(style.edge, visibility * 0.16f));
        strokePaint.setStrokeWidth(18f);
        canvas.drawPath(path, strokePaint);
        strokePaint.setColor(withAlpha(style.accent, visibility * 0.30f));
        strokePaint.setStrokeWidth(8f);
        canvas.drawPath(path, strokePaint);
        strokePaint.setColor(withAlpha(style.edge, visibility * 0.98f));
        strokePaint.setStrokeWidth(2.4f);
        canvas.drawPath(path, strokePaint);

        drawEnergyNodes(canvas, pts, raw, style, now);
        drawParticles(canvas, pts, style, now);
        return true;
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
        strokePaint.setColor(withAlpha(style.edge, visibility * 0.32f));
        strokePaint.setStrokeWidth(3.5f);
        c.drawLine(crop.left, y, crop.right, y, strokePaint);
        strokePaint.setColor(withAlpha(style.accent, visibility * 0.12f));
        strokePaint.setStrokeWidth(13f);
        c.drawLine(crop.left, y, crop.right, y, strokePaint);
    }

    private void drawEnergyNodes(Canvas c, PointF[] pts, PortalState raw, Style style, long now) {
        float pulse = 0.5f + 0.5f * (float) Math.sin(now / 120_000_000.0);
        particlePaint.setColor(withAlpha(style.edge, visibility * 0.96f));
        for (int anchor : raw.anchors) {
            if (anchor < 0 || anchor >= pts.length) continue;
            PointF p = pts[anchor];
            c.drawCircle(p.x, p.y, 5f + pulse * 2f, particlePaint);
            particlePaint.setColor(withAlpha(style.accent, visibility * 0.18f));
            c.drawCircle(p.x, p.y, 13f + pulse * 4f, particlePaint);
            particlePaint.setColor(withAlpha(style.edge, visibility * 0.96f));
        }

        if (debug) {
            particlePaint.setColor(withAlpha(0xFFFFFF54, visibility));
            for (PointF p : pts) c.drawCircle(p.x, p.y, 3.5f, particlePaint);
        }
    }

    private void drawParticles(Canvas c, PointF[] pts, Style style, long now) {
        if (pts.length < 2) return;
        particlePaint.setColor(withAlpha(style.accent, visibility * 0.70f));
        long tick = now / 40_000_000L;
        for (int k = 0; k < Math.min(12, pts.length * 2); k++) {
            int edge = (int) ((tick + k * 3L) % pts.length);
            PointF a = pts[edge];
            PointF b = pts[(edge + 1) % pts.length];
            float t = ((tick * 17L + k * 29L) % 100L) / 100f;
            float x = a.x + (b.x - a.x) * t;
            float y = a.y + (b.y - a.y) * t;
            float r = 1.2f + ((k * 7) % 5) * 0.35f;
            c.drawCircle(x, y, r, particlePaint);
        }
    }

    private static Path smoothClosedPath(PointF[] p) {
        Path path = new Path();
        if (p.length < 3) return path;
        path.moveTo(p[0].x, p[0].y);
        final float tension = 0.72f;
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

    private static PointF toBuffer(Frame frame, float displayX, float displayY) {
        float x = displayX;
        float y = displayY;
        if (frame.isMirroring()) x = 1f - x;

        float bx, by;
        switch (frame.getRotationDegrees()) {
            case 90:
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

        Rect crop = frame.getCropRect();
        return new PointF(
                crop.left + bx * crop.width(),
                crop.top + by * crop.height());
    }

    private static boolean isAnchor(PortalState s, int nodeIndex) {
        for (int i : s.anchors) if (i == nodeIndex) return true;
        return false;
    }

    private static int withAlpha(int color, float alpha) {
        int a = Math.max(0, Math.min(255, Math.round(Color.alpha(color) * alpha)));
        return (color & 0x00FFFFFF) | (a << 24);
    }

    private static final class Style {
        final String name;
        final int fillA, fillB, edge, accent;
        final boolean scan, grid;

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
