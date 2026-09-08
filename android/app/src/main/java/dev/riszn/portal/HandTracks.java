package dev.riszn.portal;

import java.util.ArrayList;
import java.util.List;

/** Two-hand assignment and independent interaction/debug filters, in input-image pixels. */
final class HandTracks {
    static final long RETAIN_NS = 500_000_000L;
    static final int[] TIPS = {4, 8, 12, 16, 20};

    static final class Observation {
        final float[] xy;
        final String label;
        final float confidence;
        Observation(float[] xy, String label, float confidence) {
            this.xy = xy; this.label = label; this.confidence = confidence;
        }
        boolean valid() {
            if (xy == null || xy.length != 42) return false;
            for (float v : xy) if (!Float.isFinite(v)) return false;
            return scale(xy) > 1f;
        }
    }

    static final class Hand {
        final int id;
        final float[] raw, points, skeleton;
        final float scale;
        final String label;
        Hand(Track t, Observation o) {
            id = t.id; raw = o.xy; points = t.filtered.clone(); skeleton = t.debug.clone();
            scale = scale(raw); label = t.label;
        }
    }

    private final List<Track> tracks = new ArrayList<>(2);
    private int nextId = 1;
    void reset() { tracks.clear(); }

    List<Hand> update(List<Observation> incoming, long now) {
        tracks.removeIf(t -> now - t.seen > RETAIN_NS);
        List<Observation> valid = new ArrayList<>(2);
        for (Observation o : incoming) if (o.valid() && valid.size() < 2) valid.add(o);
        // Exhaustive global assignment (at most two observations/tracks), including unmatched.
        int[] assignment = {-1, -1};
        double best = Double.POSITIVE_INFINITY;
        for (int a = -1; a < tracks.size(); a++) {
            for (int b = -1; b < tracks.size(); b++) {
                if (a >= 0 && a == b) continue;
                double cost = valid.isEmpty() ? 0 : cost(a, valid.get(0), now);
                if (valid.size() > 1) cost += cost(b, valid.get(1), now);
                if (cost < best) { best = cost; assignment[0] = a; assignment[1] = b; }
            }
        }
        List<Hand> result = new ArrayList<>(2);
        List<Track> observed = new ArrayList<>(2);
        for (int i = 0; i < valid.size(); i++) {
            Observation o = valid.get(i);
            Track t = assignment[i] < 0 ? new Track(nextId++) : tracks.get(assignment[i]);
            t.update(o, now);
            observed.add(t);
            result.add(new Hand(t, o));
        }
        // Retain a missing hand's identity, but reset its filters on return so the filter's
        // catch-up motion cannot leak unseen travel into a resumed grab.
        for (Track t : tracks) t.missed = !observed.contains(t);
        for (Track t : observed) if (!tracks.contains(t) && tracks.size() < 2) tracks.add(t);
        return result;
    }

    private double cost(int slot, Observation o, long now) {
        if (slot < 0) return 2.8;
        Track t = tracks.get(slot);
        float dt = Math.min(.15f, (now - t.seen) / 1e9f);
        float scale = Math.max(scale(o.xy), scale(t.raw));
        float dx = palmX(o.xy) - (palmX(t.raw) + t.vx * dt);
        float dy = palmY(o.xy) - (palmY(t.raw) + t.vy * dt);
        float d = (float) Math.hypot(dx, dy) / scale;
        if (d > 2f) return 100;
        float mismatch = !o.label.equals("?") && !t.label.equals("?") && !o.label.equals(t.label)
                ? 1.2f * o.confidence : 0f;
        return d + mismatch;
    }

    private static final class Track {
        final int id;
        String label = "?";
        long seen;
        boolean missed;
        float vx, vy;
        float[] raw, filtered = new float[42], debug = new float[42];
        final Euro[] motion = new Euro[21], visual = new Euro[21];
        Track(int id) {
            this.id = id;
            for (int i = 0; i < 21; i++) { motion[i] = new Euro(4f, .035f); visual[i] = new Euro(3f, .025f); }
        }
        void update(Observation o, long now) {
            float dt = seen == 0 || missed ? 0 : (now - seen) / 1e9f;
            missed = false;
            if (raw != null && dt > 0 && dt < .2f) {
                vx = .5f * vx + .5f * (palmX(o.xy) - palmX(raw)) / dt;
                vy = .5f * vy + .5f * (palmY(o.xy) - palmY(raw)) / dt;
            } else { vx = 0; vy = 0; }
            if (label.equals("?") || o.confidence > .85f) label = o.label;
            raw = o.xy.clone();
            for (int i = 0; i < 21; i++) {
                motion[i].filter(raw[i*2], raw[i*2+1], dt, filtered, i*2);
                visual[i].filter(raw[i*2], raw[i*2+1], dt, debug, i*2);
            }
            seen = now;
        }
    }

    /** One Euro filter uses one speed/cutoff for each XY pair, preserving rotational symmetry. */
    static final class Euro {
        final float cutoff, beta;
        float x, y, rawX, rawY, dx, dy;
        boolean initialized;
        Euro(float cutoff, float beta) { this.cutoff = cutoff; this.beta = beta; }
        void filter(float nx, float ny, float dt, float[] out, int at) {
            if (!initialized || dt <= 0 || dt > .2f) {
                x = nx; y = ny; dx = dy = 0; initialized = true;
            } else {
                float derivativeAlpha = alpha(1f, dt);
                dx += derivativeAlpha * ((nx - rawX) / dt - dx);
                dy += derivativeAlpha * ((ny - rawY) / dt - dy);
                float a = alpha(cutoff + beta * (float)Math.hypot(dx, dy), dt);
                x += a * (nx - x); y += a * (ny - y);
            }
            rawX = nx; rawY = ny; out[at] = x; out[at+1] = y;
        }
        private static float alpha(float hz, float dt) { return 1f / (1f + 1f / (2f * (float)Math.PI * hz * dt)); }
    }

    static float distance(float[] p, int a, int b) {
        return (float)Math.hypot(p[a*2]-p[b*2], p[a*2+1]-p[b*2+1]);
    }
    static float scale(float[] p) { return (distance(p, 0, 9) + distance(p, 5, 17)) * .5f; }
    private static float palmX(float[] p) { return (p[0] + p[10] + p[18] + p[34]) * .25f; }
    private static float palmY(float[] p) { return (p[1] + p[11] + p[19] + p[35]) * .25f; }
}
