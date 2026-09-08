package dev.riszn.portal;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Deterministic interaction reducer. Every length and point is in unrotated analysis pixels. */
final class PortalInteraction {
    enum Phase { IDLE, TRIGGER_CANDIDATE, SEEDED, ACTIVE, GRABBING, RESIZING }
    static final float PINCH_ON = .22f, PINCH_OFF = .38f;
    static final long TRIGGER_NS = 90_000_000L, CONTACT_NS = 70_000_000L;
    static final long RELEASE_NS = 70_000_000L, DROPOUT_NS = 250_000_000L;

    static final class Grip {
        final int handId, finger, handle;
        float x, y;
        long seen, releaseSince;
        boolean missed;
        Grip(int handId, int finger, int handle, float x, float y, long now) {
            this.handId = handId; this.finger = finger; this.handle = handle;
            this.x = x; this.y = y; seen = now;
        }
        int mask() { return handle < 4 ? 1 << handle : (1 << (handle-4)) | (1 << ((handle-3)%4)); }
    }
    private static final class Contact {
        final int handle;
        final long since;
        Contact(int handle, long since) { this.handle = handle; this.since = since; }
    }
    private Phase phase = Phase.IDLE;
    private float[] corners;
    private float minEdge, minArea;
    private int triggerHand = -1, triggerFinger = -1;
    private long triggerSince, releaseSince, previousTime;
    private final List<Grip> grips = new ArrayList<>(4);
    private final Map<Long, Contact> contacts = new HashMap<>();
    private final Set<Long> blocked = new HashSet<>();

    Phase phase() { return phase; }
    float[] corners() { return corners == null ? new float[0] : corners.clone(); }
    List<Grip> grips() { return new ArrayList<>(grips); }
    void reset() {
        phase = Phase.IDLE; corners = null; triggerHand = triggerFinger = -1;
        triggerSince = releaseSince = previousTime = 0;
        grips.clear(); contacts.clear(); blocked.clear();
    }

    void update(List<HandTracks.Hand> hands, long now) {
        if (now <= previousTime) return; // stale analysis must not move time or geometry backwards
        float dt = previousTime == 0 ? 1f/30f : Math.min(.1f, (now-previousTime)/1e9f);
        previousTime = now;
        if (phase == Phase.IDLE || phase == Phase.TRIGGER_CANDIDATE) {
            trigger(hands, now); return;
        }
        if (phase == Phase.SEEDED) {
            HandTracks.Hand hand = find(hands, triggerHand);
            boolean released = hand == null || ratio(hand.raw, triggerFinger, hand.scale) > PINCH_OFF;
            if (released) {
                if (releaseSince == 0) releaseSince = now;
                if (now-releaseSince >= RELEASE_NS) {
                    phase = Phase.ACTIVE;
                    // Trigger tips must leave the seed before being allowed to grab it again.
                    blocked.add(key(triggerHand, 4)); blocked.add(key(triggerHand, triggerFinger));
                }
            } else releaseSince = 0;
            return;
        }
        manipulate(hands, now, dt);
        phase = grips.isEmpty() ? Phase.ACTIVE : grips.size() == 1 ? Phase.GRABBING : Phase.RESIZING;
    }

    private void trigger(List<HandTracks.Hand> hands, long now) {
        HandTracks.Hand selected = find(hands, triggerHand);
        if (selected != null && triggerFinger >= 8 && ratio(selected.raw, triggerFinger, selected.scale) <= PINCH_OFF) {
            if (now-triggerSince >= TRIGGER_NS) seed(selected, triggerFinger);
            return;
        }
        triggerHand = triggerFinger = -1;
        phase = Phase.IDLE;
        float best = PINCH_ON;
        for (HandTracks.Hand h : hands) for (int finger = 8; finger <= 20; finger += 4) {
            float r = ratio(h.raw, finger, h.scale);
            if (r <= best && HandTracks.distance(h.raw, finger, finger-1) > 1f) {
                best = r; selected = h; triggerHand = h.id; triggerFinger = finger;
            }
        }
        if (triggerHand >= 0) { triggerSince = now; phase = Phase.TRIGGER_CANDIDATE; }
    }

    private void seed(HandTracks.Hand h, int finger) {
        float[] p = h.raw;
        // Use the actual distal segment, with no minimum size that can exceed the finger.
        float segment = HandTracks.distance(p, finger, finger-1);
        if (segment <= 1f) { phase = Phase.IDLE; return; }
        float ux = (p[finger*2]-p[(finger-1)*2])/segment;
        float uy = (p[finger*2+1]-p[(finger-1)*2+1])/segment;
        float cx = (p[8]+p[finger*2])*.5f, cy = (p[9]+p[finger*2+1])*.5f;
        // Even the diagonal is smaller than the visible segment.
        float width = Math.min(segment*.72f, h.scale*.22f), height = width*.65f;
        corners = new float[8];
        for (int i = 0; i < 4; i++) {
            float a = (i == 0 || i == 3 ? -.5f : .5f)*width;
            float b = (i < 2 ? -.5f : .5f)*height;
            corners[i*2] = cx + a*ux-b*uy; corners[i*2+1] = cy+a*uy+b*ux;
        }
        minEdge = Math.min(width, height)*.15f;
        minArea = width*height*.08f;
        phase = Phase.SEEDED; releaseSince = 0;
    }

    private void manipulate(List<HandTracks.Hand> hands, long now, float dt) {
        float[] desired = corners.clone();
        int occupied = 0;
        for (Iterator<Grip> it = grips.iterator(); it.hasNext();) {
            Grip g = it.next();
            HandTracks.Hand h = find(hands, g.handId);
            if (h == null) {
                g.missed = true;
                if (now-g.seen > DROPOUT_NS) { it.remove(); blocked.add(key(g.handId,g.finger)); }
                else occupied |= g.mask();
                continue;
            }
            float x = h.points[g.finger*2], y = h.points[g.finger*2+1];
            if (!extended(h, g.finger)) {
                if (g.releaseSince == 0) g.releaseSince = now;
                if (now-g.releaseSince >= RELEASE_NS) {
                    it.remove(); blocked.add(key(g.handId,g.finger)); continue;
                }
                g.missed = true; occupied |= g.mask(); continue;
            }
            g.releaseSince = 0;
            // Freeze/rebase after a miss; never apply unseen travel as a giant drag.
            float dx = x-g.x, dy = y-g.y;
            float distance = (float)Math.hypot(dx,dy);
            float maxTravel = h.scale*(.35f + 6f*dt);
            if (!g.missed && distance <= maxTravel && now-g.seen <= DROPOUT_NS) {
                for (int v = 0; v < 4; v++) if ((g.mask() & (1 << v)) != 0) {
                    desired[v*2] += dx; desired[v*2+1] += dy;
                }
            }
            g.x = x; g.y = y; g.seen = now; g.missed = false;
            occupied |= g.mask();
        }
        constrain(desired);
        Set<Long> seenContacts = new HashSet<>();
        Set<Long> visibleFingers = new HashSet<>();
        for (HandTracks.Hand h : hands) for (int finger : HandTracks.TIPS) {
            long key = key(h.id, finger); visibleFingers.add(key);
            if (owns(h.id, finger)) continue;
            float radius = Math.max(2f, h.scale*.12f);
            float x = h.points[finger*2], y = h.points[finger*2+1];
            int hit = hit(x,y,radius);
            if (blocked.contains(key)) {
                if (hit < 0) blocked.remove(key);
                continue;
            }
            if (hit < 0 || !extended(h,finger) || (mask(hit) & occupied) != 0) continue;
            seenContacts.add(key);
            Contact c = contacts.get(key);
            if (c == null || c.handle != hit) { contacts.put(key,new Contact(hit,now)); continue; }
            if (now-c.since >= CONTACT_NS) {
                Grip grip = new Grip(h.id,finger,hit,x,y,now);
                grips.add(grip); occupied |= grip.mask(); contacts.remove(key);
            }
        }
        contacts.keySet().retainAll(seenContacts);
        // IDs are never reused. Keep only current owners and visible blocked fingers.
        blocked.retainAll(visibleFingers);
    }

    private boolean extended(HandTracks.Hand h, int finger) {
        if (finger == 4) {
            for (int tip = 8; tip <= 20; tip += 4) if (ratio(h.raw,tip,h.scale) < PINCH_OFF) return false;
            return HandTracks.distance(h.raw,4,0) > HandTracks.distance(h.raw,3,0);
        }
        return ratio(h.raw,finger,h.scale) > PINCH_OFF &&
                HandTracks.distance(h.raw,finger,finger-3) > HandTracks.distance(h.raw,finger-2,finger-3)*1.15f;
    }

    /** Corners win only near endpoints; every other point on the full segment is an edge. */
    int hit(float x, float y, float radius) {
        float best = radius;
        int result = -1;
        for (int i = 0; i < 4; i++) {
            float adjacent = Math.min(edgeLength(corners,i),edgeLength(corners,(i+3)%4));
            float d = (float)Math.hypot(x-corners[i*2],y-corners[i*2+1]);
            if (d <= Math.min(radius,adjacent*.3f) && d <= best) { best = d; result = i; }
        }
        if (result >= 0) return result;
        for (int i = 0; i < 4; i++) {
            int j = (i+1)%4;
            float ax = corners[i*2], ay = corners[i*2+1];
            float dx = corners[j*2]-ax, dy = corners[j*2+1]-ay;
            float t = Math.max(0,Math.min(1,((x-ax)*dx+(y-ay)*dy)/(dx*dx+dy*dy)));
            float d = (float)Math.hypot(x-ax-t*dx,y-ay-t*dy);
            if (d <= best) { best = d; result = 4+i; }
        }
        return result;
    }

    private void constrain(float[] desired) {
        // Find the first invalid point along the drag, including mid-path folds. Corner numbering
        // and winding never change. This also keeps opposing simultaneous drags deterministic.
        float lo = 0f, hi = 1f;
        float[] trial = new float[8];
        for (int step = 1; step <= 16; step++) {
            float t = step/16f; interpolate(corners,desired,t,trial);
            if (!valid(trial,minEdge,minArea)) { hi = t; break; }
            lo = t;
        }
        if (lo < 1f) for (int n = 0; n < 12; n++) {
            float mid = (lo+hi)*.5f; interpolate(corners,desired,mid,trial);
            if (valid(trial,minEdge,minArea)) lo = mid; else hi = mid;
        }
        interpolate(corners,desired,lo,trial); corners = trial;
    }

    static boolean valid(float[] q, float minEdge, float minArea) {
        if (q == null || q.length != 8) return false;
        double area = 0;
        for (float v : q) if (!Float.isFinite(v)) return false;
        for (int i = 0; i < 4; i++) {
            int j = (i+1)%4, k = (i+2)%4;
            if (edgeLength(q,i) < minEdge) return false;
            double cross = (double)(q[j*2]-q[i*2])*(q[k*2+1]-q[j*2+1]) -
                    (double)(q[j*2+1]-q[i*2+1])*(q[k*2]-q[j*2]);
            if (cross <= 1e-5) return false;
            area += (double)q[i*2]*q[j*2+1]-(double)q[j*2]*q[i*2+1];
        }
        return area*.5 >= minArea;
    }
    private static float edgeLength(float[] q,int i) {
        int j = (i+1)%4; return (float)Math.hypot(q[i*2]-q[j*2],q[i*2+1]-q[j*2+1]);
    }
    private static void interpolate(float[] a,float[] b,float t,float[] out) {
        for (int i = 0; i < 8; i++) out[i] = a[i]+(b[i]-a[i])*t;
    }
    private boolean owns(int id,int finger) { for (Grip g : grips) if(g.handId==id && g.finger==finger) return true; return false; }
    private static int mask(int h) { return h < 4 ? 1 << h : (1 << (h-4)) | (1 << ((h-3)%4)); }
    private static long key(int hand,int finger) { return ((long)hand << 32) | finger; }
    private static float ratio(float[] p,int finger,float scale) { return HandTracks.distance(p,4,finger)/scale; }
    private static HandTracks.Hand find(List<HandTracks.Hand> hands,int id) {
        for (HandTracks.Hand h : hands) if (h.id == id) return h; return null;
    }
}
