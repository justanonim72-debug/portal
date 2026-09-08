package dev.riszn.portal;

import org.junit.Test;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import static org.junit.Assert.*;

public class PortalInteractionTest {
    private final HandTracks tracks = new HandTracks();
    private final PortalInteraction panel = new PortalInteraction();
    private long now = 1_000_000_000L;

    static float[] hand(float dx) {
        float[] p = new float[42];
        p[0]=300+dx; p[1]=550;
        for(int f=1;f<=5;f++) for(int j=0;j<4;j++) {
            int i=1+(f-1)*4+j;
            p[i*2]=190+f*45+dx; p[i*2+1]=480-j*30;
        }
        return p;
    }
    private static void tip(float[] p,int finger,float x,float y) {
        p[finger*2]=x; p[finger*2+1]=y;
        p[(finger-1)*2]=x; p[(finger-1)*2+1]=y+24;
        p[(finger-2)*2]=x; p[(finger-2)*2+1]=y+48;
        p[(finger-3)*2]=x; p[(finger-3)*2+1]=y+72;
    }
    private static HandTracks.Observation observation(float[] p,String side) {
        return new HandTracks.Observation(p,side,.99f);
    }
    private List<HandTracks.Hand> tick(float[]... observations) {
        List<HandTracks.Observation> list=new ArrayList<>();
        for(int i=0;i<observations.length;i++) list.add(observation(observations[i],i==0?"Left":"Right"));
        now+=40_000_000L;
        List<HandTracks.Hand> hands=tracks.update(list,now); panel.update(hands,now); return hands;
    }
    private float[] seed(int finger) {
        float[] p=hand(0); tip(p,finger,310,300);
        p[8]=312; p[9]=300;
        for(int i=0;i<4;i++) tick(p);
        assertEquals(PortalInteraction.Phase.SEEDED,panel.phase());
        return p;
    }
    private float[] activate() {
        float[] p=seed(8); p[8]=100; p[9]=400;
        for(int i=0;i<4;i++) tick(p);
        assertEquals(PortalInteraction.Phase.ACTIVE,panel.phase());
        return p;
    }
    private void settle(float[]... p) { for(int i=0;i<12;i++) tick(p); }

    @Test public void idleHasNoPanelWithVisibleHand() {
        settle(hand(0)); assertEquals(PortalInteraction.Phase.IDLE,panel.phase()); assertEquals(0,panel.corners().length);
    }
    @Test public void allFourPinchesCreateExactlyOneSmallSeed() {
        for(int finger=8;finger<=20;finger+=4) {
            panel.reset(); tracks.reset();
            float[] p=seed(finger), q=panel.corners();
            float cx=0,cy=0,max=0;
            for(int i=0;i<4;i++) {
                cx+=q[i*2]/4; cy+=q[i*2+1]/4;
                for(int j=0;j<4;j++) max=Math.max(max,(float)Math.hypot(q[i*2]-q[j*2],q[i*2+1]-q[j*2+1]));
            }
            assertEquals(311,cx,.001); assertEquals(300,cy,.001); assertTrue(max<=24);
            settle(p); assertArrayEquals(q,panel.corners(),0f);
        }
    }
    @Test public void shortPinchAndFingerChangesDoNotTrigger() {
        float[] p=hand(0); p[8]=p[16]; p[9]=p[17]; tick(p); tick(hand(0));
        assertEquals(PortalInteraction.Phase.IDLE,panel.phase());
        tick(p); p[8]=p[24]; p[9]=p[25]; tick(p);
        assertEquals(0,panel.corners().length);
    }
    @Test public void seedHasNoScreenSizeFloor() {
        float[] p=hand(0); tip(p,8,310,300); p[14]=310; p[15]=302; p[8]=310; p[9]=300;
        settle(p); float[] q=panel.corners(); assertEquals(8,q.length);
        for(int i=0;i<4;i++) for(int j=0;j<4;j++) assertTrue(Math.hypot(q[i*2]-q[j*2],q[i*2+1]-q[j*2+1])<=2);
    }
    @Test public void releaseAndLongAbsencePreservePanel() {
        activate(); float[] q=panel.corners();
        for(int i=0;i<100;i++) tick();
        assertEquals(PortalInteraction.Phase.ACTIVE,panel.phase()); assertArrayEquals(q,panel.corners(),0f);
        panel.reset(); assertEquals(PortalInteraction.Phase.IDLE,panel.phase()); assertEquals(0,panel.corners().length);
    }
    @Test public void anotherFingerLatchesCornerWithoutSnappingAndOnlyMovesThatCorner() {
        float[] p=activate(), q=panel.corners();
        tip(p,16,q[0]-1,q[1]-1); settle(p);
        assertEquals(1,panel.grips().size()); assertEquals(16,panel.grips().get(0).finger);
        assertEquals(0,panel.grips().get(0).handle);
        float[] before=panel.corners();
        tip(p,16,q[0]-26,q[1]-21); settle(p);
        float[] after=panel.corners();
        assertTrue(Math.hypot(after[0]-before[0],after[1]-before[1])>15);
        for(int i=2;i<8;i++) assertEquals(before[i],after[i],.001);
        assertTrue(PortalInteraction.valid(after,.01f,.01f));
    }
    @Test public void edgeHitCoversWholeSegmentAndTranslatesBothEndpoints() {
        float[] p=activate(), q=panel.corners();
        tip(p,16,(q[0]+q[2])/2,(q[1]+q[3])/2); settle(p);
        assertEquals(4,panel.grips().get(0).handle);
        float[] before=panel.corners();
        tip(p,16,p[32]+30,p[33]); settle(p);
        float[] after=panel.corners();
        assertEquals(after[0]-before[0],after[2]-before[2],.001);
        for(int i=4;i<8;i++) assertEquals(before[i],after[i],.001);
        // Larger diagonal edge: hit away from its midpoint still chooses that edge.
        float x=after[0]*.65f+after[2]*.35f,y=after[1]*.65f+after[3]*.35f;
        assertEquals(4,panel.hit(x,y,1));
    }
    @Test public void twoHandsKeepIndependentGripsThroughSecondHandMiss() {
        float[] p=activate(), q=panel.corners(), b=hand(300);
        tip(p,16,q[0],q[1]); tip(b,12,q[4],q[5]);
        settle(p,b);
        assertEquals(2,panel.grips().size()); assertEquals(PortalInteraction.Phase.RESIZING,panel.phase());
        int first=panel.grips().get(0).handId, second=panel.grips().get(1).handId;
        assertNotEquals(first,second);
        float[] before=panel.corners(); tick(p); tick(p);
        assertEquals(2,panel.grips().size()); assertArrayEquals(before,panel.corners(),.01f);
        tip(b,12,q[4]+40,q[5]+30); tick(p,b); // returning finger rebases, no unseen jump
        assertArrayEquals(before,panel.corners(),.01f);
        settle(p,b); assertEquals(second,panel.grips().get(1).handId);
    }
    @Test public void pinchReleasesGripAndDoesNotSummonAgain() {
        float[] p=activate(),q=panel.corners(); tip(p,16,q[0],q[1]); settle(p);
        p[8]=p[32]; p[9]=p[33]; settle(p);
        assertEquals(PortalInteraction.Phase.ACTIVE,panel.phase()); assertEquals(0,panel.grips().size());
        assertEquals(8,panel.corners().length);
    }
    @Test public void crossingDragStaysConvexAndCanRecover() {
        float[] p=activate(),q=panel.corners(); tip(p,16,q[0],q[1]); settle(p);
        tip(p,16,q[4]+50,q[5]+50); settle(p);
        assertTrue(java.util.Arrays.toString(panel.corners()),PortalInteraction.valid(panel.corners(),.01f,.01f));
        float fromX=p[32],fromY=p[33];
        for(int i=1;i<=20;i++){tip(p,16,fromX+(q[0]-50-fromX)*i/20,fromY+(q[1]+50-fromY)*i/20);tick(p);}
        settle(p);
        assertTrue(java.util.Arrays.toString(panel.corners()),PortalInteraction.valid(panel.corners(),.01f,.01f));
        assertTrue("q="+java.util.Arrays.toString(q)+" now="+java.util.Arrays.toString(panel.corners())+" grips="+panel.grips().size(),Math.hypot(panel.corners()[0]-q[0],panel.corners()[1]-q[1])>10);
    }
    @Test public void horizontalAndVerticalHandsAreRotationEquivalentInPortraitPixels() {
        float[] p=hand(0); tip(p,8,310,300); p[8]=312;p[9]=300;
        settle(p); float[] vertical=panel.corners();
        panel.reset(); tracks.reset();
        for(int i=0;i<21;i++) {float x=p[i*2]-310,y=p[i*2+1]-300; p[i*2]=310-y;p[i*2+1]=300+x;}
        settle(p); float[] horizontal=panel.corners();
        for(int i=0;i<4;i++) {
            assertEquals(310-(vertical[i*2+1]-300),horizontal[i*2],.002);
            assertEquals(300+(vertical[i*2]-310),horizontal[i*2+1],.002);
        }
    }
    @Test public void invalidAndOutOfOrderFramesCannotPoisonGeometry() {
        activate();float[] before=panel.corners(),bad=hand(0);bad[0]=Float.NaN;
        tick(bad); panel.update(List.of(),now-1);
        assertArrayEquals(before,panel.corners(),0f);
    }
    @Test public void arbitraryDragsNeverSelfCrossOrExplode() {
        Random random=new Random(7);float[] p=activate(),q=panel.corners();tip(p,16,q[0],q[1]);settle(p);
        for(int i=0;i<1500;i++) {
            tip(p,16,p[32]+random.nextFloat()*24-12,p[33]+random.nextFloat()*24-12);tick(p);
            assertTrue(java.util.Arrays.toString(panel.corners()),PortalInteraction.valid(panel.corners(),.01f,.01f));
            for(float v:panel.corners()) assertTrue(Float.isFinite(v));
        }
    }
}
