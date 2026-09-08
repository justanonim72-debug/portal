package dev.riszn.portal;

import org.junit.Test;
import java.util.List;
import static org.junit.Assert.*;

public class HandTracksTest {
    private HandTracks.Observation observation(float dx,String label) {
        return new HandTracks.Observation(PortalInteractionTest.hand(dx),label,.99f);
    }
    @Test public void resultOrderingAndTemporaryMissDoNotSwapIds() {
        HandTracks tracks=new HandTracks();
        var a=observation(0,"Left");var b=observation(250,"Right");
        var first=tracks.update(List.of(a,b),1_000_000_000L);
        var reversed=tracks.update(List.of(b,a),1_040_000_000L);
        assertEquals(first.get(0).id,reversed.get(1).id);assertEquals(first.get(1).id,reversed.get(0).id);
        var only=tracks.update(List.of(b),1_080_000_000L);assertEquals(first.get(1).id,only.get(0).id);
        var both=tracks.update(List.of(a,b),1_120_000_000L);
        assertEquals(first.get(0).id,both.get(0).id);assertEquals(first.get(1).id,both.get(1).id);
    }
    @Test public void nearbyCrossingUsesHandednessAndTrajectory() {
        HandTracks tracks=new HandTracks();int a=0,b=0;
        for(int i=0;i<20;i++) {
            var hands=tracks.update(List.of(observation(i*10,"Left"),observation(190-i*10,"Right")),1_000_000_000L+i*40_000_000L);
            if(i==0){a=hands.get(0).id;b=hands.get(1).id;}
            assertEquals(a,hands.get(0).id);assertEquals(b,hands.get(1).id);
        }
    }
    @Test public void expiredIdentityIsNotReused() {
        HandTracks tracks=new HandTracks();var a=observation(0,"Left");
        int id=tracks.update(List.of(a),1_000_000_000L).get(0).id;
        assertNotEquals(id,tracks.update(List.of(a),2_000_000_000L).get(0).id);
        tracks.reset();assertNotEquals(id,tracks.update(List.of(a),3_000_000_000L).get(0).id);
    }
    @Test public void newHandKeepsIdentityWhenMissingSlotsAreOccupied() {
        HandTracks tracks=new HandTracks();
        var old=tracks.update(List.of(observation(0,"Left"),observation(250,"Right")),1_000_000_000L);
        tracks.update(List.of(),1_040_000_000L);
        var newHand=observation(1000,"Right");
        int id=tracks.update(List.of(newHand),1_080_000_000L).get(0).id;
        assertNotEquals(old.get(0).id,id);assertNotEquals(old.get(1).id,id);
        var next=tracks.update(List.of(newHand,observation(0,"Left")),1_120_000_000L);
        assertEquals(id,next.get(0).id);
        assertEquals(old.get(0).id,next.get(1).id);
        var reordered=tracks.update(List.of(observation(0,"Left"),newHand),1_160_000_000L);
        assertEquals(id,reordered.get(1).id);
    }
    @Test public void oneEuroFilterIsRotationInvariantAndResponsive() {
        HandTracks.Euro a=new HandTracks.Euro(4,.035f),b=new HandTracks.Euro(4,.035f);
        float[] av=new float[2],bv=new float[2];
        for(int i=0;i<50;i++) {
            float x=i<10?0:(i-10)*10,y=i*.2f;
            a.filter(x,y,.033f,av,0);b.filter(-y,x,.033f,bv,0);
            assertEquals(-av[1],bv[0],.001f);assertEquals(av[0],bv[1],.001f);
            if(i>20)assertTrue(x-av[0]<16); // under ~1.6 tracking frames at 300 px/s
        }
    }
}
