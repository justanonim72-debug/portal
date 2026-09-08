package dev.riszn.portal;

import android.Manifest;
import android.content.ContentValues;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.media.MediaMetadataRetriever;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.widget.TextView;
import android.util.Log;
import androidx.camera.video.MediaStoreOutputOptions;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoRecordEvent;
import androidx.camera.view.LifecycleCameraController;
import androidx.camera.view.PreviewView;
import androidx.camera.view.video.AudioConfig;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.Lifecycle;
import androidx.test.core.app.ActivityScenario;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.rule.GrantPermissionRule;
import org.junit.Rule;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.lang.reflect.Field;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.Assert.*;

/** Uses the real CameraX effect, preview, lifecycle and Recorder, with deterministic panel state. */
@RunWith(AndroidJUnit4.class)
public class PortalRecordingTest {
    @Rule public GrantPermissionRule cameraPermission=GrantPermissionRule.grant(Manifest.permission.CAMERA);

    @Test public void frontAndBackCameraRecordTheFilterAfterLifecycleRebind() throws Exception {
        try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)) {
            awaitPreview(scenario);
            recordAndVerify(scenario,"front",true);
            scenario.moveToState(Lifecycle.State.CREATED);
            awaitPreviewState(scenario,PreviewView.StreamState.IDLE);
            scenario.moveToState(Lifecycle.State.RESUMED);
            awaitPreview(scenario);
            recordAndVerify(scenario,"front-resumed",true);
            scenario.onActivity(activity -> ((TextView)field(activity,"flipButton")).performClick());
            awaitPreview(scenario);
            recordAndVerify(scenario,"back",true);
        }
    }
    @Test public void unfilteredCameraXControlProducesMp4() throws Exception {
        try(ActivityScenario<MainActivity> scenario=ActivityScenario.launch(MainActivity.class)) {
            scenario.onActivity(a -> ((LifecycleCameraController)field(a,"cameraController")).setEffects(java.util.Collections.emptySet()));
            awaitPreview(scenario);
            recordAndVerify(scenario,"unfiltered-control",false);
        }
    }
    private void awaitPreview(ActivityScenario<MainActivity> scenario) {
        awaitPreviewState(scenario,PreviewView.StreamState.STREAMING);
    }
    private void awaitPreviewState(ActivityScenario<MainActivity> scenario,PreviewView.StreamState expected) {
        AtomicReference<PreviewView.StreamState> state=new AtomicReference<>();
        long deadline=SystemClock.elapsedRealtime()+20_000;
        do {
            scenario.onActivity(a -> state.set(((PreviewView)field(a,"previewView")).getPreviewStreamState().getValue()));
            if(state.get()==expected)return;
            SystemClock.sleep(100);
        } while(SystemClock.elapsedRealtime()<deadline);
        fail("Preview did not reach "+expected+": "+state.get());
    }
    @SuppressWarnings("unchecked")
    private void recordAndVerify(ActivityScenario<MainActivity> scenario,String name,boolean filtered) throws Exception {
        CountDownLatch drained=new CountDownLatch(1), started=new CountDownLatch(1), finished=new CountDownLatch(1), enoughData=new CountDownLatch(1);
        AtomicReference<MainActivity> activity=new AtomicReference<>();
        AtomicReference<Recording> recording=new AtomicReference<>();
        AtomicReference<VideoRecordEvent.Finalize> finalized=new AtomicReference<>();
        AtomicReference<String> progress=new AtomicReference<>("no events");
        scenario.onActivity(a -> {
            activity.set(a);
            ((LifecycleCameraController)field(a,"cameraController")).clearImageAnalysisAnalyzer();
            ((HandTracker)field(a,"handTracker")).executor().execute(drained::countDown);
        });
        assertTrue("Inference drain timed out",drained.await(10,TimeUnit.SECONDS));
        scenario.onActivity(a -> {
            AtomicReference<PortalState> state=(AtomicReference<PortalState>)field(a,"portalState");
            state.set(new PortalState(PortalInteraction.Phase.ACTIVE,
                    new float[]{-1000,-1000,10000,-1000,10000,10000,-1000,10000},new float[0][],new int[0],new int[0],0,
                    System.nanoTime(),System.nanoTime(),new Matrix(),640,480));
            ContentValues values=new ContentValues();
            values.put(MediaStore.Video.Media.DISPLAY_NAME,"Portal-CI-"+name+"-"+System.currentTimeMillis());
            values.put(MediaStore.Video.Media.MIME_TYPE,"video/mp4");
            MediaStoreOutputOptions output=new MediaStoreOutputOptions.Builder(a.getContentResolver(),MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
                    .setContentValues(values).build();
            recording.set(((LifecycleCameraController)field(a,"cameraController")).startRecording(output,AudioConfig.AUDIO_DISABLED,
                    ContextCompat.getMainExecutor(a), event -> {
                        var stats=event.getRecordingStats();
                        progress.set(event.getClass().getSimpleName()+" bytes="+stats.getNumBytesRecorded()+" duration="+stats.getRecordedDurationNanos());
                        Log.i("PortalRecordingTest",name+" "+progress.get());
                        if(stats.getNumBytesRecorded()>0 && stats.getRecordedDurationNanos()>=1_500_000_000L)enoughData.countDown();
                        if(event instanceof VideoRecordEvent.Start)started.countDown();
                        if(event instanceof VideoRecordEvent.Finalize){finalized.set((VideoRecordEvent.Finalize)event);finished.countDown();}
                    }));
        });
        try {
            assertTrue("Recorder did not start",started.await(15,TimeUnit.SECONDS));
            // Start announces encoder startup, not a keyframe. Wait for actual encoded media.
            enoughData.await(20,TimeUnit.SECONDS);
        } finally { scenario.onActivity(a -> recording.get().stop()); }
        assertTrue("MP4 did not finalize",finished.await(20,TimeUnit.SECONDS));
        Log.i("PortalRecordingTest",name+" finalize error="+finalized.get().getError()+" cause="+finalized.get().getCause()+" "+progress.get());
        assertFalse(name+" Recording error "+finalized.get().getError()+" cause="+finalized.get().getCause()+" "+progress.get(),finalized.get().hasError());
        assertEquals(name+" did not encode 1.5 seconds: "+progress.get(),0,enoughData.getCount());
        var uri=finalized.get().getOutputResults().getOutputUri();
        try(MediaMetadataRetriever retriever=new MediaMetadataRetriever()) {
            retriever.setDataSource(activity.get(),uri);
            assertTrue(Long.parseLong(retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION))>=1500);
            Bitmap frame=retriever.getFrameAtTime(1_000_000,MediaMetadataRetriever.OPTION_CLOSEST);
            assertNotNull("MP4 has no decodable frame",frame);
            // Violet's green channel is <= 31 even for white input; blue is >= 46 even for black.
            // A plain/unfiltered camera recording cannot satisfy this across the entire test scene.
            if(filtered)for(int x=1;x<5;x++)for(int y=1;y<5;y++) {
                int pixel=frame.getPixel(frame.getWidth()*x/5,frame.getHeight()*y/5);
                assertTrue("Recorded frame lacks violet filter: "+Integer.toHexString(pixel),
                        Color.green(pixel)<55 && Color.blue(pixel)>Color.green(pixel)+10);
            }
            TestEvidence.save(frame,"recorded-"+name);
            frame.recycle();
        } finally {activity.get().getContentResolver().delete(uri,null,null);}
    }
    private static Object field(Object target,String name) {
        try {Field f=target.getClass().getDeclaredField(name);f.setAccessible(true);return f.get(target);}
        catch(ReflectiveOperationException e){throw new AssertionError(e);}
    }
}
