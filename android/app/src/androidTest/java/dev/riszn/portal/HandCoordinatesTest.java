package dev.riszn.portal;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;
import org.junit.Test;
import org.junit.runner.RunWith;
import static org.junit.Assert.*;

/** Verifies the actual MediaPipe output contract, rather than assuming ROI coordinates. */
@RunWith(AndroidJUnit4.class)
public class HandCoordinatesTest {
    @Test public void rotationOptionsReturnLandmarksInOriginalInputBuffer() throws Exception {
        var instrumentation=InstrumentationRegistry.getInstrumentation();
        Bitmap upright;
        try(var stream=instrumentation.getContext().getAssets().open("right_hands.jpg")) {
            upright=BitmapFactory.decodeStream(stream);
        }
        assertNotNull(upright);
        Matrix rotation=new Matrix();rotation.postRotate(90);
        Bitmap sideways=Bitmap.createBitmap(upright,0,0,upright.getWidth(),upright.getHeight(),rotation,true);
        try {
            HandLandmarkerResult a=detect(upright,0),b=detect(sideways,270);
            assertEquals("Official fixture should detect two hands",2,a.landmarks().size());
            assertEquals(2,b.landmarks().size());
            for(var hand:a.landmarks()) {
                float expectedX=1-hand.get(0).y(),expectedY=hand.get(0).x();
                var match=b.landmarks().get(0);
                for(var candidate:b.landmarks()) {
                    if(Math.hypot(candidate.get(0).x()-expectedX,candidate.get(0).y()-expectedY)<
                            Math.hypot(match.get(0).x()-expectedX,match.get(0).y()-expectedY))match=candidate;
                }
                for(int i=0;i<21;i++) {
                    assertEquals("x must refer to sideways input pixels",1-hand.get(i).y(),match.get(i).x(),.035f);
                    assertEquals("y must refer to sideways input pixels",hand.get(i).x(),match.get(i).y(),.035f);
                }
            }
        } finally {sideways.recycle();upright.recycle();}
    }
    private HandLandmarkerResult detect(Bitmap bitmap,int rotation) {
        var context=InstrumentationRegistry.getInstrumentation().getTargetContext();
        var options=HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").setDelegate(Delegate.CPU).build())
                .setRunningMode(RunningMode.IMAGE).setNumHands(2).build();
        try(HandLandmarker landmarker=HandLandmarker.createFromOptions(context,options);
            MPImage image=new BitmapImageBuilder(bitmap).build()) {
            return landmarker.detect(image,ImageProcessingOptions.builder().setRotationDegrees(rotation).build());
        }
    }
}
