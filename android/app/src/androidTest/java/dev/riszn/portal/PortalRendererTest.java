package dev.riszn.portal;

import android.graphics.Canvas;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.opengl.*;
import android.view.Surface;
import androidx.test.ext.junit.runners.AndroidJUnit4;
import org.junit.Test;
import org.junit.runner.RunWith;
import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class PortalRendererTest {
    @Test public void shaderFiltersOnlyQuadAndMapsAnalysisPixelsThroughSensor() throws Exception {
        EGLDisplay display=EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        int[] v=new int[2];assertTrue(EGL14.eglInitialize(display,v,0,v,1));
        EGLConfig[] configs=new EGLConfig[1];int[] count=new int[1];
        assertTrue(EGL14.eglChooseConfig(display,new int[]{EGL14.EGL_RED_SIZE,8,EGL14.EGL_GREEN_SIZE,8,EGL14.EGL_BLUE_SIZE,8,
                EGL14.EGL_RENDERABLE_TYPE,EGL14.EGL_OPENGL_ES2_BIT,EGL14.EGL_SURFACE_TYPE,EGL14.EGL_PBUFFER_BIT,EGL14.EGL_NONE},0,configs,0,1,count,0));
        EGLContext context=EGL14.eglCreateContext(display,configs[0],EGL14.EGL_NO_CONTEXT,new int[]{EGL14.EGL_CONTEXT_CLIENT_VERSION,2,EGL14.EGL_NONE},0);
        EGLSurface output=EGL14.eglCreatePbufferSurface(display,configs[0],new int[]{EGL14.EGL_WIDTH,128,EGL14.EGL_HEIGHT,256,EGL14.EGL_NONE},0);
        assertTrue(EGL14.eglMakeCurrent(display,output,output,context));
        int[] tex=new int[1];GLES20.glGenTextures(1,tex,0);GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,tex[0]);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR);
        SurfaceTexture st=new SurfaceTexture(tex[0]);st.setDefaultBufferSize(128,256);Surface input=new Surface(st);
        AtomicReference<PortalState> state=new AtomicReference<>(PortalState.none());PortalRenderer renderer=new PortalRenderer(state);
        try {
            Canvas c=input.lockCanvas(null);c.drawColor(Color.WHITE);input.unlockCanvasAndPost(c);
            st.updateTexImage();float[] transform=new float[16];st.getTransformMatrix(transform);
            renderer.initialize();
            Matrix analysisToSensor=new Matrix();analysisToSensor.setScale(4,4);
            Matrix sensorToOutput=new Matrix();sensorToOutput.setScale(.5f,.5f);
            // A horizontal plane in a portrait buffer, off-center to detect Y flips/rotation errors.
            float[] q={5,15,45,15,42.5f,40,7.5f,40};
            state.set(new PortalState(PortalInteraction.Phase.ACTIVE,q,new float[0][],new int[0],new int[0],0,
                    System.nanoTime(),System.nanoTime(),analysisToSensor,128,256));
            renderer.draw(tex[0],transform,sensorToOutput,128,256);GLES20.glFinish();
            assertPixel(20,55,true);assertPixel(70,55,true);assertPixel(64,180,false);assertPixel(100,55,false);
            // CameraX front-camera output can reflect winding. The same sensor point must follow.
            sensorToOutput.setScale(-.5f,.5f);sensorToOutput.postTranslate(128,0);
            renderer.draw(tex[0],transform,sensorToOutput,128,256);GLES20.glFinish();
            assertPixel(108,55,true);assertPixel(20,55,false);
            ByteBuffer pixels=ByteBuffer.allocateDirect(128*256*4);
            GLES20.glReadPixels(0,0,128,256,GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,pixels);
            Bitmap evidence=Bitmap.createBitmap(128,256,Bitmap.Config.ARGB_8888);
            for(int y=0;y<256;y++)for(int x=0;x<128;x++) {
                int at=((255-y)*128+x)*4;
                evidence.setPixel(x,y,Color.rgb(pixels.get(at)&255,pixels.get(at+1)&255,pixels.get(at+2)&255));
            }
            TestEvidence.save(evidence,"portrait-horizontal-plane");evidence.recycle();
            state.set(PortalState.none());renderer.draw(tex[0],transform,sensorToOutput,128,256);GLES20.glFinish();
            assertPixel(64,55,false);
            assertEquals(GLES20.GL_NO_ERROR,GLES20.glGetError());
        } finally {
            renderer.release();input.release();st.release();GLES20.glDeleteTextures(1,tex,0);
            EGL14.eglMakeCurrent(display,EGL14.EGL_NO_SURFACE,EGL14.EGL_NO_SURFACE,EGL14.EGL_NO_CONTEXT);
            EGL14.eglDestroySurface(display,output);EGL14.eglDestroyContext(display,context);EGL14.eglTerminate(display);
        }
    }
    private void assertPixel(int x,int y,boolean violet) {
        ByteBuffer pixel=ByteBuffer.allocateDirect(4);GLES20.glReadPixels(x,255-y,1,1,GLES20.GL_RGBA,GLES20.GL_UNSIGNED_BYTE,pixel);
        int r=pixel.get(0)&255,g=pixel.get(1)&255,b=pixel.get(2)&255;
        String values="RGB "+r+","+g+","+b;
        if(violet){assertTrue(values,r>140 && r<190);assertTrue(values,g<50);assertTrue(values,b>190);}
        else assertTrue(values,r>240 && g>240 && b>240);
    }
}
