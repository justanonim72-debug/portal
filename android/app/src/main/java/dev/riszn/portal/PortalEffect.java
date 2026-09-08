package dev.riszn.portal;

import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLContext;
import android.opengl.EGLDisplay;
import android.opengl.EGLExt;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.os.Handler;
import android.os.HandlerThread;
import android.view.Surface;
import androidx.camera.core.CameraEffect;
import androidx.camera.core.SurfaceOutput;
import androidx.camera.core.SurfaceProcessor;
import androidx.camera.core.SurfaceRequest;
import androidx.core.util.Consumer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.Executor;

/** A single native GPU path for the live camera and Recorder, separate from inference. */
final class PortalEffect implements SurfaceProcessor, AutoCloseable {
    private final HandlerThread thread = new HandlerThread("Portal-Camera-GL");
    private final Handler handler;
    private final Executor executor;
    private final PortalRenderer renderer;
    private final Consumer<Throwable> errorListener;
    private final Map<SurfaceOutput,EGLSurface> outputs = new LinkedHashMap<>();
    private final ArrayList<Input> inputs = new ArrayList<>();
    private final float[] textureMatrix = new float[16], outputMatrix = new float[16];
    private EGLDisplay display = EGL14.EGL_NO_DISPLAY;
    private EGLContext context = EGL14.EGL_NO_CONTEXT;
    private EGLSurface placeholder = EGL14.EGL_NO_SURFACE;
    private EGLConfig config;
    private Input active;
    private boolean closed, initialized;

    PortalEffect(PortalRenderer renderer, Consumer<Throwable> errorListener) {
        this.renderer = renderer; this.errorListener = errorListener;
        thread.start(); handler = new Handler(thread.getLooper()); executor = command -> handler.post(command);
    }
    CameraEffect cameraEffect() {
        return new CameraEffect(CameraEffect.PREVIEW | CameraEffect.VIDEO_CAPTURE,executor,this,errorListener) {};
    }

    @Override public void onInputSurface(SurfaceRequest request) {
        if (closed) { request.willNotProvideSurface(); return; }
        try {
            initialize(); makeCurrent(placeholder);
            int[] textures = new int[1]; GLES20.glGenTextures(1,textures,0);
            GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,textures[0]);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_MAG_FILTER,GLES20.GL_LINEAR);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_WRAP_S,GLES20.GL_CLAMP_TO_EDGE);
            GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,GLES20.GL_TEXTURE_WRAP_T,GLES20.GL_CLAMP_TO_EDGE);
            Input input = new Input(textures[0]);
            input.texture.setDefaultBufferSize(request.getResolution().getWidth(),request.getResolution().getHeight());
            input.texture.setOnFrameAvailableListener(ignored -> render(input),handler);
            inputs.add(input); active = input;
            request.provideSurface(input.surface,executor,result -> {
                makeCurrent(placeholder);
                input.texture.setOnFrameAvailableListener(null);
                input.surface.release(); input.texture.release(); GLES20.glDeleteTextures(1,new int[]{input.id},0);
                inputs.remove(input); if (active == input) active = null;
                releaseWhenUnused();
            });
        } catch (RuntimeException error) { request.willNotProvideSurface(); errorListener.accept(error); }
    }

    @Override public void onOutputSurface(SurfaceOutput output) {
        if (closed) { output.close(); return; }
        try {
            initialize();
            Surface surface = output.getSurface(executor,event -> removeOutput(output));
            EGLSurface egl = EGL14.eglCreateWindowSurface(display,config,surface,new int[]{EGL14.EGL_NONE},0);
            if (egl == EGL14.EGL_NO_SURFACE) throw new IllegalStateException("Cannot create camera output: " + EGL14.eglGetError());
            outputs.put(output,egl);
        } catch (RuntimeException error) { output.close(); errorListener.accept(error); }
    }

    private void render(Input input) {
        if (closed || input != active) return;
        try {
            makeCurrent(placeholder);
            input.texture.updateTexImage(); // latest camera buffer, no inference wait or frame queue
            input.texture.getTransformMatrix(textureMatrix);
            for (Map.Entry<SurfaceOutput,EGLSurface> entry : outputs.entrySet()) {
                SurfaceOutput output = entry.getKey();
                makeCurrent(entry.getValue());
                output.updateTransformMatrix(outputMatrix,textureMatrix);
                renderer.draw(input.id,outputMatrix,output.getSensorToBufferTransform(),
                        output.getSize().getWidth(),output.getSize().getHeight());
                EGLExt.eglPresentationTimeANDROID(display,entry.getValue(),input.texture.getTimestamp());
                if (!EGL14.eglSwapBuffers(display,entry.getValue())) throw new IllegalStateException("Camera swap failed: " + EGL14.eglGetError());
            }
        } catch (RuntimeException error) { errorListener.accept(error); }
    }

    private void initialize() {
        if (initialized) return;
        display = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        int[] version = new int[2];
        if (!EGL14.eglInitialize(display,version,0,version,1)) throw new IllegalStateException("EGL initialization failed");
        int[] attributes = {EGL14.EGL_RED_SIZE,8,EGL14.EGL_GREEN_SIZE,8,EGL14.EGL_BLUE_SIZE,8,
                EGL14.EGL_ALPHA_SIZE,8,EGL14.EGL_RENDERABLE_TYPE,EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE,EGL14.EGL_WINDOW_BIT | EGL14.EGL_PBUFFER_BIT,
                0x3142,1, // EGL_RECORDABLE_ANDROID, required for MediaCodec/Recorder surfaces
                EGL14.EGL_NONE};
        EGLConfig[] configs = new EGLConfig[1]; int[] count = new int[1];
        if (!EGL14.eglChooseConfig(display,attributes,0,configs,0,1,count,0) || count[0] == 0) throw new IllegalStateException("No recordable EGL config");
        config = configs[0];
        context = EGL14.eglCreateContext(display,config,EGL14.EGL_NO_CONTEXT,new int[]{EGL14.EGL_CONTEXT_CLIENT_VERSION,2,EGL14.EGL_NONE},0);
        placeholder = EGL14.eglCreatePbufferSurface(display,config,new int[]{EGL14.EGL_WIDTH,1,EGL14.EGL_HEIGHT,1,EGL14.EGL_NONE},0);
        makeCurrent(placeholder); renderer.initialize(); initialized = true;
    }
    private void makeCurrent(EGLSurface surface) {
        if (!EGL14.eglMakeCurrent(display,surface,surface,context)) throw new IllegalStateException("Cannot make EGL current: " + EGL14.eglGetError());
    }
    private void removeOutput(SurfaceOutput output) {
        EGLSurface surface = outputs.remove(output);
        if (surface != null) { makeCurrent(placeholder); EGL14.eglDestroySurface(display,surface); }
        output.close();
    }
    @Override public void close() {
        executor.execute(() -> {
            closed = true;
            for (SurfaceOutput output : new ArrayList<>(outputs.keySet())) removeOutput(output);
            releaseWhenUnused();
        });
    }
    private void releaseWhenUnused() {
        // CameraX owns input use until provideSurface's completion callback. Never release early.
        if (!closed || !inputs.isEmpty()) return;
        if (initialized) {
            makeCurrent(placeholder); renderer.release();
            EGL14.eglMakeCurrent(display,EGL14.EGL_NO_SURFACE,EGL14.EGL_NO_SURFACE,EGL14.EGL_NO_CONTEXT);
            EGL14.eglDestroySurface(display,placeholder); EGL14.eglDestroyContext(display,context);
            EGL14.eglReleaseThread(); EGL14.eglTerminate(display); initialized = false;
        }
        thread.quitSafely();
    }
    private static final class Input {
        final int id;
        final SurfaceTexture texture;
        final Surface surface;
        Input(int id) { this.id = id; texture = new SurfaceTexture(id); surface = new Surface(texture); }
    }
}
