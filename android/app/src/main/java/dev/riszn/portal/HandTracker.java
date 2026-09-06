package dev.riszn.portal;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import com.google.mediapipe.framework.image.BitmapImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.vision.core.ImageProcessingOptions;
import com.google.mediapipe.tasks.vision.core.RunningMode;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;

import java.nio.ByteBuffer;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.AtomicReference;

final class HandTracker implements ImageAnalysis.Analyzer, AutoCloseable {
    interface Listener {
        void onBackendReady(String backend);
        void onTrackerError(String message);
        void onPerf(double inferenceMs, double detectorFps, int hands);
    }

    private static final String MODEL = "hand_landmarker.task";

    private final Context context;
    private final AtomicReference<PortalState> stateRef;
    private final PortalGeometry geometry = new PortalGeometry();
    private final Listener listener;
    private final HandlerThread thread = new HandlerThread("Portal-MediaPipe");
    private Handler handler;
    private Executor executor;
    private HandLandmarker landmarker;
    private volatile boolean ready = false;
    private volatile boolean frontCamera = true;
    private volatile String backend = "…";

    private Bitmap bitmapBuffer;
    private byte[] packedRgba;
    private long lastTimestampMs = -1L;
    private long perfWindowStartNs = System.nanoTime();
    private int perfFrames = 0;
    private double perfMsSum = 0.0;

    HandTracker(Context context, AtomicReference<PortalState> stateRef, Listener listener) {
        this.context = context.getApplicationContext();
        this.stateRef = stateRef;
        this.listener = listener;
        thread.start();
        handler = new Handler(thread.getLooper());
        executor = command -> handler.post(command);
    }

    Executor executor() { return executor; }

    void setFrontCamera(boolean front) { this.frontCamera = front; }

    void initialize() {
        handler.post(() -> {
            try {
                landmarker = create(BaseOptions.Delegate.GPU);
                backend = "GPU";
                ready = true;
                listener.onBackendReady(backend);
            } catch (Throwable gpuError) {
                try {
                    landmarker = create(BaseOptions.Delegate.CPU);
                    backend = "CPU fallback";
                    ready = true;
                    listener.onBackendReady(backend);
                } catch (Throwable cpuError) {
                    ready = false;
                    listener.onTrackerError("MediaPipe gagal: " + cpuError.getMessage());
                }
            }
        });
    }

    private HandLandmarker create(BaseOptions.Delegate delegate) {
        BaseOptions base = BaseOptions.builder()
                .setModelAssetPath(MODEL)
                .setDelegate(delegate)
                .build();

        HandLandmarker.HandLandmarkerOptions options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.45f)
                .setMinHandPresenceConfidence(0.48f)
                .setMinTrackingConfidence(0.58f)
                .build();

        return HandLandmarker.createFromOptions(context, options);
    }

    @Override
    public void analyze(@NonNull ImageProxy imageProxy) {
        if (!ready || landmarker == null) {
            imageProxy.close();
            return;
        }

        long started = System.nanoTime();
        MPImage mpImage = null;
        try {
            ensureBitmap(imageProxy.getWidth(), imageProxy.getHeight());
            copyRgba(imageProxy);

            mpImage = new BitmapImageBuilder(bitmapBuffer).build();
            int rotation = imageProxy.getImageInfo().getRotationDegrees();
            ImageProcessingOptions processing = ImageProcessingOptions.builder()
                    .setRotationDegrees(rotation)
                    .build();

            long timestampMs = imageProxy.getImageInfo().getTimestamp() / 1_000_000L;
            if (timestampMs <= lastTimestampMs) timestampMs = lastTimestampMs + 1L;
            lastTimestampMs = timestampMs;

            HandLandmarkerResult result = landmarker.detectForVideo(mpImage, processing, timestampMs);
            PortalState portal = geometry.fromResult(result, frontCamera);
            stateRef.set(portal);

            double ms = (System.nanoTime() - started) / 1_000_000.0;
            updatePerf(ms, portal.hands);
        } catch (Throwable error) {
            stateRef.set(PortalState.none());
            listener.onTrackerError("Tracking frame gagal: " + error.getMessage());
        } finally {
            if (mpImage != null) mpImage.close();
            imageProxy.close();
        }
    }

    private void ensureBitmap(int width, int height) {
        if (bitmapBuffer == null || bitmapBuffer.getWidth() != width || bitmapBuffer.getHeight() != height) {
            if (bitmapBuffer != null) bitmapBuffer.recycle();
            bitmapBuffer = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
            packedRgba = null;
        }
    }

    private void copyRgba(ImageProxy imageProxy) {
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        if (planes.length == 0) throw new IllegalStateException("RGBA plane kosong");
        ImageProxy.PlaneProxy plane = planes[0];
        ByteBuffer src = plane.getBuffer().duplicate();
        src.rewind();

        int width = imageProxy.getWidth();
        int height = imageProxy.getHeight();
        int packedRow = width * 4;
        int rowStride = plane.getRowStride();

        if (rowStride == packedRow) {
            bitmapBuffer.copyPixelsFromBuffer(src);
            return;
        }

        int required = packedRow * height;
        if (packedRgba == null || packedRgba.length != required) packedRgba = new byte[required];
        for (int y = 0; y < height; y++) {
            int srcPos = y * rowStride;
            src.position(srcPos);
            src.get(packedRgba, y * packedRow, packedRow);
        }
        bitmapBuffer.copyPixelsFromBuffer(ByteBuffer.wrap(packedRgba));
    }

    private void updatePerf(double ms, int hands) {
        perfFrames++;
        perfMsSum += ms;
        long now = System.nanoTime();
        long elapsed = now - perfWindowStartNs;
        if (elapsed >= 600_000_000L) {
            double sec = elapsed / 1_000_000_000.0;
            double fps = perfFrames / sec;
            double avgMs = perfMsSum / Math.max(1, perfFrames);
            listener.onPerf(avgMs, fps, hands);
            perfFrames = 0;
            perfMsSum = 0.0;
            perfWindowStartNs = now;
        }
    }

    @Override
    public void close() {
        ready = false;
        handler.post(() -> {
            if (landmarker != null) {
                landmarker.close();
                landmarker = null;
            }
            if (bitmapBuffer != null) {
                bitmapBuffer.recycle();
                bitmapBuffer = null;
            }
            thread.quitSafely();
        });
    }
}
