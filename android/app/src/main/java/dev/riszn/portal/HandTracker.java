package dev.riszn.portal;

import android.content.Context;
import android.os.Handler;
import android.os.HandlerThread;

import androidx.annotation.NonNull;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;

import com.google.mediapipe.framework.image.ByteBufferImageBuilder;
import com.google.mediapipe.framework.image.MPImage;
import com.google.mediapipe.tasks.core.BaseOptions;
import com.google.mediapipe.tasks.core.Delegate;
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

    // Reused direct RGBA staging buffer. ByteBuffer-backed MPImage.close() is a no-op for the
    // caller-owned buffer, unlike BitmapImageBuilder whose MPImage container recycles the Bitmap.
    private ByteBuffer rgbaBuffer;
    private int rgbaWidth = -1;
    private int rgbaHeight = -1;

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
                // GPU must be created and used on this same dedicated thread.
                landmarker = create(Delegate.GPU);
                backend = "GPU";
                ready = true;
                listener.onBackendReady(backend);
            } catch (Throwable gpuError) {
                try {
                    landmarker = create(Delegate.CPU);
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

    private HandLandmarker create(Delegate delegate) {
        BaseOptions base = BaseOptions.builder()
                .setModelAssetPath(MODEL)
                .setDelegate(delegate)
                .build();

        HandLandmarker.HandLandmarkerOptions options = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(base)
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                // Video test is dim and the second hand was frequently missed. Lower detection /
                // presence only moderately; keep tracking at 0.50 to avoid turning noise into a
                // persistent phantom hand.
                .setMinHandDetectionConfidence(0.40f)
                .setMinHandPresenceConfidence(0.42f)
                .setMinTrackingConfidence(0.50f)
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
            int width = imageProxy.getWidth();
            int height = imageProxy.getHeight();
            ensureRgbaBuffer(width, height);
            copyRgba(imageProxy, width, height);

            rgbaBuffer.rewind();
            mpImage = new ByteBufferImageBuilder(
                    rgbaBuffer,
                    width,
                    height,
                    MPImage.IMAGE_FORMAT_RGBA)
                    .build();

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

    private void ensureRgbaBuffer(int width, int height) {
        int required = width * height * 4;
        if (rgbaBuffer == null || rgbaBuffer.capacity() != required || rgbaWidth != width || rgbaHeight != height) {
            rgbaBuffer = ByteBuffer.allocateDirect(required);
            rgbaWidth = width;
            rgbaHeight = height;
        }
    }

    private void copyRgba(ImageProxy imageProxy, int width, int height) {
        ImageProxy.PlaneProxy[] planes = imageProxy.getPlanes();
        if (planes.length == 0) throw new IllegalStateException("RGBA plane kosong");

        ImageProxy.PlaneProxy plane = planes[0];
        int pixelStride = plane.getPixelStride();
        int rowStride = plane.getRowStride();
        int packedRow = width * 4;

        if (pixelStride != 4) {
            throw new IllegalStateException("RGBA pixelStride tak didukung: " + pixelStride);
        }
        if (rowStride < packedRow) {
            throw new IllegalStateException("RGBA rowStride invalid: " + rowStride + " < " + packedRow);
        }

        ByteBuffer src = plane.getBuffer().duplicate();
        rgbaBuffer.clear();

        if (rowStride == packedRow) {
            int required = packedRow * height;
            src.position(0);
            src.limit(Math.min(src.capacity(), required));
            if (src.remaining() < required) {
                throw new IllegalStateException("RGBA buffer terlalu kecil: " + src.remaining() + " < " + required);
            }
            rgbaBuffer.put(src);
        } else {
            for (int y = 0; y < height; y++) {
                int srcPos = y * rowStride;
                int srcEnd = srcPos + packedRow;
                if (srcEnd > src.capacity()) {
                    throw new IllegalStateException("RGBA row melewati buffer pada y=" + y);
                }
                src.position(srcPos);
                src.limit(srcEnd);
                rgbaBuffer.put(src);
                src.limit(src.capacity());
            }
        }

        rgbaBuffer.flip();
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
            rgbaBuffer = null;
            rgbaWidth = -1;
            rgbaHeight = -1;
            thread.quitSafely();
        });
    }
}
