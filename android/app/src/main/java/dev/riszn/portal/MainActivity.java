package dev.riszn.portal;

import android.Manifest;
import android.content.ContentValues;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.drawable.GradientDrawable;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.MediaStore;
import android.view.Gravity;
import android.view.View;
import android.view.ViewGroup;
import android.view.Window;
import android.widget.FrameLayout;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.core.CameraEffect;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.MirrorMode;
import androidx.camera.effects.OverlayEffect;
import androidx.camera.video.MediaStoreOutputOptions;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoRecordEvent;
import androidx.camera.view.CameraController;
import androidx.camera.view.LifecycleCameraController;
import androidx.camera.view.PreviewView;
import androidx.camera.view.video.AudioConfig;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicReference;

public final class MainActivity extends AppCompatActivity implements HandTracker.Listener {
    private static final int CAMERA_PERMISSION = 401;

    private final AtomicReference<PortalState> portalState = new AtomicReference<>(PortalState.none());

    private PreviewView previewView;
    private TextView statusText;
    private TextView perfText;
    private TextView effectButton;
    private TextView recordButton;
    private TextView debugButton;
    private TextView flipButton;

    private LifecycleCameraController cameraController;
    private HandTracker handTracker;
    private PortalRenderer portalRenderer;
    private HandlerThread overlayThread;
    private OverlayEffect overlayEffect;
    private Recording activeRecording;

    private boolean frontCamera = true;
    private boolean debug = false;
    private boolean cameraConfigured = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        enterImmersive();
        buildUi();

        portalRenderer = new PortalRenderer(portalState);
        setupOverlayEffect();

        handTracker = new HandTracker(this, portalState, this);
        handTracker.setFrontCamera(frontCamera);
        handTracker.initialize();

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            setupCamera();
        } else {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION);
        }
    }

    private void buildUi() {
        FrameLayout root = new FrameLayout(this);
        root.setBackgroundColor(Color.rgb(3, 5, 8));

        previewView = new PreviewView(this);
        previewView.setImplementationMode(PreviewView.ImplementationMode.PERFORMANCE);
        previewView.setScaleType(PreviewView.ScaleType.FILL_CENTER);
        root.addView(previewView, new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT));

        LinearLayout top = new LinearLayout(this);
        top.setOrientation(LinearLayout.HORIZONTAL);
        top.setGravity(Gravity.CENTER_VERTICAL);
        top.setPadding(dp(14), dp(14), dp(14), dp(14));
        FrameLayout.LayoutParams topLp = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.TOP);

        statusText = pill("MENYIAPKAN GPU", 13);
        perfText = pill("-- ms · -- FPS", 12);
        LinearLayout.LayoutParams grow = new LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f);
        top.addView(statusText, grow);
        LinearLayout.LayoutParams perfLp = new LinearLayout.LayoutParams(ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        perfLp.leftMargin = dp(10);
        top.addView(perfText, perfLp);
        root.addView(top, topLp);

        LinearLayout controls = new LinearLayout(this);
        controls.setOrientation(LinearLayout.HORIZONTAL);
        controls.setGravity(Gravity.CENTER);
        controls.setPadding(dp(10), dp(8), dp(10), dp(18));

        effectButton = control("THERMAL HOLO", true);
        recordButton = control("●", false);
        debugButton = control("⌁", false);
        flipButton = control("↺", false);

        LinearLayout.LayoutParams effectLp = new LinearLayout.LayoutParams(0, dp(54), 1f);
        controls.addView(effectButton, effectLp);
        addSmallControl(controls, recordButton);
        addSmallControl(controls, debugButton);
        addSmallControl(controls, flipButton);

        FrameLayout.LayoutParams controlsLp = new FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT, Gravity.BOTTOM);
        root.addView(controls, controlsLp);

        effectButton.setOnClickListener(v -> effectButton.setText(portalRenderer.nextStyle()));
        debugButton.setOnClickListener(v -> {
            debug = !debug;
            portalRenderer.setDebug(debug);
            debugButton.setText(debug ? "⌁·" : "⌁");
        });
        flipButton.setOnClickListener(v -> switchCamera());
        recordButton.setOnClickListener(v -> toggleRecording());

        setContentView(root);
    }

    private void setupOverlayEffect() {
        overlayThread = new HandlerThread("Portal-Overlay-GL");
        overlayThread.start();
        Handler handler = new Handler(overlayThread.getLooper());
        overlayEffect = new OverlayEffect(
                CameraEffect.PREVIEW | CameraEffect.VIDEO_CAPTURE,
                0,
                handler,
                throwable -> runOnUiThread(() -> setStatus("OVERLAY ERROR", 0xFFFF6B78)));
        overlayEffect.setOnDrawListener(frame -> portalRenderer.draw(frame));
    }

    private void setupCamera() {
        if (cameraConfigured) return;
        cameraConfigured = true;

        cameraController = new LifecycleCameraController(this);
        cameraController.setCameraSelector(frontCamera ? CameraSelector.DEFAULT_FRONT_CAMERA : CameraSelector.DEFAULT_BACK_CAMERA);
        cameraController.setEnabledUseCases(CameraController.IMAGE_ANALYSIS | CameraController.VIDEO_CAPTURE);
        cameraController.setVideoCaptureMirrorMode(MirrorMode.MIRROR_MODE_ON_FRONT_ONLY);
        cameraController.setImageAnalysisBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST);
        cameraController.setImageAnalysisOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888);
        cameraController.setImageAnalysisAnalyzer(handTracker.executor(), handTracker);
        cameraController.setEffects(Collections.singleton(overlayEffect));
        cameraController.bindToLifecycle(this);
        previewView.setController(cameraController);
    }

    private void switchCamera() {
        if (cameraController == null) return;
        frontCamera = !frontCamera;
        handTracker.setFrontCamera(frontCamera);
        portalState.set(PortalState.none());
        try {
            cameraController.setCameraSelector(frontCamera ? CameraSelector.DEFAULT_FRONT_CAMERA : CameraSelector.DEFAULT_BACK_CAMERA);
            toast(frontCamera ? "Kamera depan" : "Kamera belakang");
        } catch (Throwable t) {
            frontCamera = !frontCamera;
            handTracker.setFrontCamera(frontCamera);
            toast("Kamera itu nggak tersedia");
        }
    }

    private void toggleRecording() {
        if (cameraController == null) return;
        if (activeRecording != null) {
            activeRecording.stop();
            return;
        }

        ContentValues values = new ContentValues();
        String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
        values.put(MediaStore.Video.Media.DISPLAY_NAME, "Portal-" + stamp);
        values.put(MediaStore.Video.Media.MIME_TYPE, "video/mp4");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            values.put(MediaStore.Video.Media.RELATIVE_PATH, "Movies/Portal");
        }

        MediaStoreOutputOptions output = new MediaStoreOutputOptions.Builder(
                getContentResolver(), MediaStore.Video.Media.EXTERNAL_CONTENT_URI)
                .setContentValues(values)
                .build();

        try {
            activeRecording = cameraController.startRecording(
                    output,
                    AudioConfig.AUDIO_DISABLED,
                    ContextCompat.getMainExecutor(this),
                    event -> {
                        if (event instanceof VideoRecordEvent.Start) {
                            recordButton.setText("■");
                            recordButton.setTextColor(0xFFFF6779);
                            setStatus("REC · " + backendLabel(), 0xFFFF6779);
                        } else if (event instanceof VideoRecordEvent.Finalize) {
                            VideoRecordEvent.Finalize fin = (VideoRecordEvent.Finalize) event;
                            activeRecording = null;
                            recordButton.setText("●");
                            recordButton.setTextColor(Color.WHITE);
                            if (fin.hasError()) {
                                toast("Rekam gagal: " + fin.getError());
                            } else {
                                toast("Video tersimpan di Movies/Portal");
                            }
                        }
                    });
        } catch (Throwable t) {
            activeRecording = null;
            toast("Gagal mulai rekam: " + t.getMessage());
        }
    }

    @Override
    public void onBackendReady(String backend) {
        runOnUiThread(() -> {
            int color = backend.startsWith("GPU") ? 0xFF61F3C2 : 0xFFFFC966;
            setStatus("TRACKING · " + backend, color);
        });
    }

    @Override
    public void onTrackerError(String message) {
        runOnUiThread(() -> {
            setStatus("TRACKING ERROR", 0xFFFF6B78);
            perfText.setText(message == null ? "error" : message);
        });
    }

    @Override
    public void onPerf(double inferenceMs, double detectorFps, int hands) {
        runOnUiThread(() -> {
            perfText.setText(String.format(Locale.US, "%.1f ms · %.0f track/s · %d tangan", inferenceMs, detectorFps, hands));
            if (activeRecording == null) {
                PortalState s = portalState.get();
                String mode = s.mode == PortalState.Mode.ONE_HAND ? "PINCH LOCK" :
                        s.mode == PortalState.Mode.TWO_HAND ? "2 TANGAN" : "CARI TANGAN";
                statusText.setText(mode + " · " + backendLabel());
            }
        });
    }

    private String backendLabel() {
        String text = statusText == null ? "GPU" : statusText.getText().toString();
        return text.contains("CPU") ? "CPU" : "GPU";
    }

    private void setStatus(String text, int color) {
        statusText.setText(text);
        statusText.setTextColor(color);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == CAMERA_PERMISSION) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                setupCamera();
            } else {
                setStatus("IZIN KAMERA DIBUTUHKAN", 0xFFFF6B78);
            }
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        enterImmersive();
    }

    @Override
    protected void onDestroy() {
        if (activeRecording != null) {
            activeRecording.stop();
            activeRecording = null;
        }
        if (cameraController != null) cameraController.clearImageAnalysisAnalyzer();
        if (handTracker != null) handTracker.close();
        if (overlayEffect != null) overlayEffect.close();
        if (overlayThread != null) overlayThread.quitSafely();
        super.onDestroy();
    }

    private void enterImmersive() {
        Window window = getWindow();
        window.setStatusBarColor(Color.TRANSPARENT);
        window.setNavigationBarColor(Color.TRANSPARENT);
        window.getDecorView().setSystemUiVisibility(
                View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY |
                View.SYSTEM_UI_FLAG_FULLSCREEN |
                View.SYSTEM_UI_FLAG_HIDE_NAVIGATION |
                View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN |
                View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION |
                View.SYSTEM_UI_FLAG_LAYOUT_STABLE);
    }

    private TextView pill(String text, int sp) {
        TextView v = new TextView(this);
        v.setText(text);
        v.setTextColor(Color.WHITE);
        v.setTextSize(sp);
        v.setGravity(Gravity.CENTER);
        v.setPadding(dp(12), dp(8), dp(12), dp(8));
        v.setBackground(roundRect(0x8A071018, 0x334FE5FF, 18));
        return v;
    }

    private TextView control(String text, boolean wide) {
        TextView v = new TextView(this);
        v.setText(text);
        v.setTextColor(Color.WHITE);
        v.setTextSize(wide ? 12 : 21);
        v.setGravity(Gravity.CENTER);
        v.setSingleLine(true);
        v.setBackground(roundRect(0xB20A1118, 0x554FE5FF, 18));
        v.setClickable(true);
        v.setFocusable(true);
        return v;
    }

    private void addSmallControl(LinearLayout row, TextView view) {
        LinearLayout.LayoutParams lp = new LinearLayout.LayoutParams(dp(54), dp(54));
        lp.leftMargin = dp(8);
        row.addView(view, lp);
    }

    private GradientDrawable roundRect(int fill, int stroke, int radiusDp) {
        GradientDrawable d = new GradientDrawable();
        d.setColor(fill);
        d.setCornerRadius(dp(radiusDp));
        d.setStroke(dp(1), stroke);
        return d;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private void toast(String text) {
        Toast.makeText(this, text, Toast.LENGTH_SHORT).show();
    }
}
