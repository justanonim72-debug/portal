package dev.riszn.portal;

import android.graphics.Matrix;
import com.google.mediapipe.tasks.components.containers.Category;
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark;
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult;
import java.util.ArrayList;
import java.util.List;

/** MediaPipe adapter; interaction and filtering are platform-independent and replay-testable. */
final class PortalGeometry {
    private final HandTracks tracks = new HandTracks();
    private final PortalInteraction interaction = new PortalInteraction();
    private int width, height, rotation;
    private boolean mirrored;

    synchronized void reset() { tracks.reset(); interaction.reset(); }

    synchronized PortalState fromResult(HandLandmarkerResult result, boolean front,
            Matrix analysisToSensor, int frameWidth, int frameHeight, int frameRotation,
            long startedAtNanos) {
        if (width != frameWidth || height != frameHeight || rotation != frameRotation || mirrored != front) reset();
        width = frameWidth; height = frameHeight; rotation = frameRotation; mirrored = front;
        long now = System.nanoTime();
        List<List<NormalizedLandmark>> landmarks = result.landmarks();
        int count = landmarks == null ? 0 : Math.min(2,landmarks.size());
        List<HandTracks.Observation> observations = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            List<NormalizedLandmark> hand = landmarks.get(i);
            float[] xy = new float[hand.size()*2];
            for (int j = 0; j < hand.size(); j++) {
                // ImageProcessingOptions rotates the inference ROI. MediaPipe's projection graph
                // returns landmarks in the ORIGINAL image, not the rotated ROI. No extra rotation
                // or selfie mirror belongs here. CameraX applies those at the output boundary.
                xy[j*2] = hand.get(j).x()*width; xy[j*2+1] = hand.get(j).y()*height;
            }
            String label = "?"; float confidence = 0;
            if (i < result.handedness().size() && !result.handedness().get(i).isEmpty()) {
                Category c = result.handedness().get(i).get(0);
                label = c.categoryName(); confidence = c.score();
            }
            observations.add(new HandTracks.Observation(xy,label,confidence));
        }
        List<HandTracks.Hand> hands = tracks.update(observations,now);
        interaction.update(hands,now);
        float[][] skeletons = new float[hands.size()][];
        int[] ids = new int[hands.size()];
        for (int i = 0; i < hands.size(); i++) { skeletons[i] = hands.get(i).skeleton; ids[i] = hands.get(i).id; }
        int[] handles = interaction.grips().stream().mapToInt(g -> g.handle).toArray();
        return new PortalState(interaction.phase(),interaction.corners(),skeletons,ids,handles,
                count,now,startedAtNanos,analysisToSensor,width,height);
    }
}
