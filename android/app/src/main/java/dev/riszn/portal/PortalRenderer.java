package dev.riszn.portal;

import android.graphics.Matrix;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.atomic.AtomicReference;

/** GL camera shader shared by preview and video. Never reads camera pixels back to the CPU. */
final class PortalRenderer {
    private static final String[] STYLES = {"VIOLET", "INVERT", "MONO", "THERMAL", "CLEAR"};
    private static final int[][] BONES = {
        {0,1},{1,2},{2,3},{3,4},{0,5},{5,6},{6,7},{7,8},{5,9},{9,10},{10,11},{11,12},
        {9,13},{13,14},{14,15},{15,16},{13,17},{0,17},{17,18},{18,19},{19,20}
    };
    static final String VERTEX = """
        attribute vec2 aPosition;
        uniform mat4 uTextureMatrix;
        varying vec2 vTexture;
        void main() {
          gl_Position = vec4(aPosition, 0.0, 1.0);
          vTexture = (uTextureMatrix * vec4(aPosition * 0.5 + 0.5, 0.0, 1.0)).xy;
        }
        """;
    static final String FRAGMENT = """
        #extension GL_OES_EGL_image_external : require
        precision highp float;
        uniform samplerExternalOES uCamera;
        uniform vec2 uSize;
        uniform vec2 uCorners[4];
        uniform int uVisible;
        uniform int uStyle;
        varying vec2 vTexture;
        float cross2(vec2 a, vec2 b) { return a.x*b.y-a.y*b.x; }
        float edgeDistance(vec2 p, vec2 a, vec2 b) {
          vec2 d = b-a;
          float t = clamp(dot(p-a,d)/max(dot(d,d),0.0001),0.0,1.0);
          return length(p-a-t*d);
        }
        void main() {
          vec3 rgb = texture2D(uCamera, vTexture).rgb;
          vec2 p = vec2(gl_FragCoord.x,uSize.y-gl_FragCoord.y);
          if (uVisible == 1) {
            float a = cross2(uCorners[1]-uCorners[0],p-uCorners[0]);
            float b = cross2(uCorners[2]-uCorners[1],p-uCorners[1]);
            float c = cross2(uCorners[3]-uCorners[2],p-uCorners[2]);
            float d = cross2(uCorners[0]-uCorners[3],p-uCorners[3]);
            bool inside = min(min(a,b),min(c,d)) >= 0.0 || max(max(a,b),max(c,d)) <= 0.0;
            if (inside) {
              float luma = dot(rgb,vec3(0.299,0.587,0.114));
              if (uStyle == 0) rgb = vec3(rgb.r*0.55+0.10,rgb.g*0.12,rgb.b*0.65+0.18);
              if (uStyle == 1) rgb = vec3(1.0)-rgb;
              if (uStyle == 2) rgb = vec3(luma);
              if (uStyle == 3) rgb = clamp(vec3(1.5)-abs(4.0*luma-vec3(3.0,2.0,1.0)),0.0,1.0);
            }
            float edge = min(min(edgeDistance(p,uCorners[0],uCorners[1]),edgeDistance(p,uCorners[1],uCorners[2])),
                             min(edgeDistance(p,uCorners[2],uCorners[3]),edgeDistance(p,uCorners[3],uCorners[0])));
            rgb = mix(rgb,vec3(0.88,0.85,0.95),(1.0-smoothstep(0.5,1.5,edge))*0.7);
          }
          gl_FragColor = vec4(rgb,1.0);
        }
        """;
    private static final String LINES_VERTEX = """
        attribute vec2 aPosition;
        uniform vec2 uSize;
        uniform float uPointSize;
        void main() {
          gl_Position = vec4(aPosition.x/uSize.x*2.0-1.0,1.0-aPosition.y/uSize.y*2.0,0.0,1.0);
          gl_PointSize = uPointSize;
        }
        """;
    private static final String LINES_FRAGMENT = "precision mediump float; uniform vec3 uColor; void main(){gl_FragColor=vec4(uColor,1.0);}";
    private final AtomicReference<PortalState> stateRef;
    private final FloatBuffer screen = buffer(new float[]{-1,-1,1,-1,-1,1,1,1});
    private final FloatBuffer lines = ByteBuffer.allocateDirect(21*4*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
    private final float[] mapped = new float[42], corners = new float[8];
    private final Matrix analysisToOutput = new Matrix();
    private volatile int style;
    private volatile boolean debug;
    private volatile double responseMs;
    private long measuredState;
    private int program, lineProgram, position, textureMatrix, camera, size, quad, visible, styleUniform;
    private int linePosition, lineSize, lineColor, pointSize;

    PortalRenderer(AtomicReference<PortalState> stateRef) { this.stateRef = stateRef; }
    String nextStyle() { style = (style+1)%STYLES.length; return STYLES[style]; }
    void setDebug(boolean enabled) { debug = enabled; }
    double responseMs() { return responseMs; }
    void initialize() {
        program = link(VERTEX,FRAGMENT); lineProgram = link(LINES_VERTEX,LINES_FRAGMENT);
        position = GLES20.glGetAttribLocation(program,"aPosition");
        textureMatrix = uniform(program,"uTextureMatrix"); camera = uniform(program,"uCamera");
        size = uniform(program,"uSize"); quad = uniform(program,"uCorners");
        visible = uniform(program,"uVisible"); styleUniform = uniform(program,"uStyle");
        linePosition = GLES20.glGetAttribLocation(lineProgram,"aPosition");
        lineSize = uniform(lineProgram,"uSize"); lineColor = uniform(lineProgram,"uColor");
        pointSize = uniform(lineProgram,"uPointSize");
    }

    void draw(int texture, float[] transform, Matrix sensorToOutput, int width, int height) {
        PortalState state = stateRef.get();
        long now = System.nanoTime();
        analysisToOutput.set(state.analysisToSensor); analysisToOutput.postConcat(sensorToOutput);
        GLES20.glViewport(0,0,width,height);
        GLES20.glUseProgram(program);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,texture);
        GLES20.glUniform1i(camera,0);
        GLES20.glUniformMatrix4fv(textureMatrix,1,false,transform,0);
        GLES20.glUniform2f(size,width,height);
        GLES20.glUniform1i(visible,state.visible()?1:0);
        GLES20.glUniform1i(styleUniform,style);
        if (state.visible()) {
            analysisToOutput.mapPoints(corners,state.nodes);
            GLES20.glUniform2fv(quad,4,corners,0);
        }
        screen.position(0);
        GLES20.glEnableVertexAttribArray(position);
        GLES20.glVertexAttribPointer(position,2,GLES20.GL_FLOAT,false,0,screen);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP,0,4);
        GLES20.glDisableVertexAttribArray(position);
        if (state.visible()) {
            GLES20.glUseProgram(lineProgram); GLES20.glUniform2f(lineSize,width,height);
            GLES20.glUniform3f(lineColor,.88f,.85f,.95f); GLES20.glUniform1f(pointSize,3f);
            GLES20.glEnableVertexAttribArray(linePosition);
            lines.clear(); lines.put(corners); lines.flip();
            GLES20.glVertexAttribPointer(linePosition,2,GLES20.GL_FLOAT,false,0,lines);
            GLES20.glDrawArrays(GLES20.GL_POINTS,0,4);
            GLES20.glUniform3f(lineColor,.5f,1f,.8f); GLES20.glUniform1f(pointSize,5f);
            for (int handle : state.grabbedHandles) {
                int a = handle < 4 ? handle : handle-4, b = handle < 4 ? a : (a+1)%4;
                lines.clear(); lines.put((corners[a*2]+corners[b*2])*.5f).put((corners[a*2+1]+corners[b*2+1])*.5f); lines.flip();
                GLES20.glVertexAttribPointer(linePosition,2,GLES20.GL_FLOAT,false,0,lines);
                GLES20.glDrawArrays(GLES20.GL_POINTS,0,1);
            }
            GLES20.glDisableVertexAttribArray(linePosition);
        }
        // Geometry is already filtered and constrained. Renderer must not smooth/reshape it again.
        if (debug && now-state.producedAtNanos < 200_000_000L) {
            GLES20.glUseProgram(lineProgram); GLES20.glUniform1f(pointSize,6f); GLES20.glUniform2f(lineSize,width,height);
            GLES20.glEnableVertexAttribArray(linePosition);
            for (int h = 0; h < state.skeletons.length; h++) {
                analysisToOutput.mapPoints(mapped,state.skeletons[h]);
                if ((state.handIds[h]&1) == 1) GLES20.glUniform3f(lineColor,.35f,.91f,1f);
                else GLES20.glUniform3f(lineColor,1f,.39f,.85f);
                lines.clear();
                for (int[] bone : BONES) for (int joint : bone) lines.put(mapped[joint*2]).put(mapped[joint*2+1]);
                lines.flip(); GLES20.glVertexAttribPointer(linePosition,2,GLES20.GL_FLOAT,false,0,lines);
                GLES20.glLineWidth(2f); GLES20.glDrawArrays(GLES20.GL_LINES,0,BONES.length*2);
                lines.clear(); lines.put(mapped); lines.flip();
                GLES20.glVertexAttribPointer(linePosition,2,GLES20.GL_FLOAT,false,0,lines);
                GLES20.glDrawArrays(GLES20.GL_POINTS,0,21);
            }
            GLES20.glDisableVertexAttribArray(linePosition);
        }
        if (state.producedAtNanos != measuredState && state.analysisStartedAtNanos != 0) {
            responseMs = (now-state.analysisStartedAtNanos)/1e6; measuredState = state.producedAtNanos;
        }
    }
    void release() { GLES20.glDeleteProgram(program); GLES20.glDeleteProgram(lineProgram); }
    private static FloatBuffer buffer(float[] values) {
        FloatBuffer b = ByteBuffer.allocateDirect(values.length*4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        b.put(values).flip(); return b;
    }
    private static int uniform(int p,String name) { return GLES20.glGetUniformLocation(p,name); }
    private static int shader(int type,String source) {
        int shader = GLES20.glCreateShader(type); GLES20.glShaderSource(shader,source); GLES20.glCompileShader(shader);
        int[] ok = new int[1]; GLES20.glGetShaderiv(shader,GLES20.GL_COMPILE_STATUS,ok,0);
        if (ok[0] == 0) { String error = GLES20.glGetShaderInfoLog(shader); GLES20.glDeleteShader(shader); throw new IllegalStateException(error); }
        return shader;
    }
    private static int link(String vertex,String fragment) {
        int v = shader(GLES20.GL_VERTEX_SHADER,vertex), f = shader(GLES20.GL_FRAGMENT_SHADER,fragment);
        int p = GLES20.glCreateProgram(); GLES20.glAttachShader(p,v); GLES20.glAttachShader(p,f); GLES20.glLinkProgram(p);
        GLES20.glDeleteShader(v); GLES20.glDeleteShader(f);
        int[] ok = new int[1]; GLES20.glGetProgramiv(p,GLES20.GL_LINK_STATUS,ok,0);
        if(ok[0] == 0) { String error = GLES20.glGetProgramInfoLog(p); GLES20.glDeleteProgram(p); throw new IllegalStateException(error); }
        return p;
    }
}
