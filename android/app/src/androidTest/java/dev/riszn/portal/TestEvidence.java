package dev.riszn.portal;

import android.graphics.Bitmap;
import android.os.ParcelFileDescriptor;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.io.FileOutputStream;
import static org.junit.Assert.*;

final class TestEvidence {
    private TestEvidence() {}
    static void save(Bitmap bitmap,String name) throws Exception {
        var instrumentation=InstrumentationRegistry.getInstrumentation();
        File local=new File(instrumentation.getTargetContext().getExternalFilesDir(null),name+".png");
        try(var stream=new FileOutputStream(local)) {assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG,100,stream));}
        // AGP uninstalls the target app after instrumentation, deleting its external files.
        // UiAutomation executes argv directly, without shell operators. Wait for cp to finish,
        // then verify the exported file size under the shell identity before uninstall.
        String destination="/sdcard/Download/Portal-CI-"+name+".png";
        shell("cp "+local.getAbsolutePath()+" "+destination);
        assertEquals("Could not preserve test evidence",Long.toString(local.length()),
                shell("stat -c %s "+destination).trim());
    }
    private static String shell(String command) throws Exception {
        var automation=InstrumentationRegistry.getInstrumentation().getUiAutomation();
        try(var descriptor=automation.executeShellCommand(command);
            var stream=new ParcelFileDescriptor.AutoCloseInputStream(descriptor)) {
            StringBuilder response=new StringBuilder();int c;
            while((c=stream.read())!=-1)response.append((char)c);
            return response.toString();
        }
    }
}
