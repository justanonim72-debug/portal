package dev.riszn.portal;

import android.graphics.Bitmap;
import android.os.ParcelFileDescriptor;
import androidx.test.platform.app.InstrumentationRegistry;
import java.io.File;
import java.io.FileOutputStream;
import static org.junit.Assert.assertTrue;

final class TestEvidence {
    private TestEvidence() {}
    static void save(Bitmap bitmap,String name) throws Exception {
        var instrumentation=InstrumentationRegistry.getInstrumentation();
        File local=new File(instrumentation.getTargetContext().getExternalFilesDir(null),name+".png");
        try(var stream=new FileOutputStream(local)) {assertTrue(bitmap.compress(Bitmap.CompressFormat.PNG,100,stream));}
        // AGP uninstalls the target app after instrumentation, deleting its external files.
        // Copy under the test shell identity before uninstall so CI can preserve decoded evidence.
        String command="cp "+local.getAbsolutePath()+" /sdcard/Download/Portal-CI-"+name+".png && echo saved";
        try(var descriptor=instrumentation.getUiAutomation().executeShellCommand(command);
            var stream=new ParcelFileDescriptor.AutoCloseInputStream(descriptor)) {
            StringBuilder response=new StringBuilder();int c;
            while((c=stream.read())!=-1)response.append((char)c);
            assertTrue("Could not preserve test evidence: "+response,response.toString().contains("saved"));
        }
    }
}
