#!/bin/sh
# Preserve failure logs before android-emulator-runner shuts the emulator down.
set -eu
mkdir -p android/app/build/diagnostics android/app/build/recording-evidence
capture() {
    result=$?
    adb logcat -d -v threadtime > android/app/build/diagnostics/logcat.txt || true
    adb shell dumpsys media.codec > android/app/build/diagnostics/media-codec.txt || true
    adb pull /sdcard/Android/data/dev.riszn.portal/files/ android/app/build/recording-evidence/ || true
    exit "$result"
}
trap capture EXIT
adb logcat -c
gradle -p android --no-daemon :app:connectedDebugAndroidTest --stacktrace
