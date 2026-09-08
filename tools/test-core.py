#!/usr/bin/env python3
"""Run the same JUnit core tests without an Android SDK (use cached Gradle JUnit jars)."""
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
cache = pathlib.Path.home() / '.gradle/caches/modules-2/files-2.1'
jars = list(cache.glob('junit/junit/4.13.2/*/*.jar')) + list(cache.glob('org.hamcrest/hamcrest-core/1.3/*/*.jar'))
if len(jars) < 2:
    raise SystemExit('Run gradle :app:testDebugUnitTest once to populate JUnit dependencies.')
cp = ':'.join(map(str, jars))
out = root / 'android/app/build/core-tests'
out.mkdir(parents=True, exist_ok=True)
src = root / 'android/app/src/main/java/dev/riszn/portal'
files = [src / 'HandTracks.java', src / 'PortalInteraction.java'] + list((root / 'android/app/src/test/java/dev/riszn/portal').glob('*.java'))
subprocess.run(['javac', '-cp', cp, '-d', str(out), *map(str, files)], check=True)
subprocess.run(['java', '-cp', str(out) + ':' + cp, 'org.junit.runner.JUnitCore', 'dev.riszn.portal.PortalInteractionTest', 'dev.riszn.portal.HandTracksTest'], check=True)
