"""recognize_wav reads a wave through whichever source class is installed.

ovos-plugin-manager replaces ``speech_recognition.AudioFile`` with its own
class from 2.2 on, and importing this package installs that replacement:
``ovos_stt_plugin_pocketsphinx/__init__.py`` imports
``ovos_plugin_manager.templates.stt``, which pulls in
``ovos_plugin_manager.utils.audio``.

The replacement does not subclass ``speech_recognition.AudioSource``, and
``Recognizer.record()`` asserts exactly that, so the old
``with sr.AudioFile(path) as source: r.record(source)`` raised
``AssertionError: Source must be an audio source`` on any current install.

This package declares ``ovos-plugin-manager>=0.0.1a7`` with no upper bound,
so both source classes are in range and these tests drive the property
``record()`` checks rather than a version. Same shape as
ovos-microphone-plugin-files#26.
"""
import os
import struct
import tempfile
import unittest
import wave
from unittest.mock import patch

import speech_recognition as sr

from ovos_stt_plugin_pocketsphinx.recognizer import PocketSphinxRecognizer


def _wave_file(frames=8000):
    path = os.path.join(tempfile.mkdtemp(prefix="t4183-"), "a.wav")
    handle = wave.open(path, "wb")
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(16000)
    handle.writeframes(struct.pack(f"<{frames}h", *([0] * frames)))
    handle.close()
    return path


class _OPMStyleSource:
    """Reads itself, as ovos-plugin-manager's class does.

    Deliberately NOT an ``sr.AudioSource`` and with no attribute
    ``record()`` needs, so a test that took the wrong branch would fail
    rather than quietly pass.
    """

    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        with wave.open(self.path, "rb") as handle:
            raw = handle.readframes(handle.getnframes())
        return sr.AudioData(raw, 16000, 2)


class _UpstreamStyleSource(sr.AudioSource):
    """An ``AudioSource`` with no ``read()``, as the upstream class is."""

    def __init__(self, path):
        self._inner = sr.AudioFile.__mro__[0]
        self.path = path
        self._file = None
        self.stream = None
        self.CHUNK = 4096
        self.SAMPLE_RATE = 16000
        self.SAMPLE_WIDTH = 2

    def __enter__(self):
        self._file = wave.open(self.path, "rb")
        self.stream = _WaveStream(self._file)
        return self

    def __exit__(self, *exc):
        self._file.close()
        return False


class _WaveStream:
    def __init__(self, handle):
        self.handle = handle

    def read(self, size):
        return self.handle.readframes(size // 2)


class TestReadWaveFile(unittest.TestCase):

    def test_an_opm_style_source_is_read_through_read(self):
        """The branch that was broken. record() refuses this source."""
        path = _wave_file()
        with patch.object(sr, "AudioFile", _OPMStyleSource):
            audio = PocketSphinxRecognizer.read_wave_file(path)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(len(audio.frame_data), 16000)

    def test_an_upstream_source_is_read_through_record(self):
        """The branch that always worked, kept working."""
        path = _wave_file()
        with patch.object(sr, "AudioFile", _UpstreamStyleSource):
            audio = PocketSphinxRecognizer.read_wave_file(path)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(len(audio.frame_data), 16000)

    def test_the_real_installed_source_is_read(self):
        """No patching. Whichever class this install actually has, the
        wave comes back. This is the test that would have caught the
        defect on a normal developer machine."""
        path = _wave_file()
        audio = PocketSphinxRecognizer.read_wave_file(path)
        self.assertIsInstance(audio, sr.AudioData)
        self.assertEqual(len(audio.frame_data), 16000)

    def test_the_installed_source_really_is_the_opm_one(self):
        """The control on the test above.

        If ovos-plugin-manager ever stops replacing the class, the test
        above still passes and stops covering the defect. This says out
        loud which class was exercised, so a change upstream is visible
        rather than silent.
        """
        self.assertFalse(
            issubclass(sr.AudioFile, sr.AudioSource),
            "speech_recognition.AudioFile is NOT patched in this "
            "environment: ovos-plugin-manager no longer replaces it, or it "
            "was imported before the plugin. The no-patch branch is the "
            "one under test here, and the OPM branch is covered only by "
            "test_an_opm_style_source_is_read_through_read.")


if __name__ == "__main__":
    unittest.main()
