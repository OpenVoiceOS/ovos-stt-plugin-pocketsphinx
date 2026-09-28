"""The branches `test_plugin.py` cannot reach.

`test_plugin.py` is one live transcription of `goforward.raw` through a fully
configured plugin. It is the right test for the happy path and it reaches
none of the error branches in `recognizer.py`: the three `RequestError`
raises in `PocketSphinxRecognizer.__init__` all fire before the decoder is
built, and the keyword and grammar paths of `recognize` are never taken,
because the plugin passes neither.

reviewer-c reached these branches while reviewing #8 by stubbing
`speech_recognition` and `pocketsphinx` in `sys.modules`. That was the right
tool for a differential over two trees loaded side by side in one process.
Here the real packages are installed — `test_plugin.py` already depends on
them — so these tests use the real `sr.RequestError` and the real decoder
rather than a stub that could drift from either.

No acoustic model is needed for the three raises: each one fires before
`Decoder(...)` is reached.
"""
import unittest
from os.path import dirname, join

import speech_recognition as sr

from ovos_stt_plugin_pocketsphinx.recognizer import PocketSphinxRecognizer

AUDIO = join(dirname(__file__), "goforward.raw")


def _audio():
    with open(AUDIO, "rb") as f:
        return sr.AudioData(f.read(), 16000, 2)


def _real_model():
    return PocketSphinxRecognizer.get_default_english_model()


class TestTheConstructorRejectsMissingFiles(unittest.TestCase):
    """Each raise names what is missing, and the path it looked for.

    The message matters: it is what a deployer reads when a model is
    mis-configured, and the path in it is the only clue to which of the three
    settings is wrong.
    """

    def test_a_missing_acoustic_model_directory(self):
        hmm, lm, pho = _real_model()
        missing = "/nonexistent/acoustic-parameters"
        with self.assertRaises(sr.RequestError) as caught:
            PocketSphinxRecognizer(missing, lm, pho)
        self.assertEqual(
            str(caught.exception),
            f'missing PocketSphinx language model parameters directory: "{missing}"')

    def test_a_missing_language_model_file(self):
        hmm, lm, pho = _real_model()
        missing = join(dirname(lm), "no-such.lm.bin")
        with self.assertRaises(sr.RequestError) as caught:
            PocketSphinxRecognizer(hmm, missing, pho)
        self.assertEqual(
            str(caught.exception),
            f'missing PocketSphinx language model file: "{missing}"')

    def test_a_missing_phoneme_dictionary_file(self):
        hmm, lm, pho = _real_model()
        missing = join(dirname(pho), "no-such.dict")
        with self.assertRaises(sr.RequestError) as caught:
            PocketSphinxRecognizer(hmm, lm, missing)
        self.assertEqual(
            str(caught.exception),
            f'missing PocketSphinx phoneme dictionary file: "{missing}"')

    def test_the_directory_is_checked_before_the_files(self):
        """All three wrong reports the directory, not the dictionary: a
        deployer fixing one thing at a time should be told the first."""
        with self.assertRaises(sr.RequestError) as caught:
            PocketSphinxRecognizer("/nonexistent/dir", "/nonexistent/lm",
                                   "/nonexistent/dict")
        self.assertIn("parameters directory", str(caught.exception))

    def test_a_directory_where_a_file_belongs_is_missing(self):
        """`isfile` is the check, so a directory at the language model path
        is not a language model."""
        hmm, lm, pho = _real_model()
        a_directory = dirname(lm)
        with self.assertRaises(sr.RequestError) as caught:
            PocketSphinxRecognizer(hmm, a_directory, pho)
        self.assertIn("language model file", str(caught.exception))


class TestTheGrammarBranch(unittest.TestCase):
    """`recognize(grammar=...)` refuses a path that is not there."""

    def test_a_grammar_that_does_not_exist(self):
        recognizer = PocketSphinxRecognizer(*_real_model())
        missing = "/nonexistent/commands.jsgf"
        with self.assertRaises(ValueError) as caught:
            recognizer.recognize(_audio(), grammar=missing)
        self.assertEqual(str(caught.exception),
                         f"Grammar '{missing}' does not exist.")


class TestTheKeywordBranch(unittest.TestCase):
    """`recognize(keyword_entries=...)` takes a different path through the
    decoder, and writes a keywords file whose threshold is computed per
    keyword."""

    def test_a_keyword_that_is_in_the_audio_is_found(self):
        recognizer = PocketSphinxRecognizer(*_real_model())
        found = recognizer.recognize(_audio(), keyword_entries=[("forward", 1.0)])
        self.assertIn("forward", found)

    def test_a_keyword_that_is_not_in_the_audio_is_not_found(self):
        """The negative case, so the test above cannot pass by matching
        anything at all. The audio says "go forward ten meters"."""
        recognizer = PocketSphinxRecognizer(*_real_model())
        with self.assertRaises(sr.UnknownValueError):
            recognizer.recognize(_audio(), keyword_entries=[("aardvark", 1.0)])

    def test_sensitivity_sets_the_threshold_and_a_low_one_accepts_anything(self):
        """`sensitivity` reads like "how sensitive the spotter is" and means
        the opposite: the file written is

            f"{keyword} /1e{100 * sensitivity - 110}/\n"

        so 1.0 is a threshold of 1e-10 and 0.1 is 1e-100, and the smaller the
        number the more the spotter accepts. At 0.1 this audio, which says
        "go forward ten meters", reports the keyword "aardvark".

        This is pinned rather than reported as a defect: the mapping comes
        from `speech_recognition`'s own pocketsphinx recognizer and changing
        it would change every caller's tuning. The test exists so that the
        expression cannot be altered by accident, since a rewrite of it was
        the reason this file was asked for.
        """
        recognizer = PocketSphinxRecognizer(*_real_model())
        self.assertIn("aardvark",
                      recognizer.recognize(_audio(),
                                           keyword_entries=[("aardvark", 0.1)]))

    def test_a_sensitivity_outside_the_range_is_refused(self):
        """The assertion documents ``0 <= sensitivity <= 1``."""
        recognizer = PocketSphinxRecognizer(*_real_model())
        with self.assertRaises(AssertionError):
            recognizer.recognize(_audio(), keyword_entries=[("forward", 4.2)])

    def test_audio_that_is_not_audio_data_is_refused(self):
        recognizer = PocketSphinxRecognizer(*_real_model())
        with self.assertRaises(AssertionError):
            recognizer.recognize(b"raw bytes are not AudioData")


if __name__ == "__main__":
    unittest.main()
