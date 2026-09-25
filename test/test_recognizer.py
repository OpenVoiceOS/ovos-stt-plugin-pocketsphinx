"""Cover the branches the live transcription test never reaches.

test_plugin.py drives one happy path: a real transcription of goforward.raw.
It never reaches the three RequestError raises in PocketSphinxRecognizer's
constructor, the grammar branch of recognize(), or the keyword-file branch.

Every message here is asserted in full, and each is asserted once with a path
that holds a double quote and a brace. Those two characters are what tells a
correct rewrite of these messages from an incorrect one: the messages wrap the
path in escaped quotes, and a rewrite that drops the escaping or that lets a
brace reach a formatter changes the text. A test that only checks the path is
mentioned somewhere would pass through such a change.
"""
import unittest
from os.path import dirname, join

import speech_recognition as sr

from ovos_stt_plugin_pocketsphinx.recognizer import PocketSphinxRecognizer

AUDIO = join(dirname(__file__), "goforward.raw")

# A name no filesystem holds, carrying the two characters that break a bad
# rewrite of the messages under test.
AWKWARD = '/no/such/pa"th/{braces}'


def _audio():
    with open(AUDIO, "rb") as f:
        return sr.AudioData(f.read(), 16000, 2)


class TestConstructorRejectsMissingModel(unittest.TestCase):
    """The three RequestError branches.

    The constructor checks the three paths in order, so each test supplies
    real paths for the checks before the one it drives. That ordering is
    itself under test: if the checks were reordered, the lm and dictionary
    tests would report the directory message instead.
    """

    @classmethod
    def setUpClass(cls):
        cls.hmm, cls.lm, cls.pho = \
            PocketSphinxRecognizer.get_default_english_model()

    def test_missing_acoustic_parameters_directory(self):
        for path in ("/no/such/dir", AWKWARD):
            with self.subTest(path=path):
                with self.assertRaises(sr.RequestError) as caught:
                    PocketSphinxRecognizer(path, self.lm, self.pho)
                self.assertEqual(
                    str(caught.exception),
                    'missing PocketSphinx language model parameters '
                    f'directory: "{path}"')

    def test_missing_language_model_file(self):
        for path in ("/no/such/lm", AWKWARD):
            with self.subTest(path=path):
                with self.assertRaises(sr.RequestError) as caught:
                    PocketSphinxRecognizer(self.hmm, path, self.pho)
                self.assertEqual(
                    str(caught.exception),
                    'missing PocketSphinx language model file: '
                    f'"{path}"')

    def test_missing_phoneme_dictionary_file(self):
        for path in ("/no/such/dict", AWKWARD):
            with self.subTest(path=path):
                with self.assertRaises(sr.RequestError) as caught:
                    PocketSphinxRecognizer(self.hmm, self.lm, path)
                self.assertEqual(
                    str(caught.exception),
                    'missing PocketSphinx phoneme dictionary file: '
                    f'"{path}"')

    def test_the_bundled_model_satisfies_all_three_checks(self):
        """The control. Without it, a constructor that raised on every input
        would pass the three tests above."""
        self.assertIsNotNone(
            PocketSphinxRecognizer(self.hmm, self.lm, self.pho).decoder)


class TestRecognizeBranches(unittest.TestCase):
    """The grammar and keyword-file branches of recognize()."""

    @classmethod
    def setUpClass(cls):
        cls.recognizer = PocketSphinxRecognizer(
            *PocketSphinxRecognizer.get_default_english_model())

    def test_missing_grammar_is_rejected(self):
        for path in ("/no/such/grammar.jsgf", AWKWARD):
            with self.subTest(path=path):
                with self.assertRaises(ValueError) as caught:
                    self.recognizer.recognize(_audio(), grammar=path)
                self.assertEqual(
                    str(caught.exception),
                    f"Grammar '{path}' does not exist.")

    def test_keyword_entries_drive_the_keyword_search(self):
        """Reaches the keyword-file writer, which renders each sensitivity
        into a threshold.

        The recording says "go forward ten meters". At sensitivity 1.0 a
        keyword that is spoken is found and one that is not raises, so the
        file this branch writes did reach the decoder and was read. The pair
        is the point: the positive alone would pass against a search that
        accepts every keyword, which is what this code does at sensitivity
        0.5 and below.
        """
        self.assertIn("forward", self.recognizer.recognize(
            _audio(), keyword_entries=[("forward", 1.0)]))

        self.assertRaises(
            sr.UnknownValueError, self.recognizer.recognize,
            _audio(), keyword_entries=[("aardvark", 1.0)])

    def test_keyword_entries_are_validated(self):
        for entries in ([("forward", 1.5)], [("forward", -0.1)],
                        [(object(), 0.5)]):
            with self.subTest(entries=entries):
                self.assertRaises(AssertionError, self.recognizer.recognize,
                                  _audio(), entries)


if __name__ == "__main__":
    unittest.main()
