import speech_recognition as sr
import os
from os.path import join, isdir, dirname, isfile
from pocketsphinx import Jsgf, FsgModel, Decoder, get_model_path


class PocketSphinxRecognizer:
    def __init__(self,
                 acoustic_parameters_directory=None,
                 language_model_file=None,
                 phoneme_dictionary_file=None):

        if not isdir(acoustic_parameters_directory):
            raise sr.RequestError(
                "missing PocketSphinx language model parameters directory: "
                "\"{}\"".format(acoustic_parameters_directory))

        if not isfile(language_model_file):
            raise sr.RequestError(
                "missing PocketSphinx language model file: \"{}\"".format(
                    language_model_file))

        if not isfile(phoneme_dictionary_file):
            raise sr.RequestError(
                "missing PocketSphinx phoneme dictionary file: \"{}\"".format(
                    phoneme_dictionary_file))

        # create decoder object
        self.decoder = Decoder(hmm=acoustic_parameters_directory,
                               lm=language_model_file,
                               dict=phoneme_dictionary_file,
                               logfn=os.devnull)

    @staticmethod
    def get_default_english_model():
        # the model bundled with pocketsphinx 5 ships as model/en-us/en-us
        # (acoustic model dir nested beside the dictionary and LMs)
        language_directory = join(get_model_path(), "en-us")
        hmm = join(language_directory, "en-us")
        lm = join(language_directory, "en-us.lm.bin")
        pho = join(language_directory, "cmudict-en-us.dict")
        return hmm, lm, pho

    def recognize(self, audio_data, keyword_entries=None, grammar=None):
        assert isinstance(audio_data,
                          sr.AudioData), "``audio_data`` must be audio data"
        assert keyword_entries is None or all(
            isinstance(keyword,
                       (type(""), type(u""))) and 0 <= sensitivity <= 1
            for keyword, sensitivity in
            keyword_entries), "``keyword_entries`` must be ``None`` or" \
                              " a list of pairs of strings and " \
                              "numbers between 0 and 1"
        # obtain audio data
        raw_data = audio_data.get_raw_data(convert_rate=16000,
                                           convert_width=2)
        # obtain recognition results
        if keyword_entries is not None:  # explicitly specified set of keywords
            with sr.PortableNamedTemporaryFile("w") as f:
                # generate a keywords file
                f.writelines(
                    "{} /1e{}/\n".format(keyword, 100 * sensitivity - 110)
                    for keyword, sensitivity in keyword_entries)
                f.flush()
                # perform the speech recognition with the keywords file
                self.decoder.set_kws("keywords", f.name)
                self.decoder.set_search("keywords")
                self.decoder.start_utt()  # begin utterance processing
                self.decoder.process_raw(raw_data, False,
                                         True)
                self.decoder.end_utt()  # stop utterance processing
        elif grammar is not None:  # a path to a FSG or JSGF grammar
            if not os.path.exists(grammar):
                raise ValueError(
                    "Grammar '{0}' does not exist.".format(grammar))
            grammar_path = os.path.abspath(os.path.dirname(grammar))
            grammar_name = os.path.splitext(os.path.basename(grammar))[0]
            fsg_path = "{0}/{1}.fsg".format(grammar_path, grammar_name)
            if not os.path.exists(
                    fsg_path):  # create FSG grammar if not available
                jsgf = Jsgf(grammar)
                rule = jsgf.get_rule("{0}.{0}".format(grammar_name))
                fsg = jsgf.build_fsg(rule, self.decoder.get_logmath(), 7.5)
                fsg.writefile(fsg_path)
            else:
                fsg = FsgModel(fsg_path, self.decoder.get_logmath(), 7.5)
            self.decoder.set_fsg(grammar_name, fsg)
            self.decoder.set_search(grammar_name)
            self.decoder.start_utt()
            self.decoder.process_raw(raw_data, False,
                                     True)
            self.decoder.end_utt()  # stop utterance processing
        else:  # no keywords, perform freeform recognition
            self.decoder.start_utt()  # begin utterance processing
            self.decoder.process_raw(raw_data, False,
                                     True)
            self.decoder.end_utt()  # stop utterance processing
        # return results
        hypothesis = self.decoder.hyp()
        if hypothesis is not None:
            return hypothesis.hypstr
        raise sr.UnknownValueError()  # no transcriptions available

    @staticmethod
    def read_wave_file(file_path):
        """Read a wave file through whichever source class is installed.

        ovos-plugin-manager replaces ``speech_recognition.AudioFile`` with
        its own class (``ovos_plugin_manager.utils.audio``) from 2.2 on, and
        importing this package installs that replacement: the plugin's
        ``__init__`` imports ``ovos_plugin_manager.templates.stt``, which
        pulls it in. The replacement does not subclass
        ``speech_recognition.AudioSource``, which is exactly what
        ``Recognizer.record()`` asserts, so ``record()`` refuses the source
        it is handed and raises "Source must be an audio source". The
        replacement reads the stream itself instead.

        This package declares ``ovos-plugin-manager>=0.0.1a7`` with no upper
        bound, so both source classes are in range. The test is therefore
        for the property ``record()`` checks, not for a version.

        Same fix as ovos-microphone-plugin-files#26, same upstream cause.
        """
        with sr.AudioFile(file_path) as source:
            if isinstance(source, sr.AudioSource):
                return sr.Recognizer().record(source)
            return source.read()

    def recognize_wav(self, file_path, keyword_entries=None, grammar=None):
        audio = self.read_wave_file(file_path)
        return self.recognize(audio, keyword_entries, grammar)

