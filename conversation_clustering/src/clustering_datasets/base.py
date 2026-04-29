import abc
from itertools import groupby
import random
import re
from dataclasses import dataclass


random.seed(324)

@dataclass
class Session():
    def __init__(self, session_id: str, transcripts: dict[str, list[str]],
                 timestamps: dict[str, list[tuple[float, float]]],
                 true_labels: dict[str, int]):
        self.session_id = session_id
        self.transcripts = transcripts
        self.timestamps = timestamps
        self.true_labels = true_labels


    session_id: str
    transcripts: dict[str, list[str]]
    timestamps: dict[str, list[tuple[float, float]]]
    true_labels: dict[str, int]


class AbstractDataset(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_dev(self) -> list[Session]:
        pass

    @abc.abstractmethod
    def get_train(self) -> list[Session]:
        pass


    nodding_words = set(
        ['oh', 'ha', 'um', 'uh', 'ah', 'hmm', 'haahaa', 'mmm', 'ohhh', 'ohh', 'ahh', 'hahaha', 'ohhhh', 'haaa', 'hmmm', 'haa', 'ahhh', 'umm', 'haha', 'mmmm', 'ummm', 'hah', 'hhh', 'ahw', 'hm', 'haahaahaa', 'hahahaha', 'hmmmm', 'hmmmmmmmm', 'aah', 'haaaa', 'uhh', 'hahah', 'hai', 'uhhh', 'ohw', 'ahhhh', 'haahaaa', 'hahahah', 'hhhh', 'hahahahaha', 'mmmmm', 'ummmm', 'aaaa', 'ohhhhh', 'sss', 'uuu', '000', 'aaah', 'hhhhh', 'hmmmmm', 'hmmmmmmm', 'www', 'aaahh', 'haaaaaa', 'huu', 'ohhhhhhh', 'ohhhhhhhh', 'ohhhhhhhhhhhhhh', 'aaa', 'aahw', 'eee', 'hahahha', 'hh', 'hmmmmmm', 'hoo', 'ooo', 'uhhhh', 'uhhhhh', 'aaaaa', 'aahhh', 'haaaaa', 'haah', 'hahahahahahaha', 'hahhaha', 'ohhhhhh', 'rrr', 'ummmmm', 'uuuu', 'wwww', 'aahm', 'ahhhhhhhhh', 'er', 'haaaaaaa', 'haaaaaaaa', 'hahaa', 'hahaaaa', 'hahahaa', 'hahahahahaha', 'hahahahha', 'hahahuh', 'hahhah', 'hahhhh', 'hahuhu', 'hooo', 'mmmmmmm', 'oooo', 'ssssss', 'ummmmmmm', 'yah', 'yyyyyyyyyyyy', '999', 'aaaahhm', 'aaahhh', 'aaahhhmmm', 'aahh', 'aahmm', 'ahhhhh', 'ahhhhhhhhhh', 'ahhhhhhhhhhh', 'eeee', 'ffff', 'haaaaaaaaa', 'haaaaaaaaaa', 'haaaaaaaaaaaaaaaaaaa', 'haahaha', 'haahahaha', 'haahuuuuu', 'hahaaa', 'hahaaaaa', 'hahaaha', 'hahahaaah', 'hahahahaahahha', 'hahahahah', 'hahahahahah', 'hahahahahahahaha', 'hahahahahha', 'hahahahu', 'hahahahuh', 'hahahahuhu', 'hahahhaa', 'hahahoho', 'hahahu', 'hahahuha', 'hahha', 'hahhaaha', 'hahhh', 'hahu', 'hahuh', 'hahuhahuh', 'hahuhuhu', 'haisho', 'hap', 'haummm', 'hhhhhh', 'hhhhhhh', 'huhahihi', 'huhuhuha', 'huuu', 'huuuu', 'huuuuu', 'lll', 'mchhh', 'mmmmmm', 'nnn', 'nnnnn', 'nnnnnn', 'ohahahahhu', 'ohhhhhhhhh', 'ohhhhhhhhhhh', 'ohhhhhhhhhhhh', 'ohhhhhhhhhhhhhhhhh', 'ohhn', 'ohhp', 'ohooo', 'ooooo', 'oooooo', 'ooooooooo', 'oooooooooooooooooooooooooo', 'ppppppp', 'ssss', 'sssss', 'uhhhhhhh', 'uhhhhhhhhhhhh', 'ummmmmmmm', 'ummmmmmmmm', 'yyy', 'yyyyyyy', 'yay', 'hehehe', 'shhhhh', 'uhhhhmm', 'huh', 'hhahaha', 'huhuhuh', 'hmmhmm', 'huhhhhhhh', 'wow', 'huhuuu', 'yea', 'huhhhhh', 'huhh', 'huhuhu', 'whoa', 'huhhh', 'yeah', 'huhuu', 'sshhh', 'huhuuhhu', 'uhm', 'huhmmmm', 'huhhu', 'onnnnnn', 'huhummm', 'oohh', 'mmmhmmm', 'oohhoa', 'ss', 'mhmm', 'sshhhhh', 'uhmm', 'ssshh', 'mmhmm', 'huhuhh', 'oohhh', 'okay', 'okey', 'ok', 'alright', 'alrite', 'uhuh', 'uhuhh', 'yeahhh', 'yep', 'yup', 'yeah', "mm", "mhm", "mh", "mmh", "mmhm", "mmhmm", "mm-hmm", "uh-huh", "uhhuh", "uh-hm", "yeahyeah", "yeh", "ye", "ya", "yahh", "yess", "yesss", "yup yup", "right", "righto", "correct", "sure", "suree", "exactly", "indeed", "absolutely", "totally", "definitely", "erm", "erms", "ermm", "err", "errr", "erhm", "uhm", "uhmm", "uhmmm", "hmmhm", "hmm-hmm", "hmmmhm", "well", "welp", "lemme", "letme", "kinda", "sorta", "ohh", "ohhhhm", "ooh", "oooh", "ooooh", "ah", "aha", "ahha", "aha!", "ohkay", "okayy", "woww", "whoooa", "jeez", "gee",     "uh huh", "uh huhh", "mm yeah", "yeah mm", "right yeah", "okay yeah", "mhmm yeah", "mmkay", "mkay", "kay", "k"]) 

    def _filter_content(self, segments: list[str]):
        filtered_segments = []
        for segment in segments:
            words = re.findall(r'\b\w+\b', segment.lower())
            if any(word in self.nodding_words for word in words):
                filtered_segments.append(' '.join([word for word in words if word in self.nodding_words]))

        return filtered_segments

    def filter_nodding(self, transcript: dict[str, list[str]]) -> dict[str, list[str]]:
        for spk, segments in transcript.items():
            filtered_segments = []
            for segment in segments:
                words = re.findall(r'\b\w+\b', segment.lower())
                if not all(word in self.nodding_words for word in words):
                    filtered_segments.append(segment)

            transcript[spk] = filtered_segments

        return transcript

    def get_dev_with_passive(self, ratio_per_session: float = 0) -> list[Session]:
        dev_sessions = self.get_dev()

        for session in dev_sessions:
            clusters = groupby(session.true_labels.items(), key=lambda x: x[1])

            # sample portion of speakers from each cluster
            passive_speakers = []
            for _, spk_label_pairs in clusters:
                spk_list = [spk for spk, _ in spk_label_pairs]
                n_passive = round(len(spk_list) * ratio_per_session)
                if n_passive > len(spk_list):
                    n_passive = len(spk_list)

                passive_speakers.extend(random.sample(spk_list, k=n_passive))

            for spk in passive_speakers:
                try:
                    session.transcripts[spk] = self._filter_content(session.transcripts[spk])
                except KeyError:
                    print(f"Speaker {spk} not found in transcripts of session {session.session_id}")
                    continue

        return dev_sessions


    def get_train_with_passive(self, ratio_per_session: float = 0) -> list[Session]:
        train_sessions = self.get_train()

        for session in train_sessions:
            clusters = groupby(session.true_labels.items(), key=lambda x: x[1])

            # sample portion of speakers from each cluster
            passive_speakers = []
            for _, spk_label_pairs in clusters:
                spk_list = [spk for spk, _ in spk_label_pairs]
                n_passive = round(len(spk_list) * ratio_per_session)

                passive_speakers.extend(random.sample(spk_list, k=n_passive))

            for spk in passive_speakers:
                try:
                    session.transcripts[spk] = self._filter_content(session.transcripts[spk])
                except KeyError:
                    print(f"Speaker {spk} not found in transcripts of session {session.session_id}")
                    continue

        return train_sessions


    def get_filtered_dev(self) -> list[Session]:
        dev_sessions = self.get_dev()
        for session in dev_sessions:
            session.transcripts = self.filter_nodding(session.transcripts)
        return dev_sessions

    def get_filtered_train(self) -> list[Session]:
        train_sessions = self.get_train()
        for session in train_sessions:
            session.transcripts = self.filter_nodding(session.transcripts)
        return train_sessions
