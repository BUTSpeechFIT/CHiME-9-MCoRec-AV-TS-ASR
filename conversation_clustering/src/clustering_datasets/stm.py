from collections import defaultdict
import json
from itertools import groupby
from .base import AbstractDataset, Session
import os
import glob
from dataclasses import dataclass

@dataclass
class StmEntry:
    def __init__(self, session_id: str, speaker_id: str, start_time: float, end_time: float, word: str):
        self.session_id = session_id
        self.speaker_id = speaker_id
        self.start_time = start_time
        self.end_time = end_time
        self.word = word


class StmDataset(AbstractDataset):
    """
    Dataset class for loading data in stm format along with true speaker cluster labels from chime9 dataset.
    Limitation: currently automatically loads labels from chime9 dataset, expecting correct folder structure.
    """
    def __init__(
            self, 
            dev_path,
            train_path,
            correct_labels_path,
            concat_eps=1.5,
    ):
        """
        Initialize the StmDataset.

        Args:
            dev_path (str): Path to the development set stm file.
            train_path (str): Path to the training set stm file.
            correct_labels_path (str): Path to the directory containing correct speaker cluster labels in chime9 format.
            concat_eps (float): Maximum time gap (in seconds) between words to be concatenated into a single segment.
        """

        assert(dev_path.endswith(".stm") and train_path.endswith(".stm")), "Input files must be in stm format."

        with open(dev_path, "r", encoding="utf-8") as f:
            dev_words = f.readlines()

        with open(train_path, "r", encoding="utf-8") as f:
            train_words = f.readlines()

        self.concat_eps = concat_eps
        self.dev_stm_entries = self._parse_stm_lines(dev_words)
        self.train_stm_entries = self._parse_stm_lines(train_words)

        self.chime_session_paths = {
            "dev": glob.glob(os.path.join(correct_labels_path, "dev", "*")),
            "train": glob.glob(os.path.join(correct_labels_path, "train", "*")),
        }


    def _concat_into_segments(self, session_entries):
        segments = []
        timestamps = []
        curr_segment_words = []
        curr_start_time = session_entries[0].start_time
        curr_end_time = session_entries[0].end_time
        curr_speaker = session_entries[0].speaker_id

        for entry in session_entries:
            if entry.speaker_id == curr_speaker and entry.start_time - curr_end_time <= self.concat_eps:
                curr_segment_words.append(entry.word)
                curr_end_time = entry.end_time
            else:
                segments.append(" ".join(curr_segment_words))
                timestamps.append((curr_start_time, curr_end_time))

                curr_segment_words = [entry.word]
                curr_start_time = entry.start_time
                curr_end_time = entry.end_time
                curr_speaker = entry.speaker_id

        # append last segment
        segments.append( " ".join(curr_segment_words))
        timestamps.append((curr_start_time, curr_end_time))

        return segments, timestamps


    def _parse_stm_lines(self, stm_lines) -> list[StmEntry]:
        """Assumes stm lines are sorted by session, speaker and time."""

        def parse(line):
            parts = line.split()

            session_id = "_".join(parts[0].split("_")[:2])
            spk_id = parts[2]
            start_time = float(parts[3])
            end_time = float(parts[4])
            word = parts[5] if len(parts) > 5 else ""

            return StmEntry(session_id, spk_id, start_time, end_time, word)

        return [parse(line) for line in stm_lines]


    def _stm_to_sessions(self, stm_entries: list[StmEntry], true_labels: dict) -> list[Session]:
        sessions_dict = defaultdict(lambda: Session(
            session_id="",
            transcripts=defaultdict(list),
            timestamps=defaultdict(list),
            true_labels=dict(),
        ))


        for (session_id, speaker_id), session_lines in groupby(stm_entries, lambda x: (x.session_id, x.speaker_id)):
            session_entries = list(session_lines)
            segments, timestamps = self._concat_into_segments(session_entries)

            sessions_dict[session_id].transcripts[speaker_id] = segments
            sessions_dict[session_id].timestamps[speaker_id] = timestamps
            sessions_dict[session_id].session_id = session_id

            try:
                sessions_dict[session_id].true_labels = true_labels[session_id]
            except KeyError:
                print(f"Warning: true labels for session {session_id} not found.")
                sessions_dict[session_id].true_labels = dict()


        return list(sessions_dict.values())


    def _get_true_labels(self, split:str) -> dict[str, dict[str, int]]:
        """Get true speaker cluster labels for all sessions in the given split from chime9 dataset."""
        true_labels = dict()

        for session_dir in self.chime_session_paths[split]:
            session_id = os.path.basename(session_dir)
            labels_path = os.path.join(session_dir, "labels", "speaker_to_cluster.json")

            with open(labels_path, "r", encoding="utf-8") as f:
                session_true_labels = json.load(f)

            true_labels[session_id] = session_true_labels

        return true_labels

    def get_dev(self) -> list[Session]:
        """Create and return a list of DSPY examples using the dataset devset split."""

        true_labels = self._get_true_labels(split="dev")
        return self._stm_to_sessions(self.dev_stm_entries, true_labels)

    def get_train(self) -> list[Session]:
        """Create and return a list of DSPY examples using the dataset trainset split."""
        true_labels = self._get_true_labels(split="train")
        return self._stm_to_sessions(self.train_stm_entries, true_labels)
