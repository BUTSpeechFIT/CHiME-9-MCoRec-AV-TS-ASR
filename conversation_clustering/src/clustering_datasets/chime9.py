from collections import defaultdict
from .base import AbstractDataset, Session
import os
import json
import glob
from util import load_vtt_segments

class Chime9(AbstractDataset):
    def __init__(self, 
                 dev_path = None,
                 train_path = None,
                 labels_dir="labels",
                 concat_eps=0,
                 **kwargs):
        assert dev_path or train_path, "At least one of dev_path or train_path must be provided."

        self.dev_sessions = glob.glob(os.path.join(dev_path, "*")) if dev_path else []
        self.train_sessions = glob.glob(os.path.join(train_path, "*")) if train_path else []

        self.labels_dir = labels_dir
        self.concat_segments_eps = concat_eps


    def concat_segments(self, segments, timestamps, eps=1.5):
        """Concatenate segments that are within eps seconds of each other."""
        new_segments = defaultdict(list)
        new_timestamps = defaultdict(list)

        spk_ids = segments.keys()
        for spk_id in spk_ids:
            spk_segments = segments[spk_id]
            spk_timestamps = timestamps[spk_id]

            if not spk_segments:
                continue

            curr_segment = spk_segments[0]
            curr_start, curr_end = spk_timestamps[0]

            for i in range(1, len(spk_segments)):
                next_segment = spk_segments[i]
                next_start, next_end = spk_timestamps[i]

                if next_start - curr_end <= eps:
                    # concatenate
                    curr_segment += " " + next_segment
                    curr_end = next_end
                else:
                    # save current segment
                    new_segments[spk_id].append(curr_segment)
                    new_timestamps[spk_id].append((curr_start, curr_end))

                    # start new segment
                    curr_segment = next_segment
                    curr_start, curr_end = next_start, next_end

            # save last segment
            new_segments[spk_id].append(curr_segment)
            new_timestamps[spk_id].append((curr_start, curr_end))

        return new_segments, new_timestamps


    def _load_sessions(self, session_dirs) -> list[Session]:
        sessions = []
        for session_dir in session_dirs:
            labels_path = os.path.join(session_dir, self.labels_dir)
            metadata_path = os.path.join(session_dir, "metadata.json")

            # load speakers metadata (if exists)
            if not os.path.exists(metadata_path):
                speaker_md = None
            else:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    speaker_md = json.load(f)

            # load true labels
            with open(os.path.join(labels_path, "speaker_to_cluster.json")) as f:
                true_labels = json.load(f)

            # load speaker segments
            segments, timestamps = load_vtt_segments(labels_path, speaker_md)

            if self.concat_segments_eps > 0:
                segments, timestamps = self.concat_segments(
                    segments, timestamps, eps=self.concat_segments_eps
                )

            session = Session(
                session_id=os.path.basename(session_dir),
                transcripts=segments,
                timestamps=timestamps,
                true_labels=true_labels,
            )
            sessions.append(session)

        return sessions


    def get_dev(self) -> list[Session]:
        return self._load_sessions(self.dev_sessions)


    def get_train(self) -> list[Session]:
        return self._load_sessions(self.train_sessions)
