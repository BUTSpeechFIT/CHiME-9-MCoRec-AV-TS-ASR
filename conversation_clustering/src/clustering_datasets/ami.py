from .base import AbstractDataset, Session
from datasets import load_dataset
import random
import itertools
import numpy as np
from collections import defaultdict

class AMI(AbstractDataset):
    def __init__(self, conv_per_sess=3, max_segs_per_spk=200, num_sess=25):
        np.random.seed(12)
        random.seed(12)

        self.conv_per_sess = conv_per_sess
        self.max_segs_per_spk = max_segs_per_spk
        self.num_sess = num_sess


        # TODO: Update to use the huggingface datasets library
        # this is provisory solution
        self.ds = load_dataset(
            "parquet",
            "ihm",
            data_files=[
                "data-bin/ami/ami_dev_1.parquet",
                "data-bin/ami/ami_dev_2.parquet",
                "data-bin/ami/ami_dev_3.parquet",
                "data-bin/ami/ami_dev_4.parquet",
                "data-bin/ami/ami_train_1.parquet",
                "data-bin/ami/ami_train_2.parquet",
                "data-bin/ami/ami_train_3.parquet",
                "data-bin/ami/ami_train_4.parquet",
                "data-bin/ami/ami_train_5.parquet",
            ],
        )
        self.ds = self.ds.select_columns(["meeting_id", "text", "begin_time", "end_time", "speaker_id"])


    def _combine_meetings(self, meetings, session_id):
        """Combine multiple meetings into a single multi-conversation session."""

        session = Session(
            session_id=session_id,
            transcripts=defaultdict(list),
            timestamps=defaultdict(list),
            true_labels=dict(),
        )

        spk_id_cnter = itertools.count(start=0)
        spk_id_dict = defaultdict(lambda: next(spk_id_cnter))

        for i, meeting in enumerate(meetings):
            # add all segments grouped by speakers to the conversation
            for row in meeting:
                timestamp = (row["begin_time"], row["end_time"])
                segment = row["text"]

                # rename speaker id to avoid hinting
                original_spk_id = row["speaker_id"]
                spk_id = f"spk_{spk_id_dict[original_spk_id]}"

                session.transcripts[spk_id].append(segment)
                session.timestamps[spk_id].append(timestamp)
                session.true_labels[spk_id] = i  # ground truth cluster is the meeting index

        return session


    def _create_multiconversation_sessions(self, split):
        """Create multi-conversation sessions from the dataset split."""

        # group dataset by meetings
        meetings_iter = itertools.groupby(
            sorted(self.ds["train"], key=lambda x: x["meeting_id"]),
            key=lambda x: x["meeting_id"]
        )

        meetings = {
            meeting_id: list(rows)
            for meeting_id, rows in meetings_iter
        }

        sessions = []
        seen_sessions = set()

        for i in range(self.num_sess):
            # gather N random meetings
            while True:
                sampled_meetings = random.sample(
                    list(meetings.values()),
                    self.conv_per_sess
                )
                meeting_ids = tuple(sorted([m[0]["meeting_id"] for m in sampled_meetings]))

                if meeting_ids not in seen_sessions:
                    seen_sessions.add(meeting_ids)
                    break

            # combine meetings into a single session
            session = self._combine_meetings(sampled_meetings, session_id=f"ami_{split}_session_{i}")

            sessions.append(session)

        return sessions


    def get_dev(self) -> list[Session]:
        return self._create_multiconversation_sessions(split="dev")


    def get_train(self) -> list[Session]:
        return self._create_multiconversation_sessions(split="train")
