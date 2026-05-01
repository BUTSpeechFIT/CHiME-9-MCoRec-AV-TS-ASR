import os
from typing import Dict, Tuple
import webvtt


def load_vtt_segments(labels_dir: str, metadata: Dict|None) -> Tuple[Dict[str, list], Dict[str,list]]:
    """
    Load segments from VTT files in the specified directory.

    Args:
        labels_dir (str): Directory containing VTT files for each speaker.
        speaker_md (Dict): Metadata dictionary for speakers.
    Returns:
        Tuple[Dict[str, list], Dict[str,list]]: A tuple containing:
            - A dictionary mapping speaker IDs to their list of segments (text).
            - A dictionary mapping speaker IDs to their list of timestamps (start, end).
    """

    speaker_segments = {}
    speaker_timestamps = {}
    for filename in os.listdir(labels_dir):
        if filename.endswith(".vtt"):
            speaker_id = os.path.splitext(filename)[0]
            if metadata is None:
                uem_start = 0.0
                uem_end = float('inf')
            else:
                uem_start = metadata[speaker_id]["central"]["uem"]["start"]
                uem_end = metadata[speaker_id]["central"]["uem"]["end"]

            segments = []
            timestamps = []
            for caption in webvtt.read(os.path.join(labels_dir, filename)):
                start_time = caption.start_in_seconds
                end_time = caption.end_in_seconds

                segment = caption.text.strip()

                # Discard segments outside UEM
                if end_time < uem_start:
                    continue
                if start_time > uem_end:
                    break

                # Align segment times to UEM
                aligned_start = start_time - uem_start
                aligned_end = end_time - uem_start

                segments.append(segment)
                timestamps.append((aligned_start, aligned_end))

            speaker_segments[speaker_id] = segments
            speaker_timestamps[speaker_id] = timestamps

    return speaker_segments, speaker_timestamps
