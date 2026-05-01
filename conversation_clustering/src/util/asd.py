"""
Utility functions for processing Active Speaker Detection (ASD) data.
Taken from the MCorec baseline implementation.
"""

import json
from typing import List, Tuple
import math


CENTRAL_ASD_CHUNKING_PARAMETERS = {
    "onset": 1.0,        # start threshold
    "offset": 0.8,       # end threshold
    "min_duration_on": 1.0,  # drop
    "min_duration_off": 0.5, # fill
    "max_chunk_size": 10,
    "min_chunk_size": 1
}

EGO_ASD_CHUNKING_PARAMETERS = {
    "onset": 2.4,        # start threshold
    "offset": 1.6,       # end threshold
    "min_duration_on": 1.0,  # drop
    "min_duration_off": 0.5, # fill
    "max_chunk_size": 10,
    "min_chunk_size": 1
}

def segment_by_asd(asd, parameters={}):
    onset_threshold = parameters.get("onset", CENTRAL_ASD_CHUNKING_PARAMETERS["onset"])
    offset_threshold = parameters.get("offset", CENTRAL_ASD_CHUNKING_PARAMETERS["offset"])
    
    # Convert frame numbers to integers and sort them
    frames = sorted([int(f) for f in asd.keys()])
    if not frames:
        return []
        
    # Find the minimum frame number to normalize frame indices
    min_frame = min(frames)
    
    # Convert duration parameters from seconds to frames (assuming 25 fps)
    min_duration_on_frames = int(parameters.get("min_duration_on",  CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_on"]) * 25)
    min_duration_off_frames = int(parameters.get("min_duration_off", CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_on"]) * 25)
    max_chunk_frames = int(parameters.get("max_chunk_size", CENTRAL_ASD_CHUNKING_PARAMETERS["max_chunk_size"]) * 25)
    min_chunk_frames = int(parameters.get("min_chunk_size", CENTRAL_ASD_CHUNKING_PARAMETERS["min_chunk_size"]) * 25)
    
    # First pass: Find speech regions using hysteresis thresholding
    speech_regions = []
    current_region = None
    is_active = False
    
    for frame in frames:
        score = asd.get(str(frame), -1)
        normalized_frame = frame - min_frame
        
        if not is_active:
            # Currently inactive, check for onset
            if score > onset_threshold:
                is_active = True
                current_region = [normalized_frame]
        else:
            # Currently active, check for offset
            if score < offset_threshold:
                is_active = False
                if current_region is not None:
                    speech_regions.append(current_region)
                    current_region = None
            else:
                current_region.append(normalized_frame)
    
    # Handle case where speech continues until the end
    if current_region is not None:
        speech_regions.append(current_region)
    
    # Second pass: Merge regions separated by short non-speech gaps
    merged_regions = []
    if speech_regions:
        current_region = speech_regions[0]
        
        for next_region in speech_regions[1:]:
            gap = next_region[0] - current_region[-1] - 1
            if gap <= min_duration_off_frames:
                # Merge regions
                current_region.extend(next_region)
            else:
                merged_regions.append(current_region)
                current_region = next_region
        merged_regions.append(current_region)
    
    # Third pass: Remove short speech regions and split long ones
    final_segments = []
    for region in merged_regions:
        region_length = len(region)
        
        # Skip regions shorter than minimum duration
        if region_length < min_duration_on_frames:
            continue
            
        # Split long regions
        if region_length > max_chunk_frames:
            num_chunks = math.ceil(region_length / max_chunk_frames)
            chunk_size = math.ceil(region_length / num_chunks)
            
            for i in range(0, region_length, chunk_size):
                sub_segment = region[i:i + chunk_size]
                if len(sub_segment) >= min_chunk_frames:
                    final_segments.append(sub_segment)
        else:
            final_segments.append(region)
    
    # Convert frame indices back to original frame indices
    final_segments = [
        [frame + min_frame for frame in segment]
        for segment in final_segments
    ]
    
    return final_segments

def get_speaker_activity_segments(pycrop_asd_path: List[str], uem_start: int, uem_end: int) -> List[Tuple[float, float]]:
    pycrop_asd_path = sorted(pycrop_asd_path)
    all_frames = dict({})
    for asd_path in pycrop_asd_path:
        with open(asd_path, "r") as f:
            asd_data = json.load(f)
            all_frames.update(asd_data)
    sorted_frames = sorted(all_frames.items(), key=lambda x: int(x[0]))
    
    # activity_segments = []
    # current_segment = []
    # for frame_id, score in sorted_frames:
    #     if score > 2.5:
    #         current_segment.append(frame_id)
    #     else:
    #         if current_segment:
    #             activity_segments.append(current_segment)
    #             current_segment = []
    # if current_segment:
    #     activity_segments.append(current_segment)
    
    activity_segments = segment_by_asd(all_frames)

    activity_segments = [
        (int(segment[0])/25, int(segment[-1])/25)
        for segment in activity_segments
    ]
    
    aligned_activity_segments = []
    for segment in activity_segments:
        if segment[1] < uem_start:
            continue
        if segment[0] > uem_end:
            break
        aligned_activity_segments.append([
            # max(segment[0] - uem_start, 0),
            # min(segment[1] - uem_start, uem_end - uem_start)
            segment[0] - uem_start,
            segment[1] - uem_start
        ])
    
    return aligned_activity_segments
