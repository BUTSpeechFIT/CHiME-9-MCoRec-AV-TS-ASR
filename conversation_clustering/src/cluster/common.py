from typing import List, Tuple, Dict
import numpy as np

def remove_disfluencies(transcript: List[str]) -> List[str]:
    """
    Simple function to remove disfluencies from a transcript.
    Disfluencies include filler words like "um", "uh", "like", "you know", etc.
    
    Args:
        transcript: List of words in the transcript
        
    Returns:
        Cleaned transcript with disfluencies removed
    """
    disfluencies = set(
        ['oh', 'ha', 'um', 'uh', 'ah', 'hmm', 'haahaa', 'mmm', 'ohhh', 'ohh', 'ahh', 'hahaha', 'ohhhh', 'haaa', 'hmmm', 'haa', 'ahhh', 'umm', 'haha', 'mmmm', 'ummm', 'hah', 'hhh', 'ahw', 'hm', 'haahaahaa', 'hahahaha', 'hmmmm', 'hmmmmmmmm', 'aah', 'haaaa', 'uhh', 'hahah', 'hai', 'uhhh', 'ohw', 'ahhhh', 'haahaaa', 'hahahah', 'hhhh', 'hahahahaha', 'mmmmm', 'ummmm', 'aaaa', 'ohhhhh', 'sss', 'uuu', '000', 'aaah', 'hhhhh', 'hmmmmm', 'hmmmmmmm', 'www', 'aaahh', 'haaaaaa', 'huu', 'ohhhhhhh', 'ohhhhhhhh', 'ohhhhhhhhhhhhhh', 'aaa', 'aahw', 'eee', 'hahahha', 'hh', 'hmmmmmm', 'hoo', 'ooo', 'uhhhh', 'uhhhhh', 'aaaaa', 'aahhh', 'haaaaa', 'haah', 'hahahahahahaha', 'hahhaha', 'ohhhhhh', 'rrr', 'ummmmm', 'uuuu', 'wwww', 'aahm', 'ahhhhhhhhh', 'er', 'haaaaaaa', 'haaaaaaaa', 'hahaa', 'hahaaaa', 'hahahaa', 'hahahahahaha', 'hahahahha', 'hahahuh', 'hahhah', 'hahhhh', 'hahuhu', 'hooo', 'mmmmmmm', 'oooo', 'ssssss', 'ummmmmmm', 'yah', 'yyyyyyyyyyyy', '999', 'aaaahhm', 'aaahhh', 'aaahhhmmm', 'aahh', 'aahmm', 'ahhhhh', 'ahhhhhhhhhh', 'ahhhhhhhhhhh', 'eeee', 'ffff', 'haaaaaaaaa', 'haaaaaaaaaa', 'haaaaaaaaaaaaaaaaaaa', 'haahaha', 'haahahaha', 'haahuuuuu', 'hahaaa', 'hahaaaaa', 'hahaaha', 'hahahaaah', 'hahahahaahahha', 'hahahahah', 'hahahahahah', 'hahahahahahahaha', 'hahahahahha', 'hahahahu', 'hahahahuh', 'hahahahuhu', 'hahahhaa', 'hahahoho', 'hahahu', 'hahahuha', 'hahha', 'hahhaaha', 'hahhh', 'hahu', 'hahuh', 'hahuhahuh', 'hahuhuhu', 'haisho', 'hap', 'haummm', 'hhhhhh', 'hhhhhhh', 'huhahihi', 'huhuhuha', 'huuu', 'huuuu', 'huuuuu', 'lll', 'mchhh', 'mmmmmm', 'nnn', 'nnnnn', 'nnnnnn', 'ohahahahhu', 'ohhhhhhhhh', 'ohhhhhhhhhhh', 'ohhhhhhhhhhhh', 'ohhhhhhhhhhhhhhhhh', 'ohhn', 'ohhp', 'ohooo', 'ooooo', 'oooooo', 'ooooooooo', 'oooooooooooooooooooooooooo', 'ppppppp', 'ssss', 'sssss', 'uhhhhhhh', 'uhhhhhhhhhhhh', 'ummmmmmmm', 'ummmmmmmmm', 'yyy', 'yyyyyyy', 'yay', 'hehehe', 'shhhhh', 'uhhhhmm', 'huh', 'hhahaha', 'huhuhuh', 'hmmhmm', 'huhhhhhhh', 'wow', 'huhuuu', 'yea', 'huhhhhh', 'huhh', 'huhuhu', 'whoa', 'huhhh', 'yeah', 'huhuu', 'sshhh', 'huhuuhhu', 'uhm', 'huhmmmm', 'huhhu', 'onnnnnn', 'huhummm', 'oohh', 'mmmhmmm', 'oohhoa', 'ss', 'mhmm', 'sshhhhh', 'uhmm', 'ssshh', 'mmhmm', 'huhuhh', 'oohhh', 'okay', 'okey', 'ok', 'alright', 'alrite', 'uhuh', 'uhuhh', 'yeahhh', 'yep', 'yup', 'yeah', "mm", "mhm", "mh", "mmh", "mmhm", "mmhmm", "mm-hmm", "uh-huh", "uhhuh", "uh-hm", "yeahyeah", "yeh", "ye", "ya", "yahh", "yess", "yesss", "yup yup", "right", "righto", "correct", "sure", "suree", "exactly", "indeed", "absolutely", "totally", "definitely", "erm", "erms", "ermm", "err", "errr", "erhm", "uhm", "uhmm", "uhmmm", "hmmhm", "hmm-hmm", "hmmmhm", "well", "welp", "lemme", "letme", "kinda", "sorta", "ohh", "ohhhhm", "ooh", "oooh", "ooooh", "ah", "aha", "ahha", "aha!", "ohkay", "okayy", "woww", "whoooa", "jeez", "gee",     "uh huh", "uh huhh", "mm yeah", "yeah mm", "right yeah", "okay yeah", "mhmm yeah", "mmkay", "mkay", "kay", "k"]) 

    cleaned_transcript = []
    for segment in transcript:
        words = segment.split()
        cleaned_words = [word for word in words if word.lower() not in disfluencies]
        cleaned_transcript.append(" ".join(cleaned_words))

    return cleaned_transcript


def calculate_overlap_duration(segments1: List[Tuple[float, float]], 
                             segments2: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Taken from the MCorec baseline implementation.
    Calculate total overlap and non-overlap duration between two speakers' segments.
    
    Args:
        segments1: List of (start, end) tuples for first speaker
        segments2: List of (start, end) tuples for second speaker
        
    Returns:
        Tuple of (total_overlap_duration, total_non_overlap_duration)
    """
    total_overlap = 0.0
    total_non_overlap = 0.0
    
    # Calculate total duration of each speaker's segments
    total_duration1 = sum(end - start for start, end in segments1)
    total_duration2 = sum(end - start for start, end in segments2)
    
    # Calculate overlaps
    for start1, end1 in segments1:
        for start2, end2 in segments2:
            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            if overlap_end > overlap_start:
                total_overlap += overlap_end - overlap_start
    
    # Calculate non-overlap
    total_non_overlap = total_duration1 + total_duration2 - 2 * total_overlap
    
    return total_overlap, total_non_overlap


def calculate_conversation_scores(speaker_segments: Dict[str, List[Tuple[float, float]]], likelihood=True) -> np.ndarray:
    """
    Taken from the MCorec baseline implementation.
    Calculate conversation likelihood scores between all pairs of speakers.
    Higher score means speakers are more likely to be in the same conversation.
    
    Args:
        speaker_segments: Dictionary mapping speaker IDs to their time segments
        
    Returns:
        NxN numpy array of conversation scores
    """
    n_speakers = len(speaker_segments)
    scores = np.zeros((n_speakers, n_speakers))
    speaker_ids = list(speaker_segments.keys())
    
    for i in range(n_speakers):
        for j in range(i + 1, n_speakers):
            spk1 = speaker_ids[i]
            spk2 = speaker_ids[j]
            
            overlap, non_overlap = calculate_overlap_duration(
                speaker_segments[spk1],
                speaker_segments[spk2]
            )
            
            # Calculate conversation likelihood score
            # Higher score when there's less overlap (more likely to be in same conversation)
            if overlap + non_overlap > 0:
                # Normalize overlap by total duration to get overlap ratio
                total_duration = overlap + non_overlap
                overlap_ratio = overlap / total_duration
                # Convert to conversation likelihood (1 - overlap_ratio)
                if likelihood:
                    score = 1 - overlap_ratio
                else:
                    score = overlap_ratio
            else:
                score = 0
                
            scores[i, j] = score
            scores[j, i] = score  # Symmetric
    
    return scores

