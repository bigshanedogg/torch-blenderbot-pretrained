def split_segment_by_speaker_ids(utterances, speaker_ids):
    sequence_utterances = []
    sequence_turn_ids = []
    segment_utterances = []
    segment_turn_ids = []
    first_speaker_id = speaker_ids[0]
    replied = False
    for idx, (speaker_id, utterance) in enumerate(zip(speaker_ids, utterances)):
        if replied and speaker_id == first_speaker_id:
            sequence_utterances.append(segment_utterances)
            sequence_turn_ids.append(segment_turn_ids)
            segment_utterances = []
            segment_turn_ids = []
            replied = False
        segment_utterances.append(utterance)
        segment_turn_ids.append(speaker_id)
        if speaker_id != first_speaker_id: replied = True
    sequence_utterances.append(segment_utterances)
    sequence_turn_ids.append(segment_turn_ids)
    return sequence_utterances, sequence_turn_ids

def flatten_sequence(sequence_utterances, sequence_turn_ids):
    utterances = []
    speaker_ids = []
    for segment_utterances, segment_turn_ids in zip(sequence_utterances, sequence_turn_ids):
        for utterance, speaker_id in zip(segment_utterances, segment_turn_ids):
            utterances.append(utterance)
            speaker_ids.append(speaker_id)
    return utterances, speaker_ids

def convert_turn_ids(sequence_turn_ids, model_id=0, user_id=1):
    '''
    Regarding the last speaker of segments as conversational_model,
    convert model's speaker_id to given_model_id, and others to given_user_id.
    '''
    cur_model_id = sequence_turn_ids[0][-1]
    output = []
    for _segment_turn_ids in sequence_turn_ids:
        segment_turn_ids = []
        for _turn_id in _segment_turn_ids:
            if _turn_id == cur_model_id: segment_turn_ids.append(model_id)
            else: segment_turn_ids.append(user_id)
        output.append(segment_turn_ids)
    return output

def split_context_candidate(sequence, speaker_ids):
    candidate_speaker_id = speaker_ids[-1]
    candidate_begin_idx = -1
    for i in range(len(speaker_ids) - 1, 0, -1):
        if speaker_ids[i] != candidate_speaker_id: break
        candidate_begin_idx = i
    context_sequence = sequence[:candidate_begin_idx]
    candidate_sequence = sequence[candidate_begin_idx:]
    context_speaker_ids = speaker_ids[:candidate_begin_idx]
    candidate_speaker_ids = speaker_ids[candidate_begin_idx:]

    context = (context_sequence, context_speaker_ids)
    candidate = (candidate_sequence, candidate_speaker_ids)
    return context, candidate