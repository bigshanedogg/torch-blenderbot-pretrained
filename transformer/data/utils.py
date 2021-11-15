import numpy as np
from transformer.utils.common import get_device_index

def get_iter_range(worker_id, num_workers, device, nprocs, data_size, verbose=False):
    device_index = get_device_index(device=device, default=0)
    per_proc = int(np.ceil(data_size/int(nprocs)))
    proc_start = device_index * per_proc
    proc_end = min((device_index + 1) * per_proc, data_size)
    per_worker = int(per_proc/num_workers) # int((proc_end - proc_start)/num_workers)
    iter_start = proc_start + worker_id * per_worker
    iter_end = min(proc_start + (worker_id + 1) * per_worker, proc_end)

    if verbose:
        print("##### nrpocs:{}, device_index:{}, num_workers:{}, per_proc:{}, worker_id:{}, iter_start:{}, iter_end:{}".format(nprocs, device_index, num_workers, per_proc, worker_id, iter_start, iter_end))
    # per_worker = int(np.ceil(data_size / float(nprocs * num_workers)))
    # local_worker_id = (device_index * nprocs) + worker_id
    # iter_start = local_worker_id * per_worker
    # iter_end = min(iter_start + per_worker, data_size)
    return iter_start, iter_end

def merge_utterances(utterances, speaker_ids):
    new_utterances = []
    new_speaker_ids = []

    utterance = ""
    prev_speaker_id = -100
    for _idx, (_speaker_id, _utterance) in enumerate(zip(speaker_ids, utterances)):
        if _speaker_id != prev_speaker_id:
            if _idx > 0:
                new_utterances.append(utterance)
                new_speaker_ids.append(prev_speaker_id)
            utterance = _utterance
            prev_speaker_id = _speaker_id
        else:
            utterance = utterance + " " + _utterance

        if _idx >= len(utterances) - 1:
            new_utterances.append(utterance)
            new_speaker_ids.append(prev_speaker_id)
    return new_utterances, new_speaker_ids

def simplify_speaker_ids(speaker_ids, user_id=1, model_id=0):
    cur_model_id = speaker_ids[-1]
    new_speaker_ids = []
    for speaker_id in speaker_ids:
        if speaker_id != cur_model_id:
            new_speaker_ids.append(user_id)
        else:
            new_speaker_ids.append(model_id)
    return new_speaker_ids

def random_sampling_gen(low, high, size, replace=False):
    array = np.arange(start=low, stop=high)
    while True:
        yield np.random.choice(array, size, replace=replace)