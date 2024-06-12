import json
from vllm.sequence import Sequence, SequenceGroup, SequenceStage


def dump_sequnce(seq: Sequence) -> dict:
    assert seq.lora_request is None, "Not supported yet"

    data = {
        "seq_id": seq.seq_id,
        "inputs": {
            "prompt_token_ids": seq.prompt_token_ids,
            "prompt": seq.prompt,
        },
        "block_size": seq.block_size,
        "eos_token_id": seq.eos_token_id,
        "data": {
            "prompt_token_ids": seq.data.prompt_token_ids,
            "output_token_ids": seq.data.output_token_ids,
            "cumulative_logprob": seq.data.cumulative_logprob,
            "_num_computed_tokens": seq.data._num_computed_tokens,
            "_stage": seq.data._stage.name,
            "_prompt_token_ids_tuple": seq.data._prompt_token_ids_tuple,
        },
        "output_logprobs": [
            {
                k: {
                    "logprob": v.logprob,
                    "rank": v.rank,
                    "decoded_token": v.decoded_token,
                }
                for k, v in logprob.items()
            }
            for logprob in seq.output_logprobs
        ],
        "output_text": seq.output_text,
        "status": seq.status.name,
        "stop_reason": seq.stop_reason,
        "tokens": seq.tokens,
        "prefix_offset": seq.prefix_offset,
        "read_offset": seq.read_offset,
        "logical_token_blocks": [
            {"block_number": block.block_number, "block_size": block.block_size}
            for block in seq.logical_token_blocks
        ],
    }

    return data


def restore_sequnce(seq_data: Dict) -> Sequence:
        seq = Sequence(
            seq_id=seq_data['seq_id'],
            inputs={'prompt_token_ids': seq_data['inputs']['prompt_token_ids'], 'prompt': seq_data['inputs']['prompt']},
            block_size=seq_data['block_size'],
            eos_token_id=seq_data.get('eos_token_id')
        )
        seq.data = SequenceData(
            prompt_token_ids=seq_data['data']['prompt_token_ids'],
            output_token_ids=seq_data['data']['output_token_ids']
        )
        seq.data.cumulative_logprob = seq_data['data']['cumulative_logprob']
        seq.data._num_computed_tokens = seq_data['data']['_num_computed_tokens']
        seq.data._stage = SequenceStage[seq_data['data']['_stage']]
        seq.data._prompt_token_ids_tuple = tuple(seq_data['data']['_prompt_token_ids_tuple'])
        seq.output_logprobs = [{k: Logprob(logprob=v['logprob'], rank=v.get('rank'), decoded_token=v.get('decoded_token')) for k, v in logprob.items()} for logprob in seq_data['output_logprobs']]
        seq.output_text = seq_data['output_text']
        seq.status = SequenceStatus[seq_data['status']]
        seq.stop_reason = seq_data['stop_reason']
        seq.tokens = seq_data.get('tokens')
        seq.prefix_offset = seq_data['prefix_offset']
        seq.read_offset = seq_data['read_offset']
        seq.logical_token_blocks = [LogicalTokenBlock(block_number=block['block_number'], block_size=block['block_size']) for block in seq_data['logical_token_blocks']]
        return seq



def dump_seq_group(seq_group: SequenceGroup, sampling_params: dict) -> str:
    data = {
        "request_id": seq_group.request_id,
        "seqs": "TODO",
        "arrival_time": seq_group.metrics.arrival_time,
        "sampling_params": sampling_params,
    }

    assert seq_group.lora_request is None, "Not supported yet"
    assert seq_group.embeddings is None, "Not supported yet"
    assert seq_group.pooling_params is None, "Not supported yet"
    assert seq_group.encoder_seq is None, "Not supported yet"
    assert seq_group.state.generator is None, "Not supported yet"
    return json.dumps(data)

def restore_seq_group(data: str) -> SequenceGroup:
    data = json.loads(data)
    seq_group = SequenceGroup(
        request_id=data["request_id"],
        seqs="TODO",
        arrival_time=data["arrival_time"],
        sampling_params=data["sampling_params"],
    )
    return seq_group
