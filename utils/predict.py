import torch 

def predict(input_ids, attention_masks, model):
    chunk_outputs = []

    for i in range(input_ids.size(1)):
        chunk_input_ids = input_ids[:, i, :]
        chunk_attention_mask = attention_masks[:, i, :]

        chunk_output = model(
            input_ids=chunk_input_ids,
            attention_mask=chunk_attention_mask
        )

        chunk_outputs.append(chunk_output.logits.unsqueeze(1))

    return chunk_outputs