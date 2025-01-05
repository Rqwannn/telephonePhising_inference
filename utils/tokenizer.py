import os
from transformers import AutoTokenizer, BertForSequenceClassification

def load_model_from_huggingface():
    repo_name = os.getenv("REPO_ID")   
    tokenizer = AutoTokenizer.from_pretrained(f"{repo_name}/tokenizer")    
    model = BertForSequenceClassification.from_pretrained(f"{repo_name}/models")
    
    return tokenizer, model

def split_text_into_chunks_with_overlap(text, tokenizer, max_length=512, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    step_size = max_length - 2 - overlap  

    chunks = [tokens[i:i + step_size] for i in range(0, len(tokens), step_size)]
    return chunks

def tokenize_with_special_tokens_and_overlap(text, tokenizer, max_length=512, overlap=50):
    chunks = split_text_into_chunks_with_overlap(text, tokenizer, max_length=max_length, overlap=overlap)
    input_ids = []
    attention_masks = []

    for idx, chunk in enumerate(chunks):
        if idx == 0:
            chunk_ids = (
                tokenizer.encode("[CHUNK_START]", add_special_tokens=False) +
                chunk +
                tokenizer.encode("[CHUNK_END]", add_special_tokens=False)
            )
        else:
            chunk_ids = (
                tokenizer.encode("[CHUNK_CONTINUATION]", add_special_tokens=False) +
                chunk +
                tokenizer.encode("[CHUNK_END]", add_special_tokens=False)
            )

        attention_mask = [1] * len(chunk_ids)

        padding_length = max_length - len(chunk_ids)
        chunk_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length

        input_ids.append(chunk_ids)
        attention_masks.append(attention_mask)

    return input_ids, attention_masks

def process_dataset_with_overlap(data, max_length=512, overlap=50):
    return tokenize_with_special_tokens_and_overlap(data, max_length=max_length, overlap=overlap) 
