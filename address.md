# ADDRESS API SPEC

## CREATE ADDRESS

ENDPOINT: POST [Inference](http://127.0.0.1:5000/stream)

REQUEST HEADER:

- CONTENT-TYPE: multipart/form-data

REQUEST BODY:

```json
{
  "audio": "file",
  "danoised": "int",
  "text_previous": "str", // only in realtime analysis
}
```

RESPONSE BODY: (SUCCESS)

```json
{
    "message": "Prediction successful",
    "confidence_scores": [0.90, 0.59, 0.20, 0.72, 0.33],
    "predicted_labels": ["1_p_p_o", "2_p_b_k_k", "3_p_i_i", "4_p_j_b_j", "5_p_h", "6_n_p"],
    "transcription": "Ini Transcription",
    "text_previous": "text_previous", // only in realtime analysis
    "processing_time": "in seconds format"
}
```

RESPONSE BODY: (FAILED)

```json
{
  "message": "Error during prediction"
}
```

## GET ADDRESS

ENDPOINT: GET [Save Model](http://127.0.0.1:5000/save_model)

RESPONSE BODY: (SUCCESS)

```json
{
  "message": "Model Berhasil Di simpan"
}
```

RESPONSE BODY: (FAILED)

```json
{
  "errors": "Error there was a problem with the API"
}
```