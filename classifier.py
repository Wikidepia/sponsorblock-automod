import bisect

import numpy as np
import onnxruntime as ort
import requests
from tokenizers import BertWordPieceTokenizer

import utils.transcript as transcript_api

bert_tokenizer = BertWordPieceTokenizer.from_file("model/bert_vocab.txt")
bert_tokenizer.enable_truncation(max_length=512)
bert_tokenizer.add_special_tokens(["[CLS]", "[SEP]"])
bert_session = ort.InferenceSession("model/bert_model.onnx")


def req_api(uuid):
    r = requests.get(f"https://sponsor.ajay.app/api/segmentInfo?UUID={uuid}")
    r_json = r.json()[0]
    video_id = r_json["videoID"]
    start_time = r_json["startTime"]
    end_time = r_json["endTime"]
    category = r_json["category"]
    return video_id, start_time, end_time, category


def get_transcript(video_id, start_time, end_time):
    player_info = transcript_api.parse_player_info(video_id)
    video_captions = transcript_api.parse_caption(player_info)
    if video_captions == []:
        # Use silero if no captions
        transcript = transcript_api.parse_caption_offline(
            video_id, start_time, end_time
        )
        return transcript

    # Remove [Music], [Laughter], etc
    video_captions = [
        ts
        for ts in video_captions
        if ts["text"].strip() != "" and "[" not in ts["text"]
    ]
    show_s = [x["show_s"] for x in video_captions]
    index_start = bisect.bisect_left(show_s, start_time)
    index_end = bisect.bisect_left(show_s, end_time)
    return " ".join(x["text"].strip() for x in video_captions[index_start:index_end])


def classify(text):
    inputs = bert_tokenizer.encode("[CLS]" + text + "[SEP]")
    inputs_onnx = {
        "input_ids": np.array([inputs.ids]),
        "attention_mask": np.array([inputs.attention_mask]),
        "token_type_ids": np.array([inputs.type_ids]),
    }
    output = bert_session.run(None, inputs_onnx)
    return np.argmax(output[0], axis=1)


def classify_uuid(uuid):
    try:
        video_id, start_time, end_time, category = req_api(uuid)
        if category != "sponsor" and category != "selfpromo":
            return {"error": "Submission category is not sponsor / self promo"}
    except:
        return {"error": "Submission not found"}

    transcript = get_transcript(video_id, start_time, end_time)
    result = classify(transcript.lower())
    return {
        "is_sponsored": bool(result[0]),
        "video_id": video_id,
        "start_time": start_time,
        "end_time": end_time,
        "text": transcript,
    }
