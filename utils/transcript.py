import bisect
import json
import os

import ffmpeg
import requests
import yt_dlp

from utils.offline_asr import OfflineASR

offline_asr = OfflineASR("model/silero_model_xl.onnx", "model/silero_vocab.json")
ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": "%(id)s.%(ext)s",
    "extractor_args": {
        "youtube": {
            "player_skip": ["webpage", "configs", "js"],
            "player_client": ["android"],
        }
    },
    "quiet": True,
    "no_warnings": True,
}


def parse_player_info(video_id):
    player_params = {"key": "AIzaSyAO_FJ2SlqU8Q4STEHLGCilw_Y9_11qcW8"}
    data = {
        "videoId": video_id,
        "context": {
            "client": {
                "clientName": "WEB_EMBEDDED_PLAYER",
                "clientVersion": "1.20211019.01.00",
            },
        },
    }
    response = requests.post(
        "https://www.youtube-nocookie.com/youtubei/v1/player",
        params=player_params,
        data=json.dumps(data),
    )
    return response.json()


def parse_caption(response):
    sentences = []
    if (
        "captions" not in response
        or "playerCaptionsTracklistRenderer" not in response["captions"]
    ):
        return sentences
    caption_tracks = response["captions"]["playerCaptionsTracklistRenderer"][
        "captionTracks"
    ]
    json3_url = None
    for cap in caption_tracks:
        if "kind" in cap and cap["kind"] == "asr":
            json3_url = "https://www.youtube.com" + cap["baseUrl"] + "&fmt=json3"
            if cap["languageCode"] != "en":
                json3_url += "&tlang=en"
            break

    if json3_url is None:
        return sentences
    r = requests.get(json3_url)
    for event in r.json()["events"]:
        if "segs" not in event:
            continue
        segs = event["segs"]
        start_ms = event["tStartMs"]
        for seg in segs:
            if "tOffsetMs" in seg:
                seg_ms = start_ms + seg["tOffsetMs"]
            else:
                seg_ms = start_ms
            sentences.append({"text": seg["utf8"], "show_s": seg_ms / 1000})
    return sentences


def parse_caption_offline(video_id, start_time, end_time):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_id])
    ffmpeg.input(f"{video_id}.webm", ss=start_time, to=end_time).output(
        f"{video_id}.wav", ac=1, ar=16000
    ).overwrite_output().run()
    sentences = offline_asr(f"{video_id}.wav")
    os.remove(f"{video_id}.wav")
    os.remove(f"{video_id}.webm")
    return sentences
