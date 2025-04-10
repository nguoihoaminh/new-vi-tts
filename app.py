import csv
import datetime
import os
import re
import time
import uuid
from io import StringIO

import gradio as gr
import spaces
import torch
import torchaudio
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from vinorm import TTSnorm

# download for mecab
os.system("python -m unidic download")

HF_TOKEN = os.environ.get("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# This will trigger downloading model
print("Downloading if not downloaded viXTTS")
checkpoint_dir = "model/"
repo_id = "capleaf/viXTTS"
use_deepspeed = False

os.makedirs(checkpoint_dir, exist_ok=True)

required_files = ["model.pth", "config.json", "vocab.json", "speakers_xtts.pth"]
files_in_dir = os.listdir(checkpoint_dir)
if not all(file in files_in_dir for file in required_files):
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=checkpoint_dir,
    )
    hf_hub_download(
        repo_id="coqui/XTTS-v2",
        filename="speakers_xtts.pth",
        local_dir=checkpoint_dir,
    )

xtts_config = os.path.join(checkpoint_dir, "config.json")
config = XttsConfig()
config.load_json(xtts_config)
MODEL = Xtts.init_from_config(config)
MODEL.load_checkpoint(
    config, checkpoint_dir=checkpoint_dir, use_deepspeed=use_deepspeed
)
if torch.cuda.is_available():
    MODEL.cuda()

supported_languages = config.languages
if not "vi" in supported_languages:
    supported_languages.append("vi")


def normalize_vietnamese_text(text):
    text = (
        TTSnorm(text, unknown=False, lower=False, rule=True)
        .replace("..", ".")
        .replace("!.", "!")
        .replace("?.", "?")
        .replace(" .", ".")
        .replace(" ,", ",")
        .replace('"', "")
        .replace("'", "")
        .replace("AI", "Ây Ai")
        .replace("A.I", "Ây Ai")
    )
    return text


def calculate_keep_len(text, lang):
    """Simple hack for short sentences"""
    if lang in ["ja", "zh-cn"]:
        return -1

    word_count = len(text.split())
    num_punct = text.count(".") + text.count("!") + text.count("?") + text.count(",")

    if word_count < 5:
        return 15000 * word_count + 2000 * num_punct
    elif word_count < 10:
        return 13000 * word_count + 2000 * num_punct
    return -1


@spaces.GPU
def predict(
    prompt,
    language,
    audio_file_pth,
    normalize_text=True,
):
    if language not in supported_languages:
        metrics_text = gr.Warning(
            f"Language you put {language} in is not in is not in our Supported Languages, please choose from dropdown"
        )

        return (None, metrics_text)

    speaker_wav = audio_file_pth

    if len(prompt) < 2:
        metrics_text = gr.Warning("Please give a longer prompt text")
        return (None, metrics_text)

    # if len(prompt) > 250:
    #     metrics_text = gr.Warning(
    #         str(len(prompt))
    #         + " characters.\n"
    #         + "Your prompt is too long, please keep it under 250 characters\n"
    #         + "Văn bản quá dài, vui lòng giữ dưới 250 ký tự."
    #     )
    #     return (None, metrics_text)

    try:
        metrics_text = ""
        t_latent = time.time()

        try:
            (
                gpt_cond_latent,
                speaker_embedding,
            ) = MODEL.get_conditioning_latents(
                audio_path=speaker_wav,
                gpt_cond_len=30,
                gpt_cond_chunk_len=4,
                max_ref_length=60,
            )

        except Exception as e:
            print("Speaker encoding error", str(e))
            metrics_text = gr.Warning(
                "It appears something wrong with reference, did you unmute your microphone?"
            )
            return (None, metrics_text)

        prompt = re.sub("([^\x00-\x7F]|\w)(\.|\。|\?)", r"\1 \2\2", prompt)

        if normalize_text and language == "vi":
            prompt = normalize_vietnamese_text(prompt)

        print("I: Generating new audio...")
        t0 = time.time()
        out = MODEL.inference(
            prompt,
            language,
            gpt_cond_latent,
            speaker_embedding,
            repetition_penalty=5.0,
            temperature=0.75,
            enable_text_splitting=True,
        )
        inference_time = time.time() - t0
        print(f"I: Time to generate audio: {round(inference_time*1000)} milliseconds")
        metrics_text += (
            f"Time to generate audio: {round(inference_time*1000)} milliseconds\n"
        )
        real_time_factor = (time.time() - t0) / out["wav"].shape[-1] * 24000
        print(f"Real-time factor (RTF): {real_time_factor}")
        metrics_text += f"Real-time factor (RTF): {real_time_factor:.2f}\n"

        # Temporary hack for short sentences
        keep_len = calculate_keep_len(prompt, language)
        out["wav"] = out["wav"][:keep_len]

        torchaudio.save("output.wav", torch.tensor(out["wav"]).unsqueeze(0), 24000)

    except RuntimeError as e:
        if "device-side assert" in str(e):
            # cannot do anything on cuda device side error, need to restart
            print(
                f"Exit due to: Unrecoverable exception caused by language:{language} prompt:{prompt}",
                flush=True,
            )
            gr.Warning("Unhandled Exception encounter, please retry in a minute")
            print("Cuda device-assert Runtime encountered need restart")

            error_time = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
            error_data = [
                error_time,
                prompt,
                language,
                audio_file_pth,
            ]
            error_data = [str(e) if type(e) != str else e for e in error_data]
            print(error_data)
            print(speaker_wav)
            write_io = StringIO()
            csv.writer(write_io).writerows([error_data])
            csv_upload = write_io.getvalue().encode()

            filename = error_time + "_" + str(uuid.uuid4()) + ".csv"
            print("Writing error csv")
            error_api = HfApi()
            error_api.upload_file(
                path_or_fileobj=csv_upload,
                path_in_repo=filename,
                repo_id="coqui/xtts-flagged-dataset",
                repo_type="dataset",
            )

            # speaker_wav
            print("Writing error reference audio")
            speaker_filename = error_time + "_reference_" + str(uuid.uuid4()) + ".wav"
            error_api = HfApi()
            error_api.upload_file(
                path_or_fileobj=speaker_wav,
                path_in_repo=speaker_filename,
                repo_id="coqui/xtts-flagged-dataset",
                repo_type="dataset",
            )

            # HF Space specific.. This error is unrecoverable need to restart space
            space = api.get_space_runtime(repo_id=repo_id)
            if space.stage != "BUILDING":
                api.restart_space(repo_id=repo_id)
            else:
                print("TRIED TO RESTART but space is building")

        else:
            if "Failed to decode" in str(e):
                print("Speaker encoding error", str(e))
                metrics_text = gr.Warning(
                    metrics_text="It appears something wrong with reference, did you unmute your microphone?"
                )
            else:
                print("RuntimeError: non device-side assert error:", str(e))
                metrics_text = gr.Warning(
                    "Something unexpected happened please retry again."
                )
            return (None, metrics_text)
    return ("output.wav", metrics_text)


with gr.Blocks(analytics_enabled=False) as demo:
    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """
                # viXTTS Demo ✨
                - Github: https://github.com/thinhlpg/vixtts-demo/
                """
            )
        with gr.Column():
            # placeholder to align the image
            pass

    with gr.Row():
        with gr.Column():
            input_text_gr = gr.Textbox(
                label="Text Prompt (Văn bản cần đọc)",
                info="Mỗi câu nên từ 10 từ trở lên.",
                value="Xin chào, tôi là một mô hình chuyển đổi văn bản thành giọng nói tiếng Việt.",
            )
            language_gr = gr.Dropdown(
                label="Language (Ngôn ngữ)",
                choices=[
                    "vi",
                    "en",
                    "es",
                    "fr",
                    "de",
                    "it",
                    "pt",
                    "pl",
                    "tr",
                    "ru",
                    "nl",
                    "cs",
                    "ar",
                    "zh-cn",
                    "ja",
                    "ko",
                    "hu",
                    "hi",
                ],
                max_choices=1,
                value="vi",
            )
            normalize_text = gr.Checkbox(
                label="Chuẩn hóa văn bản tiếng Việt",
                info="Normalize Vietnamese text",
                value=True,
            )
            ref_gr = gr.Audio(
                label="Reference Audio (Giọng mẫu)",
                type="filepath",
                value="model/samples/nu-luu-loat.wav",
            )
            tts_button = gr.Button(
                "Đọc 🗣️🔥",
                elem_id="send-btn",
                visible=True,
                variant="primary",
            )

        with gr.Column():
            audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
            out_text_gr = gr.Text(label="Metrics")

    tts_button.click(
        predict,
        [
            input_text_gr,
            language_gr,
            ref_gr,
            normalize_text,
        ],
        outputs=[audio_gr, out_text_gr],
        api_name="predict",
    )

demo.queue()
demo.launch(debug=True, show_api=True, share=True)
