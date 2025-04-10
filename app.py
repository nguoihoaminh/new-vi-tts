# Ubuntu: sudo apt install ffmpeg
# Windows please refer to https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/

import os
import requests
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] =  "0" # Tell it which GPU to use (or ignore if you're CPU-bound and patient!)

from vinorm import TTSnorm # Gotta normalize that Vietnamese text first
from infer.f5tts_wrapper import F5TTSWrapper # Our handy wrapper class


default_dir = "/kaggle/working/new-vi-tts"

# --- Config ---
MODEL_URL = "https://cdn-lfs-us-1.hf.co/repos/ab/76/ab761a9d373aa0b6e54886fd8cf42675589ec29e5f3a1bf14d38b59eb8ad99a7/f2122cedcffc532d6048847092414e638cfb4db402881cb8146a606008e9ff56?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27seamlessM4T_v2_large.pt%3B+filename%3D%22seamlessM4T_v2_large.pt%22%3B&Expires=1744276208&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0NDI3NjIwOH19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zL2FiLzc2L2FiNzYxYTlkMzczYWEwYjZlNTQ4ODZmZDhjZjQyNjc1NTg5ZWMyOWU1ZjNhMWJmMTRkMzhiNTllYjhhZDk5YTcvZjIxMjJjZWRjZmZjNTMyZDYwNDg4NDcwOTI0MTRlNjM4Y2ZiNGRiNDAyODgxY2I4MTQ2YTYwNjAwOGU5ZmY1Nj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=iLzsf6AWbQRTPZHxWq7SF0Iq6ZVhoGHf3lem5up6iH0Cun8-M1N%7EqTpH4lFJAPa%7ELDA-%7EhNTbRdJK1e0YpwxH1NsQPI-qyjint9ESu-rhQow-CQ82wrBhQjea9nRAthHyiOd9PTZEOAe9nCy1V-%7EGhhfxZpWzfDCaLfMyFAhSnR-ZbPPZTSh4MtdowzbQRcE9C-5IfhhbiuKT2Q5za9AT0J8DT6EJYevab%7Emp%7EkY2cDHZO3rL5nxJqCUufeHGCcMOQQsRCw7czFAY9qE0lvQDGH8kDONLxE35snIboT4tJMqNXm2qCyARwUTOqlDAQc05qE2CI9UFhNZenZ1P83pjw__&Key-Pair-Id=K24J24Z295AEI9"

def download_model(url: str, save_path: str):
    """Download model file with progress bar"""
    if os.path.exists(save_path):
        print(f"Model already exists at {save_path}")
        return
    
    print(f"Downloading model to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as file, tqdm(
        desc="Downloading",
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

# Download model if needed
download_model(MODEL_URL, f"{default_dir}/infer/model/")

eraX_ckpt_path = f"{default_dir}/infer/model/seamlessM4T_v2_large.pt"

# Path to the voice you want to clone
ref_audio_path = f"{default_dir}/infer/audio.wav" # <-- CHANGE THIS!

# Path to the vocab file from this repo
vocab_file = f"{default_dir}/infer/model/vocab.txt" # <-- CHANGE THIS!

# Where to save the generated sound
output_dir = f"{default_dir}/audio"

# --- Texts ---
# Text matching the reference audio (helps the model learn the voice). Please make sure it match with the referrence audio!
ref_text = "Và nếu họ có quan điểm chính trị trái ngược , bạn vẫn có thể mỉm cười với họ trong nhà thờ nhưng ngoài đời thì khó lòng mà làm được . Những suy nghĩ này có quen thuộc với bạn không ?"

# --- Let's Go! ---
print("Initializing the TTS engine... (Might take a sec)")
tts = F5TTSWrapper(
    vocoder_name="vocos", # Using Vocos vocoder
    ckpt_path=eraX_ckpt_path,
    vocab_file=vocab_file,
    use_ema=False, # ALWAYS False as we converted from .pt to safetensors and EMA (where there is or not) was in there
)

# Normalize the reference text (makes it easier for the model)
ref_text_norm = TTSnorm(ref_text)

# Prepare the output folder
os.makedirs(output_dir, exist_ok=True)

print("Processing the reference voice...")
# Feed the model the reference voice ONCE
# Provide ref_text for better quality, or set ref_text="" to use Whisper for auto-transcription (if installed)
tts.preprocess_reference(
    ref_audio_path=ref_audio_path,
    ref_text=ref_text_norm,
    clip_short=True # Keeps reference audio to a manageable length (~12s)
)
print(f"Reference audio duration used: {tts.get_current_audio_length():.2f} seconds")

# --- Generate New Speech ---
print("Generating new speech with the cloned voice...")

def speak(text,filename):
    # Normalize the text we want to speak
    text_norm = TTSnorm(text)

    # You can generate multiple sentences easily
    # Just add more normalized strings to this list
    sentences = [text_norm]

    for i, sentence in enumerate(sentences):
        output_path = os.path.join(output_dir, f"{filename}_{i+1}.wav")

        # THE ACTUAL GENERATION HAPPENS HERE!
        tts.generate(
            text=sentence,
            output_path=output_path,
            nfe_step=20,               # Denoising steps. More = slower but potentially better? (Default: 32)
            cfg_strength=2.0,          # How strongly to stick to the reference voice style? (Default: 2.0)
            speed=1.0,                 # Make it talk faster or slower (Default: 1.0)
            cross_fade_duration=0.15,  # Smooths transitions if text is split into chunks (Default: 0.15)
        )

        print(f"Boom! Audio saved to: {output_path}")

    print("\nAll done! Check your output folder.")

dir = f'{default_dir}/books'

for filename in os.listdir(dir):
    fs = open(dir + '/'+filename, "r")
    text = fs.read()
    speak(text,filename.split('.')[0])
    fs.close()
    print('Saved: '+filename)