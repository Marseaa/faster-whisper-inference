from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

audio_file = "audioteste.mp3"
auth_token = "seu token aqui"

print("\n\n------------ Faster Whisper Transcription --------------\n\n")

# faster whisper transcription
model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe(audio_file, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

print("\n")

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

print("\n\n\n")

# pyannote diarizization:
print("------------ Pyannote Diarization --------------\n\n")

# huggingface
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token=auth_token
)

#pipeline.to(torch.device("cuda"))

# run the pipeline on an audio file
diarization = pipeline(audio_file)

# print diarization
print("\n\n")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

print("\n\n\n")
