from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pydub import AudioSegment
import re, io

audio_file = "audioteste.mp3"
auth_token = "seu token aqui"

# -----------------------------------------------------------------

# pyannote diarizization:
print("------------ Pyannote Diarization --------------\n\n")

# huggingface
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token=auth_token
)

#pipeline.to(torch.device("cuda"))

# Pppeline
diarization = pipeline(audio_file)

# imprimir diarization
print("\n\n")
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

# salvar diarization
with open("diarization.txt", "w") as arquivo_texto:
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        arquivo_texto.write(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}\n")
print("\n")

# -----------------------------------------------------------------

# converte segundos em milisegundos (para busca em arquivo de áudio)
def seconds_to_ms(seconds):
    return int(seconds * 1000)

# -----------------------------------------------------------------

print("\n\n------------ Faster Whisper Raw Transcription --------------\n\n")

# faster whisper transcription
model_size = "large-v3"

# run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe(audio_file, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))
print("\n")

# imprime transcriçao bruta
for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))

print("\n")

#for line in info:
#    print(line)

print("\n")

# -----------------------------------------------------------------

print("\n\n------------ Final Transcription --------------\n\n")

with open("diarization.txt", "r") as diarization_file:
    with open("transcriptions.txt", "w") as transcription_file:
        for line in diarization_file:
            start_time, end_time = tuple(re.findall(r'[0-9]+\.[0-9]+', line))
            speaker = re.findall(r'speaker_([^ \n]+)', line)

            # converte em float (fix problema pydub)
            start_time = float(start_time)
            end_time = float(end_time)

            # carrega intervalo do audio definito na diarizaçao
            audio = AudioSegment.from_file(audio_file)
            audio_part = audio[seconds_to_ms(start_time):seconds_to_ms(end_time)]

            audio_bytes = io.BytesIO() # primeiro cria um objeto BytesIO para armazenar os dados de áudio exportados
            audio_part.export(audio_bytes, format="mp3") # depois exporta o segmento para objeto e define o formato do arquivo
            audio_bytes.seek(0) # move o ponteiro para o inicio do audio

            # transcreve o segmento de audio
            segments, info = model.transcribe(audio_bytes, beam_size=5)

            text = ""

            for segment in segments:
                text = text + " " + segment.text

            # salva dados e texto no arquivo final
            transcription_file.write(f"start={start_time:.1f}s stop={end_time:.1f}s speaker={speaker} text={text}\n")

# imprime transcriçao final
with open("transcriptions.txt", "r") as file:
    for l in file:
        print(l)


print("\n\n\n")

# -----------------------------------------------------------------
