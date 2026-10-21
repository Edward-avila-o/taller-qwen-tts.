import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

if torch.cuda.is_available():
    device, dtype = "cuda:0", torch.bfloat16
else:
    device, dtype = "cpu", torch.float32
print("Usando:", device)

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map=device,
    dtype=dtype,
)
wavs, sr = model.generate_custom_voice(
    text="Hola, este es mi taller de voz con Qwen TTS, ejecutado desde PowerShell.",
    language="Spanish",
    speaker="Ryan",
)
sf.write("salida_qwen.wav", wavs[0], sr)
print("Listo: salida_qwen.wav")
