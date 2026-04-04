import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def get_ffmpeg_executable() -> str:
    """
    Retorna o caminho absoluto do ffmpeg que veio do imageio-ffmpeg.
    Não depende do PATH do Windows.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    if not ffmpeg_exe or not Path(ffmpeg_exe).exists():
        raise FileNotFoundError(
            "Não foi possível localizar o ffmpeg do imageio-ffmpeg dentro do ambiente virtual."
        )

    return ffmpeg_exe


def convert_audio_to_wav(
    input_file: Path,
    output_file: Path,
    ffmpeg_exe: str,
) -> None:
    """
    Converte qualquer arquivo de áudio/vídeo aceito para WAV mono 16kHz,
    formato ótimo para o Whisper.
    """
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_file),
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Erro ao converter arquivo com ffmpeg.\n\n"
            f"Arquivo: {input_file.name}\n\n"
            f"Saída do ffmpeg:\n{result.stderr}"
        )


def load_wav_as_numpy(wav_path: Path) -> np.ndarray:
    """
    Lê um WAV PCM 16-bit mono e retorna um array float32 normalizado
    no formato esperado pelo Whisper.
    """
    with wave.open(str(wav_path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()

        if num_channels != 1:
            raise ValueError(
                f"O WAV precisa ser mono, mas veio com {num_channels} canal(is)."
            )

        if sample_width != 2:
            raise ValueError(
                f"O WAV precisa ser PCM 16-bit, mas veio com sample width {sample_width}."
            )

        if sample_rate != 16000:
            raise ValueError(
                f"O WAV precisa estar em 16000 Hz, mas veio com {sample_rate} Hz."
            )

        audio_bytes = wf.readframes(num_frames)

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return audio