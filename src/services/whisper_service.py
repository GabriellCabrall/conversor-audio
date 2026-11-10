from pathlib import Path
from typing import Any

import torch
import whisperx

import os

from src.config import TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD

if TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD:
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "true"

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_compute_type(device: str) -> str:
    if device == "cuda":
        return "float16"
    return "int8"


def load_whisperx_model(model_name: str) -> tuple[Any, str, str]:
    device = get_device()
    compute_type = get_compute_type(device)

    print(f"[WhisperX] Dispositivo selecionado: {device}")
    print(f"[WhisperX] Compute type: {compute_type}")

    model = whisperx.load_model(
        model_name,
        device=device,
        compute_type=compute_type,
        language="pt",
    )

    return model, device, compute_type


def transcribe_audio_array(
    model: Any,
    audio_array,
    batch_size: int = 16,
) -> dict:
    """
    Transcreve um áudio em memória usando WhisperX.
    """
    result = model.transcribe(
        audio_array,
        batch_size=batch_size,
        language="pt",
    )
    return result


def align_transcription(
    segments: list[dict],
    audio_array,
    device: str,
) -> dict:
    """
    Faz alinhamento para melhorar timestamps.
    """
    align_model, metadata = whisperx.load_align_model(
        language_code="pt",
        device=device,
    )

    aligned_result = whisperx.align(
        segments,
        align_model,
        metadata,
        audio_array,
        device,
        return_char_alignments=False,
    )

    return aligned_result


def build_plain_text(transcription_result: dict) -> str:
    """
    Junta os segmentos em texto corrido.
    """
    segments = transcription_result.get("segments", [])
    lines: list[str] = []

    for segment in segments:
        text = segment.get("text", "").strip()
        if text:
            lines.append(text)

    return "\n".join(lines).strip()


def transcribe_with_whisperx(
    model_name: str,
    audio_array,
) -> tuple[dict, str, str]:
    model, device, compute_type = load_whisperx_model(model_name)

    transcription = transcribe_audio_array(model, audio_array)
    aligned = align_transcription(
        segments=transcription["segments"],
        audio_array=audio_array,
        device=device,
    )

    return aligned, device, compute_type