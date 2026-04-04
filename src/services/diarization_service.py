from typing import Any

from pyannote.audio import Pipeline

from src.config import HF_TOKEN, DEFAULT_NUM_SPEAKERS


def load_diarization_pipeline() -> Any:
    """
    Carrega o pipeline de diarização do pyannote.
    """
    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN não configurado. Adicione seu token no arquivo .env para usar diarização."
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HF_TOKEN,
    )

    return pipeline


def diarize_audio(
    audio_path: str,
    num_speakers: int | None = None,
) -> Any:
    """
    Executa diarização em um arquivo de áudio.
    """
    pipeline = load_diarization_pipeline()

    diarization_kwargs: dict[str, Any] = {}

    if num_speakers is not None:
        diarization_kwargs["num_speakers"] = num_speakers
    elif DEFAULT_NUM_SPEAKERS is not None:
        diarization_kwargs["num_speakers"] = DEFAULT_NUM_SPEAKERS

    diarization = pipeline(audio_path, **diarization_kwargs)
    return diarization


def diarization_to_segments(diarization: Any) -> list[dict]:
    """
    Converte a saída do pyannote em uma lista simples de segmentos.
    """
    segments: list[dict] = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
        )

    return segments