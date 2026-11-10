from typing import List, Dict


def map_speakers(speaker_segments: List[Dict]) -> Dict[str, str]:
    """
    Mapeia SPEAKER_00 → Pessoa 1, etc.
    """
    speaker_map: Dict[str, str] = {}
    current_index = 1

    for segment in speaker_segments:
        speaker = segment["speaker"]

        if speaker not in speaker_map:
            speaker_map[speaker] = f"Pessoa {current_index}"
            current_index += 1

    return speaker_map


def find_speaker_for_segment(
    segment_start: float,
    segment_end: float,
    speaker_segments: List[Dict],
) -> str:
    """
    Encontra qual speaker corresponde a um segmento de transcrição.
    """
    for spk in speaker_segments:
        if (
            segment_start >= spk["start"]
            and segment_end <= spk["end"]
        ):
            return spk["speaker"]

    # fallback: tenta interseção parcial
    for spk in speaker_segments:
        if not (segment_end < spk["start"] or segment_start > spk["end"]):
            return spk["speaker"]

    return "UNKNOWN"


def build_speaker_transcript(
    whisper_segments: List[Dict],
    speaker_segments: List[Dict],
) -> str:
    """
    Junta transcrição + diarização
    """
    speaker_map = map_speakers(speaker_segments)

    lines: List[str] = []

    for segment in whisper_segments:
        start = segment.get("start", 0.0)
        end = segment.get("end", 0.0)
        text = segment.get("text", "").strip()

        if not text:
            continue

        speaker_id = find_speaker_for_segment(start, end, speaker_segments)
        speaker_name = speaker_map.get(speaker_id, "Desconhecido")

        lines.append(f"{speaker_name}: {text}")

    return "\n".join(lines)