from pathlib import Path

from src.constants import AUDIO_EXTENSIONS


def list_audio_files(folder_path: str) -> list[Path]:
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        return []

    return sorted(
        [
            file
            for file in folder.iterdir()
            if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS
        ],
        key=lambda x: x.name.lower(),
    )


def list_audio_file_names(folder_path: str) -> list[str]:
    return [file.name for file in list_audio_files(folder_path)]