import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from src.audio.ffmpeg_utils import (
    convert_audio_to_wav,
    get_ffmpeg_executable,
    load_wav_as_numpy,
)
from src.audio.file_scanner import list_audio_files
from src.services.whisper_service import (
    transcribe_with_whisperx,
    build_plain_text,
)
from src.services.diarization_service import diarize_audio, diarization_to_segments
from src.services.transcriptor_formatter import build_speaker_transcript


class TranscriptionWorker(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(
        self,
        folder_path: str,
        model_name: str,
        enable_speakers: bool = False,
    ) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.model_name = model_name
        self._running = True
        self.enable_speakers = enable_speakers

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            folder = Path(self.folder_path)
            if not folder.exists() or not folder.is_dir():
                self.failed.emit("A pasta selecionada é inválida.")
                return

            audio_files = list_audio_files(self.folder_path)

            if not audio_files:
                self.failed.emit("Nenhum arquivo de áudio foi encontrado na pasta.")
                return

            output_dir = folder / "transcricoes"
            output_dir.mkdir(exist_ok=True)

            temp_dir = folder / "_temp_wav"
            temp_dir.mkdir(exist_ok=True)

            ffmpeg_exe = get_ffmpeg_executable()
            self.log.emit(f"FFmpeg do ambiente virtual encontrado em: {ffmpeg_exe}")

            self.log.emit(f"Carregando modelo WhisperX: {self.model_name}")

            total_files = len(audio_files)

            for index, audio_file in enumerate(audio_files, start=1):
                if not self._running:
                    self.log.emit("Transcrição interrompida pelo usuário.")
                    self.finished.emit()
                    return

                self.log.emit(f"Processando ({index}/{total_files}): {audio_file.name}")

                temp_wav = temp_dir / f"{audio_file.stem}.wav"
                self.log.emit(f"Convertendo para WAV: {audio_file.name}")

                convert_audio_to_wav(
                    input_file=audio_file,
                    output_file=temp_wav,
                    ffmpeg_exe=ffmpeg_exe,
                )

                self.log.emit(f"Transcrevendo: {audio_file.name}")

                audio_array = load_wav_as_numpy(temp_wav)

                self.log.emit(f"Transcrevendo com WhisperX: {audio_file.name}")

                result = transcribe_with_whisperx(
                    model_name=self.model_name,
                    audio_array=audio_array,
                )

                if self.enable_speakers:
                    self.log.emit(f"Identificando falantes: {audio_file.name}")

                    diarization = diarize_audio(str(temp_wav))
                    speaker_segments = diarization_to_segments(diarization)

                    text_output = build_speaker_transcript(
                        whisper_segments=result.get("segments", []),
                        speaker_segments=speaker_segments,
                    )
                else:
                    text_output = build_plain_text(result)

                output_file = output_dir / f"{audio_file.stem}.txt"
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(text_output)

                try:
                    temp_wav.unlink(missing_ok=True)
                except Exception:
                    pass

                progress_value = int((index / total_files) * 100)
                self.progress.emit(progress_value)
                self.log.emit(f"Arquivo salvo: {output_file.name}")

            try:
                temp_dir.rmdir()
            except Exception:
                pass

            self.finished.emit()

        except Exception:
            self.failed.emit(traceback.format_exc())