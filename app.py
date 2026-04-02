import os
import sys
import subprocess
import traceback
from pathlib import Path
import wave
import numpy as np

import imageio_ffmpeg
import whisper

from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QFileDialog,
    QListWidget,
    QTextEdit,
    QComboBox,
    QMessageBox,
    QProgressBar,
)

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".wma",
    ".mp4",
    ".opus",
}


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


class TranscriptionWorker(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    failed = Signal(str)

    def __init__(self, folder_path: str, model_name: str) -> None:
        super().__init__()
        self.folder_path = folder_path
        self.model_name = model_name
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            folder = Path(self.folder_path)
            if not folder.exists() or not folder.is_dir():
                self.failed.emit("A pasta selecionada é inválida.")
                return

            audio_files = sorted(
                [
                    file
                    for file in folder.iterdir()
                    if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS
                ],
                key=lambda x: x.name.lower(),
            )

            if not audio_files:
                self.failed.emit("Nenhum arquivo de áudio foi encontrado na pasta.")
                return

            output_dir = folder / "transcricoes"
            output_dir.mkdir(exist_ok=True)

            temp_dir = folder / "_temp_wav"
            temp_dir.mkdir(exist_ok=True)

            ffmpeg_exe = get_ffmpeg_executable()
            self.log.emit(f"FFmpeg do ambiente virtual encontrado em: {ffmpeg_exe}")

            self.log.emit(f"Carregando modelo Whisper: {self.model_name}")
            model = whisper.load_model(self.model_name)

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

                result = model.transcribe(
                    audio_array,
                    language="pt",
                    task="transcribe",
                    verbose=False,
                    fp16=False,
                )

                output_file = output_dir / f"{audio_file.stem}.txt"
                with open(output_file, "w", encoding="utf-8") as file:
                    file.write(result["text"].strip())

                # remove o wav temporário após transcrever
                try:
                    temp_wav.unlink(missing_ok=True)
                except Exception:
                    pass

                progress_value = int((index / total_files) * 100)
                self.progress.emit(progress_value)
                self.log.emit(f"Arquivo salvo: {output_file.name}")

            # tenta remover a pasta temporária no final
            try:
                temp_dir.rmdir()
            except Exception:
                pass

            self.finished.emit()

        except Exception:
            self.failed.emit(traceback.format_exc())


class MainWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("Transcritor de Áudios com Whisper")
        self.resize(800, 600)

        self.selected_folder: str = ""
        self.thread: QThread | None = None
        self.worker: TranscriptionWorker | None = None

        self.title_label = QLabel("Selecione uma pasta com áudios")
        self.folder_label = QLabel("Nenhuma pasta selecionada")

        self.select_button = QPushButton("Selecionar pasta")
        self.select_button.clicked.connect(self.select_folder)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large"])

        self.file_list = QListWidget()

        self.start_button = QPushButton("Iniciar transcrição")
        self.start_button.clicked.connect(self.start_transcription)

        self.stop_button = QPushButton("Parar")
        self.stop_button.clicked.connect(self.stop_transcription)
        self.stop_button.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        self.setup_layout()

    def setup_layout(self) -> None:
        layout = QVBoxLayout()

        layout.addWidget(self.title_label)
        layout.addWidget(self.folder_label)

        top_buttons_layout = QHBoxLayout()
        top_buttons_layout.addWidget(self.select_button)
        top_buttons_layout.addWidget(QLabel("Modelo:"))
        top_buttons_layout.addWidget(self.model_combo)

        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.addWidget(self.start_button)
        action_buttons_layout.addWidget(self.stop_button)

        layout.addLayout(top_buttons_layout)
        layout.addWidget(QLabel("Arquivos encontrados:"))
        layout.addWidget(self.file_list)
        layout.addLayout(action_buttons_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Logs:"))
        layout.addWidget(self.log_output)

        self.setLayout(layout)

    def log(self, message: str) -> None:
        self.log_output.append(message)

    def select_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Selecione a pasta com os áudios")
        if not folder:
            return

        self.selected_folder = folder
        self.folder_label.setText(folder)
        self.load_audio_files()

    def load_audio_files(self) -> None:
        self.file_list.clear()

        folder = Path(self.selected_folder)
        if not folder.exists():
            return

        audio_files = sorted(
            [
                file.name
                for file in folder.iterdir()
                if file.is_file() and file.suffix.lower() in AUDIO_EXTENSIONS
            ]
        )

        if not audio_files:
            self.log("Nenhum áudio encontrado nessa pasta.")
            return

        self.file_list.addItems(audio_files)
        self.log(f"{len(audio_files)} arquivo(s) encontrado(s).")

    def start_transcription(self) -> None:
        if not self.selected_folder:
            QMessageBox.warning(self, "Aviso", "Selecione uma pasta primeiro.")
            return

        if self.file_list.count() == 0:
            QMessageBox.warning(self, "Aviso", "Nenhum áudio encontrado na pasta.")
            return

        self.progress_bar.setValue(0)
        self.log("Iniciando transcrição...")

        self.thread = QThread()
        self.worker = TranscriptionWorker(
            folder_path=self.selected_folder,
            model_name=self.model_combo.currentText(),
        )

        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.log)
        self.worker.finished.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)

        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.thread.deleteLater)

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

        self.thread.start()

    def stop_transcription(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.log("Solicitação de parada enviada...")

    def on_finished(self) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log("Processo finalizado.")
        QMessageBox.information(self, "Concluído", "Transcrição finalizada com sucesso.")

    def on_failed(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log("Erro durante a transcrição:")
        self.log(message)
        QMessageBox.critical(self, "Erro", message)

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())