# Conversor de Áudio — Transcritor com Whisper

Aplicativo desktop (PySide6) que transcreve em lote todos os arquivos de áudio de uma pasta usando **WhisperX** (em português), com opção de **identificar falantes** via **pyannote**.

## Funcionalidades

- Seleção de uma pasta e listagem automática dos áudios suportados: `.mp3`, `.wav`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.wma`, `.mp4`, `.opus`
- Escolha do modelo Whisper: `tiny`, `base`, `small`, `medium`, `large`
- Identificação de falantes opcional (saída no formato `Pessoa 1: ...`, `Pessoa 2: ...`)
- Barra de progresso e log em tempo real
- Usa GPU (CUDA) automaticamente quando disponível; caso contrário, roda na CPU

## Pré-requisitos

- **Python 3.11** (o projeto foi desenvolvido com 3.11.9)
- **Git**
- Não é preciso instalar o FFmpeg: ele vem junto pelo pacote `imageio-ffmpeg`
- Para identificar falantes: uma conta no [Hugging Face](https://huggingface.co)

## Instalação

```powershell
# 1. Clone o repositório
git clone <url-do-repositorio>
cd conversor-audio

# 2. Crie e ative o ambiente virtual
python -m venv .venv
.venv\Scripts\Activate.ps1        # PowerShell
# .venv\Scripts\activate.bat      # CMD
# source .venv/bin/activate       # Linux/macOS

# 3. Instale as dependências
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> Se o PowerShell bloquear a ativação do venv, rode uma vez:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### (Opcional) Usar GPU NVIDIA

O `torch` instalado pelo `requirements.txt` é a versão para CPU. Para usar CUDA, reinstale o PyTorch com a build da sua versão de CUDA (veja [pytorch.org](https://pytorch.org/get-started/locally/)), por exemplo:

```powershell
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

## Configuração (`.env`)

Crie um arquivo `.env` na raiz do projeto:

```env
# Token do Hugging Face (obrigatório apenas para identificar falantes)
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxx

# Número fixo de falantes (opcional; deixe vazio para detecção automática)
DEFAULT_NUM_SPEAKERS=

# Use "true" se aparecer erro de "weights_only" ao carregar modelos com PyTorch 2.6+
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true
```

### Liberando a identificação de falantes

1. Gere um token de leitura em <https://huggingface.co/settings/tokens> e coloque em `HF_TOKEN`.
2. Logado no Hugging Face, aceite os termos de uso dos modelos:
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>

Sem isso, a transcrição simples funciona normalmente, mas a opção **Identificar falantes** vai falhar.

## Como rodar

Com o ambiente virtual ativado, na raiz do projeto:

```powershell
python app.py
```

Na janela:

1. Clique em **Selecionar pasta** e escolha a pasta com os áudios.
2. Escolha o modelo (o padrão `small` é um bom equilíbrio entre velocidade e qualidade).
3. Marque **Identificar falantes**, se quiser.
4. Clique em **Iniciar transcrição** (use **Parar** para interromper).

As transcrições são salvas em `.txt` numa subpasta `transcricoes/` dentro da pasta escolhida, com o mesmo nome de cada áudio.

> Na primeira execução os modelos são baixados da internet, o que pode demorar. Modelos maiores (`medium`, `large`) são bem mais lentos na CPU.

## Estrutura do projeto

```
app.py                         # Ponto de entrada (abre a janela)
src/
├── config.py                  # Configurações e leitura do .env
├── constants.py               # Extensões de áudio suportadas
├── audio/
│   ├── ffmpeg_utils.py        # Conversão para WAV mono 16 kHz
│   └── file_scanner.py        # Listagem dos áudios da pasta
├── services/
│   ├── whisper_service.py     # Transcrição e alinhamento com WhisperX
│   ├── diarization_service.py # Identificação de falantes com pyannote
│   └── transcriptor_formatter.py # Junta transcrição + falantes
├── ui/
│   └── main_window.py         # Interface PySide6
└── workers/
    └── transcription_worker.py # Processamento em segundo plano
```

## Problemas comuns

| Problema | Solução |
| --- | --- |
| `HF_TOKEN não configurado` | Adicione o token no `.env` (veja acima). |
| Erro 401/403 ao baixar o pyannote | Aceite os termos dos modelos no Hugging Face. |
| Erro com `weights_only` ao carregar modelo | Defina `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=true` no `.env`. |
| Log mostra `Dispositivo selecionado: cpu` tendo GPU | Instale o PyTorch com CUDA (seção GPU). |
