# Video to Text

Upload a video and get a transcript with speaker labels (Person 1, Person 2, …). Works with OpenAI Whisper API or fully offline using Python.

## Features

- **Video upload** – Drag & drop or browse (MP4, WebM, MP3, M4A, WAV; up to 200 MB)
- **Speaker diarization** – Person-wise output in local mode
- **Dual mode** – API (OpenAI) or local (Python, no external services)
- **No build tools** – Pure Python/Node; no Visual C++ or webrtcvad

## Setup

### 1. Install Node.js dependencies

```bash
npm install
```

### 2. Choose transcription mode

**Option A: Local (offline, no API key)**

1. Install Python 3.9+ if needed.
2. Install Python dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

   Or: `python3 -m pip install -r requirements.txt`

3. Copy env and set local mode:

   ```bash
   # Windows (PowerShell)
   Copy-Item .env.example .env
   ```

   In `.env`:

   ```
   TRANSCRIPTION_MODE=local
   ```

**Option B: OpenAI API**

1. Copy the example env file (see above).
2. In `.env`:

   ```
   TRANSCRIPTION_MODE=api
   OPENAI_API_KEY=sk-your-api-key-here
   ```

   Get your key from [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

### 3. Run the server

```bash
npm start
```

4. Open [http://localhost:3000](http://localhost:3000)

## Usage

- Drag and drop a video or click to browse
- Supported formats: MP4, WebM, MP3, MPEG, MPGA, M4A, WAV
- Max file size: 200 MB (larger files are chunked in API mode)

## Local mode notes

- **Speaker diarization:** MFCC-based clustering; fully local, no Hugging Face or build tools
- First run downloads the Whisper model (~150 MB)
- Model size: edit `transcribe_local.py` and change `model_size` (`tiny`, `base`, `small`, `medium`, `large-v2`)
- If `python` is not in PATH, set `PYTHON_PATH` in `.env` (e.g. `PYTHON_PATH=C:\Python314\python.exe`)

## Development

```bash
npm run dev
```

## Tech stack

- **Node.js** – Express, Multer, FFmpeg (bundled)
- **Python** (local mode) – faster-whisper, librosa, scikit-learn

## License

MIT
