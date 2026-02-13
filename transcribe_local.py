#!/usr/bin/env python3
"""
Local Whisper transcription with speaker diarization (person-wise output).
Uses faster-whisper + MFCC-based clustering. Fully local—no build tools required.

Usage: python transcribe_local.py <file_path> [--model base]
Output: Person 1: ... \n Person 2: ...
"""
import json
import sys
from pathlib import Path


def load_audio(file_path: str, sr: int = 16000):
    """Load audio as mono float32 at given sample rate."""
    import librosa
    import numpy as np

    wav, _ = librosa.load(str(file_path), sr=sr, mono=True)
    return wav.astype(np.float32)


def extract_mfcc_features(audio_chunk: "np.ndarray", sr: int = 16000) -> "np.ndarray":
    """Extract MFCC features for speaker clustering (no webrtcvad/build tools)."""
    import librosa
    import numpy as np

    if len(audio_chunk) < sr * 0.2:  # Too short
        return None
    mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13, n_fft=512, hop_length=160)
    return np.mean(mfcc, axis=1)  # Average over time


def transcribe_with_diarization(file_path: str, model_size: str) -> str:
    """Transcribe and assign speakers using faster-whisper + MFCC clustering."""
    from faster_whisper import WhisperModel
    from sklearn.cluster import AgglomerativeClustering
    import numpy as np

    SAMPLE_RATE = 16000

    # 1. Transcribe with timestamps
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments_gen, _ = model.transcribe(file_path, word_timestamps=False)
    segments = list(segments_gen)

    if not segments:
        return "(No speech detected)"

    # 2. Load audio
    try:
        audio = load_audio(file_path, SAMPLE_RATE)
    except Exception as e:
        return _format_single_speaker(segments) + f"\n(Note: Could not load audio for diarization: {e})"

    # 3. Get MFCC features per segment
    features = []
    valid_indices = []
    for i, seg in enumerate(segments):
        start = int(seg.start * SAMPLE_RATE)
        end = int(seg.end * SAMPLE_RATE)
        if end - start < SAMPLE_RATE * 0.2:
            continue
        chunk = audio[start:end]
        feats = extract_mfcc_features(chunk, SAMPLE_RATE)
        if feats is not None:
            features.append(feats)
            valid_indices.append(i)

    # 4. Cluster speakers
    if len(features) < 2:
        return _format_single_speaker(segments)

    n_speakers = min(10, max(2, len(features) // 3))
    clustering = AgglomerativeClustering(
        n_clusters=n_speakers, metric="cosine", linkage="average"
    )
    labels = clustering.fit_predict(np.stack(features))

    # 5. Map indices and assign speakers (unassigned get previous)
    seg_to_speaker = {idx: int(lab) for idx, lab in zip(valid_indices, labels)}
    prev_speaker = 0
    speaker_seq = []
    for i in range(len(segments)):
        if i in seg_to_speaker:
            prev_speaker = seg_to_speaker[i]
        speaker_seq.append(prev_speaker)

    # 6. Remap to Person 1, 2, 3...
    order = []
    speaker_ids = []
    for s in speaker_seq:
        if s not in order:
            order.append(s)
        speaker_ids.append(order.index(s) + 1)

    # 7. Format output
    lines = []
    for i, seg in enumerate(segments):
        text = (seg.text or "").strip()
        if text:
            lines.append(f"Person {speaker_ids[i]}: {text}")

    return "\n".join(lines) if lines else "(No speech detected)"


def _format_single_speaker(segments) -> str:
    text = " ".join((s.text or "").strip() for s in segments if (s.text or "").strip())
    return f"Person 1: {text}" if text else "(No speech detected)"


def main():
    if len(sys.argv) < 2:
        print(
            json.dumps({"error": "Usage: transcribe_local.py <file_path> [--model base]"}),
            file=sys.stderr,
        )
        sys.exit(1)

    file_path = sys.argv[1]
    model_size = "base"
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_size = sys.argv[idx + 1]

    if not Path(file_path).exists():
        print(json.dumps({"error": f"File not found: {file_path}"}), file=sys.stderr)
        sys.exit(1)

    try:
        text = transcribe_with_diarization(file_path, model_size)
        print(json.dumps({"text": text}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
