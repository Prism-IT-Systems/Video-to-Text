import { pipeline } from '@huggingface/transformers';
import { readFileSync } from 'fs';
import WaveFile from 'wavefile';

const WHISPER_CHUNK_SEC = 30; // Transformers.js Whisper processes ~30s chunks

let transcriberPromise = null;

async function getTranscriber() {
  if (!transcriberPromise) {
    transcriberPromise = pipeline(
      'automatic-speech-recognition',
      'Xenova/whisper-tiny.en',
      { quantized: true }
    );
  }
  return transcriberPromise;
}

function wavToFloat32(wavPath) {
  const buffer = readFileSync(wavPath);
  const wav = new WaveFile(buffer);
  wav.toBitDepth('32f');
  wav.toSampleRate(16000);
  wav.toMono();
  let audioData = wav.getSamples();
  if (Array.isArray(audioData)) {
    audioData = audioData[0] ?? audioData;
  }
  return new Float32Array(audioData);
}

export async function transcribeWavFile(transcriber, wavPath) {
  const audioData = wavToFloat32(wavPath);
  const result = await transcriber(audioData);
  return result?.text?.trim() || '';
}

export async function transcribeLocal(inputPath, extractToWav) {
  const transcriber = await getTranscriber();
  const wavPath = await extractToWav(inputPath);
  const duration = await getDurationFromWav(wavPath);
  const sampleRate = 16000;
  const samplesPerChunk = sampleRate * WHISPER_CHUNK_SEC;
  const totalSamples = duration * sampleRate;
  const texts = [];

  const buffer = readFileSync(wavPath);
  const wav = new WaveFile(buffer);
  wav.toBitDepth('32f');
  wav.toSampleRate(16000);
  wav.toMono();
  let fullAudio = wav.getSamples();
  if (Array.isArray(fullAudio)) fullAudio = fullAudio[0];
  const audioData = new Float32Array(fullAudio);

  for (let start = 0; start < audioData.length; start += samplesPerChunk) {
    const end = Math.min(start + samplesPerChunk, audioData.length);
    const chunk = audioData.slice(start, end);
    const result = await transcriber(chunk);
    const t = result?.text?.trim();
    if (t) texts.push(t);
  }

  return texts.join(' ').trim() || '(No speech detected)';
}

function getDurationFromWav(wavPath) {
  const buffer = readFileSync(wavPath);
  const wav = new WaveFile(buffer);
  const numSamples = wav.getSamples().length;
  const sampleRate = wav.format.sampleRate;
  return numSamples / sampleRate;
}
