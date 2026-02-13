import 'dotenv/config';
import express from 'express';
import multer from 'multer';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';
import fs from 'fs/promises';
import { createReadStream } from 'fs';
import OpenAI from 'openai';
import ffmpeg from 'fluent-ffmpeg';
import ffmpegStatic from 'ffmpeg-static';
import ffprobeStatic from 'ffprobe-static';

// Use bundled FFmpeg binaries (no system install required)
if (ffmpegStatic) ffmpeg.setFfmpegPath(ffmpegStatic);
if (ffprobeStatic?.path) ffmpeg.setFfprobePath(ffprobeStatic.path);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;

// Upload limit: 200 MB (Whisper API limit per request: 25 MB - we chunk larger files)
const MAX_FILE_SIZE = 200 * 1024 * 1024;
const WHISPER_MAX_SIZE = 25 * 1024 * 1024;
const SEGMENT_DURATION_SEC = 600; // 10 min chunks (~9 MB MP3 each)
const UPLOAD_DIR = path.join(__dirname, 'uploads');
const TEMP_DIR = path.join(__dirname, 'uploads', 'temp');
const PYTHON_SCRIPT = path.join(__dirname, 'transcribe_local.py');

// api = OpenAI API (needs OPENAI_API_KEY), local = Python Whisper (no API)
const TRANSCRIPTION_MODE = (process.env.TRANSCRIPTION_MODE || 'api').toLowerCase();

await fs.mkdir(UPLOAD_DIR, { recursive: true });
await fs.mkdir(TEMP_DIR, { recursive: true });

const storage = multer.diskStorage({
  destination: UPLOAD_DIR,
  filename: (req, file, cb) => {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1e9);
    const ext = path.extname(file.originalname) || '.mp4';
    cb(null, `video-${uniqueSuffix}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: Number(MAX_FILE_SIZE) },
  fileFilter: (req, file, cb) => {
    const allowed = /\.(mp4|webm|mp3|mpeg|mpga|m4a|wav)$/i;
    if (allowed.test(path.extname(file.originalname))) {
      cb(null, true);
    } else {
      cb(new Error('Invalid file type. Use mp4, webm, mp3, mpeg, mpga, m4a, or wav.'));
    }
  },
});

function getDuration(inputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg.ffprobe(inputPath, (err, data) => {
      if (err) reject(err);
      else resolve(data.format.duration || 0);
    });
  });
}

function extractAudio(inputPath, outputPath) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .noVideo()
      .audioCodec('libmp3lame')
      .audioBitrate(128)
      .audioChannels(1)
      .output(outputPath)
      .on('end', () => resolve(outputPath))
      .on('error', reject)
      .run();
  });
}

function extractSegment(inputPath, outputPath, startSec, durationSec) {
  return new Promise((resolve, reject) => {
    ffmpeg(inputPath)
      .setStartTime(startSec)
      .setDuration(durationSec)
      .output(outputPath)
      .on('end', () => resolve(outputPath))
      .on('error', reject)
      .run();
  });
}

async function transcribeFile(openai, filePath) {
  const stat = await fs.stat(filePath);
  if (stat.size > WHISPER_MAX_SIZE) {
    throw new Error(`Segment too large (${Math.round(stat.size / 1024 / 1024)} MB). Try shorter segments.`);
  }
  return openai.audio.transcriptions.create({
    file: createReadStream(filePath),
    model: 'whisper-1',
  });
}

function runLocalWhisper(filePath) {
  return new Promise((resolve, reject) => {
    // Windows: 'py -3' (launcher) or 'python'; Unix: 'python3'
    const python = process.env.PYTHON_PATH || (process.platform === 'win32' ? 'py' : 'python3');
    const args = python === 'py' ? ['-3', PYTHON_SCRIPT, filePath] : [PYTHON_SCRIPT, filePath];
    const env = { ...process.env };
    const child = spawn(python, args, {
      stdio: ['ignore', 'pipe', 'pipe'],
      env,
    });

    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (d) => (stdout += d.toString()));
    child.stderr.on('data', (d) => (stderr += d.toString()));

    child.on('close', (code) => {
      if (code !== 0) {
        try {
          const err = JSON.parse(stderr.trim() || '{}');
          reject(new Error(err.error || stderr || 'Python transcription failed'));
        } catch {
          const msg = stderr || 'Python transcription failed';
          if (msg.includes('No module named') || msg.includes('ModuleNotFoundError')) {
            reject(new Error('Python packages missing. Run: pip install -r requirements.txt'));
          } else {
            reject(new Error(msg));
          }
        }
        return;
      }
      try {
        const out = JSON.parse(stdout.trim());
        resolve(out.text || '(No speech detected)');
      } catch {
        reject(new Error('Invalid output from Python script'));
      }
    });

    child.on('error', (err) => {
      reject(new Error(`Failed to run Python: ${err.message}. Is Python installed? Run: pip install -r requirements.txt`));
    });
  });
}

async function transcribeLargeFile(openai, filePath) {
  const jobId = path.basename(filePath, path.extname(filePath));
  const audioPath = path.join(TEMP_DIR, `${jobId}-audio.mp3`);
  const segments = [];

  try {
    await extractAudio(filePath, audioPath);
    const duration = await getDuration(audioPath);
    let start = 0;

    while (start < duration) {
      const segDuration = Math.min(SEGMENT_DURATION_SEC, duration - start);
      const segPath = path.join(TEMP_DIR, `${jobId}-seg-${segments.length}.mp3`);
      await extractSegment(audioPath, segPath, start, segDuration);
      segments.push(segPath);
      start += segDuration;
    }

    const texts = [];
    for (const segPath of segments) {
      const result = await transcribeFile(openai, segPath);
      texts.push(result.text?.trim() || '');
    }

    return texts.filter(Boolean).join('\n\n') || '(No speech detected)';
  } finally {
    const toDelete = [audioPath, ...segments];
    for (const p of toDelete) {
      try {
        await fs.unlink(p);
      } catch (_) {}
    }
  }
}

// Increase request size limit for large uploads (multer handles multipart separately)
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ limit: '1mb', extended: true }));

app.use(express.static(path.join(__dirname, 'public')));

app.get('/api/mode', (req, res) => {
  res.json({ mode: TRANSCRIPTION_MODE });
});

app.post('/api/transcribe', upload.single('video'), async (req, res) => {
  let filePath = null;

  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No file uploaded' });
    }

    filePath = req.file.path;
    const fileSize = req.file.size;

    let text;

    if (TRANSCRIPTION_MODE === 'local') {
      text = await runLocalWhisper(filePath);
    } else {
      const apiKey = process.env.OPENAI_API_KEY;
      if (!apiKey) {
        return res.status(500).json({
          error:
            'OPENAI_API_KEY is not set. Add it to .env for API mode, or set TRANSCRIPTION_MODE=local for local Whisper.',
        });
      }

      const openai = new OpenAI({
        apiKey,
        timeout: 120000,
        maxRetries: 4,
      });

      if (fileSize <= WHISPER_MAX_SIZE) {
        const transcription = await transcribeFile(openai, filePath);
        text = transcription.text || '(No speech detected)';
      } else {
        text = await transcribeLargeFile(openai, filePath);
      }
    }

    res.json({ text });
  } catch (err) {
    console.error('Transcription error:', err);

    if (err.message?.toLowerCase().includes('ffmpeg') || err.message?.toLowerCase().includes('ffprobe')) {
      return res.status(500).json({
        error: 'FFmpeg is required for large videos. Install it from https://ffmpeg.org and ensure it is in your PATH.',
      });
    }
    if (err.status === 401) {
      return res.status(401).json({ error: 'Invalid API key. Check your OPENAI_API_KEY.' });
    }
    if (err.status === 429) {
      return res.status(429).json({
        error: 'Rate limit exceeded. Please wait a moment and try again.',
      });
    }
    const isConnectionError =
      err.name === 'APIConnectionError' ||
      err.cause?.code === 'ECONNRESET' ||
      err.cause?.code === 'ETIMEDOUT' ||
      err.cause?.code === 'ENOTFOUND';
    if (isConnectionError) {
      return res.status(503).json({
        error:
          'Connection to OpenAI failed. Check your internet, try again, or use a VPN if OpenAI is blocked in your region.',
      });
    }
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        error: 'File too large. Maximum size is 200 MB.',
      });
    }
    if (err.message?.includes('Invalid file type')) {
      return res.status(400).json({ error: err.message });
    }

    res.status(500).json({
      error: err.message || 'Transcription failed. Please try again.',
    });
  } finally {
    if (filePath) {
      try {
        await fs.unlink(filePath);
      } catch (e) {
        console.warn('Could not delete temp file:', e.message);
      }
    }
  }
});

// Error handler - catches Multer LIMIT_FILE_SIZE and other unhandled errors
app.use((err, req, res, next) => {
  if (err?.code === 'LIMIT_FILE_SIZE') {
    return res.status(400).json({
      error: 'File too large. Maximum size is 200 MB.',
    });
  }
  console.error('Unhandled error:', err);
  res.status(500).json({ error: err.message || 'An error occurred' });
});

app.listen(PORT, () => {
  console.log(`\n  Video-to-Text running at http://localhost:${PORT}`);
  console.log(`  Mode: ${TRANSCRIPTION_MODE === 'local' ? 'Local Whisper (Python)' : 'OpenAI API'}\n`);
});
