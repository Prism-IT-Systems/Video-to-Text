const dropzone = document.getElementById('dropzone');
const dropzoneContent = document.getElementById('dropzoneContent');
const dropzoneLoading = document.getElementById('dropzoneLoading');
const fileInput = document.getElementById('fileInput');
const resultSection = document.getElementById('resultSection');
const errorSection = document.getElementById('errorSection');
const transcriptEl = document.getElementById('transcript');
const copyBtn = document.getElementById('copyBtn');
const retryBtn = document.getElementById('retryBtn');
const errorMessage = document.getElementById('errorMessage');
const modeBadge = document.getElementById('modeBadge');

fetch('/api/mode')
  .then((r) => r.json())
  .then((d) => (modeBadge.textContent = d.mode === 'local' ? 'Local mode' : 'API mode'))
  .catch(() => (modeBadge.textContent = ''));

function hideAll() {
  resultSection.hidden = true;
  errorSection.hidden = true;
}

function showLoading(show) {
  dropzoneContent.hidden = show;
  dropzoneLoading.hidden = !show;
}

function showResult(text) {
  hideAll();
  transcriptEl.textContent = text || '(No speech detected)';
  resultSection.hidden = false;
}

function showError(msg) {
  hideAll();
  errorMessage.textContent = msg;
  errorSection.hidden = false;
}

async function transcribe(file) {
  hideAll();
  showLoading(true);

  const formData = new FormData();
  formData.append('video', file);

  try {
    const res = await fetch('/api/transcribe', {
      method: 'POST',
      body: formData,
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || 'Transcription failed');
    }

    showResult(data.text);
  } catch (err) {
    showError(err.message);
  } finally {
    showLoading(false);
  }
}

// Click to browse
dropzone.addEventListener('click', (e) => {
  if (!e.target.closest('.dropzone-loading')) {
    fileInput.click();
  }
});

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (file) transcribe(file);
  fileInput.value = '';
});

// Drag and drop
dropzone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropzone.classList.add('dragover');
});

dropzone.addEventListener('dragleave', () => {
  dropzone.classList.remove('dragover');
});

dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) transcribe(file);
});

copyBtn.addEventListener('click', async () => {
  const text = transcriptEl.textContent;
  try {
    await navigator.clipboard.writeText(text);
    copyBtn.textContent = 'Copied!';
    setTimeout(() => (copyBtn.textContent = 'Copy text'), 2000);
  } catch {
    showError('Could not copy to clipboard');
  }
});

retryBtn.addEventListener('click', () => {
  errorSection.hidden = true;
});
