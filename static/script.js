async function fetchWithTimeout(url, options, timeoutMs = 20000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => {
    console.error(`[MIS-Lab] Request ke ${url} timeout setelah ${timeoutMs}ms, dibatalkan.`);
    controller.abort();
  }, timeoutMs);

  console.log(`[MIS-Lab] Mengirim request ke ${url} ...`);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    console.log(`[MIS-Lab] Response diterima dari ${url}, status: ${res.status}`);
    return res;
  } catch (err) {
    if (err.name === 'AbortError') {
      throw new Error(`Server tidak merespons dalam ${timeoutMs / 1000} detik (timeout). ` +
        `Kemungkinan ada firewall/antivirus yang memblokir koneksi ke localhost:5000, ` +
        `atau server backend berhenti merespons.`);
    }
    console.error(`[MIS-Lab] Fetch error ke ${url}:`, err);
    throw new Error(`Gagal terhubung ke server (${err.message}). Pastikan server Flask ` +
      `masih berjalan di terminal, dan tidak ada firewall/antivirus yang memblokir.`);
  } finally {
    clearTimeout(timeoutId);
  }
}

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewImg = document.getElementById('previewImg');
const runEnhanceBtn = document.getElementById('runEnhance');
const runSegmentBtn = document.getElementById('runSegment');
const loader = document.getElementById('loader');
const loaderText = document.getElementById('loaderText');

const cdfMin = document.getElementById('cdfMin');
const cdfMax = document.getElementById('cdfMax');
const clipLimit = document.getElementById('clipLimit');
const cdfMinVal = document.getElementById('cdfMinVal');
const cdfMaxVal = document.getElementById('cdfMaxVal');
const clipLimitVal = document.getElementById('clipLimitVal');

let currentFile = null;

const METHOD_LABELS = {
  original: 'Original',
  he: 'HE',
  clahe: 'CLAHE',
  he_clahe: 'HE → CLAHE',
  clahe_he: 'CLAHE → HE',
};
const BEST_METHOD = 'clahe_he';

[cdfMin, cdfMax, clipLimit].forEach((el) => {
  el.addEventListener('input', () => {
    cdfMinVal.textContent = cdfMin.value;
    cdfMaxVal.textContent = cdfMax.value;
    clipLimitVal.textContent = clipLimit.value;
  });
});

dropzone.addEventListener('click', () => fileInput.click());
dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.style.borderColor = '#3fd0c9'; });
dropzone.addEventListener('dragleave', () => { dropzone.style.borderColor = ''; });
dropzone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropzone.style.borderColor = '';
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewImg.hidden = false;
  };
  reader.readAsDataURL(file);
  runEnhanceBtn.disabled = false;
}

function showLoader(text) {
  loaderText.textContent = text;
  loader.hidden = false;
}
function hideLoader() { loader.hidden = true; }

runEnhanceBtn.addEventListener('click', async () => {
  console.log('[MIS-Lab] Tombol Enhance diklik. currentFile:', currentFile);
  if (!currentFile) { console.warn('[MIS-Lab] Tidak ada file dipilih.'); return; }
  showLoader('Menjalankan HE / CLAHE / hibrida…');

  const formData = new FormData();
  formData.append('image', currentFile);
  formData.append('cdf_min', cdfMin.value);
  formData.append('cdf_max', cdfMax.value);
  formData.append('clip_limit', clipLimit.value);

  try {
    const res = await fetchWithTimeout('/api/enhance', { method: 'POST', body: formData });
    const data = await res.json();
    console.log('[MIS-Lab] Data enhance:', data);
    if (data.error) throw new Error(data.error);
    renderLightbox(data.results);
    document.getElementById('results-panel').hidden = false;
    document.getElementById('segment-panel').hidden = false;
    document.getElementById('results-panel').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    alert('Gagal memproses citra: ' + err.message);
  } finally {
    hideLoader();
  }
});

function renderLightbox(results) {
  const lightbox = document.getElementById('lightbox');
  lightbox.innerHTML = '';
  let i = 1;
  for (const [method, payload] of Object.entries(results)) {
    const frame = document.createElement('div');
    frame.className = 'frame' + (method === BEST_METHOD ? ' best' : '');
    frame.dataset.idx = String(i).padStart(2, '0');
    frame.innerHTML = `
      <img class="thumb" src="${payload.image}" alt="${METHOD_LABELS[method]}">
      <span class="frame-label">${METHOD_LABELS[method]}${method === BEST_METHOD ? ' ★' : ''}</span>
      <img class="plot" src="${payload.hist_cdf_plot}" alt="histogram & CDF ${method}">
    `;
    lightbox.appendChild(frame);
    i++;
  }
}

runSegmentBtn.addEventListener('click', async () => {
  console.log('[MIS-Lab] Tombol Segment diklik. currentFile:', currentFile);
  if (!currentFile) { alert('Unggah citra terlebih dahulu.'); return; }
  showLoader('Menjalankan segmentasi CNN (U-Net)…');

  const method = document.getElementById('methodSelect').value;
  const gtInput = document.getElementById('gtInput');

  const formData = new FormData();
  formData.append('image', currentFile);
  formData.append('method', method);
  formData.append('cdf_min', cdfMin.value);
  formData.append('cdf_max', cdfMax.value);
  formData.append('clip_limit', clipLimit.value);
  if (gtInput.files.length) formData.append('ground_truth', gtInput.files[0]);

  try {
    const res = await fetchWithTimeout('/api/segment', { method: 'POST', body: formData }, 60000);
    const data = await res.json();
    console.log('[MIS-Lab] Data segment:', data);
    if (data.error) throw new Error(data.error);

    document.getElementById('segInput').src = data.processed_image;
    document.getElementById('segMask').src = data.mask;
    document.getElementById('segOverlay').src = data.overlay;
    document.getElementById('segmentResults').hidden = false;

    const badge = document.getElementById('modelStatusBadge');
    badge.hidden = false;
    if (data.model_status === 'trained') {
      badge.className = 'status-badge trained';
      badge.textContent = '● MODEL TERLATIH DIGUNAKAN';
    } else {
      badge.className = 'status-badge demo';
      badge.textContent = '◐ ARSITEKTUR DEMO (belum dilatih pada dataset — jalankan train.py untuk hasil akurat)';
    }

    if (data.dsc !== undefined) {
      document.getElementById('dscVal').textContent = data.dsc.toFixed(4);
      document.getElementById('ssimVal').textContent = data.ssim.toFixed(4);
      document.getElementById('metricsRow').hidden = false;
    } else {
      document.getElementById('metricsRow').hidden = true;
    }

    document.getElementById('segmentResults').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    alert('Gagal menjalankan segmentasi: ' + err.message);
  } finally {
    hideLoader();
  }
});