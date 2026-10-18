'use strict';

const $ = (id) => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const octx = overlay.getContext('2d');
const preview = $('preview');
const pctx = preview.getContext('2d');
const timeline = $('timeline');
const tctx = timeline.getContext('2d');

const S = {
  pid: null,
  project: null,
  meta: null, // {width, height, fps, frames, cuts}
  dets: null, // {off: Int32Array, box: Float32Array, tid: Int32Array}
  members: [],
  tracks: {}, // member id -> {det: Int32Array, score: Float32Array, box: Array<[x1,y1,x2,y2]|null>}
  busy: {}, // member id -> true while the server is re-tracking
  focus: null,
  opts: { format: 'vertical', framing: 'full', smoothness: 'medium', size: 1 },
  cam: null, // Float32Array of x,y,w,h per frame for the focused member
  camSize: 1, // box size the camera path was planned with (the slider rescales it live until re-planned)
  camInfo: null, // {upscale, has_pose} for the current plan
  camKey: '',
  inF: null,
  outF: null,
  frame: 0,
  hoverDet: -1,
  view: { s: 1, ox: 0, oy: 0, dpr: 1 },
  timelineDirty: true,
  showBoxes: true,
  showCrop: true,
};

// ------------------------------------------------------------------ helpers

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch { /* not json */ }
    throw new Error(msg);
  }
  return res.json();
}

let toastTimer = null;
function toast(msg, ms = 4500) {
  const t = $('toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, ms);
}

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const fmtTime = (sec) => {
  sec = Math.max(0, sec);
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
};
const frameToTime = (f) => (f + 0.5) / S.meta.fps; // middle of the frame: seeking lands on it reliably
const frameFromTime = (t) => clamp(Math.floor(t * S.meta.fps + 1e-3), 0, S.meta.frames - 1);
const member = (id) => S.members.find((m) => m.id === id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const VIEWS = ['homeView', 'analyzeView', 'editorView', 'studioView', 'captionView'];

function show(view, tab = 'clipper') {
  for (const v of VIEWS) if ($(v)) $(v).hidden = v !== view;
  $('newVideoBtn').hidden = !['analyzeView', 'editorView'].includes(view);
  if (tab !== 'clipper') $('projectName').textContent = '';
  else document.title = 'PopClip';
  for (const a of document.querySelectorAll('.tabs a')) a.classList.toggle('on', a.dataset.tab === tab);
}

// ------------------------------------------------------------------ routing

window.addEventListener('hashchange', route);
$('newVideoBtn').onclick = () => { location.hash = ''; };

function route() {
  const m = location.hash.match(/^#\/p\/([0-9a-f]{10})/);
  const tool = location.hash.match(/^#\/(split|overlay|caption)\b/);
  video.pause();
  if (window.leaveStudio) window.leaveStudio();
  if (tool && tool[1] === 'caption' && window.openCaption) window.openCaption();
  else if (tool && window.openStudio) window.openStudio(tool[1]);
  else if (m) openProject(m[1]);
  else openHome();
}

// ------------------------------------------------------------------ home + upload

async function openHome() {
  S.pid = null;
  $('projectName').textContent = '';
  show('homeView');
  $('uploadProgress').hidden = true;
  const list = await api('GET', '/api/projects').catch(() => []);
  $('recentSection').hidden = !list.length;
  $('recentList').innerHTML = '';
  for (const p of list) {
    const li = document.createElement('li');
    li.innerHTML = `<span class="name">${esc(p.name)}</span><span class="status">${esc(p.status)}</span>
      <button class="del" title="Delete">✕</button>`;
    li.onclick = () => { location.hash = `#/p/${p.id}`; };
    li.querySelector('.del').onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${p.name}" and its clips?`)) return;
      await api('DELETE', `/api/projects/${p.id}`);
      openHome();
    };
    $('recentList').appendChild(li);
  }
}

$('chooseBtn').onclick = () => $('fileInput').click();
$('fileInput').onchange = (e) => e.target.files[0] && upload(e.target.files[0]);
const drop = $('dropZone');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  const f = e.dataTransfer.files[0];
  if (f) upload(f);
});

function upload(file) {
  const fd = new FormData();
  fd.append('file', file);
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/projects');
  $('uploadProgress').hidden = false;
  xhr.upload.onprogress = (e) => {
    if (!e.lengthComputable) return;
    $('uploadBar').style.width = `${(100 * e.loaded) / e.total}%`;
    $('uploadText').textContent = `Uploading ${file.name}… ${Math.round((100 * e.loaded) / e.total)}%`;
  };
  xhr.onload = () => {
    if (xhr.status !== 200) { toast(`Upload failed: ${xhr.statusText}`); return; }
    location.hash = `#/p/${JSON.parse(xhr.responseText).id}`;
  };
  xhr.onerror = () => toast('Upload failed');
  xhr.send(fd);
}

// ------------------------------------------------------------------ project / analysis

let pollTimer = null;

async function openProject(pid) {
  clearTimeout(pollTimer);
  S.pid = pid;
  let p;
  try {
    p = await api('GET', `/api/projects/${pid}`);
  } catch (e) {
    toast(e.message);
    location.hash = '';
    return;
  }
  S.project = p;
  $('projectName').textContent = p.name;
  if (p.status === 'ready') return openEditor(p);
  show('analyzeView');
  if (p.status === 'failed') {
    $('analyzeText').innerHTML = `Analysis failed. <a href="#" id="retry">Try again</a>`;
    $('retry').onclick = async (e) => {
      e.preventDefault();
      await api('POST', `/api/projects/${pid}/reanalyze`);
      openProject(pid);
    };
    return;
  }
  const job = p.job || {};
  $('analyzeBar').style.width = `${100 * (job.progress || 0)}%`;
  $('analyzeText').textContent = job.status === 'error' ? job.message : (job.message || 'Starting…');
  if (job.status !== 'error') pollTimer = setTimeout(() => S.pid === pid && openProject(pid), 1000);
}

async function openEditor(p) {
  show('editorView');
  S.meta = p.video;
  S.members = p.members || [];
  S.tracks = {};
  S.cam = null;
  S.camInfo = null;
  S.camKey = '';
  S.inF = S.outF = null;
  S.timelineDirty = true;
  const saved = JSON.parse(localStorage.getItem(`clipper:${p.id}`) || '{}');
  Object.assign(S.opts, saved.opts || {});
  S.focus = member(saved.focus) ? saved.focus : (S.members[0] || {}).id || null;
  syncOptionButtons();
  updateRangeText();

  $('stage').style.aspectRatio = `${S.meta.width} / ${S.meta.height}`;
  $('stage').style.width = `min(100%, calc((100vh - 290px) * ${S.meta.width / S.meta.height}))`;
  video.src = `/api/projects/${p.id}/video`;

  const d = await api('GET', `/api/projects/${p.id}/detections`);
  S.dets = { off: Int32Array.from(d.offsets), box: Float32Array.from(d.boxes), tid: Int32Array.from(d.tids) };
  renderMembers();
  renderRenders(p.renders);
  layout();
  // Load every member's tracking (cached on the server after the first time).
  for (const m of S.members) await refreshTrack(m.id);
  refreshCamera();
}

function saveLocal() {
  if (S.pid) localStorage.setItem(`clipper:${S.pid}`, JSON.stringify({ focus: S.focus, opts: S.opts }));
}

// ------------------------------------------------------------------ tracks + camera

function setTrack(t) {
  S.tracks[t.member] = {
    det: Int32Array.from(t.det),
    score: Float32Array.from(t.score),
    box: t.boxes,
  };
  S.timelineDirty = true;
}

async function refreshTrack(mid) {
  if (!member(mid) || !member(mid).seeds.length) return;
  S.busy[mid] = true;
  renderMembers();
  try {
    setTrack(await api('GET', `/api/projects/${S.pid}/members/${mid}/track`));
  } catch (e) {
    toast(e.message);
  } finally {
    delete S.busy[mid];
    renderMembers();
  }
}

let camSeq = 0;
async function refreshCamera() {
  const mid = S.focus;
  updateRenderButton();
  if (!mid || !S.tracks[mid]) {
    S.cam = null;
    S.camInfo = null;
    S.camKey = '';
    updateQualityHint();
    setupPreview();
    return;
  }
  const m = member(mid);
  const q = new URLSearchParams({ member: mid, ...S.opts });
  const key = `${q}#${m.seeds}:${m.not_dets}:${m.absent_frames}`;
  if (key === S.camKey && S.cam) return;
  const seq = ++camSeq;
  const size = S.opts.size;
  $('previewNote').textContent = 'planning camera…';
  try {
    const r = await api('GET', `/api/projects/${S.pid}/camera?${q}`);
    if (seq !== camSeq) return; // a newer request superseded this one
    S.cam = Float32Array.from(r.rects);
    S.camSize = size;
    S.camInfo = r;
    S.camKey = key;
    updateQualityHint();
  } catch (e) {
    if (seq === camSeq) toast(e.message);
  } finally {
    if (seq === camSeq) $('previewNote').textContent = '';
  }
  setupPreview();
}

// While the box-size slider moves, rescale the last planned path around its centres so the preview
// reacts instantly; the exact plan replaces it a moment later.
function camRect(f) {
  const r = S.cam.subarray(4 * f, 4 * f + 4);
  const k = S.opts.size / S.camSize;
  if (Math.abs(k - 1) < 1e-3) return r;
  const aspect = r[2] / r[3];
  const hMax = Math.min(S.meta.height, S.meta.width / aspect);
  const h = clamp(r[3] * k, Math.min(40, hMax), hMax);
  const w = h * aspect;
  const x = clamp(r[0] + r[2] / 2 - w / 2, 0, S.meta.width - w);
  const y = clamp(r[1] + r[3] / 2 - h / 2, 0, S.meta.height - h);
  return [x, y, w, h];
}

function updateQualityHint() {
  const hint = $('qualityHint');
  const info = S.camInfo;
  hint.hidden = true;
  if (!info || !S.cam) return;
  if (!info.has_pose && ['face', 'middle', 'legs'].includes(S.opts.framing)) {
    hint.innerHTML = 'This video was analysed before body-part tracking was added, so this framing uses estimated '
      + 'body proportions. <a href="#" id="reanalyzeLink">Analyse it again</a> for precise framing (your members are kept).';
    hint.hidden = false;
    $('reanalyzeLink').onclick = async (e) => {
      e.preventDefault();
      if (!confirm('Analyse this video again? It takes as long as the first time.')) return;
      await api('POST', `/api/projects/${S.pid}/reanalyze`);
      openProject(S.pid);
    };
  } else if (info.upscale && info.upscale >= 3) {
    hint.textContent = `The video is enlarged about ${info.upscale}× for this framing, so the clip will look soft. `
      + 'A bigger box or a higher-resolution source looks sharper.';
    hint.hidden = false;
  }
}

// ------------------------------------------------------------------ members panel

function renderMembers() {
  const ul = $('memberList');
  ul.innerHTML = '';
  $('memberHint').hidden = S.members.length > 0;
  $('stageHint').hidden = S.members.length > 0;
  for (const m of S.members) {
    const t = S.tracks[m.id];
    let cov = '';
    if (t) {
      let n = 0;
      for (const b of t.box) if (b) n++;
      cov = `visible ${Math.round((100 * n) / t.box.length)}%`;
    }
    const fixes = m.seeds.length - 1 + m.not_dets.length + m.absent_frames.length;
    const li = document.createElement('li');
    li.className = m.id === S.focus ? 'focus' : '';
    li.style.setProperty('--mc', m.color);
    li.title = 'Click to focus the camera on this member';
    li.innerHTML = `
      <span class="chip" style="background:${m.color}"></span>
      <input class="name" value="${esc(m.name)}" aria-label="Member name" spellcheck="false">
      <button class="del" title="Remove member">✕</button>
      <div class="meta">
        ${S.busy[m.id] ? '<span class="busy">tracking…</span>' : `<span>${cov}</span>`}
        ${fixes > 0 ? `<span>${fixes} fix${fixes > 1 ? 'es' : ''}</span><button data-reset>undo fixes</button>` : ''}
        ${m.id === S.focus ? '<span style="color:var(--text)">◉ camera</span>' : ''}
      </div>`;
    li.onclick = (e) => {
      if (m.id === 'pending' || e.target.closest('input,button')) return;
      setFocus(m.id);
    };
    const input = li.querySelector('.name');
    input.onkeydown = (e) => { if (e.key === 'Enter') input.blur(); };
    input.onchange = async () => {
      const r = await api('PATCH', `/api/projects/${S.pid}/members/${m.id}`, { name: input.value });
      Object.assign(m, r.member);
    };
    li.querySelector('.del').onclick = async () => {
      if (m.id === 'pending' || !confirm(`Remove ${m.name}?`)) return;
      await api('DELETE', `/api/projects/${S.pid}/members/${m.id}`);
      S.members = S.members.filter((x) => x.id !== m.id);
      delete S.tracks[m.id];
      if (S.focus === m.id) setFocus((S.members[0] || {}).id || null);
      renderMembers();
      // Its clicks were "not this person" hints for the others: refresh them.
      for (const o of S.members) await refreshTrack(o.id);
      refreshCamera();
    };
    const reset = li.querySelector('[data-reset]');
    if (reset) reset.onclick = () => applyFix(m.id, { action: 'reset' });
    ul.appendChild(li);
  }
}

function setFocus(mid) {
  S.focus = mid;
  S.timelineDirty = true;
  saveLocal();
  renderMembers();
  refreshCamera();
}

async function createMember(name, det) {
  // Show the member right away; the first tracking pass takes a few seconds.
  const pending = { id: 'pending', name, color: '#8d93a3', seeds: [det], not_dets: [], absent_frames: [] };
  S.members.push(pending);
  S.busy.pending = true;
  renderMembers();
  try {
    const r = await api('POST', `/api/projects/${S.pid}/members`, { name, det });
    S.members[S.members.indexOf(pending)] = r.member;
    if (r.track) setTrack(r.track);
    // The new member's click can take a box away from someone else.
    for (const o of S.members) if (o.id !== r.member.id && o.seeds.includes(det)) o.seeds = o.seeds.filter((d) => d !== det);
    delete S.busy.pending;
    setFocus(r.member.id);
    await refreshOthers(r.member.id);
  } catch (e) {
    S.members = S.members.filter((m) => m !== pending);
    delete S.busy.pending;
    renderMembers();
    toast(e.message);
  }
}

async function applyFix(mid, fix) {
  S.busy[mid] = true;
  renderMembers();
  try {
    const r = await api('POST', `/api/projects/${S.pid}/members/${mid}/fix`, fix);
    Object.assign(member(mid), r.member);
    setTrack(r.track);
  } catch (e) {
    toast(e.message);
  } finally {
    delete S.busy[mid];
    renderMembers();
  }
  if (mid === S.focus) refreshCamera();
  await refreshOthers(mid);
}

async function refreshOthers(mid) {
  // Server only recomputes members that were affected; the rest return instantly from cache.
  const summary = await api('GET', `/api/projects/${S.pid}`);
  for (const m of summary.members) {
    const local = member(m.id);
    if (local) Object.assign(local, m);
  }
  for (const o of S.members) if (o.id !== mid) await refreshTrack(o.id);
  refreshCamera();
}

// ------------------------------------------------------------------ output options

function syncOptionButtons() {
  for (const seg of document.querySelectorAll('.seg')) {
    const key = seg.dataset.opt;
    for (const b of seg.querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === S.opts[key]);
  }
  $('sizeSlider').value = Math.round(S.opts.size * 100);
  $('sizeText').textContent = `${Math.round(S.opts.size * 100)}%`;
}

let sizeTimer = null;
function setSize(size) {
  S.opts.size = Math.round(clamp(size, 0.5, 2) * 100) / 100;
  syncOptionButtons();
  saveLocal();
  clearTimeout(sizeTimer);
  sizeTimer = setTimeout(refreshCamera, 350); // re-plan once the slider settles
}
$('sizeSlider').addEventListener('input', (e) => setSize(e.target.value / 100));
$('sizeDown').onclick = () => setSize(S.opts.size - 0.1);
$('sizeUp').onclick = () => setSize(S.opts.size + 0.1);

for (const seg of document.querySelectorAll('.seg')) {
  seg.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    S.opts[seg.dataset.opt] = b.dataset.v;
    syncOptionButtons();
    saveLocal();
    updateQualityHint();
    refreshCamera();
  });
}

function updateRangeText() {
  if (S.inF == null && S.outF == null) {
    $('rangeText').textContent = 'whole video';
    return;
  }
  const a = S.inF ?? 0;
  const b = S.outF ?? S.meta.frames;
  $('rangeText').textContent = `${fmtTime(a / S.meta.fps)} – ${fmtTime(b / S.meta.fps)}`;
  S.timelineDirty = true;
}

function setIn() {
  S.inF = S.frame;
  if (S.outF != null && S.outF <= S.inF) S.outF = null;
  updateRangeText();
}
function setOut() {
  S.outF = S.frame + 1;
  if (S.inF != null && S.inF >= S.outF) S.inF = null;
  updateRangeText();
}
$('setIn').onclick = setIn;
$('setOut').onclick = setOut;
$('clearRange').onclick = () => {
  S.inF = S.outF = null;
  S.timelineDirty = true;
  updateRangeText();
};

// ------------------------------------------------------------------ render

function updateRenderButton() {
  $('renderBtn').disabled = !(S.focus && S.tracks[S.focus]);
}

$('renderBtn').onclick = async () => {
  const m = member(S.focus);
  if (!m) return;
  $('renderBtn').disabled = true;
  $('renderProgress').hidden = false;
  $('renderBar').style.width = '0%';
  $('renderText').textContent = 'Starting…';
  try {
    const { job } = await api('POST', `/api/projects/${S.pid}/render`, {
      member: m.id, ...S.opts, start: S.inF ?? 0, end: S.outF ?? null,
    });
    await pollRender(job);
  } catch (e) {
    toast(e.message);
  } finally {
    $('renderProgress').hidden = true;
    updateRenderButton();
  }
};

async function pollRender(job) {
  for (;;) {
    const j = await api('GET', `/api/jobs/${job}`);
    $('renderBar').style.width = `${100 * j.progress}%`;
    $('renderText').textContent = j.status === 'queued' ? 'Waiting for another job to finish…' : j.message;
    if (j.status === 'done') {
      const p = await api('GET', `/api/projects/${S.pid}`);
      renderRenders(p.renders);
      $('rendersPanel').scrollIntoView({ behavior: 'smooth', block: 'start' });
      toast('Clip ready!');
      return;
    }
    if (j.status === 'error') throw new Error(j.message);
    await new Promise((r) => setTimeout(r, 700));
  }
}

function renderRenders(list) {
  const ul = $('renderList');
  ul.innerHTML = '';
  $('rendersPanel').hidden = !list || !list.length;
  for (const r of list || []) {
    const url = `/api/projects/${S.pid}/renders/${encodeURIComponent(r.file)}`;
    const li = document.createElement('li');
    li.innerHTML = `<video src="${url}" controls preload="metadata"></video>
      <div class="row"><span class="fname" title="${esc(r.file)}">${esc(r.file)}</span>
      <span class="muted">${(r.size / 1e6).toFixed(1)} MB</span><a href="${url}?download=1">Download</a></div>`;
    ul.appendChild(li);
  }
}

// ------------------------------------------------------------------ player controls

$('playBtn').onclick = togglePlay;
$('prevBtn').onclick = () => stepFrames(-1);
$('nextBtn').onclick = () => stepFrames(1);
$('showBoxes').onchange = (e) => { S.showBoxes = e.target.checked; };
$('showCrop').onchange = (e) => { S.showCrop = e.target.checked; };
video.addEventListener('play', () => { $('playBtn').textContent = '❚❚'; });
video.addEventListener('pause', () => { $('playBtn').textContent = '▶'; });

function togglePlay() {
  hidePopover();
  if (video.paused) video.play();
  else video.pause();
}

function seekFrame(f) {
  if (!S.meta) return;
  f = clamp(f, 0, S.meta.frames - 1);
  S.frame = f;
  video.currentTime = frameToTime(f);
}

function stepFrames(n) {
  video.pause();
  hidePopover();
  seekFrame(S.frame + n);
}

document.addEventListener('keydown', (e) => {
  if ($('editorView').hidden || e.target.closest('input,textarea')) return;
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'ArrowLeft') { e.preventDefault(); stepFrames(e.shiftKey ? -Math.round(S.meta.fps) : -1); }
  else if (e.key === 'ArrowRight') { e.preventDefault(); stepFrames(e.shiftKey ? Math.round(S.meta.fps) : 1); }
  else if (e.key === 'i' || e.key === 'I') setIn();
  else if (e.key === 'o' || e.key === 'O') setOut();
  else if (e.key === '[') setSize(S.opts.size - 0.1);
  else if (e.key === ']') setSize(S.opts.size + 0.1);
  else if (e.key === 'Escape') hidePopover();
});

// Precise frame sync: use the presented frame's timestamp where the browser offers it.
const hasRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype;
if (hasRVFC) {
  const onFrame = (_now, md) => {
    if (S.meta) S.frame = frameFromTime(md.mediaTime);
    video.requestVideoFrameCallback(onFrame);
  };
  video.requestVideoFrameCallback(onFrame);
}
video.addEventListener('seeked', () => { if (S.meta && !hasRVFC) S.frame = frameFromTime(video.currentTime); });

// ------------------------------------------------------------------ overlay drawing

function layout() {
  const r = $('stage').getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = Math.round(r.width * dpr);
  overlay.height = Math.round(r.height * dpr);
  if (S.meta) {
    const s = Math.min(r.width / S.meta.width, r.height / S.meta.height);
    S.view = { s, ox: (r.width - S.meta.width * s) / 2, oy: (r.height - S.meta.height * s) / 2, dpr };
  }
  const tr = timeline.getBoundingClientRect();
  timeline.width = Math.round(tr.width * dpr);
  timeline.height = Math.round(tr.height * dpr);
  S.timelineDirty = true;
}
new ResizeObserver(layout).observe($('stage'));
window.addEventListener('resize', layout);

const toScreen = (x, y) => [S.view.ox + x * S.view.s, S.view.oy + y * S.view.s];

function detsAt(f) {
  // People are detected on every 2nd frame; in between, show and click the nearest analysed frame.
  if (!S.dets) return [];
  for (const g of [f, f - 1, f + 1, f - 2, f + 2]) {
    if (g < 0 || g >= S.meta.frames || S.dets.off[g] === S.dets.off[g + 1]) continue;
    const out = [];
    for (let d = S.dets.off[g]; d < S.dets.off[g + 1]; d++) out.push(d);
    return out;
  }
  return [];
}
const detBox = (d) => [S.dets.box[4 * d], S.dets.box[4 * d + 1], S.dets.box[4 * d + 2], S.dets.box[4 * d + 3]];

function memberBoxesAt(f) {
  const out = [];
  for (const m of S.members) {
    const t = S.tracks[m.id];
    if (t && t.box[f]) out.push({ m, box: t.box[f], det: t.det[f], score: t.score[f] });
  }
  return out;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawOverlay() {
  const { dpr } = S.view;
  octx.setTransform(dpr, 0, 0, dpr, 0, 0);
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!S.meta || !S.dets) return;
  const f = S.frame;

  // Virtual camera window
  if (S.showCrop && S.cam && S.focus) {
    const [x, y, w, h] = camRect(f);
    const [sx, sy] = toScreen(x, y);
    const sw = w * S.view.s;
    const sh = h * S.view.s;
    const [vx, vy] = toScreen(0, 0);
    octx.fillStyle = 'rgba(0,0,0,0.5)';
    octx.beginPath();
    octx.rect(vx, vy, S.meta.width * S.view.s, S.meta.height * S.view.s);
    octx.rect(sx, sy, sw, sh);
    octx.fill('evenodd');
    octx.setLineDash([6, 4]);
    octx.strokeStyle = 'rgba(255,255,255,0.85)';
    octx.lineWidth = 1.5;
    octx.strokeRect(sx, sy, sw, sh);
    octx.setLineDash([]);
  }

  const named = memberBoxesAt(f);
  const namedDets = new Set(named.map((n) => n.det));

  if (S.showBoxes) {
    for (const d of detsAt(f)) {
      if (namedDets.has(d) && d !== S.hoverDet) continue;
      const [x1, y1, x2, y2] = detBox(d);
      const [a, b] = toScreen(x1, y1);
      const [c, e] = toScreen(x2, y2);
      octx.strokeStyle = d === S.hoverDet ? 'rgba(255,255,255,0.95)' : 'rgba(255,255,255,0.35)';
      octx.lineWidth = d === S.hoverDet ? 2.5 : 1;
      octx.strokeRect(a, b, c - a, e - b);
    }
  }

  // Named members, focused one last (on top)
  named.sort((p, q) => (p.m.id === S.focus) - (q.m.id === S.focus));
  octx.font = '600 13px system-ui, sans-serif';
  octx.textBaseline = 'middle';
  for (const { m, box } of named) {
    const focus = m.id === S.focus;
    if (!S.showBoxes && !focus) continue;
    const [a, b] = toScreen(box[0], box[1]);
    const [c, e] = toScreen(box[2], box[3]);
    octx.strokeStyle = m.color;
    octx.lineWidth = focus ? 3 : 2;
    octx.strokeRect(a, b, c - a, e - b);
    const label = m.name;
    const tw = octx.measureText(label).width + 12;
    const lx = clamp(c - tw, 0, overlay.width / S.view.dpr - tw);
    const ly = Math.max(b - 20, 0);
    octx.fillStyle = m.color;
    roundRect(octx, lx, ly, tw, 20, 4);
    octx.fill();
    octx.fillStyle = '#111';
    octx.fillText(label, lx + 6, ly + 10.5);
  }
}

// ------------------------------------------------------------------ live preview of the output

function setupPreview() {
  const vertical = S.opts.format === 'vertical';
  const dpr = window.devicePixelRatio || 1;
  const cw = vertical ? 216 : 316;
  const ch = vertical ? 384 : 178;
  preview.style.width = `${cw}px`;
  preview.style.height = `${ch}px`;
  preview.width = Math.round(cw * dpr);
  preview.height = Math.round(ch * dpr);
  $('previewEmpty').hidden = !!S.cam;
  preview.style.visibility = S.cam ? 'visible' : 'hidden';
}

function drawPreview() {
  if (!S.cam || video.readyState < 2) return;
  const f = S.frame;
  const [x, y, w, h] = camRect(f);
  const k = video.videoWidth / S.meta.width; // the player may show a smaller preview copy
  pctx.imageSmoothingQuality = 'high';
  pctx.drawImage(video, x * k, y * k, w * k, h * k, 0, 0, preview.width, preview.height);
}

// ------------------------------------------------------------------ timeline

const tlCache = document.createElement('canvas');

function drawTimelineBase() {
  const W = timeline.width;
  const H = timeline.height;
  tlCache.width = W;
  tlCache.height = H;
  const c = tlCache.getContext('2d');
  c.fillStyle = '#16181f';
  c.fillRect(0, 0, W, H);
  if (!S.meta) return;
  const N = S.meta.frames;
  const m = member(S.focus);
  const t = m && S.tracks[m.id];
  const barY = H * 0.2;
  const barH = H * 0.6;
  if (t) {
    for (let x = 0; x < W; x++) {
      const f0 = Math.floor((x / W) * N);
      const f1 = Math.max(f0 + 1, Math.floor(((x + 1) / W) * N));
      let seen = 0;
      let unsure = 0;
      for (let f = f0; f < f1; f++) {
        if (t.box[f]) {
          seen++;
          if (t.det[f] >= 0 && t.score[f] < 0.5) unsure++;
        }
      }
      const n = f1 - f0;
      c.fillStyle = seen < n / 2 ? '#3a3f4d' : unsure > seen * 0.3 ? '#fbbf24' : m.color;
      c.fillRect(x, barY, 1, barH);
    }
  } else {
    c.fillStyle = '#23262f';
    c.fillRect(0, barY, W, barH);
  }
  // camera cuts
  c.fillStyle = 'rgba(255,255,255,0.35)';
  for (const cut of S.meta.cuts) c.fillRect(Math.round((cut / N) * W), 0, 1, H * 0.2);
  // user fixes
  if (m && S.dets) {
    c.fillStyle = '#fff';
    const frames = [...m.seeds.map((d) => frameOfDet(d)), ...m.not_dets.map((d) => frameOfDet(d)), ...m.absent_frames];
    for (const f of frames) c.fillRect(Math.round((f / N) * W) - 1, barY - 4, 2, barH + 8);
  }
  // clip range
  if (S.inF != null || S.outF != null) {
    const a = ((S.inF ?? 0) / N) * W;
    const b = ((S.outF ?? N) / N) * W;
    c.fillStyle = 'rgba(0,0,0,0.6)';
    c.fillRect(0, 0, a, H);
    c.fillRect(b, 0, W - b, H);
    c.fillStyle = '#ff4d8d';
    c.fillRect(a, H - 3, b - a, 3);
  }
}

function frameOfDet(d) {
  // binary search in frame offsets
  let lo = 0;
  let hi = S.meta.frames - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (S.dets.off[mid] <= d) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function drawTimeline() {
  if (S.timelineDirty) {
    drawTimelineBase();
    S.timelineDirty = false;
  }
  tctx.clearRect(0, 0, timeline.width, timeline.height);
  if (tlCache.width && tlCache.height) tctx.drawImage(tlCache, 0, 0);
  if (!S.meta) return;
  const x = Math.round((S.frame / S.meta.frames) * timeline.width);
  tctx.fillStyle = '#fff';
  tctx.fillRect(x - 1, 0, 2 * (window.devicePixelRatio || 1), timeline.height);
}

let scrubbing = false;
function timelineSeek(e) {
  const r = timeline.getBoundingClientRect();
  seekFrame(Math.floor(((e.clientX - r.left) / r.width) * S.meta.frames));
}
timeline.addEventListener('pointerdown', (e) => {
  if (!S.meta) return;
  scrubbing = true;
  video.pause();
  hidePopover();
  timeline.setPointerCapture(e.pointerId);
  timelineSeek(e);
});
timeline.addEventListener('pointermove', (e) => { if (scrubbing) timelineSeek(e); });
timeline.addEventListener('pointerup', () => { scrubbing = false; });

// ------------------------------------------------------------------ clicking people

function detUnder(clientX, clientY) {
  const r = overlay.getBoundingClientRect();
  const x = (clientX - r.left - S.view.ox) / S.view.s;
  const y = (clientY - r.top - S.view.oy) / S.view.s;
  let best = -1;
  let bestScore = Infinity;
  for (const d of detsAt(S.frame)) {
    const [x1, y1, x2, y2] = detBox(d);
    if (x < x1 || x > x2 || y < y1 || y > y2) continue;
    // Prefer the smallest box, and the one whose centre is closest (overlapping members).
    const score = (x2 - x1) * (y2 - y1) * (1 + Math.abs(x - (x1 + x2) / 2) / (x2 - x1));
    if (score < bestScore) { bestScore = score; best = d; }
  }
  return best;
}

overlay.addEventListener('pointermove', (e) => {
  if (!$('popover').hidden) return;
  S.hoverDet = detUnder(e.clientX, e.clientY);
  overlay.classList.toggle('hovering', S.hoverDet >= 0);
});
overlay.addEventListener('pointerleave', () => { if ($('popover').hidden) S.hoverDet = -1; });

overlay.addEventListener('click', (e) => {
  if (!S.dets) return;
  if (!$('popover').hidden) { hidePopover(); return; }
  video.pause();
  const det = detUnder(e.clientX, e.clientY);
  S.hoverDet = det;
  openPopover(e, det);
});

function hidePopover() {
  $('popover').hidden = true;
  S.hoverDet = -1;
}

function openPopover(e, det) {
  const pop = $('popover');
  const f = S.frame;
  pop.innerHTML = '';
  const add = (html, cls, fn) => {
    const b = document.createElement('button');
    b.innerHTML = html;
    if (cls) b.className = cls;
    b.onclick = (ev) => { ev.stopPropagation(); fn(); };
    pop.appendChild(b);
    return b;
  };
  const sep = () => pop.appendChild(Object.assign(document.createElement('div'), { className: 'sep' }));

  if (det >= 0) {
    pop.insertAdjacentHTML('beforeend', '<h3>Who is this?</h3>');
    const named = S.members.filter((m) => m.id !== 'pending');
    const current = named.filter((m) => S.tracks[m.id] && S.tracks[m.id].det[f] === det);
    for (const m of named) {
      const mark = current.includes(m) ? ' <span class="muted">✓ tracked</span>' : '';
      add(`<span class="chip" style="background:${m.color}"></span>${esc(m.name)}${mark}`, '', () => {
        hidePopover();
        applyFix(m.id, { action: 'is', det });
      });
    }
    if (named.length) sep();
    const form = document.createElement('form');
    form.innerHTML = `<input placeholder="New member name" value="">
      <button type="submit">Add</button>`;
    form.onsubmit = (ev) => {
      ev.preventDefault();
      const name = form.querySelector('input').value.trim() || `Member ${S.members.length + 1}`;
      hidePopover();
      createMember(name, det);
    };
    form.onclick = (ev) => ev.stopPropagation();
    pop.appendChild(form);
    for (const m of current) {
      add(`Not ${esc(m.name)}`, 'danger', () => {
        hidePopover();
        applyFix(m.id, { action: 'is_not', det });
      });
    }
  } else {
    const m = member(S.focus);
    if (!m) { toast('Click directly on a person to add them.'); return; }
    pop.insertAdjacentHTML('beforeend', `<h3>${esc(m.name)}</h3>`);
    add(`<span class="chip" style="background:${m.color}"></span>${esc(m.name)} isn't visible here`, '', () => {
      hidePopover();
      applyFix(m.id, { action: 'absent', frame: f });
    });
  }

  pop.hidden = false;
  const r = $('stage').getBoundingClientRect();
  const pw = pop.offsetWidth;
  const ph = pop.offsetHeight;
  pop.style.left = `${clamp(e.clientX - r.left + 8, 4, r.width - pw - 4)}px`;
  pop.style.top = `${clamp(e.clientY - r.top + 8, 4, r.height - ph - 4)}px`;
  const input = pop.querySelector('input');
  if (input && !S.members.length) input.focus();
}

document.addEventListener('click', (e) => {
  if (!$('popover').hidden && !e.target.closest('#popover') && e.target !== overlay) hidePopover();
});

// ------------------------------------------------------------------ main loop

function loop() {
  requestAnimationFrame(loop); // schedule first: a drawing error must not stop the player
  if (S.meta && !$('editorView').hidden) {
    if (!hasRVFC || video.paused) S.frame = frameFromTime(video.currentTime);
    drawOverlay();
    drawPreview();
    drawTimeline();
    $('timeText').textContent = `${fmtTime(video.currentTime)} / ${fmtTime(S.meta.duration)}  ·  frame ${S.frame}`;
  }
}

// Tool scripts load after this file; route once they are all in.
document.addEventListener('DOMContentLoaded', route);
requestAnimationFrame(loop);
