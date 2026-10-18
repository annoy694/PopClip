'use strict';
// Split Screen and Overlay. The preview mirrors clipper/compose.py exactly: cropRect, clipFrames,
// outputFrames, splitGeometry and the seam curve, so what you see is what gets rendered.

(() => {
  const FPS = 60;
  const SHAPES = { '16:9': [1920, 1080], '9:16': [1080, 1920], '1:1': [1080, 1080], '4:5': [1080, 1350], '21:9': [2560, 1080] };
  const CANVAS_MODE = {
    normal: 'source-over', screen: 'screen', multiply: 'multiply', overlay: 'overlay',
    lighten: 'lighten', darken: 'darken', difference: 'difference', add: 'lighter',
  };
  const NAMES = {
    split: { title: 'Split Screen', layers: ['Left side', 'Right side'], hide: 'Black' },
    overlay: { title: 'Overlay', layers: ['Bottom layer', 'Top layer'], hide: 'Hide' },
  };

  const canvas = $('stCanvas');
  const ctx = canvas.getContext('2d');
  const states = {}; // layout -> state
  let S = null; // current state
  let loopRunning = false;

  // ---------------------------------------------------------------- maths shared with compose.py

  const even = (v) => Math.round(v / 2) * 2;

  function cropRect(sw, sh, bw, bh, px, py, zoom) {
    const s = Math.max(bw / sw, bh / sh) * zoom;
    const w = Math.min(sw, Math.max(1, Math.round(bw / s)));
    const h = Math.min(sh, Math.max(1, Math.round(bh / s)));
    return [Math.round((sw - w) * px), Math.round((sh - h) * py), w, h];
  }

  const isVideo = (L) => L.media && L.media.kind === 'video';
  const clipFrames = (L) => Math.max(1, Math.round((L.trimEnd - L.trimStart) * FPS));

  function decidingLayer(st) {
    let best = null;
    st.layers.forEach((L, i) => {
      if (isVideo(L) && L.onEnd !== 'loop' && (best === null || clipFrames(L) < clipFrames(st.layers[best]))) best = i;
    });
    return best;
  }

  function outputFrames(st) {
    const d = decidingLayer(st);
    if (st.autoDuration && d !== null) return Math.max(FPS, clipFrames(st.layers[d]));
    return Math.round(st.duration * FPS);
  }

  function splitGeometry(width, blend, fade) {
    const half = width / 2;
    const b = clamp(blend, 0, 0.9 * half);
    if (fade === 'left') return [Math.round(half), Math.round(half + b)];
    if (fade === 'right') return [Math.round(half - b), Math.round(half)];
    return [Math.round(half - b / 2), Math.round(half + b / 2)];
  }

  function layerBoxes(st) {
    if (st.layout === 'split') {
      const [z0, z1] = splitGeometry(st.width, st.blend, st.fade);
      return [[0, 0, z1, st.height], [z0, 0, st.width - z0, st.height]];
    }
    return [[0, 0, st.width, st.height], [0, 0, st.width, st.height]];
  }

  const smoothstep = (t) => t * t * (3 - 2 * t);

  /** Which moment of a layer shows at output frame k: {show, time} (time in the source video). */
  function layerAt(L, k) {
    if (!isVideo(L)) return { show: !!L.media, time: 0 };
    const n = clipFrames(L);
    if (k >= n && L.onEnd !== 'loop') {
      return L.onEnd === 'hide' ? { show: false, time: 0 } : { show: true, time: L.trimStart + (n - 1) / FPS, frozen: true };
    }
    return { show: true, time: L.trimStart + (k % n) / FPS };
  }

  // ---------------------------------------------------------------- state

  function newLayer() {
    return { media: null, el: null, posX: 0.5, posY: 0.5, zoom: 1, trimStart: 0, trimEnd: 0, onEnd: 'freeze', volume: 1, scrub: 0 };
  }

  function defaults(layout) {
    return {
      layout, shape: '16:9', width: 1920, height: 1080, autoDuration: true, duration: 10,
      blend: 240, fade: 'both', opacity: 0.5, mode: 'normal',
      layers: [newLayer(), newLayer()], t: 0, playing: false, clock: 0, job: null,
    };
  }

  function save() {
    if (!S) return;
    const keep = { ...S, playing: false, job: null, layers: S.layers.map((L) => ({ ...L, el: null, media: L.media ? L.media.id : null })) };
    try { localStorage.setItem(`popclip:${S.layout}`, JSON.stringify(keep)); } catch { /* storage full or blocked */ }
  }

  async function restore(layout) {
    const st = defaults(layout);
    let saved = null;
    try { saved = JSON.parse(localStorage.getItem(`popclip:${layout}`) || 'null'); } catch { /* ignore */ }
    if (!saved) return st;
    Object.assign(st, saved, { playing: false, job: null, t: 0 });
    st.layers = await Promise.all(saved.layers.map(async (l) => {
      const L = { ...newLayer(), ...l, el: null, media: null };
      if (!l.media) return L;
      try {
        L.media = await api('GET', `/api/media/${l.media}`);
        await attachElement(L);
      } catch {
        return newLayer(); // the upload was deleted
      }
      return L;
    }));
    return st;
  }

  // ---------------------------------------------------------------- open / close

  window.openStudio = async (layout) => {
    show('studioView', layout);
    document.title = `${NAMES[layout].title} · PopClip`;
    if (!states[layout]) {
      states[layout] = defaults(layout); // placeholder while uploads are re-attached
      S = states[layout];
      buildUI();
      states[layout] = await restore(layout);
    }
    S = states[layout];
    buildUI();
    if (!loopRunning) {
      loopRunning = true;
      requestAnimationFrame(frameLoop);
    }
  };

  window.leaveStudio = () => {
    for (const st of Object.values(states)) {
      st.playing = false;
      for (const L of st.layers) if (L.el && L.el.pause) L.el.pause();
    }
  };

  // ---------------------------------------------------------------- media

  function attachElement(L) {
    return new Promise((resolve, reject) => {
      const m = L.media;
      if (m.kind === 'image') {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('Could not load the picture'));
        img.src = m.url;
        L.el = img;
        return;
      }
      const v = document.createElement('video');
      v.muted = true;
      v.playsInline = true;
      v.preload = 'auto';
      v.crossOrigin = 'anonymous';
      v.onloadeddata = () => resolve(v);
      v.onerror = () => reject(new Error('This browser cannot preview the video'));
      v.src = `${m.url}?v=${m.preview_ready ? 1 : 0}`;
      L.el = v;
    });
  }

  async function waitForPreview(L) {
    while (L.media && !L.media.preview_ready) {
      await new Promise((r) => setTimeout(r, 1500));
      if (!L.media) return;
      L.media = await api('GET', `/api/media/${L.media.id}`);
    }
  }

  function uploadFile(file, idx) {
    const L = S.layers[idx];
    const card = $(`stLayer${idx}`);
    const fd = new FormData();
    fd.append('file', file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/media');
    const bar = card.querySelector('.lc-uploading');
    card.querySelector('.lc-empty').hidden = true;
    card.querySelector('.lc-filled').hidden = false;
    card.querySelector('.fname').textContent = `Uploading ${file.name}…`;
    bar.hidden = false;
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) bar.querySelector('span').style.width = `${(100 * e.loaded) / e.total}%`;
    };
    xhr.onload = async () => {
      bar.hidden = true;
      if (xhr.status !== 200) {
        let msg = xhr.statusText;
        try { msg = JSON.parse(xhr.responseText).detail || msg; } catch { /* not json */ }
        toast(`Upload failed: ${msg}`);
        renderCard(idx);
        return;
      }
      const info = JSON.parse(xhr.responseText);
      if (L.el && L.el.pause) L.el.pause();
      // Replacing a file resets its framing and trim.
      Object.assign(L, newLayer(), { media: info, volume: defaultVolume(info, idx) });
      L.trimEnd = info.duration;
      renderCard(idx);
      try {
        if (!info.preview_ready) {
          card.querySelector('.lc-prep').hidden = false;
          await waitForPreview(L);
          card.querySelector('.lc-prep').hidden = true;
        }
        await attachElement(L);
      } catch (e) {
        toast(e.message);
      }
      restart();
      refreshAll();
    };
    xhr.onerror = () => { toast('Upload failed'); renderCard(idx); };
    xhr.send(fd);
  }

  /** Only one layer is heard at first: the first video with sound. */
  function defaultVolume(info, idx) {
    if (info.kind !== 'video' || !info.has_audio) return 1;
    const other = S.layers[1 - idx];
    return other.media && other.media.has_audio && other.volume > 0 ? 0 : 1;
  }

  // ---------------------------------------------------------------- UI building

  const segValue = (seg, v) => {
    for (const b of seg.querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === v);
  };

  function bindPair(range, num, get, set, fmt = (v) => v) {
    range.oninput = () => { set(parseFloat(range.value)); num.value = fmt(get()); changed(); };
    num.onchange = () => { set(parseFloat(num.value)); range.value = get(); num.value = fmt(get()); changed(); };
  }

  let built = false;
  function buildUI() {
    for (const idx of [0, 1]) buildCard(idx);
    if (!built) {
      built = true;
      bindGlobalControls();
    }
    syncGlobalControls();
    renderCard(0);
    renderCard(1);
    refreshAll();
  }

  function bindGlobalControls() {
    $('stShape').onclick = (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      S.shape = b.dataset.v;
      if (SHAPES[S.shape]) [S.width, S.height] = SHAPES[S.shape];
      syncGlobalControls();
      changed();
    };
    bindPair($('stW'), $('stWn'), () => S.width, (v) => { S.width = even(clamp(v || 320, 320, 3840)); });
    bindPair($('stH'), $('stHn'), () => S.height, (v) => { S.height = even(clamp(v || 320, 320, 3840)); });
    $('stDurMode').onclick = (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      S.autoDuration = b.dataset.v === 'auto';
      syncGlobalControls();
      changed();
    };
    bindPair($('stDur'), $('stDurN'), () => S.duration, (v) => { S.duration = Math.round(clamp(v || 1, 1, 120) * 2) / 2; });
    bindPair($('stBlend'), $('stBlendN'), () => S.blend, (v) => { S.blend = Math.round(clamp(v || 0, 0, maxBlend())); });
    $('stFade').onclick = (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      S.fade = b.dataset.v;
      segValue($('stFade'), S.fade);
      changed();
    };
    bindPair($('stOpacity'), $('stOpacityN'), () => Math.round(S.opacity * 100), (v) => { S.opacity = clamp(v || 0, 0, 100) / 100; });
    $('stMode').onchange = () => { S.mode = $('stMode').value; changed(); };
    $('stSwap').onclick = () => {
      S.layers.reverse();
      renderCard(0);
      renderCard(1);
      changed();
    };
    $('stPlay').onclick = togglePlay;
    $('stScrub').oninput = () => {
      S.playing = false;
      S.t = parseFloat($('stScrub').value);
      updatePlayButton();
    };
    $('stSound').onchange = () => { S.sound = $('stSound').checked; };
    $('stRender').onclick = startRender;
    bindStageDrag();
    document.addEventListener('keydown', (e) => {
      if ($('studioView').hidden || e.target.closest('input,select,textarea')) return;
      if (e.key === ' ') { e.preventDefault(); togglePlay(); }
    });
  }

  const maxBlend = () => Math.floor(0.9 * (S.width / 2));

  function syncGlobalControls() {
    const split = S.layout === 'split';
    $('stSeamPanel').hidden = !split;
    $('stTopPanel').hidden = split;
    segValue($('stShape'), S.shape);
    $('stCustom').hidden = S.shape !== 'custom';
    $('stCustomLabel').textContent = S.shape === 'custom' ? `${S.width}×${S.height}` : 'any size';
    for (const [r, n, v] of [['stW', 'stWn', S.width], ['stH', 'stHn', S.height]]) { $(r).value = v; $(n).value = v; }
    segValue($('stDurMode'), S.autoDuration ? 'auto' : 'manual');
    $('stDur').value = S.duration;
    $('stDurN').value = S.duration;
    $('stBlend').max = maxBlend();
    $('stBlendN').max = maxBlend();
    S.blend = Math.min(S.blend, maxBlend());
    $('stBlend').value = S.blend;
    $('stBlendN').value = S.blend;
    segValue($('stFade'), S.fade);
    $('stOpacity').value = Math.round(S.opacity * 100);
    $('stOpacityN').value = Math.round(S.opacity * 100);
    $('stMode').value = S.mode;
    $('stSound').checked = !!S.sound;
    $('stHint').textContent = split
      ? 'Drag a side of the preview to move its framing, scroll to zoom. The preview uses the same framing and seam maths as the rendered video.'
      : 'Drag the preview to move the top layer, scroll to zoom it. The preview uses the same framing and blending maths as the rendered video.';
  }

  function buildCard(idx) {
    const card = $(`stLayer${idx}`);
    if (card.dataset.built) return;
    card.dataset.built = '1';
    card.appendChild($('layerCardTpl').content.cloneNode(true));
    const q = (sel) => card.querySelector(sel);
    const L = () => S.layers[idx];
    const fileInput = q('.lc-file');
    q('.lc-choose').onclick = () => fileInput.click();
    q('.lc-replace').onclick = () => fileInput.click();
    fileInput.onchange = () => { if (fileInput.files[0]) uploadFile(fileInput.files[0], idx); fileInput.value = ''; };
    card.addEventListener('dragover', (e) => { e.preventDefault(); card.classList.add('over'); });
    card.addEventListener('dragleave', () => card.classList.remove('over'));
    card.addEventListener('drop', (e) => {
      e.preventDefault();
      card.classList.remove('over');
      if (e.dataTransfer.files[0]) uploadFile(e.dataTransfer.files[0], idx);
    });
    q('.lc-clear').onclick = () => {
      if (L().el && L().el.pause) L().el.pause();
      S.layers[idx] = newLayer();
      renderCard(idx);
      changed();
    };
    bindPair(q('.lc-zoom'), q('.lc-zoomN'), () => L().zoom, (v) => { L().zoom = Math.round(clamp(v || 1, 1, 4) * 20) / 20; });
    bindPair(q('.lc-px'), q('.lc-pxN'), () => Math.round(L().posX * 1000) / 10, (v) => { L().posX = clamp(v || 0, 0, 100) / 100; });
    bindPair(q('.lc-py'), q('.lc-pyN'), () => Math.round(L().posY * 1000) / 10, (v) => { L().posY = clamp(v || 0, 0, 100) / 100; });
    q('.lc-reset').onclick = () => {
      Object.assign(L(), { posX: 0.5, posY: 0.5, zoom: 1 });
      renderCard(idx);
      changed();
    };
    // Trim: find a moment with the scrubber, or type exact times.
    q('.lc-scrub').oninput = () => {
      S.playing = false;
      updatePlayButton();
      L().scrub = parseFloat(q('.lc-scrub').value);
      L().scrubbing = true;
      if (L().el) L().el.currentTime = L().scrub;
      q('.lc-scrubtime').textContent = fmtTime(L().scrub);
    };
    q('.lc-setstart').onclick = () => setTrim(idx, L().scrub, L().trimEnd);
    q('.lc-setend').onclick = () => setTrim(idx, L().trimStart, L().scrub);
    q('.lc-ts').onchange = () => setTrim(idx, parseFloat(q('.lc-ts').value) || 0, L().trimEnd);
    q('.lc-te').onchange = () => setTrim(idx, L().trimStart, parseFloat(q('.lc-te').value) || 0);
    q('.lc-end').onclick = (e) => {
      const b = e.target.closest('button');
      if (!b) return;
      L().onEnd = b.dataset.v;
      segValue(q('.lc-end'), L().onEnd);
      restart();
      changed();
    };
    bindPair(q('.lc-vol'), q('.lc-volN'), () => Math.round(L().volume * 100), (v) => { L().volume = clamp(v || 0, 0, 150) / 100; });
    bindFrameViewDrag(idx, q('.frame-view'));
  }

  function setTrim(idx, start, end) {
    const L = S.layers[idx];
    const dur = L.media.duration;
    start = Math.round(clamp(start, 0, Math.max(0, dur - 0.1)) * 20) / 20;
    end = Math.round(clamp(end, 0, dur) * 20) / 20;
    if (end - start < 0.1) {
      if (end === L.trimEnd) end = Math.min(dur, start + 0.1);
      else start = Math.max(0, end - 0.1);
    }
    L.trimStart = start;
    L.trimEnd = Math.min(dur, Math.max(end, start + 0.1));
    L.scrubbing = false;
    renderCard(idx);
    restart(); // changing the trim restarts the preview at the selected beginning
    changed();
  }

  function renderCard(idx) {
    const card = $(`stLayer${idx}`);
    const q = (sel) => card.querySelector(sel);
    const L = S.layers[idx];
    const names = NAMES[S.layout];
    q('.lc-title').textContent = names.layers[idx];
    q('.lc-hide-label').textContent = names.hide;
    const m = L.media;
    q('.lc-empty').hidden = !!m;
    q('.lc-filled').hidden = !m;
    q('.lc-meta').textContent = '';
    if (!m) return;
    q('.lc-meta').textContent = m.kind === 'video'
      ? ` · video ${fmtTime(m.duration)} · ${m.width}×${m.height}${m.has_audio ? ' · sound' : ''}`
      : ` · picture ${m.width}×${m.height}`;
    q('.fname').textContent = m.name;
    q('.fname').title = m.name;
    q('.lc-prep').hidden = !!m.preview_ready;
    for (const [r, n, v] of [['.lc-zoom', '.lc-zoomN', L.zoom], ['.lc-px', '.lc-pxN', Math.round(L.posX * 1000) / 10], ['.lc-py', '.lc-pyN', Math.round(L.posY * 1000) / 10]]) {
      q(r).value = v;
      q(n).value = v;
    }
    q('.lc-video').hidden = m.kind !== 'video';
    if (m.kind === 'video') {
      q('.lc-scrub').max = m.duration;
      q('.lc-scrub').value = L.scrub;
      q('.lc-scrubtime').textContent = fmtTime(L.scrub);
      q('.lc-ts').value = L.trimStart.toFixed(2);
      q('.lc-te').value = L.trimEnd.toFixed(2);
      q('.lc-ts').max = m.duration;
      q('.lc-te').max = m.duration;
      const sel = q('.trim-sel');
      sel.style.left = `${(100 * L.trimStart) / m.duration}%`;
      sel.style.width = `${(100 * (L.trimEnd - L.trimStart)) / m.duration}%`;
      segValue(q('.lc-end'), L.onEnd);
      q('.lc-sound').hidden = !m.has_audio;
      q('.lc-nosound').hidden = m.has_audio;
      q('.lc-vol').value = Math.round(L.volume * 100);
      q('.lc-volN').value = Math.round(L.volume * 100);
    }
  }

  // ---------------------------------------------------------------- notes, duration, output info

  function refreshAll() {
    const total = outputFrames(S);
    const D = total / FPS;
    const d = decidingLayer(S);
    const names = NAMES[S.layout].layers;
    $('stDurRow').hidden = S.autoDuration && d !== null;
    if (S.autoDuration && d !== null) {
      $('stDurNote').textContent = `${D.toFixed(2)} s, the length of the ${names[d].toLowerCase()}'s trimmed clip (the shortest one that doesn't loop).`;
    } else if (S.autoDuration) {
      $('stDurNote').textContent = 'No video decides the length (only pictures or looping clips), so it is set manually.';
    } else {
      $('stDurNote').textContent = '';
    }
    S.layers.forEach((L, idx) => {
      const note = $(`stLayer${idx}`).querySelector('.lc-note');
      if (!isVideo(L)) return;
      const P = clipFrames(L) / FPS;
      note.classList.remove('warn');
      if (L.onEnd === 'loop') {
        note.textContent = P < D - 1e-6 ? `Clip ${P.toFixed(2)} s · repeats ${(D / P).toFixed(1)}× in the ${D.toFixed(2)} s video.` : `Clip ${P.toFixed(2)} s · fills the whole ${D.toFixed(2)} s video.`;
      } else if (P < D - 1e-6) {
        const what = L.onEnd === 'freeze' ? 'freezes on its last frame' : (S.layout === 'split' ? 'goes black' : 'disappears');
        note.textContent = `Clip ${P.toFixed(2)} s is shorter than the ${D.toFixed(2)} s video: it ${what} for the last ${(D - P).toFixed(2)} s.`;
        note.classList.add('warn');
      } else if (P > D + 1e-6) {
        note.textContent = `Clip ${P.toFixed(2)} s · only the first ${D.toFixed(2)} s are used.`;
      } else {
        note.textContent = `Clip ${P.toFixed(2)} s · matches the video length.`;
      }
    });
    const both = S.layers.every((L) => L.media);
    const sound = S.layers.some((L) => isVideo(L) && L.media.has_audio && L.volume > 0);
    $('stOutInfo').textContent = `${S.width} × ${S.height} · 60 fps · ${D.toFixed(2)} s · ${sound ? 'with sound' : 'no sound'}`;
    $('stRender').disabled = !both || !!S.job;
    $('stRender').title = both ? '' : 'Add something to both sides first';
    $('stEmpty').hidden = S.layers.some((L) => L.media);
    $('stScrub').max = D;
    layoutCanvas();
  }

  function changed() {
    syncGlobalControls();
    refreshAll();
    save();
  }

  function restart() {
    S.t = 0;
    S.clock = performance.now();
    for (const L of S.layers) L.scrubbing = false;
  }

  // ---------------------------------------------------------------- transport

  function togglePlay() {
    S.playing = !S.playing;
    S.clock = performance.now() - S.t * 1000;
    for (const L of S.layers) L.scrubbing = false;
    updatePlayButton();
  }

  function updatePlayButton() {
    $('stPlay').textContent = S.playing ? '❚❚' : '▶';
  }

  /** Keep each video at the moment the shared playhead asks for. */
  function syncVideos(k) {
    for (const L of S.layers) {
      if (!isVideo(L) || !L.el || L.el.readyState < 1) continue;
      const v = L.el;
      if (L.scrubbing) continue; // the layer's own trim scrubber is in charge
      const at = layerAt(L, k);
      v.muted = !S.sound || L.volume <= 0;
      v.volume = clamp(L.volume, 0, 1);
      const wantPlay = S.playing && at.show && !at.frozen;
      if (wantPlay) {
        if (v.paused) v.play().catch(() => {});
        if (Math.abs(v.currentTime - at.time) > 0.15 && !v.seeking) seek(L, at.time);
      } else {
        if (!v.paused) v.pause();
        // Seek only when the wanted moment changes: decoded frame times never match 60 fps exactly.
        if (Math.abs((L.target ?? -1) - at.time) > 0.5 / FPS) seek(L, at.time);
      }
    }
  }

  function seek(L, time) {
    L.target = time;
    L.el.currentTime = time;
  }

  // ---------------------------------------------------------------- drawing

  let stamp = 0; // animation frame counter, so each video frame is cached once per frame

  /** What to draw for a layer: the video while it has a frame, else the last good frame. */
  function layerSource(L) {
    const el = L.el;
    if (!el) return null;
    if (!isVideo(L)) return el.complete && el.naturalWidth ? el : null;
    if (el.readyState >= 2 && !el.seeking) {
      if (L.cacheStamp !== stamp) {
        L.cache = L.cache || document.createElement('canvas');
        const w = Math.min(el.videoWidth, 1280);
        const h = Math.round((w * el.videoHeight) / el.videoWidth);
        if (L.cache.width !== w || L.cache.height !== h) { L.cache.width = w; L.cache.height = h; }
        L.cache.getContext('2d').drawImage(el, 0, 0, w, h);
        L.cacheStamp = stamp;
      }
      return el;
    }
    return L.cache && L.cache.width ? L.cache : null;
  }

  const pixelWidth = (src) => src.videoWidth || src.naturalWidth || src.width;

  function layoutCanvas() {
    const stage = $('stStage');
    const maxW = stage.clientWidth || 800;
    const maxH = Math.max(240, window.innerHeight - 300);
    const scale = Math.min(maxW / S.width, maxH / S.height);
    const cssW = Math.round(S.width * scale);
    const cssH = Math.round(S.height * scale);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
  }
  window.addEventListener('resize', () => { if (S && !$('studioView').hidden) layoutCanvas(); });

  const srcSize = (L) => [L.media.width, L.media.height];

  /** Draw one layer's crop into the box (output pixels) on ctx scaled by k. */
  function drawLayer(c, L, src, box, k) {
    const [bx, by, bw, bh] = box;
    const [sw, sh] = srcSize(L);
    const [x, y, w, h] = cropRect(sw, sh, bw, bh, L.posX, L.posY, L.zoom);
    // The browser may show a smaller copy (preview file or frame cache): scale the crop into its pixels.
    const f = pixelWidth(src) / sw;
    c.drawImage(src, x * f, y * f, w * f, h * f, bx * k, by * k, bw * k, bh * k);
  }

  const seamCanvas = document.createElement('canvas');
  const seamCtx = seamCanvas.getContext('2d');

  function drawComposite(c, k, frame) {
    c.save();
    c.fillStyle = '#000';
    c.fillRect(0, 0, S.width * k, S.height * k);
    c.imageSmoothingQuality = 'high';
    const boxes = layerBoxes(S);
    const srcs = S.layers.map((L) => (L.media && layerAt(L, frame).show ? layerSource(L) : null));
    if (S.layout === 'split') {
      const [left, right] = S.layers;
      const [z0, z1] = splitGeometry(S.width, S.blend, S.fade);
      if (srcs[1]) drawLayer(c, right, srcs[1], boxes[1], k);
      if (srcs[0]) {
        // Left side drawn off-screen, then faded across the seam with the same smoothstep curve.
        seamCanvas.width = Math.max(1, Math.round(z1 * k));
        seamCanvas.height = Math.max(1, Math.round(S.height * k));
        seamCtx.imageSmoothingQuality = 'high';
        drawLayer(seamCtx, left, srcs[0], boxes[0], k);
        if (z1 > z0) {
          const g = seamCtx.createLinearGradient(z0 * k, 0, z1 * k, 0);
          for (let i = 0; i <= 32; i++) g.addColorStop(i / 32, `rgba(0,0,0,${1 - smoothstep(i / 32)})`);
          seamCtx.globalCompositeOperation = 'destination-in';
          seamCtx.fillStyle = g;
          seamCtx.fillRect(0, 0, seamCanvas.width, seamCanvas.height);
          seamCtx.globalCompositeOperation = 'source-over';
        }
        c.drawImage(seamCanvas, 0, 0);
      }
    } else {
      const [bottom, top] = S.layers;
      if (srcs[0]) drawLayer(c, bottom, srcs[0], boxes[0], k);
      if (srcs[1] && S.opacity > 0) {
        c.globalAlpha = S.opacity;
        c.globalCompositeOperation = CANVAS_MODE[S.mode];
        drawLayer(c, top, srcs[1], boxes[1], k);
      }
    }
    c.restore();
  }

  function drawFrameView(idx) {
    const L = S.layers[idx];
    const cv = $(`stLayer${idx}`).querySelector('.frame-view');
    const src = L.media ? layerSource(L) : null;
    if (!src) return;
    const [sw, sh] = srcSize(L);
    const cssW = cv.clientWidth || 280;
    const scale = Math.min(cssW / sw, 220 / sh);
    const w = Math.round(sw * scale);
    const h = Math.round(sh * scale);
    const dpr = window.devicePixelRatio || 1;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
      cv.style.height = `${h}px`;
    }
    const c = cv.getContext('2d');
    const s = scale * dpr;
    c.drawImage(src, 0, 0, cv.width, cv.height);
    const [, , bw, bh] = layerBoxes(S)[idx];
    const [x, y, cw, ch] = cropRect(sw, sh, bw, bh, L.posX, L.posY, L.zoom);
    c.fillStyle = 'rgba(0,0,0,0.55)';
    c.beginPath();
    c.rect(0, 0, cv.width, cv.height);
    c.rect(x * s, y * s, cw * s, ch * s);
    c.fill('evenodd');
    c.strokeStyle = '#ff4d8d';
    c.lineWidth = 2 * dpr;
    c.strokeRect(x * s, y * s, cw * s, ch * s);
    L.viewScale = s / dpr;
  }

  function frameLoop(now) {
    requestAnimationFrame(frameLoop);
    if (!S || $('studioView').hidden) return;
    stamp++;
    const total = outputFrames(S);
    const D = total / FPS;
    if (S.playing) {
      S.t = (now - S.clock) / 1000;
      if (S.t >= D) { // the preview repeats the whole video
        S.t = 0;
        S.clock = now;
      }
    }
    S.t = clamp(S.t, 0, D);
    const frame = Math.min(total - 1, Math.floor(S.t * FPS + 1e-6));
    syncVideos(frame);
    const k = canvas.width / S.width;
    drawComposite(ctx, k, frame);
    drawFrameView(0);
    drawFrameView(1);
    if (document.activeElement !== $('stScrub')) $('stScrub').value = S.t;
    $('stTime').textContent = `${fmtTime(S.t)} / ${fmtTime(D)}`;
  }

  // ---------------------------------------------------------------- dragging to reframe

  /** Move a layer's crop by a drag of (dx, dy) source pixels. */
  function panLayer(idx, dxSrc, dySrc) {
    const L = S.layers[idx];
    const [sw, sh] = srcSize(L);
    const [, , bw, bh] = layerBoxes(S)[idx];
    const [, , cw, ch] = cropRect(sw, sh, bw, bh, L.posX, L.posY, L.zoom);
    if (sw > cw) L.posX = clamp(L.posX + dxSrc / (sw - cw), 0, 1);
    if (sh > ch) L.posY = clamp(L.posY + dySrc / (sh - ch), 0, 1);
    renderCard(idx);
  }

  function zoomLayer(idx, delta) {
    const L = S.layers[idx];
    L.zoom = Math.round(clamp(L.zoom + delta, 1, 4) * 20) / 20;
    renderCard(idx);
    changed();
  }

  function bindFrameViewDrag(idx, cv) {
    let last = null;
    cv.addEventListener('pointerdown', (e) => { if (S.layers[idx].media) { last = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); } });
    cv.addEventListener('pointermove', (e) => {
      if (!last) return;
      const s = S.layers[idx].viewScale || 1;
      panLayer(idx, (e.clientX - last[0]) / s, (e.clientY - last[1]) / s); // drag the window over the source
      last = [e.clientX, e.clientY];
    });
    cv.addEventListener('pointerup', () => { if (last) { last = null; changed(); } });
    cv.addEventListener('wheel', (e) => { if (S.layers[idx].media) { e.preventDefault(); zoomLayer(idx, e.deltaY < 0 ? 0.05 : -0.05); } }, { passive: false });
  }

  function layerUnder(e) {
    if (S.layout === 'overlay') return S.layers[1].media ? 1 : 0;
    const r = canvas.getBoundingClientRect();
    return (e.clientX - r.left) / r.width < 0.5 ? 0 : 1;
  }

  function bindStageDrag() {
    let drag = null;
    canvas.addEventListener('pointerdown', (e) => {
      const idx = layerUnder(e);
      if (!S.layers[idx].media) return;
      drag = { idx, x: e.clientX, y: e.clientY };
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const L = S.layers[drag.idx];
      const [sw, sh] = srcSize(L);
      const [, , bw, bh] = layerBoxes(S)[drag.idx];
      const [, , cw] = cropRect(sw, sh, bw, bh, L.posX, L.posY, L.zoom);
      const r = canvas.getBoundingClientRect();
      const srcPerScreen = (cw / bw) * (S.width / r.width); // screen px -> output px -> source px
      // Dragging the picture moves its content with the pointer, so the crop moves the other way.
      panLayer(drag.idx, -(e.clientX - drag.x) * srcPerScreen, -(e.clientY - drag.y) * srcPerScreen);
      drag.x = e.clientX;
      drag.y = e.clientY;
    });
    canvas.addEventListener('pointerup', () => { if (drag) { drag = null; changed(); } });
    canvas.addEventListener('wheel', (e) => {
      const idx = layerUnder(e);
      if (!S.layers[idx].media) return;
      e.preventDefault();
      zoomLayer(idx, e.deltaY < 0 ? 0.05 : -0.05);
    }, { passive: false });
  }

  // Full-resolution preview frame and the settings it was drawn with, for automated preview/export checks.
  window.studioDebug = {
    snapshot() {
      const c = document.createElement('canvas');
      c.width = S.width;
      c.height = S.height;
      drawComposite(c.getContext('2d'), 1, Math.floor(S.t * FPS));
      return c.toDataURL('image/png');
    },
    settings: () => JSON.parse(JSON.stringify({ ...S, layers: S.layers.map((L) => ({ ...L, el: null, cache: null })) })),
  };

  // ---------------------------------------------------------------- render

  async function startRender() {
    const st = S;
    const body = {
      layout: st.layout, width: st.width, height: st.height, auto_duration: st.autoDuration, duration: st.duration,
      blend: st.blend, fade: st.fade, opacity: st.opacity, mode: st.mode,
      layers: st.layers.map((L) => ({
        media: L.media.id, pos_x: L.posX, pos_y: L.posY, zoom: L.zoom,
        trim_start: L.trimStart, trim_end: L.trimEnd, on_end: L.onEnd, volume: L.volume,
      })),
    };
    st.job = 'starting';
    refreshAll();
    $('stProgress').hidden = false;
    $('stResult').hidden = true;
    $('stBar').style.width = '0%';
    $('stProgText').textContent = 'Starting…';
    try {
      const { job } = await api('POST', '/api/compose/render', body);
      for (;;) {
        const j = await api('GET', `/api/jobs/${job}`);
        $('stBar').style.width = `${100 * j.progress}%`;
        $('stProgText').textContent = j.status === 'queued' ? 'Waiting for another job to finish…' : j.message;
        if (j.status === 'done') {
          $('stResultVideo').src = j.result.url;
          $('stDownload').href = `${j.result.url}?download=1`;
          $('stDownload').setAttribute('download', j.result.file);
          $('stResult').hidden = false;
          toast('Video ready!');
          break;
        }
        if (j.status === 'error') throw new Error(j.message);
        await new Promise((r) => setTimeout(r, 700));
      }
    } catch (e) {
      toast(e.message);
    } finally {
      st.job = null;
      $('stProgress').hidden = true;
      refreshAll();
    }
  }
})();
