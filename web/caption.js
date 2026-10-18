'use strict';
// Caption Maker: captions and blur regions on a picture, exported as PNG.
// One drawing function (drawScene) makes both the on-screen preview and the export, and text is
// wrapped once at the picture's own resolution, so the PNG matches the preview line for line.

(() => {
  const FONTS = [
    ['Impact', 'Impact, "Anton", "Arial Narrow", sans-serif'],
    ['Anton', '"Anton", Impact, sans-serif'],
    ['Bebas Neue', '"Bebas Neue", Impact, sans-serif'],
    ['Oswald', '"Oswald", sans-serif'],
    ['Bangers', '"Bangers", cursive'],
    ['Permanent Marker', '"Permanent Marker", cursive'],
    ['Playfair Display', '"Playfair Display", Georgia, serif'],
    ['Lobster', '"Lobster", cursive'],
    ['Pacifico', '"Pacifico", cursive'],
    ['Caveat', '"Caveat", cursive'],
    ['Montserrat', '"Montserrat", sans-serif'],
    ['Inter', '"Inter", sans-serif'],
    ['Courier New', '"Courier New", Courier, monospace'],
    ['Georgia', 'Georgia, "Times New Roman", serif'],
    ['Arial', 'Arial, Helvetica, sans-serif'],
  ];
  const STACK = Object.fromEntries(FONTS);
  const STYLE_KEYS = ['font', 'size', 'color', 'outline', 'outlineWidth', 'align', 'bold', 'italic', 'bg', 'lineHeight', 'upper'];
  const BG = { dark: 'rgba(0,0,0,0.55)', light: 'rgba(255,255,255,0.7)' };
  const HANDLE = 7; // handle half-size in screen pixels

  const canvas = $('capCanvas');
  const overlay = $('capOverlay');
  const measure = document.createElement('canvas').getContext('2d');

  const C = {
    img: null, name: '', W: 0, H: 0,
    items: [], sel: null, sync: false,
    undo: [], redo: [], snapshot: '[]',
    disp: 1, // screen pixels per picture pixel
    guides: [],
    fontsVersion: 0,
  };
  let nextId = 1;

  // ---------------------------------------------------------------- items

  function newText() {
    return {
      id: nextId++, type: 'text', text: 'YOUR CAPTION', x: 0.5, y: 0.2, w: 0.8,
      font: 'Impact', size: 72, align: 'center', bold: true, italic: false, upper: true,
      color: '#ffffff', outline: '#000000', outlineWidth: 4, lineHeight: 1.1, rotation: 0, bg: 'none',
    };
  }

  function newBlur() {
    return { id: nextId++, type: 'blur', shape: 'circle', x: 0.5, y: 0.5, w: 0.3, h: 0.3, strength: 20, fade: 0.3 };
  }

  const selected = () => C.items.find((it) => it.id === C.sel) || null;
  const minSide = () => Math.min(C.W, C.H);

  // ---------------------------------------------------------------- undo / redo

  function commit() {
    const now = JSON.stringify(C.items);
    if (now === C.snapshot) return;
    C.undo.push(C.snapshot);
    if (C.undo.length > 200) C.undo.shift();
    C.redo = [];
    C.snapshot = now;
    syncToolbar();
  }

  function restoreFrom(stack, other) {
    if (!stack.length) return;
    other.push(C.snapshot);
    C.snapshot = stack.pop();
    C.items = JSON.parse(C.snapshot);
    if (!selected()) C.sel = null;
    refresh();
  }

  // ---------------------------------------------------------------- fonts

  const fontString = (it, px) => `${it.italic ? 'italic ' : ''}${it.bold ? 700 : 400} ${px}px ${STACK[it.font] || STACK.Impact}`;
  const loadedFonts = new Set();

  /** Make sure every font in use is loaded; re-layout and redraw when one arrives. */
  function ensureFonts() {
    const jobs = [];
    for (const it of C.items) {
      if (it.type !== 'text') continue;
      const key = fontString(it, 40);
      if (loadedFonts.has(key)) continue;
      jobs.push(document.fonts.load(key, it.text || 'A').then(() => {
        loadedFonts.add(key);
        C.fontsVersion++;
        layoutCache.clear();
      }).catch(() => loadedFonts.add(key)));
    }
    if (!jobs.length) return Promise.resolve();
    return Promise.all(jobs).then(requestDraw);
  }

  // ---------------------------------------------------------------- text layout (picture pixels)

  const layoutCache = new Map();

  function layoutText(it) {
    const key = JSON.stringify([it.text, it.upper, it.font, it.size, it.bold, it.italic, it.w, it.lineHeight, C.W, C.fontsVersion]);
    if (layoutCache.has(key)) return layoutCache.get(key);
    measure.font = fontString(it, it.size);
    const width = (s) => measure.measureText(s).width;
    const maxW = Math.max(1, it.w * C.W);
    const text = it.upper ? it.text.toUpperCase() : it.text;
    const lines = [];
    for (const para of text.split('\n')) {
      let line = '';
      for (const token of para.split(/(\s+)/)) {
        if (!token) continue;
        const candidate = line + token;
        if (width(candidate.trimEnd()) <= maxW || !line.trim()) {
          line = candidate;
        } else {
          lines.push(line.trimEnd());
          line = /^\s+$/.test(token) ? '' : token;
        }
        // A single word wider than the box is broken between letters.
        while (width(line.trimEnd()) > maxW && line.trim().length > 1) {
          let cut = line.length - 1;
          while (cut > 1 && width(line.slice(0, cut)) > maxW) cut--;
          lines.push(line.slice(0, cut));
          line = line.slice(cut);
        }
      }
      lines.push(line.trimEnd());
    }
    const lineH = it.size * it.lineHeight;
    const out = { lines, widths: lines.map(width), lineH, boxW: maxW, height: Math.max(lines.length * lineH, it.size * 0.5) };
    layoutCache.set(key, out);
    return out;
  }

  /** Item size in picture pixels (before rotation). */
  function itemSize(it) {
    if (it.type === 'text') {
      const L = layoutText(it);
      return [L.boxW, L.height];
    }
    const d = it.w * minSide();
    return [d, it.shape === 'circle' ? d : it.h * minSide()];
  }

  // ---------------------------------------------------------------- drawing (shared by preview and export)

  function roundRect(c, x, y, w, h, r) {
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  function drawText(c, it, s) {
    const L = layoutText(it);
    c.save();
    c.translate(it.x * C.W * s, it.y * C.H * s);
    c.rotate((it.rotation * Math.PI) / 180);
    const boxW = L.boxW * s;
    const top = (-L.height * s) / 2;
    const anchor = it.align === 'left' ? -boxW / 2 : it.align === 'right' ? boxW / 2 : 0;
    if (it.bg !== 'none' && L.lines.some((l) => l.trim())) {
      const w = Math.max(...L.widths) * s;
      const padX = 0.3 * it.size * s;
      const padY = 0.12 * it.size * s;
      const left = it.align === 'left' ? -boxW / 2 : it.align === 'right' ? boxW / 2 - w : -w / 2;
      roundRect(c, left - padX, top - padY, w + 2 * padX, L.height * s + 2 * padY, 0.18 * it.size * s);
      c.fillStyle = BG[it.bg];
      c.fill();
    }
    c.font = fontString(it, it.size * s);
    c.textAlign = it.align;
    c.textBaseline = 'middle';
    c.lineJoin = 'round';
    c.miterLimit = 2;
    L.lines.forEach((line, i) => {
      const y = top + (i + 0.5) * L.lineH * s;
      if (it.outlineWidth > 0) {
        c.lineWidth = 2 * it.outlineWidth * s; // stroke is centred on the glyph edge: half shows outside
        c.strokeStyle = it.outline;
        c.strokeText(line, anchor, y);
      }
      c.fillStyle = it.color;
      c.fillText(line, anchor, y);
    });
    c.restore();
  }

  const smoothstep = (t) => t * t * (3 - 2 * t);
  const tmp = document.createElement('canvas');
  const maskCanvas = document.createElement('canvas');

  /** Soft-edged mask of the region (1 inside, smooth fade over `fade` of the half-size at the edge). */
  function drawMask(m, it, rx, ry, bw, bh) {
    const g = m.getContext('2d');
    g.clearRect(0, 0, m.width, m.height);
    if (it.shape === 'circle') {
      const r = bw / 2;
      const grad = g.createRadialGradient(rx, ry, 0, rx, ry, r);
      const inner = 1 - it.fade;
      for (let i = 0; i <= 24; i++) {
        const p = i / 24;
        const a = p <= inner || it.fade === 0 ? 1 : 1 - smoothstep((p - inner) / it.fade);
        grad.addColorStop(p, `rgba(0,0,0,${a})`);
      }
      g.fillStyle = grad;
      g.beginPath();
      g.arc(rx, ry, r, 0, Math.PI * 2);
      g.fill();
      return;
    }
    const f = (it.fade * Math.min(bw, bh)) / 2;
    const ramp = (grad, len) => {
      if (f <= 0) { grad.addColorStop(0, '#000'); grad.addColorStop(1, '#000'); return; }
      const e = Math.min(0.5, f / len);
      for (let i = 0; i <= 12; i++) grad.addColorStop((e * i) / 12, `rgba(0,0,0,${smoothstep(i / 12)})`);
      for (let i = 0; i <= 12; i++) grad.addColorStop(1 - e + (e * i) / 12, `rgba(0,0,0,${smoothstep(1 - i / 12)})`);
    };
    const x0 = rx - bw / 2;
    const y0 = ry - bh / 2;
    const gx = g.createLinearGradient(x0, 0, x0 + bw, 0);
    ramp(gx, bw);
    g.fillStyle = gx;
    g.fillRect(x0, y0, bw, bh);
    g.globalCompositeOperation = 'destination-in'; // multiply by the vertical ramp
    const gy = g.createLinearGradient(0, y0, 0, y0 + bh);
    ramp(gy, bh);
    g.fillStyle = gy;
    g.fillRect(x0, y0, bw, bh);
    g.globalCompositeOperation = 'source-over';
  }

  /** Blur whatever is already drawn under the region (picture and lower layers). */
  function drawBlur(c, it, s) {
    const [w0, h0] = itemSize(it);
    const bw = w0 * s;
    const bh = h0 * s;
    const blur = it.strength * s;
    const pad = Math.ceil(blur * 2);
    const cx = it.x * C.W * s;
    const cy = it.y * C.H * s;
    const x0 = Math.floor(cx - bw / 2) - pad;
    const y0 = Math.floor(cy - bh / 2) - pad;
    const w = Math.ceil(bw) + 2 * pad + 1;
    const h = Math.ceil(bh) + 2 * pad + 1;
    if (w < 2 || h < 2) return;
    tmp.width = w;
    tmp.height = h;
    const t = tmp.getContext('2d');
    t.filter = `blur(${blur}px)`;
    t.drawImage(c.canvas, x0, y0, w, h, 0, 0, w, h);
    t.filter = 'none';
    maskCanvas.width = w;
    maskCanvas.height = h;
    drawMask(maskCanvas, it, cx - x0, cy - y0, bw, bh);
    t.globalCompositeOperation = 'destination-in';
    t.drawImage(maskCanvas, 0, 0);
    c.drawImage(tmp, x0, y0);
  }

  /** The finished picture at scale s (picture pixels -> canvas pixels). */
  function drawScene(c, s) {
    c.save();
    c.imageSmoothingEnabled = true;
    c.imageSmoothingQuality = 'high';
    c.clearRect(0, 0, c.canvas.width, c.canvas.height);
    c.drawImage(C.img, 0, 0, C.W * s, C.H * s);
    for (const it of C.items) {
      if (it.type === 'text') drawText(c, it, s);
      else drawBlur(c, it, s);
    }
    c.restore();
  }

  // ---------------------------------------------------------------- editor overlay (never exported)

  /** Corners of an item on screen. */
  function itemFrame(it) {
    const [w, h] = itemSize(it);
    const d = C.disp;
    const cx = it.x * C.W * d;
    const cy = it.y * C.H * d;
    const a = it.type === 'text' ? (it.rotation * Math.PI) / 180 : 0;
    const rot = (x, y) => [cx + x * Math.cos(a) - y * Math.sin(a), cy + x * Math.sin(a) + y * Math.cos(a)];
    const hw = (w * d) / 2;
    const hh = (h * d) / 2;
    const handle = it.type === 'blur' && it.shape === 'circle' ? rot(hw, 0) : rot(hw, hh);
    return { cx, cy, a, hw, hh, corners: [rot(-hw, -hh), rot(hw, -hh), rot(hw, hh), rot(-hw, hh)], handle };
  }

  function drawOverlay() {
    const dpr = window.devicePixelRatio || 1;
    const g = overlay.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, overlay.width, overlay.height);
    const it = selected();
    if (it) {
      const f = itemFrame(it);
      g.strokeStyle = '#ff4d8d';
      g.lineWidth = 1.5;
      g.setLineDash([5, 4]);
      g.beginPath();
      if (it.type === 'blur' && it.shape === 'circle') g.arc(f.cx, f.cy, f.hw, 0, Math.PI * 2);
      else f.corners.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
      g.closePath();
      g.stroke();
      g.setLineDash([]);
      g.fillStyle = '#ff4d8d';
      g.fillRect(f.handle[0] - HANDLE, f.handle[1] - HANDLE, HANDLE * 2, HANDLE * 2);
    }
    g.strokeStyle = 'rgba(80,200,255,0.9)';
    g.lineWidth = 1;
    for (const [axis, v] of C.guides) {
      g.beginPath();
      if (axis === 'x') { g.moveTo(v * C.W * C.disp, 0); g.lineTo(v * C.W * C.disp, C.H * C.disp); }
      else { g.moveTo(0, v * C.H * C.disp); g.lineTo(C.W * C.disp, v * C.H * C.disp); }
      g.stroke();
    }
  }

  let drawQueued = false;
  function requestDraw() {
    if (drawQueued) return;
    drawQueued = true;
    requestAnimationFrame(() => {
      drawQueued = false;
      if (!C.img || $('captionView').hidden) return;
      drawScene(canvas.getContext('2d'), C.disp * (window.devicePixelRatio || 1));
      drawOverlay();
    });
  }

  function layoutStage() {
    if (!C.img) return;
    const stage = $('capStage');
    const maxW = Math.max(200, stage.clientWidth - 24);
    const maxH = Math.max(240, window.innerHeight - 270);
    C.disp = Math.min(maxW / C.W, maxH / C.H, 2);
    const w = Math.round(C.W * C.disp);
    const h = Math.round(C.H * C.disp);
    const dpr = window.devicePixelRatio || 1;
    for (const cv of [canvas, overlay]) {
      cv.style.width = `${w}px`;
      cv.style.height = `${h}px`;
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    requestDraw();
  }
  window.addEventListener('resize', () => { if (!$('captionView').hidden) layoutStage(); });

  // ---------------------------------------------------------------- pointer editing

  function toPicture(e) {
    const r = overlay.getBoundingClientRect();
    return [(e.clientX - r.left) / C.disp, (e.clientY - r.top) / C.disp]; // picture pixels
  }

  function hit(it, px, py) {
    const [w, h] = itemSize(it);
    const dx = px - it.x * C.W;
    const dy = py - it.y * C.H;
    if (it.type === 'blur' && it.shape === 'circle') return Math.hypot(dx, dy) <= w / 2;
    const a = it.type === 'text' ? (-it.rotation * Math.PI) / 180 : 0;
    const lx = dx * Math.cos(a) - dy * Math.sin(a);
    const ly = dx * Math.sin(a) + dy * Math.cos(a);
    const slack = 6 / C.disp;
    return Math.abs(lx) <= w / 2 + slack && Math.abs(ly) <= h / 2 + slack;
  }

  let drag = null;

  overlay.addEventListener('pointerdown', (e) => {
    if (!C.img) return;
    const r = overlay.getBoundingClientRect();
    const sx = e.clientX - r.left;
    const sy = e.clientY - r.top;
    const [px, py] = toPicture(e);
    const cur = selected();
    if (cur) {
      const hnd = itemFrame(cur).handle;
      if (Math.abs(sx - hnd[0]) <= HANDLE + 3 && Math.abs(sy - hnd[1]) <= HANDLE + 3) {
        drag = { mode: 'resize', it: cur };
        overlay.setPointerCapture(e.pointerId);
        return;
      }
    }
    const it = [...C.items].reverse().find((i) => hit(i, px, py));
    C.sel = it ? it.id : null;
    if (it) {
      drag = { mode: 'move', it, dx: px - it.x * C.W, dy: py - it.y * C.H };
      overlay.setPointerCapture(e.pointerId);
    }
    refresh();
  });

  overlay.addEventListener('pointermove', (e) => {
    const [px, py] = toPicture(e);
    if (!drag) {
      const cur = selected();
      const r = overlay.getBoundingClientRect();
      const onHandle = cur && (() => { const h = itemFrame(cur).handle; return Math.abs(e.clientX - r.left - h[0]) <= HANDLE + 3 && Math.abs(e.clientY - r.top - h[1]) <= HANDLE + 3; })();
      overlay.style.cursor = onHandle ? 'nwse-resize' : C.items.some((i) => hit(i, px, py)) ? 'move' : 'default';
      return;
    }
    const it = drag.it;
    if (drag.mode === 'move') {
      let x = (px - drag.dx) / C.W;
      let y = (py - drag.dy) / C.H;
      // Snap the centre to the picture's centre lines and to other items' centres.
      C.guides = [];
      const tolX = 6 / C.disp / C.W;
      const tolY = 6 / C.disp / C.H;
      const xs = [0.5, ...C.items.filter((o) => o !== it).map((o) => o.x)];
      const ys = [0.5, ...C.items.filter((o) => o !== it).map((o) => o.y)];
      const sx = xs.find((v) => Math.abs(v - x) < tolX);
      const sy = ys.find((v) => Math.abs(v - y) < tolY);
      if (sx !== undefined) { x = sx; C.guides.push(['x', sx]); }
      if (sy !== undefined) { y = sy; C.guides.push(['y', sy]); }
      it.x = clamp(x, -0.5, 1.5);
      it.y = clamp(y, -0.5, 1.5);
    } else {
      const dx = px - it.x * C.W;
      const dy = py - it.y * C.H;
      if (it.type === 'text') {
        const a = (-it.rotation * Math.PI) / 180;
        const lx = dx * Math.cos(a) - dy * Math.sin(a);
        it.w = clamp((2 * Math.abs(lx)) / C.W, 0.05, 1);
      } else if (it.shape === 'circle') {
        it.w = clamp((2 * Math.hypot(dx, dy)) / minSide(), 0.01, 1);
      } else {
        it.w = clamp((2 * Math.abs(dx)) / minSide(), 0.01, 1);
        it.h = clamp((2 * Math.abs(dy)) / minSide(), 0.01, 1);
      }
    }
    syncPanels();
    requestDraw();
  });

  overlay.addEventListener('pointerup', () => {
    if (!drag) return;
    drag = null;
    C.guides = [];
    commit();
    requestDraw();
  });

  // ---------------------------------------------------------------- panels

  const fixed = (v, d = 1) => Math.round(v * 10 ** d) / 10 ** d;
  const seg = (el, v) => { for (const b of el.querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === v); };

  function syncPanels() {
    const it = selected();
    $('capTextPanel').hidden = !it || it.type !== 'text';
    $('capBlurPanel').hidden = !it || it.type !== 'blur';
    if (it && it.type === 'text') {
      if (document.activeElement !== $('capText')) $('capText').value = it.text;
      $('capFont').value = it.font;
      for (const [r, n, v] of [['capSize', 'capSizeN', it.size], ['capWidth', 'capWidthN', Math.round(it.w * 100)],
        ['capOutlineW', 'capOutlineWN', it.outlineWidth], ['capLine', 'capLineN', it.lineHeight], ['capRot', 'capRotN', Math.round(it.rotation)]]) {
        $(r).value = v;
        $(n).value = v;
      }
      seg($('capAlign'), it.align);
      seg($('capBg'), it.bg);
      $('capBold').checked = it.bold;
      $('capItalic').checked = it.italic;
      $('capUpper').checked = it.upper;
      $('capColor').value = it.color;
      $('capColorHex').value = it.color;
      $('capOutline').value = it.outline;
      $('capOutlineHex').value = it.outline;
    }
    if (it && it.type === 'blur') {
      seg($('capShape'), it.shape);
      for (const [r, n, v] of [['capBX', 'capBXN', fixed(it.x * 100)], ['capBY', 'capBYN', fixed(it.y * 100)],
        ['capBW', 'capBWN', fixed(it.w * 100)], ['capBH', 'capBHN', fixed(it.h * 100)], ['capBS', 'capBSN', it.strength], ['capBF', 'capBFN', Math.round(it.fade * 100)]]) {
        $(r).value = v;
        $(n).value = v;
      }
      const circle = it.shape === 'circle';
      $('capBWLabel').textContent = circle ? 'Diameter' : 'Width';
      $('capBHRow').hidden = circle;
      const [w, h] = itemSize(it);
      $('capBSizeNote').textContent = `Sizes are % of the picture's shorter side (${minSide()} px): ${circle ? `${Math.round(w)} px across` : `${Math.round(w)} × ${Math.round(h)} px`}.`;
    }
    $('capSync').checked = C.sync;
  }

  function renderLayers() {
    const ul = $('capLayers');
    ul.innerHTML = '';
    [...C.items].reverse().forEach((it) => {
      const li = document.createElement('li');
      li.className = it.id === C.sel ? 'on' : '';
      const label = it.type === 'text' ? (it.text.split('\n')[0] || '(empty text)') : `Blur ${it.shape === 'circle' ? 'circle' : 'rectangle'}`;
      li.innerHTML = `<span class="ic">${it.type === 'text' ? 'T' : it.shape === 'circle' ? '◯' : '▭'}</span><span class="lbl"></span>
        <button data-a="up" title="Bring forward">▲</button><button data-a="down" title="Send backward">▼</button><button data-a="del" title="Delete">✕</button>`;
      li.querySelector('.lbl').textContent = label;
      li.onclick = (e) => {
        const a = e.target.closest('button')?.dataset.a;
        C.sel = it.id;
        if (a === 'up') move(1);
        else if (a === 'down') move(-1);
        else if (a === 'del') remove();
        refresh();
      };
      ul.appendChild(li);
    });
    if (!C.items.length) ul.innerHTML = '<li class="muted">Nothing yet. Add text or a blur region.</li>';
  }

  function syncToolbar() {
    const has = !!selected();
    $('capUndo').disabled = !C.undo.length;
    $('capRedo').disabled = !C.redo.length;
    for (const id of ['capDup', 'capFwd', 'capBack', 'capDel']) $(id).disabled = !has;
    for (const id of ['capAddText', 'capAddBlur']) $(id).disabled = !C.img;
    $('capExport').disabled = !C.img;
  }

  function refresh() {
    syncPanels();
    renderLayers();
    syncToolbar();
    ensureFonts();
    requestDraw();
  }

  // ---------------------------------------------------------------- actions

  function add(it) {
    C.items.push(it);
    C.sel = it.id;
    commit();
    refresh();
  }

  function remove() {
    const it = selected();
    if (!it) return;
    C.items = C.items.filter((i) => i !== it);
    C.sel = null;
    commit();
    refresh();
  }

  function duplicate() {
    const it = selected();
    if (!it) return;
    const copy = { ...JSON.parse(JSON.stringify(it)), id: nextId++, x: it.x + 0.03, y: it.y + 0.03 };
    C.items.splice(C.items.indexOf(it) + 1, 0, copy);
    C.sel = copy.id;
    commit();
    refresh();
  }

  function move(dir) {
    const it = selected();
    if (!it) return;
    const i = C.items.indexOf(it);
    const j = clamp(i + dir, 0, C.items.length - 1);
    if (i === j) return;
    C.items.splice(i, 1);
    C.items.splice(j, 0, it);
    commit();
    refresh();
  }

  /** Change one property of the selected item (style properties follow "Sync styles"). */
  function setProp(key, value, { record = false } = {}) {
    const it = selected();
    if (!it) return;
    const targets = it.type === 'text' && C.sync && STYLE_KEYS.includes(key) ? C.items.filter((i) => i.type === 'text') : [it];
    for (const t of targets) t[key] = value;
    syncPanels();
    renderLayers();
    ensureFonts();
    requestDraw();
    if (record) commit();
  }

  function bindNum(range, num, key, toValue = (v) => v) {
    $(range).oninput = () => { setProp(key, toValue(parseFloat($(range).value))); };
    $(range).onchange = () => commit();
    $(num).onchange = () => {
      const v = parseFloat($(num).value);
      if (Number.isFinite(v)) setProp(key, toValue(clamp(v, parseFloat($(num).min), parseFloat($(num).max))), { record: true });
    };
  }

  function bindColor(picker, hex, key) {
    $(picker).oninput = () => setProp(key, $(picker).value);
    $(picker).onchange = () => commit();
    $(hex).onchange = () => {
      let v = $(hex).value.trim();
      if (!v.startsWith('#')) v = `#${v}`;
      if (/^#[0-9a-fA-F]{6}$/.test(v)) setProp(key, v.toLowerCase(), { record: true });
      else if (/^#[0-9a-fA-F]{3}$/.test(v)) setProp(key, `#${[...v.slice(1)].map((c) => c + c).join('')}`.toLowerCase(), { record: true });
      else syncPanels();
    };
  }

  function bindControls() {
    for (const [name] of FONTS) {
      const o = document.createElement('option');
      o.value = name;
      o.textContent = name;
      o.style.fontFamily = STACK[name];
      $('capFont').appendChild(o);
    }
    let typing = null;
    $('capText').oninput = () => {
      setProp('text', $('capText').value);
      clearTimeout(typing);
      typing = setTimeout(commit, 600);
    };
    $('capFont').onchange = () => setProp('font', $('capFont').value, { record: true });
    bindNum('capSize', 'capSizeN', 'size', (v) => Math.round(v));
    bindNum('capWidth', 'capWidthN', 'w', (v) => v / 100);
    bindNum('capOutlineW', 'capOutlineWN', 'outlineWidth');
    bindNum('capLine', 'capLineN', 'lineHeight', (v) => Math.round(v * 100) / 100);
    bindNum('capRot', 'capRotN', 'rotation', (v) => Math.round(v));
    bindNum('capBX', 'capBXN', 'x', (v) => v / 100);
    bindNum('capBY', 'capBYN', 'y', (v) => v / 100);
    bindNum('capBW', 'capBWN', 'w', (v) => v / 100);
    bindNum('capBH', 'capBHN', 'h', (v) => v / 100);
    bindNum('capBS', 'capBSN', 'strength', (v) => Math.round(v));
    bindNum('capBF', 'capBFN', 'fade', (v) => v / 100);
    bindColor('capColor', 'capColorHex', 'color');
    bindColor('capOutline', 'capOutlineHex', 'outline');
    for (const [id, key] of [['capAlign', 'align'], ['capBg', 'bg'], ['capShape', 'shape']]) {
      $(id).onclick = (e) => {
        const b = e.target.closest('button');
        if (b) setProp(key, b.dataset.v, { record: true });
      };
    }
    for (const [id, key] of [['capBold', 'bold'], ['capItalic', 'italic'], ['capUpper', 'upper']]) {
      $(id).onchange = () => setProp(key, $(id).checked, { record: true });
    }
    $('capSync').onchange = () => { C.sync = $('capSync').checked; };
    $('capApplyAll').onclick = () => {
      const it = selected();
      if (!it || it.type !== 'text') return;
      for (const t of C.items) if (t.type === 'text') for (const k of STYLE_KEYS) t[k] = it[k];
      commit();
      refresh();
      toast('Style copied to every text box');
    };
    $('capAddText').onclick = () => add(newText());
    $('capAddBlur').onclick = () => add(newBlur());
    $('capDup').onclick = duplicate;
    $('capFwd').onclick = () => move(1);
    $('capBack').onclick = () => move(-1);
    $('capDel').onclick = remove;
    $('capUndo').onclick = () => restoreFrom(C.undo, C.redo);
    $('capRedo').onclick = () => restoreFrom(C.redo, C.undo);
    $('capChoose').onclick = () => $('capFile').click();
    $('capReplace').onclick = () => $('capFile').click();
    $('capFile').onchange = () => { if ($('capFile').files[0]) loadPicture($('capFile').files[0]); $('capFile').value = ''; };
    const stage = $('capStage');
    stage.addEventListener('dragover', (e) => { e.preventDefault(); stage.classList.add('over'); });
    stage.addEventListener('dragleave', () => stage.classList.remove('over'));
    stage.addEventListener('drop', (e) => {
      e.preventDefault();
      stage.classList.remove('over');
      if (e.dataTransfer.files[0]) loadPicture(e.dataTransfer.files[0]);
    });
    $('capExportSize').onchange = updateExportLabel;
    $('capExport').onclick = exportPNG;
    document.addEventListener('keydown', onKey);
  }

  function onKey(e) {
    if ($('captionView').hidden || e.target.closest('input,textarea,select')) return;
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) restoreFrom(C.redo, C.undo); else restoreFrom(C.undo, C.redo); return; }
    if (mod && e.key.toLowerCase() === 'y') { e.preventDefault(); restoreFrom(C.redo, C.undo); return; }
    if (mod && e.key.toLowerCase() === 'd') { e.preventDefault(); duplicate(); return; }
    const it = selected();
    if (!it) return;
    if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); remove(); return; }
    const step = e.shiftKey ? 0.01 : 0.001;
    const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
    if (moves[e.key]) {
      e.preventDefault();
      it.x += moves[e.key][0];
      it.y += moves[e.key][1];
      syncPanels();
      requestDraw();
      clearTimeout(onKey.t);
      onKey.t = setTimeout(commit, 400);
    }
  }

  // ---------------------------------------------------------------- picture + export

  function loadPicture(file) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const first = !C.img;
      C.img = img;
      C.name = file.name;
      C.W = img.naturalWidth;
      C.H = img.naturalHeight;
      layoutCache.clear();
      $('capName').textContent = file.name;
      $('capName').title = file.name;
      $('capDims').textContent = `${C.W} × ${C.H} px`;
      $('capDrop').hidden = true;
      $('capHolder').hidden = false;
      if (first && !C.items.length) {
        const t = newText();
        t.size = Math.max(8, Math.round(minSide() / 12)); // the 72 px default, relative to a ~860 px picture
        C.items = [t];
        C.sel = t.id;
        C.snapshot = JSON.stringify(C.items);
      }
      fillExportSizes();
      layoutStage();
      refresh();
    };
    img.onerror = () => toast('That file could not be opened as a picture.');
    img.src = url;
  }

  function exportSizes() {
    const long = Math.max(C.W, C.H);
    const opts = [];
    const add = (id, label, s) => opts.push({ id, s, label: `${label} · ${Math.round(C.W * s)} × ${Math.round(C.H * s)} px` });
    if (long < 3840) add('sharp', long * 2 <= 3840 ? 'Sharper text (2×)' : 'Sharper text (up to 3840 px)', Math.min(2, 3840 / long));
    add('orig', 'Original size', 1);
    for (const L of [1080, 2160, 3840]) if (Math.abs(L - long) > 1) add(`l${L}`, `Long side ${L} px`, L / long);
    return opts;
  }

  function fillExportSizes() {
    const sel = $('capExportSize');
    const prev = sel.value;
    sel.innerHTML = '';
    for (const o of exportSizes()) {
      const opt = document.createElement('option');
      opt.value = o.id;
      opt.textContent = o.label;
      sel.appendChild(opt);
    }
    if ([...sel.options].some((o) => o.value === prev)) sel.value = prev;
    updateExportLabel();
  }

  function exportScale() {
    const o = exportSizes().find((x) => x.id === $('capExportSize').value) || exportSizes()[0];
    return o.s;
  }

  function updateExportLabel() {
    if (!C.img) return;
    const s = exportScale();
    $('capExport').textContent = `Download PNG · ${Math.round(C.W * s)} × ${Math.round(C.H * s)}`;
  }

  async function exportPNG() {
    if (!C.img) return;
    await ensureFonts(); // never draw the export with a fallback font
    const s = exportScale();
    const out = document.createElement('canvas');
    out.width = Math.round(C.W * s);
    out.height = Math.round(C.H * s);
    drawScene(out.getContext('2d'), s);
    out.toBlob((blob) => {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = `${C.name.replace(/\.[^.]+$/, '') || 'caption'}-caption.png`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    }, 'image/png');
  }

  // Exposed for automated preview/export comparisons.
  window.captionDebug = {
    render: async (s) => {
      await ensureFonts();
      const c = document.createElement('canvas');
      c.width = Math.round(C.W * s);
      c.height = Math.round(C.H * s);
      drawScene(c.getContext('2d'), s);
      return c.toDataURL('image/png');
    },
    state: () => C,
  };

  // ---------------------------------------------------------------- open

  let bound = false;
  window.openCaption = () => {
    show('captionView', 'caption');
    document.title = 'Caption Maker · PopClip';
    if (!bound) {
      bound = true;
      bindControls();
    }
    layoutStage();
    refresh();
  };
})();
