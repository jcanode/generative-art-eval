// Playground: freestyle scene making with a drag-and-drop layout editor and live multi-style previews.
(function () {
  const P = { meta: null, scene: null, sel: -1, seq: 0, busy: false, pending: null, timer: null, ready: false };
  const REGION = new Set(['sea', 'field', 'table', 'river', 'hills']);
  const DEFAULT_SIZE = { sun: 0.16, moon: 0.14, cloud: 0.12, bird: 0.07, stars: 0.5, rain: 0.3, mountain: 0.4, tree: 0.45, pine: 0.3,
    cypress: 0.5, bamboo: 0.9, reeds: 0.3, house: 0.2, lighthouse: 0.45, windmill: 0.5, boat: 0.25, vase: 0.45, fruit: 0.12,
    teapot: 0.45, cup: 0.28, cliff: 0.4, river: 0.5, hills: 0.1 };
  const NS = 'http://www.w3.org/2000/svg';
  const $p = s => document.querySelector(s);

  function W() { return P.scene.canvas.width; }
  function H() { return P.scene.canvas.height; }

  let initPromise = null;
  window.playInit = function () {
    if (!initPromise) initPromise = init().catch(e => { initPromise = null; $p('#p-status').innerHTML = `<span class="bad">${esc(e.message)}</span>`; });
    return initPromise;
  };
  async function init() {
    for (let i = 0; i < 100 && !STATE; i++) await new Promise(r => setTimeout(r, 100));  // main script still loading state
    if (!STATE) throw new Error('studio state did not load');
    P.meta = await api('/api/play/meta');
    $p('#p-setting').innerHTML += P.meta.settings.map(s => `<option>${esc(s)}</option>`).join('');
    $p('#p-addkind').innerHTML = P.meta.kinds.map(k => `<option>${esc(k)}</option>`).join('');
    $p('#p-palpreset').innerHTML += P.meta.palettes.map((p, i) => `<option value="${i}">${esc(p.name)}</option>`).join('');
    const styles = Object.keys(STATE.styles);
    const on = new Set(['screenprint', 'sumie', 'impasto']);
    $p('#p-styles').innerHTML = styles.map(s => `<label class="chip stylechip"><input type="checkbox" value="${esc(s)}" ${on.has(s) ? 'checked' : ''}>${esc(s)}</label>`).join('');
    $p('#p-styles').onchange = () => schedule();
    wire();
    await surprise();
  }

  function wire() {
    $p('#p-prompt').onclick = fromPrompt;
    $p('#p-text').onkeydown = e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) fromPrompt(); };
    $p('#p-surprise').onclick = surprise;
    $p('#p-claude').disabled = !STATE.key.set;
    $p('#p-claude').checked = STATE.key.set;
    $p('#p-claude').parentElement.title = STATE.key.set ? '' : 'add an API key in Settings to let Claude write scenes';
    $p('#p-title').oninput = e => { P.scene.title = e.target.value; changed(false); };
    $p('#p-canvas').onchange = e => { const [w, h] = e.target.value.split('x').map(Number); P.scene.canvas = { width: w, height: h }; changed(); };
    $p('#p-seed').onchange = e => { P.scene.seed = Number(e.target.value) || 0; changed(); };
    $p('#p-dice').onclick = () => { P.scene.seed = Math.floor(Math.random() * 10000); changed(); };
    $p('#p-horizon').oninput = e => setHorizon(Number(e.target.value));
    $p('#p-az').oninput = e => { P.scene.light.azimuth = Number(e.target.value); changed(); };
    $p('#p-mood').onchange = e => { P.scene.palette.mood = e.target.value; changed(); };
    $p('#p-palpreset').onchange = e => {
      if (e.target.value === '') return;
      const pal = P.meta.palettes[Number(e.target.value)];
      P.scene.palette = { hints: [...pal.hints], mood: pal.mood }; changed();
    };
    $p('#p-add').onclick = addObject;
    $p('#p-applyjson').onclick = () => {
      try { P.scene = normalise(JSON.parse($p('#p-json').value)); P.sel = -1; changed(); $p('#p-status').textContent = ''; }
      catch (e) { $p('#p-status').innerHTML = `<span class="bad">Not valid JSON: ${esc(e.message)}</span>`; }
    };
    $p('#p-full').onclick = () => render(false);
    $p('#p-live').onchange = e => { if (e.target.checked) schedule(); };
    $p('#p-save').onclick = async () => {
      try { const r = await api('/api/play/save', { scene: P.scene, name: $p('#p-savename').value }); $p('#p-status').innerHTML = `Saved to <b>${esc(r.path)}</b>; it now appears in the Render and Loop scene lists after a reload.`; }
      catch (e) { $p('#p-status').innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
    };
    document.addEventListener('keydown', e => {
      if (!document.getElementById('playground').classList.contains('active')) return;
      if ((e.key === 'Delete' || e.key === 'Backspace') && P.sel >= 0 && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) { e.preventDefault(); removeSel(); }
    });
  }

  function normalise(s) {
    s.canvas = s.canvas || { width: 1024, height: 768 };
    s.light = s.light || { azimuth: 225, elevation: 35, warmth: 0.5 };
    s.palette = s.palette || { hints: ['#f2c14e', '#e8643c', '#2a8c88', '#1d2d4a'], mood: 'warm' };
    s.palette.hints = (s.palette.hints || []).slice(0, 5);
    s.horizon = s.horizon ?? 0.62;
    s.objects = s.objects || [];
    s.seed = s.seed ?? 0;
    return s;
  }

  async function surprise() {
    $p('#p-status').textContent = 'composing…';
    const [w, h] = $p('#p-canvas').value.split('x').map(Number);
    const r = await api('/api/play/surprise', { setting: $p('#p-setting').value, width: w, height: h });
    P.scene = normalise(r.scene); P.sel = -1;
    $p('#p-source').textContent = r.scene.notes || '';
    changed();
  }

  async function fromPrompt() {
    const text = $p('#p-text').value.trim();
    if (!text) { $p('#p-text').focus(); return; }
    const b = $p('#p-prompt'); b.disabled = true;
    $p('#p-status').textContent = $p('#p-claude').checked ? 'Claude is writing the scene…' : 'parsing…';
    try {
      const [w, h] = $p('#p-canvas').value.split('x').map(Number);
      const r = await api('/api/play/prompt', { text, seed: Math.floor(Math.random() * 10000), use_claude: $p('#p-claude').checked, width: w, height: h });
      P.scene = normalise(r.scene); P.sel = -1;
      $p('#p-source').textContent = r.source === 'claude' ? 'Written by Claude.' : (r.scene.notes || 'Offline keyword parser.');
      changed();
    } catch (e) { $p('#p-status').innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
    b.disabled = false;
  }

  function setHorizon(h) {
    const old = P.scene.horizon;
    P.scene.horizon = Math.round(h * 1000) / 1000;
    for (const o of P.scene.objects) if (REGION.has(o.kind) && o.kind !== 'river' && (o.y === undefined || Math.abs(o.y - old) < 0.02)) o.y = P.scene.horizon;
    changed();
  }

  function addObject() {
    const k = $p('#p-addkind').value;
    const h = P.scene.horizon;
    const o = { kind: k, size: DEFAULT_SIZE[k] ?? 0.2 };
    if (REGION.has(k) && k !== 'river') o.y = h;
    else if (['sun', 'moon', 'cloud', 'bird'].includes(k)) { o.x = 0.5; o.y = Math.max(0.12, h * 0.45); }
    else if (k === 'stars') o.count = 14;
    else if (k === 'rain') o.count = 90;
    else if (k === 'mountain') { o.x = 0.5; o.y = h - o.size / 2; }
    else if (k === 'river') o.x = 0.5;
    else { o.x = 0.5; o.y = Math.min(0.95, h + 0.15); o.size = Math.min(o.size, (o.y - 0.04) / (P.meta.reach[k] || 1)); }
    P.scene.objects.push(o);
    P.sel = P.scene.objects.length - 1;
    changed();
  }

  function removeSel() { if (P.sel < 0) return; P.scene.objects.splice(P.sel, 1); P.sel = -1; changed(); }

  // Rough on-canvas footprint of each object, for the editor boxes.
  function box(o) {
    const w = W(), h = H(), s = (o.size ?? 0.2) * h, x = (o.x ?? 0.5) * w, y = (o.y ?? 0.5) * h;
    const reach = P.meta.reach[o.kind];
    if (reach) {
      const aspect = { house: 0.75, lighthouse: 0.3, windmill: 1.0, tree: 0.6, pine: 0.6, cypress: 0.3, bamboo: 0.5, reeds: 0.6, vase: 0.6, teapot: 1.6, cup: 1.2, fruit: 1.0 }[o.kind] ?? 0.6;
      const bw = s * aspect * (o.kind === 'fruit' ? (o.count || 1) : 1);
      return { x: x - bw / 2, y: y - s * reach, w: bw, h: s * reach, ax: x, ay: y };
    }
    if (o.kind === 'boat') return { x: x - s * 0.5, y: y - s * (o.variant === 'row' ? 0.2 : 1.05), w: s, h: s * (o.variant === 'row' ? 0.3 : 1.15), ax: x, ay: y };
    if (o.kind === 'cliff') { const right = (o.x ?? 0.5) >= 0.5; return { x: right ? x - s * 0.35 : 0, y, w: right ? w - x + s * 0.35 : x + s * 0.35, h: h - y, ax: x, ay: y }; }
    if (o.kind === 'mountain') { const n = o.count || 1; return { x: x - s * (0.6 + 0.45 * (n - 1)) - s * 0.3, y: y - s / 2, w: s * (1.8 + 0.9 * (n - 1)), h: s, ax: x, ay: y }; }
    if (o.kind === 'bird' && (o.count || 1) > 1) return { x: x - 0.14 * w, y: y - 0.09 * h, w: 0.28 * w, h: 0.18 * h, ax: x, ay: y };
    return { x: x - s / 2, y: y - s / 2, w: s, h: s, ax: x, ay: y };
  }

  function draw() {
    const svg = $p('#p-svg');
    const w = W(), h = H(), hz = P.scene.horizon * h;
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
    const pal = P.scene.palette.hints;
    const sky = pal[0] || '#ddd', ground = pal[Math.min(2, pal.length - 1)] || '#999';
    let out = `<rect x="0" y="0" width="${w}" height="${h}" fill="${esc(sky)}" opacity="0.55"/>`;
    const regions = P.scene.objects.filter(o => REGION.has(o.kind) && o.kind !== 'river').map(o => o.kind);
    if (regions.length) out += `<rect x="0" y="${hz}" width="${w}" height="${h - hz}" fill="${esc(ground)}" opacity="0.45"/>`;
    out += `<line id="p-hz" x1="0" x2="${w}" y1="${hz}" y2="${hz}" stroke="currentColor" stroke-width="3" stroke-dasharray="14 10" style="cursor:ns-resize;color:var(--fg)"/>`;
    out += `<text x="10" y="${hz - 10}" fill="currentColor" style="color:var(--fg)">horizon · ${esc(regions.join(', ') || 'no ground')}</text>`;
    P.scene.objects.forEach((o, i) => {
      if (REGION.has(o.kind) && o.kind !== 'river') return;
      if (o.kind === 'stars' || o.kind === 'rain') {
        out += `<g class="obj ${i === P.sel ? 'sel' : ''}" data-i="${i}"><rect x="6" y="${o.kind === 'stars' ? 6 : 36}" width="120" height="26" rx="6" fill="var(--panel)" stroke="var(--line)"/><text x="16" y="${o.kind === 'stars' ? 24 : 54}" fill="var(--fg)">${o.kind} ×${o.count || ''}</text></g>`;
        return;
      }
      const b = box(o);
      out += `<g class="obj ${i === P.sel ? 'sel' : ''}" data-i="${i}">
        <rect x="${b.x}" y="${b.y}" width="${Math.max(b.w, 8)}" height="${Math.max(b.h, 8)}" rx="4" fill="rgba(255,255,255,0.35)" stroke="rgba(0,0,0,0.55)" stroke-width="1.5"/>
        <circle cx="${b.ax}" cy="${b.ay}" r="${Math.max(6, w / 140)}" fill="var(--accent)"/>
        <text x="${b.x + 5}" y="${b.y + w / 45}" fill="#111">${esc(o.kind)}${o.count > 1 ? ' ×' + o.count : ''}</text></g>`;
    });
    svg.innerHTML = out;
    svg.querySelectorAll('text').forEach(t => t.setAttribute('font-size', Math.round(w / 55)));
    svg.querySelectorAll('.obj').forEach(g => g.addEventListener('pointerdown', startDrag));
    svg.querySelector('#p-hz').addEventListener('pointerdown', startHorizon);
  }

  function svgPoint(e) {
    const svg = $p('#p-svg'); const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
    return pt.matrixTransform(svg.getScreenCTM().inverse());
  }
  function startDrag(e) {
    const i = Number(e.currentTarget.dataset.i); P.sel = i; inspector(); draw();
    const o = P.scene.objects[i];
    if (o.x === undefined && o.y === undefined) return;
    const p0 = svgPoint(e), x0 = o.x ?? 0.5, y0 = o.y ?? 0.5;
    const svg = $p('#p-svg'); svg.setPointerCapture(e.pointerId);
    const move = ev => {
      const p = svgPoint(ev);
      o.x = Math.round(Math.min(1, Math.max(0, x0 + (p.x - p0.x) / W())) * 1000) / 1000;
      if (o.kind !== 'river') o.y = Math.round(Math.min(1.1, Math.max(0, y0 + (p.y - p0.y) / H())) * 1000) / 1000;
      draw(); inspector(); syncJson();
    };
    const up = () => { svg.removeEventListener('pointermove', move); svg.removeEventListener('pointerup', up); schedule(); };
    svg.addEventListener('pointermove', move); svg.addEventListener('pointerup', up);
  }
  function startHorizon(e) {
    e.stopPropagation();
    const svg = $p('#p-svg'); svg.setPointerCapture(e.pointerId);
    const move = ev => { const p = svgPoint(ev); setHorizonLive(Math.min(0.85, Math.max(0.3, p.y / H()))); };
    const up = () => { svg.removeEventListener('pointermove', move); svg.removeEventListener('pointerup', up); schedule(); };
    svg.addEventListener('pointermove', move); svg.addEventListener('pointerup', up);
  }
  function setHorizonLive(h) {
    const old = P.scene.horizon; P.scene.horizon = Math.round(h * 1000) / 1000;
    for (const o of P.scene.objects) if (REGION.has(o.kind) && o.kind !== 'river' && (o.y === undefined || Math.abs(o.y - old) < 0.02)) o.y = P.scene.horizon;
    fields(); draw(); syncJson();
  }

  function inspector() {
    const el = $p('#p-inspector');
    const o = P.scene.objects[P.sel];
    if (!o) { el.innerHTML = 'Select an object to edit it. Delete removes it.'; el.classList.add('hint'); return; }
    el.classList.remove('hint');
    const vars = P.meta.variants[o.kind];
    const num = (k, v, step, min, max) => `<div><label>${k}</label><input type="number" data-f="${k}" value="${v ?? ''}" step="${step}" min="${min}" max="${max}"></div>`;
    el.innerHTML = `<div class="row" style="justify-content:space-between"><b>${esc(o.kind)}</b>
        <div class="row"><button class="btn secondary" id="p-dup">Duplicate</button><button class="btn secondary" id="p-del">Delete</button></div></div>
      <label>size <output>${o.size ?? ''}</output></label><input type="range" data-f="size" min="0.02" max="1" step="0.01" value="${o.size ?? 0.2}">
      <div class="row2">${num('x', o.x, 0.01, 0, 1)}${num('y', o.y, 0.01, 0, 1.1)}</div>
      <div class="row2">${num('count', o.count ?? 1, 1, 1, 200)}
        <div><label>variant</label><select data-f="variant"><option value="">default</option>${(vars || []).map(v => `<option ${o.variant === v ? 'selected' : ''}>${v}</option>`).join('')}</select></div></div>
      ${o.kind === 'boat' ? `<label><input type="checkbox" data-f="facing" ${o.facing === 'left' ? 'checked' : ''}> facing left</label>` : ''}`;
    el.querySelectorAll('[data-f]').forEach(inp => inp.addEventListener(inp.type === 'range' ? 'input' : 'change', () => {
      const f = inp.dataset.f;
      if (f === 'facing') { if (inp.checked) o.facing = 'left'; else delete o.facing; }
      else if (f === 'variant') { if (inp.value) o.variant = inp.value; else delete o.variant; }
      else if (inp.value === '') delete o[f];
      else o[f] = f === 'count' ? Math.max(1, Math.round(Number(inp.value))) : Number(inp.value);
      if (f === 'size') inp.previousElementSibling.querySelector('output').value = inp.value;
      draw(); syncJson(); schedule();
    }));
    $p('#p-del').onclick = removeSel;
    $p('#p-dup').onclick = () => { const c = JSON.parse(JSON.stringify(o)); if (c.x !== undefined) c.x = Math.min(0.95, c.x + 0.08); P.scene.objects.push(c); P.sel = P.scene.objects.length - 1; changed(); };
  }

  function fields() {
    const s = P.scene;
    $p('#p-title').value = s.title || '';
    const cv = `${s.canvas.width}x${s.canvas.height}`;
    if (![...$p('#p-canvas').options].some(o => o.value === cv)) $p('#p-canvas').insertAdjacentHTML('beforeend', `<option value="${cv}">${cv}</option>`);
    $p('#p-canvas').value = cv;
    $p('#p-seed').value = s.seed;
    $p('#p-horizon').value = s.horizon; $p('#p-horizon-o').value = s.horizon.toFixed(2);
    $p('#p-az').value = s.light.azimuth; $p('#p-az-o').value = `${Math.round(s.light.azimuth)}° ${compass(s.light.azimuth)}`;
    $p('#p-mood').value = s.palette.mood || 'neutral';
    const sw = $p('#p-pal');
    sw.innerHTML = s.palette.hints.map((c, i) => `<input type="color" value="${esc(c)}" data-i="${i}" aria-label="palette colour ${i + 1}">`).join('')
      + (s.palette.hints.length < 5 ? '<button class="btn secondary" id="p-addcol" aria-label="add colour">+</button>' : '')
      + (s.palette.hints.length > 2 ? '<button class="btn secondary" id="p-rmcol" aria-label="remove colour">−</button>' : '');
    sw.querySelectorAll('input').forEach(inp => inp.onchange = () => { s.palette.hints[Number(inp.dataset.i)] = inp.value; $p('#p-palpreset').value = ''; changed(); });
    const add = $p('#p-addcol'); if (add) add.onclick = () => { s.palette.hints.push('#888888'); changed(); };
    const rm = $p('#p-rmcol'); if (rm) rm.onclick = () => { s.palette.hints.pop(); changed(); };
  }
  function compass(a) { return ['from right', 'from lower right', 'from below', 'from lower left', 'from left', 'from upper left', 'from above', 'from upper right'][Math.round(((a % 360) + 360) % 360 / 45) % 8]; }
  function syncJson() { $p('#p-json').value = JSON.stringify(P.scene, null, 2); }

  function changed(rerender = true) { fields(); draw(); inspector(); syncJson(); if (rerender) schedule(); }

  function styles() { return [...document.querySelectorAll('#p-styles input:checked')].map(x => x.value); }

  function schedule() {
    if (!$p('#p-live').checked) return;
    clearTimeout(P.timer);
    P.timer = setTimeout(() => render(true), 650);
  }

  async function render(preview) {
    const st = styles();
    if (!st.length) { $p('#p-status').textContent = 'Pick at least one style.'; return; }
    if (P.busy) { P.pending = preview; return; }  // coalesce: re-run once the current render finishes
    P.busy = true;
    const my = ++P.seq;
    $p('#p-status').textContent = preview ? `previewing ${st.length} style(s)…` : `rendering ${st.length} style(s) at full size…`;
    $p('#p-full').disabled = true;
    try {
      const r = await api('/api/play/render', { scene: P.scene, styles: st, preview });
      if (my === P.seq) {
        $p('#p-results').innerHTML = r.results.map(x => x.ok
          ? `<figure><img src="${esc(x.image)}" alt="${esc(x.style)} render" onclick="window.open(this.src)"><figcaption>${esc(x.style)} · ${x.seconds.toFixed(1)}s${preview ? ' · preview' : ''}</figcaption></figure>`
          : `<figure><div class="bad">${esc(x.style)} failed</div><figcaption>${esc(x.error || '')}</figcaption></figure>`).join('');
        $p('#p-status').textContent = preview ? 'Preview at reduced size. "Render full size" for the real thing.' : 'Full-size renders done. Click to open.';
      }
    } catch (e) { $p('#p-status').innerHTML = `<span class="bad">${esc(e.message)}</span>`; }
    P.busy = false; $p('#p-full').disabled = false;
    if (P.pending !== null) { const pv = P.pending; P.pending = null; render(pv); }
  }
})();
