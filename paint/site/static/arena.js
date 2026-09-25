// Model arena tab.
(function () {
  const $a = s => document.querySelector(s);
  let meta = null, started = false;

  window.arenaInit = async function () {
    if (!started) {
      started = true;
      for (let i = 0; i < 100 && !STATE; i++) await new Promise(r => setTimeout(r, 100));
      meta = await api('/api/arena/meta');
      $a('#a-models').innerHTML = meta.models.map(m => `<label class="hint" style="display:block"><input type="checkbox" value="${esc(m.id)}" ${meta.default.includes(m.id) ? 'checked' : ''}>
        ${esc(m.id)} <span class="hint">($${m.price[0]}/$${m.price[1]} per M tokens)</span></label>`).join('');
      $a('#a-tasks').innerHTML = meta.tasks.map(t => `<label class="hint" style="display:block"><input type="checkbox" value="${esc(t.id)}" checked> ${esc(t.title || t.id)}</label>`).join('');
      if (!meta.docker) { const o = $a('#a-sandbox option[value=docker]'); o.disabled = true; o.textContent = 'docker (build the image: paint sandbox --build)'; }
      document.querySelectorAll('#arena input, #arena select').forEach(el => el.addEventListener('change', estimate));
      $a('#a-go').onclick = start;
      estimate();
    }
    loadArena();
  };

  function selection() {
    const models = [...document.querySelectorAll('#a-models input:checked')].map(x => x.value);
    const custom = $a('#a-custom').value.trim();
    if (custom) models.push(custom);
    if ($a('#a-naive').checked) models.push('baseline:naive');
    const tasks = [...document.querySelectorAll('#a-tasks input:checked')].map(x => x.value);
    return { models, tasks };
  }

  function estimate() {
    // Rough: per version ~12k input (cached history + image) and ~8k output (thinking + program).
    const { models, tasks } = selection();
    const it = Math.min(4, Math.max(1, Number($a('#a-iters').value) || 4));
    let total = 0;
    for (const m of models) {
      const p = (meta.models.find(x => x.id === m) || { price: [5, 25] }).price;
      if (m.startsWith('baseline')) continue;
      total += tasks.length * it * (12000 * p[0] + 8000 * p[1]) / 1e6;
    }
    const judgeCalls = tasks.length * (models.length + 1) * 4;
    $a('#a-estimate').textContent = `${models.length} contestant(s) × ${tasks.length} task(s) × up to ${it} versions. Rough model cost ≈ $${total.toFixed(2)} (plus ~${judgeCalls} judge calls). ${STATE.key.set ? '' : 'No API key: only the naive baseline and house styles can run, with the mock judge.'}`;
  }

  async function start() {
    const { models, tasks } = selection();
    if (!models.length || !tasks.length) { $a('#a-estimate').textContent = 'Pick at least one contestant and one task.'; return; }
    const b = $a('#a-go'); b.disabled = true;
    try {
      const { job } = await api('/api/arena', { models, tasks, iterations: Number($a('#a-iters').value), size: $a('#a-size').value,
        effort: $a('#a-effort').value, max_cost: $a('#a-cap').value, sandbox: $a('#a-sandbox').value, judge: $a('#a-judge').value,
        label: $a('#a-label').value, house: $a('#a-house').checked });
      await poll(job, $a('#a-log'), j => { loadArena(); if (j.result && j.result.report) window.open(j.result.report); });
    } catch (e) { $a('#a-log').hidden = false; $a('#a-log').textContent = e.message; }
    b.disabled = false;
  }

  const pct = v => v == null ? '–' : `${Math.round(v * 100)}%`;
  const num = (v, d = 2) => v == null ? '–' : Number(v).toFixed(d);

  async function loadArena() {
    const { runs, elo } = await api('/api/arena/runs');
    $a('#a-elo').innerHTML = elo.length ? `<table><tr><th>Player</th><th>Elo</th></tr>${elo.map(r => `<tr><td>${esc(r.player)}</td><td class="num">${r.elo.toFixed(0)}</td></tr>`).join('')}</table>` : '<div class="empty">No arena runs yet.</div>';
    $a('#a-runs').innerHTML = runs.length ? runs.map(r => `<div class="panel" style="margin-bottom:10px"><div class="row" style="justify-content:space-between"><b>${esc(r.label)}</b>
        <span class="hint">${esc(r.judge.name)}${r.judge.mock ? ' (mock)' : ''} · ${r.tasks.length} task(s) · <a href="${esc(r.report)}" target="_blank" rel="noopener">report</a></span></div>
      <table><tr><th>Player</th><th>Rendered</th><th>First try</th><th>Subjects</th><th>Style</th><th>Style (first)</th><th>Cost</th></tr>
      ${r.leaderboard.map(x => `<tr><td>${esc(x.player)}</td><td class="num">${pct(x.rendered)}</td><td class="num">${pct(x.first_try)}</td><td class="num">${num(x.fidelity)}</td><td class="num">${num(x.style)}</td><td class="num">${num(x.style_first)}</td><td class="num">${x.cost == null ? '–' : '$' + num(x.cost)}</td></tr>`).join('')}</table></div>`).join('') : '<div class="empty">No arena runs yet.</div>';
  }
})();
