// Agent History and Decision Replay. Reads the registry datasets; draws only inside drawings.
window.twHist = window.twHist || { agent: null, evalId: null, idx: 0, filter: 'all' };

function twH_rows(id) {
  const s = dash.data(id);
  return s.status === 'ok' ? s.data : [];
}
function twH_ok(id) { return dash.data(id).status === 'ok'; }
window.twH_agent = function () {
  const ag = twH_rows('registry_agents');
  if (!ag.length) return null;
  let cur = ag.find((r) => r.agent_id === window.twHist.agent);
  if (!cur) {
    const ev = twH_rows('registry_evals');
    const worst = d3.greatest(ev, (e) => +e.deaths_pct);           // open on the most instructive failure
    cur = (worst && ag.find((r) => r.agent_id === worst.agent_id)) || ag[0];
    window.twHist.agent = cur.agent_id;
  }
  return cur;
};
window.twH_evals = function () {
  const a = window.twH_agent();
  return a ? twH_rows('registry_evals').filter((e) => e.agent_id === a.agent_id) : [];
};
window.twH_eval = function () {
  const ev = window.twH_evals();
  if (!ev.length) return null;
  let cur = ev.find((e) => e.eval_id === window.twHist.evalId);
  if (!cur) { cur = ev[ev.length - 1]; window.twHist.evalId = cur.eval_id; window.twHist.idx = 0; }
  return cur;
};
window.twH_decisions = function () {
  const e = window.twH_eval();
  if (!e) return [];
  return twH_rows('registry_decisions').filter((d) => d.eval_id === e.eval_id).map((d) => ({
    ...d, step: +d.step, price: +d.price, size_chosen: +d.size_chosen, leverage: +d.leverage,
    equity_before: +d.equity_before, equity_after: +d.equity_after, reward: +d.reward, costs: +d.costs,
    market_move_pct: +d.market_move_pct, expected_move_pct: d.expected_move_pct === '' || d.expected_move_pct == null ? null : +d.expected_move_pct,
  }));
};
window.twH_filtered = function (rows) {
  const f = window.twHist.filter;
  if (f === 'all') return rows.map((r, i) => i);
  const levs = rows.map((r) => r.leverage).sort((a, b) => a - b);
  const p95 = levs.length ? levs[Math.floor(0.95 * (levs.length - 1))] : Infinity;
  const keep = (r) => f === 'change' ? !['hold', 'stay flat'].includes(r.position_change)
    : f === 'loss' ? r.equity_after < r.equity_before
      : f === 'event' ? (r.died === 'yes' || (r.rejected || '') !== '')
        : r.leverage >= p95 && r.leverage > 0;
  const out = [];
  rows.forEach((r, i) => { if (keep(r)) out.push(i); });
  return out;
};
window.twH_redraw = function (names) {
  names.forEach((n) => dash.draw(n));
};
window.twH_selectAgent = function (id) {
  window.twHist.agent = id; window.twHist.evalId = null; window.twHist.idx = 0;
  window.twH_redraw(['hist-agent-select', 'hist-status', 'hist-field', 'hist-versions', 'hist-evals', 'hist-improvement-chart',
    'hist-training-chart', 'hist-diagnosis', 'hist-checkpoints', 'dr-eval-select', 'dr-chart', 'dr-step', 'dr-detail']);
};
window.twH_selectIdx = function (i) {
  window.twHist.idx = i;
  window.twH_redraw(['dr-chart', 'dr-step', 'dr-detail']);
};
window.twH_step = function (dir) {
  const rows = window.twH_decisions();
  if (!rows.length) return;
  const idxs = window.twH_filtered(rows);
  if (!idxs.length) return;
  const cur = window.twHist.idx;
  const next = dir > 0 ? idxs.find((i) => i > cur) : [...idxs].reverse().find((i) => i < cur);
  window.twH_selectIdx(next ?? (dir > 0 ? idxs[0] : idxs[idxs.length - 1]));
};

// Plain controls: listen only.
document.querySelectorAll('.tw-dr-filter').forEach((el) => el.addEventListener('change', () => {
  window.twHist.filter = el.value;
  const rows = window.twH_decisions();
  const idxs = window.twH_filtered(rows);
  window.twH_selectIdx(idxs.length ? idxs[0] : 0);
}));
document.querySelectorAll('.tw-dr-prev').forEach((el) => el.addEventListener('click', () => window.twH_step(-1)));
document.querySelectorAll('.tw-dr-next').forEach((el) => el.addEventListener('click', () => window.twH_step(1)));

const twH_res = { pass: 'Passed', fail: 'Failed', incomplete: 'Incomplete' };
const twH_statusClass = (s) => ['passed', 'improved'].includes(s) ? 'tw-pass' : s === 'failed' ? 'tw-fail' : 'tw-pending';
const twH_fmt = {
  pct: (v) => v == null || v === '' ? '—' : d3.format('+.1f')(+v) + '%',
  pct0: (v) => v == null || v === '' ? '—' : d3.format('.0f')(+v) + '%',
  x: (v) => v == null || v === '' ? '—' : d3.format('.2~f')(+v) + 'x',
  int: (v) => v == null || v === '' ? '—' : d3.format(',')(+v),
};

dash.draw('hist-agent-select', (sel) => {
  const ag = twH_rows('registry_agents');
  const cur = window.twH_agent();
  sel.replaceChildren(...(ag.length ? ag.map((r) => new Option(`${r.name} (${r.agent_id})`, r.agent_id))
    : [new Option('No agents in the registry yet', '')]));
  sel.disabled = !ag.length;
  if (cur) sel.value = cur.agent_id;
  sel.onchange = () => window.twH_selectAgent(sel.value);
});
dash.draw('hist-status', (el) => {
  const a = window.twH_agent();
  el.classList.remove('tw-pass', 'tw-fail', 'tw-pending');
  if (!a) { el.textContent = '—'; el.removeAttribute('data-where'); return; }
  el.setAttribute('data-where', 'agent_id=' + encodeURIComponent(a.agent_id));
  el.classList.add(twH_statusClass(a.status));
  el.textContent = a.status.charAt(0).toUpperCase() + a.status.slice(1);
});
dash.draw('hist-field', (el) => {
  const a = window.twH_agent();
  if (!a) { el.textContent = '—'; el.removeAttribute('data-where'); return; }
  el.setAttribute('data-where', 'agent_id=' + encodeURIComponent(a.agent_id));
  const k = el.dataset.key;
  let v = a[k];
  if (k === 'created_at' && v) v = v.replace('T', ' ').replace('Z', ' UTC');
  if ((k === 'passed_on_version' || k === 'champion_version') && v !== '' && v != null) v = 'v' + v;
  el.textContent = v === '' || v == null ? (k === 'passed_on_version' ? 'not yet' : k === 'champion_version' ? 'none' : '—') : String(v);
});

function twH_table(tb, rows, key, cells) {
  const tr = d3.select(tb).selectAll('tr').data(rows, key).join('tr');
  tr.attr('data-key', (r) => key(r).split('|').map(encodeURIComponent).join(','));
  tr.selectAll('td').data(cells).join('td')
    .attr('class', (c) => c.num ? 'tw-num' : null)
    .each(function (c) {
      const td = d3.select(this);
      td.selectAll('*').remove();
      if (c.pill) td.text('').append('span').attr('class', 'tw-pill ' + c.pill).text(c.t);
      else td.text(c.t);
    });
  return tr;
}
dash.draw('hist-versions', (tb) => {
  const a = window.twH_agent();
  const rows = a ? twH_rows('registry_versions').filter((v) => v.agent_id === a.agent_id) : [];
  twH_table(tb, rows, (r) => `${r.agent_id}|${r.version}`, (r) => [
    { t: 'v' + r.version, num: true }, { t: r.parent === '' ? '—' : 'v' + r.parent, num: true }, { t: r.kind },
    { t: twH_fmt.int(r.steps_added), num: true }, { t: twH_fmt.int(r.total_steps), num: true }, { t: r.changes },
    { t: r.checkpoints, num: true }, { t: r.history }, { t: r.reason || '—' }]);
});
dash.draw('hist-evals', (tb) => {
  const rows = window.twH_evals();
  twH_table(tb, rows, (r) => r.eval_id, (r) => [
    { t: 'v' + r.version, num: true }, { t: r.round_id }, { t: r.original },
    { t: twH_res[r.result] || r.result, pill: r.result === 'pass' ? 'tw-pass' : r.result === 'fail' ? 'tw-fail' : 'tw-pending' },
    { t: twH_fmt.pct(r.growth_pct), num: true }, { t: twH_fmt.pct0(r.deaths_pct), num: true },
    { t: twH_fmt.x(r.avg_leverage), num: true }, { t: twH_fmt.x(r.median_leverage), num: true },
    { t: twH_fmt.x(r.max_leverage), num: true }, { t: twH_fmt.pct(r.share_over_fatal_pct).replace('+', ''), num: true },
    { t: twH_fmt.pct0(r.aligned_pct), num: true }, { t: r.reason }]);
});
dash.draw('hist-checkpoints', (tb) => {
  const a = window.twH_agent();
  const rows = a ? twH_rows('registry_checkpoints').filter((c) => c.agent_id === a.agent_id) : [];
  twH_table(tb, rows, (r) => r.checksum, (r) => [
    { t: 'v' + r.version, num: true }, { t: twH_fmt.int(r.step), num: true }, { t: r.kind }, { t: r.checksum }, { t: r.rng_state }]);
});
dash.draw('hist-diagnosis', (ul) => {
  const ev = window.twH_eval();
  const a = window.twH_agent();
  const rows = a ? twH_rows('registry_diagnosis').filter((d) => d.agent_id === a.agent_id) : [];
  ul.replaceChildren();
  if (!rows.length) {
    const li = document.createElement('li');
    li.textContent = twH_ok('registry_diagnosis') ? 'No problems were detected in this agent’s recorded decisions.' : '—';
    ul.append(li);
    return;
  }
  const groups = d3.groups(rows, (d) => `v${d.version} · ${d.round_id}`);
  groups.reverse().forEach(([label, items]) => {
    const head = document.createElement('li');
    const b = document.createElement('b');
    b.textContent = label;
    head.append(b);
    ul.append(head);
    items.forEach((d) => {
      const li = document.createElement('li');
      const k = document.createElement('span');
      k.className = 'tw-kind' + (d.kind === 'Observed' ? ' tw-kind-obs' : '');
      k.textContent = d.kind;
      const t = document.createElement('span');
      t.textContent = d.text;
      li.append(k, t);
      ul.append(li);
    });
  });
});

function twH_axisStyle(g) { g.selectAll('text').style('fill', 'var(--cds-text-secondary)'); }
function twH_panels(el, panels, xs, xLabel, xAxis) {
  const W = Math.max(300, el.clientWidth || 600);
  const m = { l: 56, r: 16, t: 22, b: 40 };
  const ph = 100, gap = 34;
  const H = m.t + panels.length * ph + (panels.length - 1) * gap + m.b;
  const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%');
  const x = xs.range([m.l, W - m.r]);
  panels.forEach((p, i) => {
    const top = m.t + i * (ph + gap);
    p.y.range([top + ph, top]);
    svg.append('g').attr('class', 'tw-grid').attr('transform', `translate(${m.l},0)`)
      .call(d3.axisLeft(p.y).ticks(4).tickSize(-(W - m.l - m.r)).tickFormat(''));
    const ax = d3.axisLeft(p.y).tickFormat(p.fmt);
    if (p.tickValues) ax.tickValues(p.tickValues); else ax.ticks(4);
    svg.append('g').attr('transform', `translate(${m.l},0)`).call(ax).call(twH_axisStyle);
    svg.append('text').attr('class', 'tw-panel-title').attr('x', m.l).attr('y', top - 8).text(p.title);
    p.top = top;
  });
  svg.append('g').attr('transform', `translate(0,${H - m.b})`).call(xAxis ? xAxis(d3.axisBottom(x)) : d3.axisBottom(x).ticks(Math.max(2, Math.floor(W / 120))).tickFormat(d3.format('~s'))).call(twH_axisStyle);
  svg.append('text').attr('x', W - m.r).attr('y', H - 4).attr('text-anchor', 'end').text(xLabel);
  return { svg, x, W, H, m, ph };
}

dash.draw('hist-training-chart', (el) => {
  el.replaceChildren();
  const a = window.twH_agent();
  if (!twH_ok('registry_training') || !a) { el.textContent = '—'; return; }
  const rows = twH_rows('registry_training').filter((r) => r.agent_id === a.agent_id)
    .map((r) => ({ version: +r.version, step: +r.step, growth: +r.growth, lev: +r.avg_leverage }));
  el.style.minHeight = rows.length ? '' : 'auto';
  if (!rows.length) {
    const p = document.createElement('p');
    p.className = 'tw-muted tw-small';
    p.textContent = 'No training history recorded for this agent yet. The first gate run did not record it; every version trained from now on records its full history.';
    el.append(p);
    return;
  }
  const vs = [...new Set(rows.map((r) => r.version))].sort((p, q) => p - q);
  const panels = [
    { key: 'growth', title: 'Growth per training episode (log)', fmt: d3.format('+.2~f'), y: d3.scaleLinear().domain(d3.extent(rows, (r) => r.growth)).nice(4) },
    { key: 'lev', title: 'Average position size (x account)', fmt: (v) => d3.format('.3~g')(v) + 'x', y: d3.scaleLinear().domain([0, d3.max(rows, (r) => r.lev) || 1]).nice(4) },
  ];
  const c = twH_panels(el, panels, d3.scaleLinear().domain(d3.extent(rows, (r) => r.step)), 'training steps');
  vs.forEach((v, i) => {
    const vr = rows.filter((r) => r.version === v);
    panels.forEach((p) => c.svg.append('path').datum(vr).attr('d', d3.line().x((r) => c.x(r.step)).y((r) => p.y(r[p.key])))
      .style('fill', 'none').style('stroke', dash.colors[i % dash.colors.length]).style('stroke-width', 2));
  });
  const lg = d3.select(el).append('div').attr('class', 'tw-legend');
  vs.forEach((v, i) => {
    const s = lg.append('span');
    s.append('span').attr('class', 'tw-swatch').style('background', dash.colors[i % dash.colors.length]);
    s.append('span').text('v' + v);
  });
});

dash.draw('hist-improvement-chart', (el) => {
  el.replaceChildren();
  const ev = window.twH_evals().filter((e) => e.original === 'first evaluation');
  if (!twH_ok('registry_evals') || !ev.length) { el.textContent = '—'; return; }
  const rows = ev.map((e) => ({ version: +e.version, growth: +e.growth_pct, deaths: +e.deaths_pct, result: e.result }));
  const vmax = d3.max(rows, (r) => r.version);
  const panels = [
    { key: 'growth', title: 'Growth per test run (%)', fmt: (v) => d3.format('+~f')(v) + '%', y: d3.scaleLinear().domain(d3.extent([0, ...rows.map((r) => r.growth)])).nice(4) },
    { key: 'deaths', title: 'Test runs ending in death (%)', fmt: (v) => d3.format('~f')(v) + '%', y: d3.scaleLinear().domain([0, Math.max(10, d3.max(rows, (r) => r.deaths))]).nice(4) },
  ];
  const c = twH_panels(el, panels, d3.scaleLinear().domain([0.5, Math.max(1.5, vmax + 0.5)]), 'version',
    (ax) => ax.tickValues(d3.range(1, vmax + 1)).tickFormat((v) => 'v' + v));
  const tip = d3.select(el).append('div').attr('class', 'tw-tip').style('display', 'none');
  panels.forEach((p) => {
    if (rows.length > 1) c.svg.append('path').datum(rows).attr('d', d3.line().x((r) => c.x(r.version)).y((r) => p.y(r[p.key])))
      .style('fill', 'none').style('stroke', dash.colors[0]).style('stroke-width', 2);
    c.svg.selectAll(null).data(rows).join('circle').attr('cx', (r) => c.x(r.version)).attr('cy', (r) => p.y(r[p.key])).attr('r', 6)
      .style('fill', (r) => r.result === 'pass' ? 'var(--color-ok)' : r.result === 'fail' ? 'var(--color-bad)' : 'var(--color-fg-muted)')
      .style('stroke', 'var(--color-panel)').style('stroke-width', 2)
      .on('pointerenter', (e, r) => {
        tip.style('display', null).text(`v${r.version}: ${twH_res[r.result] || r.result}, growth ${d3.format('+.1f')(r.growth)}%, deaths ${r.deaths}%`);
        const box = el.getBoundingClientRect();
        tip.style('left', `${Math.min((c.x(r.version) / c.W) * box.width + 10, box.width - 220)}px`).style('top', `${((p.y(r[p.key])) / c.H) * box.height}px`);
      })
      .on('pointerleave', () => tip.style('display', 'none'));
  });
});

dash.draw('dr-eval-select', (sel) => {
  const ev = window.twH_evals();
  const cur = window.twH_eval();
  sel.replaceChildren(...(ev.length ? ev.map((e) => new Option(`v${e.version} · ${e.round_id} · ${twH_res[e.result] || e.result}`, e.eval_id))
    : [new Option('No evaluation recorded', '')]));
  sel.disabled = !ev.length;
  if (cur) sel.value = cur.eval_id;
  sel.onchange = () => { window.twHist.evalId = sel.value; window.twHist.idx = 0; window.twH_redraw(['dr-chart', 'dr-step', 'dr-detail', 'hist-diagnosis']); };
});

dash.draw('dr-chart', (el) => {
  el.replaceChildren();
  if (!twH_ok('registry_decisions')) { el.textContent = '—'; return; }
  const rows = window.twH_decisions();
  if (!rows.length) { el.textContent = 'No recorded decisions for this evaluation.'; return; }
  const idx = Math.min(window.twHist.idx, rows.length - 1);
  const panels = [
    { key: 'equity_after', title: 'Account after each decision ($)', fmt: (v) => '$' + d3.format(',.2f')(v), y: d3.scaleLinear().domain([0, Math.max(10, d3.max(rows, (r) => r.equity_after))]).nice(4) },
    { key: 'leverage', title: 'Position size chosen (x account, log scale)', fmt: (v) => d3.format('~g')(v) + 'x',
      y: d3.scaleSymlog().constant(0.1).domain([0, Math.max(1, d3.max(rows, (r) => r.leverage))]),
      tickValues: [0, 0.1, 1, 10, 100, 1000, 10000].filter((v) => v <= Math.max(1, d3.max(rows, (r) => r.leverage)) * 1.01) },
  ];
  const c = twH_panels(el, panels, d3.scaleLinear().domain(d3.extent(rows, (r) => r.step)), 'decision');
  panels.forEach((p) => c.svg.append('path').datum(rows).attr('d', d3.line().x((r) => c.x(r.step)).y((r) => p.y(r[p.key])))
    .style('fill', 'none').style('stroke', dash.colors[0]).style('stroke-width', 1.5));
  const deaths = rows.filter((r) => r.died === 'yes');
  deaths.forEach((r) => c.svg.append('line').attr('x1', c.x(r.step)).attr('x2', c.x(r.step)).attr('y1', c.m.t).attr('y2', c.H - c.m.b)
    .style('stroke', 'var(--cds-chart-status-critical)').style('stroke-width', 2));
  const sel = rows[idx];
  c.svg.append('line').attr('x1', c.x(sel.step)).attr('x2', c.x(sel.step)).attr('y1', c.m.t).attr('y2', c.H - c.m.b)
    .style('stroke', 'var(--color-fg)').style('stroke-dasharray', '3 3');
  panels.forEach((p) => c.svg.append('circle').attr('cx', c.x(sel.step)).attr('cy', p.y(sel[p.key])).attr('r', 5)
    .style('fill', dash.colors[0]).style('stroke', 'var(--color-panel)').style('stroke-width', 2));
  const bis = d3.bisector((r) => r.step).center;
  c.svg.append('rect').attr('x', c.m.l).attr('y', c.m.t).attr('width', c.W - c.m.l - c.m.r).attr('height', c.H - c.m.t - c.m.b)
    .style('fill', 'transparent').style('cursor', 'pointer')
    .on('click', (e) => { const [px] = d3.pointer(e); window.twH_selectIdx(bis(rows, c.x.invert(px))); });
});

dash.draw('dr-step', (el) => {
  const rows = window.twH_decisions();
  const e = window.twH_eval();
  if (!rows.length || !e) { el.textContent = '—'; el.removeAttribute('data-where'); return; }
  const r = rows[Math.min(window.twHist.idx, rows.length - 1)];
  el.setAttribute('data-where', `eval_id=${encodeURIComponent(e.eval_id)}&step=${r.step}`);
  el.textContent = `${d3.format(',')(r.step)} of ${d3.format(',')(rows.length)}`;
});

dash.draw('dr-detail', (el) => {
  el.replaceChildren();
  const rows = window.twH_decisions();
  if (!rows.length) { el.textContent = '—'; return; }
  const r = rows[Math.min(window.twHist.idx, rows.length - 1)];
  const pnl = r.equity_after - r.equity_before;
  const items = [
    ['Account before', '$' + d3.format(',.4f')(r.equity_before)],
    ['Direction chosen', r.direction],
    ['Size chosen', d3.format('.4~g')(r.size_chosen) + 'x'],
    ['Position after the order', d3.format('.3~g')(r.leverage) + 'x of account'],
    ['Order', r.position_change],
    ['Market move next', d3.format('+.3f')(r.market_move_pct) + '%'],
    ['Expected move (known to us, not the agent)', r.expected_move_pct == null ? 'unknown' : d3.format('+.4f')(r.expected_move_pct) + '%'],
    ['Trading costs', '$' + d3.format(',.5f')(r.costs)],
    ['Result', (pnl >= 0 ? '+$' : '−$') + d3.format(',.4f')(Math.abs(pnl))],
    ['Account after', '$' + d3.format(',.4f')(r.equity_after)],
    ['Reward received', d3.format('+.6f')(r.reward)],
    ['Rejected order', r.rejected || 'none'],
    ['Account died', r.died === 'yes' ? `yes (${r.death_reason})` : 'no'],
  ];
  items.forEach(([k, v]) => {
    const d = document.createElement('div');
    const a = document.createElement('span');
    a.textContent = k;
    const b = document.createElement('span');
    b.textContent = v;
    d.append(a, b);
    el.append(d);
  });
});
