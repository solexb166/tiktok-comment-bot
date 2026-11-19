// Road to demo trading and Learning experiments. Reads roadmap, experiments and experiment_seeds; draws only inside drawings.
window.twExp = window.twExp || {};

function twE_rows(id) {
  const s = dash.data(id);
  return s.status === 'ok' ? s.data : [];
}
const twE_statusClass = { passed: 'tw-pass', failed: 'tw-fail' };
const twE_statusText = { passed: 'Passed', failed: 'Failed', pending: 'Running', running: 'Running', 'not started': 'Not started' };
const twE_short = { exp1: 'Exp 1', exp1b: 'Exp 1b', rfl: 'R-FL', m1: 'M1', m2: 'M2' };

dash.draw('roadmap-passed', (el) => {
  const r = twE_rows('roadmap');
  el.textContent = r.length ? `${r.filter((x) => x.status === 'passed').length} of ${r.length}` : '—';
});

dash.draw('roadmap-list', (ol) => {
  const s = dash.data('roadmap');
  ol.replaceChildren();
  if (s.status === 'error') { ol.textContent = 'Failed to load'; return; }
  if (s.status !== 'ok') return;
  s.data.slice().sort((a, b) => a.order - b.order).forEach((r) => {
    const li = d3.select(ol).append('li').attr('class', 'tw-stage').attr('data-key', r.order);
    li.append('span').attr('class', 'tw-stage-n').text(r.order);
    const body = li.append('div');
    const head = body.append('div').attr('class', 'tw-row');
    head.append('span').attr('class', 'tw-stage-title').text(r.stage);
    head.append('span').attr('class', 'tw-pill ' + (twE_statusClass[r.status] || 'tw-pending')).text(twE_statusText[r.status] || r.status);
    if (r.evidence) body.append('p').attr('class', 'tw-muted tw-small').text(r.evidence);
  });
});

dash.draw('exp-rows', (tb) => {
  const g = twE_rows('experiments');
  const rows = d3.select(tb).selectAll('tr').data(g, (r) => r.experiment + '/' + r.group).join('tr');
  rows.attr('data-key', (r) => encodeURIComponent(r.experiment + '/' + r.group));
  const f2 = (v) => v == null ? '—' : d3.format('.2f')(v) + 'x';
  rows.selectAll('td').data((r) => [
    { t: twE_short[r.experiment] || r.experiment },
    { t: r.agent }, { t: `${r.market_name} (${r.market})` },
    { t: r.successes == null ? 'not scored per agent' : `${r.successes} of ${r.n}`, num: true },
    { t: r.verdict, pill: r.passes ? 'tw-pass' : 'tw-fail' },
    { t: f2(r.median_size), num: true }, { t: (r.market === 'S4' || r.market === 'S3') ? 'none (0x)' : r.optimal_size == null ? 'changes every bar' : f2(r.optimal_size), num: true },
    { t: r.ruinous_pct == null ? '—' : d3.format('.1f')(r.ruinous_pct) + '%', num: true },
  ]).join('td')
    .attr('class', (c) => c.num ? 'tw-num' : null)
    .each(function (c) {
      const td = d3.select(this);
      td.selectAll('*').remove();
      if (c.pill) td.text('').append('span').attr('class', 'tw-pill ' + c.pill).text(c.t);
      else td.text(c.t);
    });
});

dash.draw('exp-chart', (el) => {
  const s = dash.data('experiment_seeds');
  const gs = twE_rows('experiments');
  el.replaceChildren();
  if (s.status === 'error') { el.textContent = 'Failed to load'; return; }
  if (s.status !== 'ok' || !gs.length) { el.textContent = '—'; return; }
  const keyOf = (r) => r.experiment + '/' + r.group;
  const groups = gs.map((g) => ({ key: keyOf(g), label: `${twE_short[g.experiment]} · ${g.agent} · ${g.market}`, g }));
  const W = Math.max(320, el.clientWidth || 640);
  const narrow = W < 560;
  const m = { l: narrow ? 8 : 300, r: 16, t: 26, b: 40 };
  const rh = narrow ? 40 : 26;
  const H = m.t + groups.length * rh + m.b;
  const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${W} ${H}`).attr('width', '100%')
    .attr('role', 'img').attr('aria-label', 'Typical position size of every agent in every experiment, against the best size for its market');
  const x = d3.scaleSymlog().constant(0.5).domain([0, 30]).range([m.l, W - m.r]);
  const ticks = [0, 0.5, 1, 2, 3, 5, 10, 20, 30];
  svg.append('g').attr('class', 'tw-grid').attr('transform', `translate(0,${m.t})`)
    .call(d3.axisTop(x).tickValues(ticks).tickSize(-(groups.length * rh)).tickFormat(''));
  svg.append('g').attr('transform', `translate(0,${H - m.b})`)
    .call(d3.axisBottom(x).tickValues(ticks).tickFormat((v) => v + 'x'));
  svg.append('text').attr('x', W - m.r).attr('y', H - 4).attr('text-anchor', 'end').text('typical (median) position size, x account');
  // the ruin line on the crash market: a 10% crash wipes out an account at 10x
  const ruinX = x(9.9);
  const color = { true: 'var(--color-ok)', false: 'var(--color-bad)', null: 'var(--cds-chart-muted)' };
  groups.forEach((G, i) => {
    const y0 = m.t + i * rh;
    const cy = narrow ? y0 + rh - 12 : y0 + rh / 2;
    if (i % 2 === 0) svg.append('rect').attr('x', 0).attr('y', y0).attr('width', W).attr('height', rh)
      .style('fill', 'var(--color-fg)').style('opacity', 0.03);
    svg.append('text').attr('x', narrow ? m.l : m.l - 10).attr('y', narrow ? y0 + 12 : cy + 4)
      .attr('text-anchor', narrow ? 'start' : 'end').text(G.label);
    if (G.g.market === 'S6') svg.append('line').attr('x1', ruinX).attr('x2', ruinX).attr('y1', cy - 9).attr('y2', cy + 9)
      .style('stroke', 'var(--cds-chart-status-critical)').style('stroke-width', 2).style('stroke-dasharray', '3 2');
    if (G.g.optimal_size != null) svg.append('path').attr('d', d3.symbol(d3.symbolDiamond, 70)()).attr('transform', `translate(${x(G.g.optimal_size)},${cy})`)
      .style('fill', 'none').style('stroke', 'var(--color-fg)').style('stroke-width', 1.5);
  });
  const tip = d3.select(el).append('div').attr('class', 'tw-tip').style('display', 'none');
  const rowIndex = new Map(groups.map((G, i) => [G.key, i]));
  const pts = s.data.filter((r) => rowIndex.has(keyOf(r)));
  svg.selectAll('circle.seed').data(pts).join('circle').attr('class', 'seed')
    .attr('cx', (r) => x(Math.min(30, Math.abs(r.median_size || 0))))
    .attr('cy', (r) => {
      const i = rowIndex.get(keyOf(r));
      const y0 = m.t + i * rh;
      return (narrow ? y0 + rh - 12 : y0 + rh / 2) + (r.seed_index - 3.5) * 1.6;
    })
    .attr('r', 4.5).style('fill', (r) => color[r.success]).style('opacity', 0.85)
    .style('stroke', 'var(--color-panel)').style('stroke-width', 1)
    .on('pointerenter', (e, r) => {
      tip.style('display', null).html('');
      tip.append('div').append('b').text(`${twE_short[r.experiment]} · ${r.group} · seed ${r.seed_index}`);
      tip.append('div').text(`Agent ${r.agent_id}`);
      tip.append('div').text(`Typical size ${d3.format('.2f')(Math.abs(r.median_size || 0))}x (best ${r.optimal_size}x)`);
      if (r.ruinous_pct != null) tip.append('div').text(`Decisions at a ruinous size ${d3.format('.1f')(r.ruinous_pct)}%`);
      if (r.deaths != null) tip.append('div').text(`Test deaths ${r.deaths}${r.runs ? ' of ' + r.runs + ' runs' : ''}`);
      if (r.growth != null) tip.append('div').text(`Growth per run ${d3.format('+.3f')(r.growth)}` +
        (r.growth_lo != null ? ` (${d3.format('+.3f')(r.growth_lo)} to ${d3.format('+.3f')(r.growth_hi)})` : ''));
      tip.append('div').text(r.success == null ? 'Not scored per agent' : r.success ? 'Met every criterion' : `Missed: ${(r.failed || []).join(', ')}`);
      const box = el.getBoundingClientRect();
      const [px, py] = d3.pointer(e, el);
      tip.style('left', `${Math.min(px + 12, box.width - 230)}px`).style('top', `${py + 12}px`);
    })
    .on('pointerleave', () => tip.style('display', 'none'));
});
