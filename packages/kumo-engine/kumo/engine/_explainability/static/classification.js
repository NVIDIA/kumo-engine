/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Classification: the cutoff card and Scores against outcomes. */

function cutoffState(card) {
  const cut = {};
  for (const c of card.classes) cut[c.name] = card.default;
  return {cls: card.classes[0].name, cut};
}

function mix(color, percent, base) {
  return `color-mix(in srgb, ${color} ${percent.toFixed(1)}%, ${base || COLOR.bg})`;
}

function matrixView(host, width, cls, at) {
  /* counts are [tp, fp, fn, tn]; rows are actual positive / negative,
     columns predicted positive / negative. */
  const [tp, fp, fn, tn] = at.counts;
  const cells = [[tp, fn, at.cells[0], at.cells[2]], [fp, tn, at.cells[1], at.cells[3]]];
  const n = tp + fp + fn + tn || 1;
  const rowLab = 104, totW = 76, gap = 6;
  const cellW = Math.max(96, Math.min(220, (width - rowLab - totW - gap * 3) / 2));
  const cellH = 64, top = 30;
  const totalW = rowLab + 2 * cellW + gap + totW + gap;
  const height = top + 2 * cellH + gap + 30 + 26;
  const svg = svgRoot(host, Math.min(width, totalW), height, 'Confusion matrix');
  const g = S('g', {}, svg);
  const cx = j => rowLab + j * (cellW + gap);
  const cy = i => top + i * (cellH + gap);
  cls.matrix_columns.forEach((label, j) => T(g, cx(j) + cellW / 2, top - 12, label, 'kx-t-axis', 'middle',
    {style: {fill: COLOR.ink, fontWeight: 600}}));
  T(g, cx(2) + totW / 2, top - 12, 'total', 'kx-t-axis', 'middle');
  cls.matrix_rows.forEach((label, i) => T(g, rowLab - 12, cy(i) + cellH / 2 + 4, label, 'kx-t-axis', 'end',
    {style: {fill: COLOR.ink, fontWeight: 600}}));
  for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) {
    const count = cells[i][j], share = count / n;
    const flagged = j === 0;
    const strength = 14 + 86 * Math.sqrt(share);
    const fillColor = flagged ? mix(COLOR.green, strength) : mix(COLOR.base, strength);
    S('rect', {x: cx(j), y: cy(i), width: cellW, height: cellH, rx: 8, style: {fill: fillColor}}, g);
    const text = cells[i][2 + j].split(' · ');
    const dark = flagged && strength > 55;
    const ink = dark ? 'var(--kx-on-green)' : COLOR.ink;
    T(g, cx(j) + cellW / 2, cy(i) + cellH / 2 + 1, text[0], 'kx-t-cell', 'middle',
      {style: {fill: ink, fontSize: '17px', fontWeight: 700}});
    if (text[1]) T(g, cx(j) + cellW / 2, cy(i) + cellH / 2 + 19, text[1], null, 'middle',
      {style: {fill: ink, fontSize: '12px', opacity: 0.85}});
  }
  const rowTotals = [tp + fn, fp + tn], colTotals = [tp + fp, fn + tn];
  rowTotals.forEach((v, i) => T(g, cx(2) + totW / 2, cy(i) + cellH / 2 + 5, v.toLocaleString('en-US'),
    'kx-t-axis', 'middle', {style: {fill: COLOR.ink}}));
  const ty = cy(2) + 14;
  colTotals.forEach((v, j) => T(g, cx(j) + cellW / 2, ty, v.toLocaleString('en-US'), 'kx-t-axis', 'middle',
    {style: {fill: COLOR.ink}}));
  T(g, rowLab - 12, ty, 'total', 'kx-t-axis', 'end');
  T(g, cx(0) + cellW / 2, ty + 20, at.matrix_callout, 'kx-t-note', 'middle');
}

function curveView(host, width, kind, cls, at, view) {
  const opts = {height: 340, maxRatio: 2.3, extraRight: 0, label: kind};
  const lines = [];
  let point, xt, yt;
  if (kind === 'roc') {
    lines.push({points: [[0, 0], [1, 1]], color: COLOR.axis, width: 1.4, dash: '5 4'});
    lines.push({points: cls.roc, color: ROLE.actual, width: 2.4});
    point = [at.fpr, at.recall]; xt = view.x; yt = view.y;
  } else if (kind === 'pr') {
    lines.push({points: [[0, cls.share], [1, cls.share]], color: COLOR.axis, width: 1.4, dash: '5 4'});
    lines.push({points: cls.pr, color: ROLE.actual, width: 2.4});
    point = [at.recall, at.precision]; xt = view.x; yt = view.y;
    /* The axis starts just under the lowest precision reached, so the
       curve's shape fills the plot instead of its top fifth. */
    const low = Math.min(cls.share, ...cls.pr.map(p => p[1]).filter(v => isFinite(v)));
    opts.yDomain = [Math.max(0, Math.floor((low - 0.03) * 10) / 10), 1];
  } else {
    lines.push({points: [[0, 0], [1, 1]], color: COLOR.axis, width: 1.4, dash: '5 4'});
    lines.push({points: [[0, 0], [cls.share, 1], [1, 1]], color: COLOR.base, width: 1.6, dash: '2 3'});
    lines.push({points: cls.gain, color: ROLE.actual, width: 2.4});
    point = [at.flagged, at.recall];
    xt = view.x;
    yt = 'share of ' + cls.name + ' captured';
  }
  const fmt = v => Math.round(v * 100) + '%';
  const shareAxes = kind === 'gain';
  const c = curves(host, width, Object.assign(opts, {lines, xTitle: xt, yTitle: yt,
    xFmt: shareAxes ? fmt : null, yFmt: shareAxes ? fmt : null}));
  const {x, y, notes, space} = c;
  /* The cutoff point is placed first, then the fixed line labels, which
     stay even when nothing is flagged and the point is undefined. */
  const ok = point[0] !== null && point[1] !== null && isFinite(point[0]) && isFinite(point[1]);
  if (ok) cutoffNote(c, kind, at, x(point[0]), y(point[1]));
  refNotes(c, kind, cls, view);
  if (ok) marker(c.g, x(point[0]), y(point[1]));
}

function cutoffNote(c, kind, at, px, py) {
  const {notes, space} = c;
  space.add(px - 8, py - 8, 16, 16);
  const callout = kind === 'roc' ? at.roc_callout : kind === 'pr' ? at.pr_callout : at.gain_callout;
  const right = px < (c.x0 + c.x1) / 2;
  const box = note(notes, space, twoLines(callout), [
    {x: px + 12, y: py + 18}, {x: px - 12, y: py + 18, anchor: 'end'},
    {x: px + 12, y: py - 10}, {x: px - 12, y: py - 10, anchor: 'end'},
    {x: right ? px + 12 : px - 12, y: py + 34, anchor: right ? 'start' : 'end'},
    {x: right ? px + 12 : px - 12, y: py - 26, anchor: right ? 'start' : 'end'},
    {x: px - 12, y: py + 50, anchor: 'end'}, {x: px + 12, y: py + 50},
    {x: right ? px + 12 : px - 12, y: py + 66, anchor: right ? 'start' : 'end'},
    {x: right ? px + 12 : px - 12, y: py - 42, anchor: right ? 'start' : 'end'}]);
  leader(c.g, px, py, box);
}

/* Reference-line labels, at fixed places on their lines. */
function refNotes(c, kind, cls, view) {
  const {x, y, notes, space} = c;
  if (kind === 'roc') {
    /* On the curve: just under it, or just over it, near its bend. */
    const on = f => {
      const pts = cls.roc;
      for (let i = 1; i < pts.length; i++) {
        if (pts[i][0] >= f) {
          const [a, b] = [pts[i - 1], pts[i]];
          const k = b[0] > a[0] ? (f - a[0]) / (b[0] - a[0]) : 1;
          return a[1] + k * (b[1] - a[1]);
        }
      }
      return 1;
    };
    const spots = [];
    for (const f of [0.3, 0.38, 0.22, 0.46, 0.16, 0.55]) {
      spots.push({x: x(f) + 8, y: y(on(f)) + 20}, {x: x(f) - 8, y: y(on(f)) - 9, anchor: 'end'});
    }
    note(notes, space, cls.auroc_callout, spots);
    note(notes, space, view.diagonal, [{x: x(0.64), y: y(0.55)}, {x: x(0.78), y: y(0.7)},
      {x: x(0.5), y: y(0.4)}], 'kx-t-note-plain');
  } else if (kind === 'pr') {
    const by = y(cls.share);
    note(notes, space, cls.base_rate_callout, [{x: c.x1 - 4, y: by - 7, anchor: 'end'},
      {x: c.x1 - 4, y: by + 16, anchor: 'end'}, {x: c.x0 + 6, y: by + 16}], 'kx-t-note-plain');
    note(notes, space, cls.ap_callout, [{x: x(0.5), y: y(Math.min(1, cls.share + 0.5 * (1 - cls.share))) - 8, anchor: 'middle'},
      {x: x(0.2), y: y(0.98)}, {x: x(0.5), y: c.y0 + 14, anchor: 'middle'}]);
  } else {
    note(notes, space, view.diagonal, [{x: x(0.7), y: y(0.62)}, {x: x(0.8), y: y(0.72)}], 'kx-t-note-plain');
    note(notes, space, cls.ideal_callout, [{x: x(cls.share) + 8, y: y(1) + 16},
      {x: x(cls.share) - 8, y: y(1) - 6, anchor: 'end'}, {x: x(cls.share) - 8, y: y(1) + 16, anchor: 'end'},
      {x: x(cls.share) + 8, y: y(1) + 32}, {x: x(cls.share) - 8, y: y(1) + 32, anchor: 'end'}], 'kx-t-note-plain');
  }
}


KX.cards.cutoff = (parent, card, ctx) => {
  const state = cutoffState(card);
  ctx.state.set('cutoff', {cls: state.cls, index: state.cut[state.cls], card});
  const views = card.views;
  let refresh = () => {};
  const controls = root => {
    const box = H('div', 'kx-controls', root);
    const row = H('div', 'kx-slider-row', box);
    const label = H('label', null, row, 'Cutoff');
    const range = H('div', 'kx-range', row);
    const input = H('input', null, range);
    input.type = 'range'; input.min = '0'; input.max = String(card.grid.length - 1); input.step = '1';
    input.id = ctx.host.id + '-cutoff';
    label.htmlFor = input.id;
    const tickBox = H('div', 'kx-ticks', range);
    for (const t of [0, 0.25, 0.5, 0.75, 1]) {
      const i = card.grid.findIndex(v => Math.abs(v - t) < 1e-9);
      if (i < 0) continue;
      const span = H('span', i === card.default ? 'kx-default' : null, tickBox, card.grid_text[i]);
      span.style.left = (100 * i / (card.grid.length - 1)) + '%';
    }
    const value = H('span', 'kx-cut', row);
    const ctl = H('div', 'kx-ctl-row', box);
    const stops = H('div', 'kx-group', ctl);
    H('b', null, stops, 'Stops');
    const classes = H('div', 'kx-group', ctl);
    H('b', null, classes, 'Positive class');
    const classButtons = card.classes.map(c => {
      const b = H('button', 'kx-chipbtn', classes, c.name);
      b.type = 'button';
      b.addEventListener('click', () => { state.cls = c.name; update(); }, {signal: ctx.signal});
      return b;
    });
    const strip = H('div', 'kx-strip', box);
    strip.setAttribute('aria-live', 'polite');
    const stopButtons = [];
    const renderStops = cls => {
      stopButtons.splice(0).forEach(b => b.remove());
      const items = [{index: card.default, extra: null}];
      if (cls.preset !== null && cls.preset !== undefined && cls.preset !== card.default) {
        items.push({index: cls.preset, extra: pct(cls.cutoffs[cls.preset].flagged, 1) + ' flagged'});
      }
      for (const item of items) {
        const b = H('button', 'kx-chipbtn', stops, card.grid_text[item.index]);
        b.type = 'button';
        b.dataset.index = item.index;
        if (item.extra) H('small', null, b, item.extra);
        b.addEventListener('click', () => { state.cut[state.cls] = item.index; update(); }, {signal: ctx.signal});
        stopButtons.push(b);
      }
    };
    const update = () => {
      const cls = card.classes.find(c => c.name === state.cls);
      const index = state.cut[state.cls];
      const at = cls.cutoffs[index];
      input.value = String(index);
      input.setAttribute('aria-valuetext', card.grid_text[index]);
      range.style.setProperty('--kx-fill', `calc(9px + (100% - 18px) * ${index / (card.grid.length - 1)})`);
      value.textContent = card.grid_text[index];
      classButtons.forEach((b, i) => b.setAttribute('aria-pressed', String(card.classes[i].name === state.cls)));
      renderStops(cls);
      stopButtons.forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.index) === index)));
      strip.replaceChildren();
      for (const item of at.strip) {
        const span = H('span', null, strip, item.label.charAt(0).toUpperCase() + item.label.slice(1));
        H('b', null, span, item.text);
      }
      ctx.state.set('cutoff', {cls: state.cls, index, card});
      refresh();
    };
    input.addEventListener('input', () => { state.cut[state.cls] = Number(input.value); update(); },
      {signal: ctx.signal});
    box.update = update;
    return box;
  };
  let box;
  const frame = cardFrame(parent, card, ctx, views, (body, view) => {
    const draw = (host, width) => {
      const cls = card.classes.find(c => c.name === state.cls);
      const at = cls.cutoffs[state.cut[state.cls]];
      if (view.id === 'matrix') matrixView(host, width, cls, at);
      else curveView(host, width, view.id, cls, at, view);
    };
    const chart = chartHost(body, draw, ctx);
    refresh = () => chart.redraw();
  }, root => { box = controls(root); });
  box.update();
  return frame;
};

KX.cards.scores = (parent, card, ctx) => {
  let chart = null;
  ctx.state.on('cutoff', () => { if (chart && chart.isConnected && chart.dataset.view === 'outcome') chart.redraw(); });
  return cardFrame(parent, card, ctx, card.views, (body, view) => {
    if (view.id === 'tenths') {
      legend(body, [{label: view.series[0], color: ROLE.model}, {label: view.series[1], color: ROLE.actual},
        {label: 'all ' + ENTITIES, kind: 'dash', color: COLOR.ink, width: 1.4}]);
      chart = chartHost(body, (host, width) => columns(host, width, {
        categories: view.tenths.map((t, i) => ({label: t.label, values: [t.score, t.observed],
          texts: endLabels(view, i) ? [t.score_text, t.observed_text] : []})),
        series: [{color: ROLE.model}, {color: ROLE.actual}],
        highlight: view.highlight, highlightBand: true,
        ref: {value: view.reference, text: view.reference_callout},
        notes: [{index: view.highlight, text: view.gap_callout}],
        yTitle: view.y, xTitle: view.x, height: 300, label: card.title,
      }), ctx);
    } else {
      const [pos, neg] = view.series;
      legend(body, [{label: pos.name, color: ROLE.actual}, {label: neg.name, color: ROLE.other},
        {label: 'cutoff', kind: 'line', color: COLOR.green, width: 2.5}]);
      chart = chartHost(body, (host, width) => histogramView(host, width, view, ctx), ctx);
    }
    chart.dataset.view = view.id;
  });
};

function histogramView(host, width, view, ctx) {
  const edges = view.edges;
  const series = view.series;
  const totals = edges.slice(1).map((e, i) => series.reduce((a, s) => a + s.counts[i], 0));
  const maxCount = Math.max(...totals);
  const state = ctx.state.get('cutoff');
  let pos = null;
  if (state) {
    const t = state.card.grid[state.index];
    pos = state.cls === state.card.classes[0].name ? t : 1 - t;
  }
  const lo = Math.min(edges[0], pos === null ? edges[0] : pos);
  const hi = Math.max(edges[edges.length - 1], pos === null ? 0 : pos);
  const c = curves(host, width, {height: 300, xDomain: [lo, hi],
    yDomain: [0, maxCount * 1.12], lines: [], xTitle: view.x, yTitle: view.y,
    xFmt: v => v.toFixed(2), yFmt: v => v.toLocaleString('en-US'), label: 'Scores by outcome'});
  const {x, y, g, space, notes} = c;
  const colors = [ROLE.actual, ROLE.other];
  /* One solid bar per bin, the two outcomes stacked, so every bar's height
     is the bin's whole count and no colour is ever blended. */
  totals.forEach((total, i) => {
    const bx = x(edges[i]) + 0.5, bw = Math.max(1, x(edges[i + 1]) - x(edges[i]) - 1);
    let base = 0;
    series.forEach((s, k) => {
      const count = s.counts[i];
      if (!count) return;
      const top = base + count;
      S('rect', {x: bx, y: y(top), width: bw, height: Math.max(0.8, y(base) - y(top)),
        style: {fill: colors[k]}}, g);
      base = top;
    });
    if (total) space.add(bx, y(total), bw, y(0) - y(total));
  });
  /* Each class's median: a thin line in its colour on a halo of the panel,
     labelled at its top. */
  const binOf = v => Math.max(0, Math.min(totals.length - 1, edges.findIndex(e => e > v) - 1));
  const under = Math.max(0, ...series.filter(s => typeof s.median === 'number').map(s => totals[binOf(s.median)]));
  const medianTop = Math.min(c.y0 + 34, y(under) - 16);
  series.forEach((s, k) => {
    if (typeof s.median !== 'number') return;
    const mx = x(s.median);
    S('path', {d: `M${mx} ${c.y1}V${medianTop}`, style: {stroke: COLOR.panel, strokeWidth: 4.5}}, g);
    S('path', {d: `M${mx} ${c.y1}V${medianTop}`, style: {stroke: colors[k], strokeWidth: 1.6,
      strokeDasharray: '4 3'}}, g);
    space.add(mx - 2, medianTop, 4, c.y1 - medianTop);
  });
  if (state) {
    const text = state.card.classes.find(cl => cl.name === state.cls).cutoffs[state.index].line_callout;
    const px = x(pos);
    S('path', {d: `M${px} ${c.y0 - 4}V${c.y1}`, style: {stroke: COLOR.panel, strokeWidth: 5.5}}, g);
    S('path', {d: `M${px} ${c.y0 - 4}V${c.y1}`, style: {stroke: COLOR.green, strokeWidth: 2.5}}, g);
    space.add(px - 2, c.y0, 4, c.y1 - c.y0);
    note(notes, space, text, [{x: px + 6, y: c.y0 + 10}, {x: px - 6, y: c.y0 + 10, anchor: 'end'},
      {x: px + 6, y: c.y0 + 40}, {x: px - 6, y: c.y0 + 40, anchor: 'end'}]);
  }
  const i = view.tallest;
  if (i !== null && i !== undefined) {
    const cx = (x(edges[i]) + x(edges[i + 1])) / 2;
    const ty = y(totals[i]) - 8;
    const bl = x(edges[i]) - 6, br = x(edges[i + 1]) + 6, by = y(totals[i]) + 12;
    /* Count on the first line, the score range on the second. */
    const text = view.tallest_callout.replace(/ (score \d)/, '\n$1');
    note(notes, space, text, [{x: bl, y: by, anchor: 'end'}, {x: br, y: by},
      {x: bl, y: by + 18, anchor: 'end'}, {x: br, y: by + 18}, {x: cx, y: ty, anchor: 'middle'}]);
  }
  series.forEach((s, k) => {
    if (typeof s.median !== 'number') return;
    const mx = x(s.median);
    note(notes, space, s.median_callout, [{x: mx, y: medianTop - 6, anchor: 'middle'},
      {x: mx + 6, y: medianTop + 10}, {x: mx - 6, y: medianTop + 10, anchor: 'end'},
      {x: mx + 6, y: medianTop + 28}, {x: mx - 6, y: medianTop + 28, anchor: 'end'}], 'kx-t-note-plain');
  });
}
