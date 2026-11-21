/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Classification: the cutoff card and Scores against outcomes. */

/* Python's formats of a nonnegative double, which round its exact value
   half to even: `fixed` to some decimals, `rate` to three significant figures
   and `share` as a one-decimal percentage. */
function scaled(v, places) {
  const view = new DataView(new ArrayBuffer(8));
  view.setFloat64(0, v);
  const bits = view.getBigUint64(0), exp = Number(bits >> 52n);
  const shift = BigInt(Math.max(exp, 1) - 1075);
  let num = ((bits & (1n << 52n) - 1n) | (exp ? 1n << 52n : 0n)) * 10n ** BigInt(places), den = 1n;
  if (shift < 0) den <<= -shift; else num <<= shift;
  const q = num / den, twice = 2n * (num - q * den);
  return q + (twice > den || twice === den && q % 2n ? 1n : 0n);
}
function fixed(v, places) {
  const text = String(scaled(v, places)).padStart(places + 1, '0');
  return places ? text.slice(0, -places) + '.' + text.slice(-places) : text;
}
function rate(v) {
  if (!v) return '0.00';
  const places = 2 - Math.floor(Math.log10(v));
  return fixed(v, scaled(v, places) >= 1000n ? places - 1 : places);
}
const share = v => fixed(100 * v, 1) + '%';

/* Rates and texts at grid cutoff i of a class, from its [tp, fp, fn, tn]. */
function cutoffAt(card, cls, i) {
  const counts = cls.counts[i], [tp, fp, fn, tn] = counts, n = tp + fp + fn + tn;
  const at = {counts, flagged: (tp + fp) / n, precision: tp + fp ? tp / (tp + fp) : 1,
    recall: tp + fn ? tp / (tp + fn) : 0, fpr: fp + tn ? fp / (fp + tn) : 1};
  return Object.assign(at, {
    cells: counts.map(c => c.toLocaleString('en-US') + ' \u00b7 ' + share(c / n)),
    strip: [['precision', rate(at.precision)], ['recall', rate(at.recall)], ['accuracy', rate((tp + tn) / n)],
      ['flagged', share(at.flagged)], ['F1', rate(2 * tp / (2 * tp + fp + fn || 1))]],
    /* The score histogram's axis is True_PROB, so the cutoff line names the
       value on that axis. */
    line_callout: 'cutoff ' + card.grid_text[cls.name === card.classes[0].name ? i : card.grid.length - 1 - i]});
}

/* The class's curve through every grid cutoff, at two decimals. */
const curvePoints = (card, cls, x, y) => cls.counts.map((_, i) => cutoffAt(card, cls, i))
  .map(at => [Number(fixed(at[x], 2)), Number(fixed(at[y], 2))]);

function matrixView(host, width, cls, at) {
  /* counts are [tp, fp, fn, tn]; rows are actual positive / negative,
     columns predicted positive / negative. */
  const [tp, fp, fn, tn] = at.counts;
  const cells = [[tp, fn, at.cells[0], at.cells[2]], [fp, tn, at.cells[1], at.cells[3]]];
  const n = tp + fp + fn + tn || 1;
  const rowLab = 104, totW = 76, gap = 6;
  /* Cells shrink to 64px so the totals column never runs past the chart. */
  const cellW = Math.max(64, Math.min(168, (width - rowLab - totW - gap * 3) / 2));
  const cellH = 56, stacked = cellW < 130, top = stacked ? 46 : 30;
  const totalW = rowLab + 2 * cellW + gap + totW + gap;
  const height = top + 2 * cellH + gap + 30;
  const svg = svgRoot(host, Math.min(width, totalW), height, 'Confusion matrix');
  const g = S('g', {}, svg);
  const cx = j => rowLab + j * (cellW + gap);
  const cy = i => top + i * (cellH + gap);
  /* A narrow matrix stacks each column label on two lines. */
  cls.matrix_columns.forEach((label, j) => (stacked ? label.split(' ') : [label]).forEach((part, i, parts) =>
    T(g, cx(j) + cellW / 2, top - 12 - 14 * (parts.length - 1 - i), part, 'kx-t-axis', 'middle',
      {style: {fill: COLOR.ink, fontWeight: 600}})));
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
}

function curveView(host, width, kind, cls, at, view) {
  const opts = {height: 290, maxRatio: 2.3, extraRight: 0, label: kind};
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
  const {x, y} = c;
  refNote(c, kind, cls);
  const ok = point[0] !== null && point[1] !== null && isFinite(point[0]) && isFinite(point[1]);
  if (ok) marker(c.g, x(point[0]), y(point[1]));
}

/* The curve's own score, AUROC or AP, is its only label. */
function refNote(c, kind, cls) {
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
  } else if (kind === 'pr') {
    note(notes, space, cls.ap_callout, [{x: x(0.5), y: y(Math.min(1, cls.share + 0.5 * (1 - cls.share))) - 8, anchor: 'middle'},
      {x: x(0.2), y: y(0.98)}, {x: x(0.5), y: c.y0 + 14, anchor: 'middle'}]);
  }
}


KX.cards.cutoff = (parent, card, ctx) => {
  const cut = {}, byName = {};
  for (const c of card.classes) {
    cut[c.name] = card.default;
    const flagged = c.counts.map(([tp, fp, fn, tn]) => (tp + fp) / (tp + fp + fn + tn));
    const gaps = flagged.map(f => Math.abs(f - c.share));
    /* The preset stop flags the share closest to the class's own share. */
    byName[c.name] = {...c, roc: curvePoints(card, c, 'fpr', 'recall'), pr: curvePoints(card, c, 'recall', 'precision'),
      gain: curvePoints(card, c, 'flagged', 'recall'), preset: gaps.indexOf(Math.min(...gaps))};
  }
  const state = ctx.state.ui('cutoff', {cls: card.classes[0].name, cut});
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
    for (const i of [0, 25, 50, 75, 100]) {
      H('span', i === card.default ? 'kx-default' : null, tickBox, card.grid_text[i]).style.left = i + '%';
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
      if (cls.preset !== card.default) {
        items.push({index: cls.preset, extra: share(cutoffAt(card, cls, cls.preset).flagged) + ' flagged'});
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
      const cls = byName[state.cls], index = state.cut[state.cls], at = cutoffAt(card, cls, index);
      input.value = String(index);
      input.setAttribute('aria-valuetext', card.grid_text[index]);
      range.style.setProperty('--kx-fill', `calc(9px + (100% - 18px) * ${index / (card.grid.length - 1)})`);
      value.textContent = card.grid_text[index];
      classButtons.forEach((b, i) => b.setAttribute('aria-pressed', String(card.classes[i].name === state.cls)));
      renderStops(cls);
      stopButtons.forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.index) === index)));
      strip.replaceChildren();
      for (const [label, text] of at.strip) {
        H('b', null, H('span', null, strip, label.charAt(0).toUpperCase() + label.slice(1)), text);
      }
      ctx.state.set('cutoff', state);
      refresh();
    };
    input.addEventListener('input', () => { state.cut[state.cls] = Number(input.value); update(); },
      {signal: ctx.signal});
    box.update = update;
    return box;
  };
  let box;
  /* Confusion matrix beside the cutoff controls, then the three curves. */
  const frame = cardFrame(parent, card, ctx, null, body => {
    const draw = (id, host, width) => {
      const cls = byName[state.cls], at = cutoffAt(card, cls, state.cut[state.cls]);
      if (id === 'matrix') matrixView(host, width, cls, at);
      else curveView(host, width, id, cls, at, card.views.find(v => v.id === id));
    };
    const top = H('div', 'kx-cutoff-top', body), curvesRow = H('div', 'kx-cutoff-curves', body);
    const charts = card.views.map((view, i) => {
      const cell = H('div', i ? 'kx-cutoff-curve' : 'kx-cutoff-matrix', i ? curvesRow : top);
      H('h4', null, cell, view.label);
      return chartHost(cell, (host, width) => draw(view.id, host, width), ctx);
    });
    box = controls(top);
    box.classList.add('kx-cutoff-slider');
    refresh = () => charts.forEach(chart => chart.redraw());
  });
  box.update();
  return frame;
};

KX.cards.scores = (parent, card, ctx) => {
  let chart = null;
  ctx.state.on('cutoff', () => { if (chart && chart.isConnected && chart.dataset.view === 'outcome') chart.redraw(); });
  return cardFrame(parent, card, ctx, card.views, (body, view) => {
    if (view.id === 'tenths') chart = tenthsView(body, view, card.title, ctx);
    else {
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
  const state = ctx.state.get('cutoff'), cutoff = ctx.cards.cutoff;
  const at = state && cutoff ? cutoffAt(cutoff, cutoff.classes.find(cl => cl.name === state.cls), state.cut[state.cls]) : null;
  let pos = null;
  if (at) {
    const t = cutoff.grid[state.cut[state.cls]];
    pos = state.cls === cutoff.classes[0].name ? t : 1 - t;
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
  const binOf = v => {
    const i = edges.findIndex(e => e > v);
    return i < 0 ? totals.length - 1 : Math.max(0, i - 1);
  };
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
  if (at) {
    const text = at.line_callout;
    const px = x(pos);
    S('path', {d: `M${px} ${c.y0 - 4}V${c.y1}`, style: {stroke: COLOR.panel, strokeWidth: 5.5}}, g);
    S('path', {d: `M${px} ${c.y0 - 4}V${c.y1}`, style: {stroke: COLOR.green, strokeWidth: 2.5}}, g);
    space.add(px - 2, c.y0, 4, c.y1 - c.y0);
    note(notes, space, text, [{x: px + 6, y: c.y0 + 10}, {x: px - 6, y: c.y0 + 10, anchor: 'end'},
      {x: px + 6, y: c.y0 + 40}, {x: px - 6, y: c.y0 + 40, anchor: 'end'}]);
  }
  const i = view.tallest, bl = x(edges[i]) - 6, br = x(edges[i + 1]) + 6, by = y(totals[i]) + 12;
  /* Count on the first line, the score range on the second. */
  note(notes, space, view.tallest_callout.replace(/ (score \d)/, '\n$1'), [{x: bl, y: by, anchor: 'end'},
    {x: br, y: by}, {x: bl, y: by + 18, anchor: 'end'}, {x: br, y: by + 18},
    {x: (bl + br) / 2, y: by - 20, anchor: 'middle'}]);
  series.forEach((s, k) => {
    if (typeof s.median !== 'number') return;
    const mx = x(s.median);
    note(notes, space, s.median_callout, [{x: mx, y: medianTop - 6, anchor: 'middle'},
      {x: mx + 6, y: medianTop + 10}, {x: mx - 6, y: medianTop + 10, anchor: 'end'},
      {x: mx + 6, y: medianTop + 28}, {x: mx - 6, y: medianTop + 28, anchor: 'end'}], 'kx-t-note-plain');
  });
}
