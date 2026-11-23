/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Regression: Predicted against actual. */

function heatmapView(host, width, view) {
  const nx = view.columns.length, ny = view.rows.length;
  const xEdges = view.x_edges, yEdges = view.y_edges;
  const decimals = Math.min(12, Math.max(0, 1 - Math.floor(Math.log10(view.width))));
  const format = value => value.toLocaleString('en-US', {maximumFractionDigits: decimals}).replace('-', '−');
  const longest = edges => Math.max(...edges.map(value => format(value).length)) * 7;
  const left = Math.max(48, longest(yEdges) + 31), right = Math.max(12, longest(xEdges) / 2 + 4);
  const calloutW = measure(T(svgRoot(host, 10, 10), 0, 0, view.densest_callout, 'kx-t-note'));
  host.replaceChildren();
  /* The densest cell's callout sits right at that cell: under the axis when
     the cell is on the bottom row, over the grid when it is on the top row,
     and otherwise in a lane right of the grid, joined by a short leader. */
  const [dr, dc] = view.densest;
  const place = dr === 0 ? 'below' : dr === ny - 1 ? 'above' : 'side';
  const laneW = calloutW + 34;
  const side = place === 'side' && (width - left - right - laneW) / nx >= 44;
  const cell = Math.max(4, Math.min(64, (width - left - right - (side ? laneW : 0)) / nx));
  const top = 14 + (place === 'above' || (place === 'side' && !side) ? 30 : 0);
  const bottom = 52 + (place === 'below' ? 24 : 0);
  const plotW = cell * nx, cellH = Math.max(26, Math.min(46, cell * 0.75));
  const height = top + cellH * ny + bottom;
  const svgW = left + plotW + right + (side ? laneW : 0);
  const svg = svgRoot(host, svgW, height, 'Heatmap');
  const g = S('g', {}, svg);
  const x0 = left, y1 = top + cellH * ny;
  const max = Math.max(...view.counts.flat());
  const space = new Space(x0, svgW - 2, 2, y1);
  const labels = [];
  for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) {
    const count = view.counts[r][c];
    const cx = x0 + c * cell, cy = y1 - (r + 1) * cellH;
    const level = count ? 12 + 78 * Math.log1p(count) / Math.log1p(max) : 0;
    const densest = r === dr && c === dc;
    const color = densest ? COLOR.green : count ? mix(COLOR.hold, level, COLOR.panel) : COLOR.soft;
    const rect = S('rect', {x: cx + 1, y: cy + 1, width: cell - 2, height: cellH - 2, rx: 3, style: {fill: color,
      opacity: count ? 1 : 0.5}}, g);
    S('title', {text: `${view.x}: ${view.columns[c]}, ${view.y}: ${view.rows[r]}, count: ${count}`}, rect);
    if (count && cell >= 10 + String(count).length * 7) {
      const dark = !densest && level > 52;
      const text = count.toLocaleString('en-US');
      const fs = cell < 44 || text.length > 4 && cell < 52 ? 10.5 : 11.5;
      labels.push([cx + cell / 2, cy + cellH / 2 + 4, text, {style: {
        fill: densest ? 'var(--kx-on-green)' : dark ? 'var(--kx-panel)' : COLOR.ink, stroke: color,
        strokeWidth: 5, paintOrder: 'stroke', strokeLinejoin: 'round',
        fontWeight: densest ? 700 : 500, fontSize: fs + 'px'}}]);
      space.add(cx + cell / 2 - text.length * 3.6, cy + cellH / 2 - 6, text.length * 7.2, 13);
    }
  }
  const everyX = Math.ceil(70 / cell), everyY = Math.ceil(24 / cellH);
  for (let c = 0; c < nx; c += everyX) {
    if ((nx - c) * cell >= 54) T(g, x0 + c * cell, y1 + 16, format(xEdges[c]), 'kx-t-axis', 'middle');
  }
  T(g, x0 + nx * cell, y1 + 16, format(xEdges[nx]), 'kx-t-axis', 'middle');
  for (let r = xEdges[0] === yEdges[0] ? everyY : 0; r < ny; r += everyY) {
    if ((ny - r) * cellH >= 18) T(g, x0 - 8, y1 - r * cellH + 4, format(yEdges[r]), 'kx-t-axis', 'end');
  }
  T(g, x0 - 8, y1 - ny * cellH + 4, format(yEdges[ny]), 'kx-t-axis', 'end');
  T(g, x0 + plotW / 2, height - 8, view.x, 'kx-t-axis', 'middle');
  yTitle(g, view.y, 12, (top + y1) / 2);
  const x = scale(xEdges[0], xEdges[nx], x0, x0 + plotW);
  const y = scale(yEdges[0], yEdges[ny], y1, top);
  const lo = Math.max(xEdges[0], yEdges[0]), hi = Math.min(xEdges[nx], yEdges[ny]);
  const sx = x(lo), sy = y(lo), ex = x(hi), ey = y(hi);
  if (lo <= hi) {
    S('path', {d: `M${sx} ${sy}L${ex} ${ey}`, style: {stroke: COLOR.ink,
      strokeWidth: 1.4, strokeDasharray: '5 4', fill: 'none'}}, g);
    for (let t = 0; t <= 1; t += 0.02) space.add(sx + (ex - sx) * t - 2, sy + (ey - sy) * t - 2, 4, 4);
  }
  /* Counts sit above the diagonal, each on a halo of its own cell's fill. */
  for (const [lx, ly, text, attrs] of labels) {
    T(g, lx, ly, text, null, 'middle', attrs);
  }
  const notes = S('g', {}, svg);
  /* The callout is joined to the green cell by a leader that runs along
     the gutter above the cell's row, clear of every count. */
  const gy = y1 - (dr + 1) * cellH;
  const rx = x0 + (dc + 1) * cell - 1;
  const lx = x0 + dc * cell + cell / 2;
  const fits = lx - 2 + calloutW <= svgW - 2;
  if (place === 'below' || place === 'above') {
    const ty = place === 'below' ? y1 + 40 : top - 12;
    const from = place === 'below' ? y1 : gy;
    const to = place === 'below' ? ty - 12 : ty + 4;
    S('path', {d: `M${lx} ${from}V${to}`, class: 'kx-leader'}, notes);
    T(notes, fits ? lx - 2 : lx + 2, ty, view.densest_callout, 'kx-t-note', fits ? 'start' : 'end');
  } else if (side) {
    const tx = x0 + plotW + 12;
    S('path', {d: `M${rx} ${gy}H${tx - 4}`, class: 'kx-leader'}, notes);
    T(notes, tx, gy + 4, view.densest_callout, 'kx-t-note');
  } else {
    const tx = Math.min(svgW - 4, Math.max(lx + calloutW / 2, calloutW + 3)), ty = top - 12;
    S('path', {d: `M${rx} ${gy}V${ty + 4}H${tx - calloutW / 2}`, class: 'kx-leader'}, notes);
    T(notes, tx, ty, view.densest_callout, 'kx-t-note', 'end');
  }
  const box = notes.lastChild.getBBox();
  space.add(box.x - 3, box.y - 3, box.width + 6, box.height + 6);
  if (lo <= hi) note(notes, space, view.diagonal, [{x: ex + 6, y: ey + 14},
    {x: ex - 6, y: ey - 6, anchor: 'end'}, {x: ex + 6, y: ey - 6},
    {x: (sx + ex) / 2 + 10, y: (sy + ey) / 2 + 4}], 'kx-t-note-plain');
}

function distributionsView(body, view, ctx) {
  const colors = [ROLE.actual, ROLE.model];
  legend(body, view.series.map((s, k) => ({label: s.name, color: colors[k]})));
  chartHost(body, (host, width) => {
    const n = view.bin_count, start = view.start;
    const index = i => view.indices?.[i] ?? i;
    const step = Math.max(1, Math.ceil(n / Math.max(1, Math.floor((width - 70) / 60))));
    const edge = index => tickText(start + index * view.width, Math.min(0.1, view.width));
    const maxCount = view.series.reduce((max, s) => s.counts.reduce((m, count) => Math.max(m, count), max), 0);
    const log = v => Math.log10(1 + v);
    const top = Math.ceil(Math.log10(maxCount + 1));
    const yTicks = [0];
    for (let p = 1; p <= top; p++) yTicks.push(log(10 ** p));
    const c = curves(host, width, {height: 290, extraRight: 44, xDomain: [0, n], yDomain: [0, log(10 ** top) * 1.04],
      yTicks, lines: [], yTitle: view.y + ' (log scale)', xTitle: view.x, label: 'Distributions',
      xTicks: Array.from({length: Math.ceil(n / step)}, (_, i) => i * step),
      xFmt: v => (v < n - (view.folded ? 2 : 0) ? edge(v) : ''),
      yFmt: v => (v ? Math.round(10 ** v - 1).toLocaleString('en-US') : '')});
    const {x, y, g, space, notes} = c;
    const yc = v => y(log(v));
    const bw = x(1) - x(0);
    const inner = Math.min(bw * 0.8, 44), gap = Math.min(2, inner / 5), barW = (inner - gap) / 2;
    const marks = [];
    view.series.forEach((s, k) => s.counts.forEach((count, i) => {
      if (!count) return;
      const bx = x(index(i)) + (bw - inner) / 2 + k * (barW + gap);
      S('rect', {x: bx, y: yc(count), width: barW, height: Math.max(1, yc(0) - yc(count)), rx: 1.5,
        style: {fill: colors[k]}}, g);
      space.add(bx, yc(count), barW, yc(0) - yc(count));
      marks.push({x: bx + barW / 2, y: yc(count), text: count.toLocaleString('en-US')});
    }));
    if (view.folded) {
      T(g, x(n - 1), c.y1 + 17, '\u2265 ' + edge(n - 1), 'kx-t-axis', 'start', {dx: -8});
    }
    view.series.forEach((s, k) => {
      const max = s.stats.values.max;
      const i = Math.max(0, Math.min(n - 1, Math.floor(max / view.width) - Math.round(start / view.width)));
      const bx = x(i) + (bw - inner) / 2 + k * (barW + gap) + barW / 2;
      const yy = yc(s.counts[view.indices ? view.indices.indexOf(i) : i] ?? 0);
      /* Directly above the bar first; when a taller neighbour is in the way,
         higher up the same column, joined to the bar by a thin leader. */
      const candidates = [];
      for (const anchor of ['middle', 'end', 'start']) {
        for (let lift = 24; yy - lift > c.y0 + 12; lift += 16) candidates.push({x: bx, y: yy - lift, anchor});
      }
      const placed = note(notes, space, s.max_callout, candidates.length ? candidates : [{x: bx, y: yy - 24}]);
      if (placed && placed.y + placed.h < yy - 22) {
        S('path', {d: `M${bx} ${placed.y + placed.h + 1}V${yy - 16}`, class: 'kx-leader'}, notes);
        space.add(bx - 1, placed.y + placed.h, 2, yy - placed.y - placed.h);
      }
    });
    /* Every bar carries its count, or none does: a partly labelled series
       reads as an accident. */
    const placed = [];
    for (const m of marks) {
      const box = note(notes, space, m.text, [{x: m.x, y: m.y - 5, anchor: 'middle'}, {x: m.x, y: m.y - 18, anchor: 'middle'}], 'kx-t-value', true);
      if (!box) { placed.forEach(node => node.remove()); break; }
      placed.push(notes.lastChild);
    }
  }, ctx);
  const stats = H('div', 'kx-stats', body);
  H('span', 'kx-sh', stats, '');
  for (const item of view.series[0].stats.items) H('span', 'kx-sh', stats, item.label);
  view.series.forEach((s, k) => {
    const label = H('span', 'kx-sl', stats);
    swatch(label, {w: 10, h: 10, rx: 2, color: colors[k]});
    H('span', null, label, s.name);
    for (const item of s.stats.items) H('span', null, stats, item.text);
  });
}

KX.cards.predicted = (parent, card, ctx) => {
  cardFrame(parent, card, ctx, card.views, (body, view, ui) => {
    if (view.id === 'tenths') tenthsView(body, view, card.title, ctx);
    else if (view.id === 'bands') {
      chartHost(body, (host, width) => columns(host, width, {
        categories: view.bands,
        series: [{color: ROLE.actual}], highlight: view.highlight, highlightBar: 0,
        notes: [{index: view.highlight, text: view.share_callout}],
        yTitle: view.y, xTitle: view.x + ' \u00b7 share of squared error', height: 300,
        label: card.title}), ctx);
    } else if (view.id === 'heatmap') {
      const label = H('label', 'kx-heatmap-toggle', body);
      const hideOutliers = H('input', null, label);
      hideOutliers.type = 'checkbox';
      hideOutliers.checked = !!ui.hideTails;
      label.append(document.createTextNode('Hide outliers'));
      const chart = chartHost(body, (host, width) => {
        const selected = hideOutliers.checked ? {...view, ...view.trimmed} : view;
        heatmapView(host, width, selected);
      }, ctx);
      chart.classList.add('kx-heatmap');
      hideOutliers.addEventListener('change', () => {
        ui.hideTails = hideOutliers.checked;
        chart.redraw();
      }, {signal: ctx.signal});
      if (view.coverage) H('p', 'kx-coverage', body, view.coverage);
    } else {
      distributionsView(body, view, ctx);
    }
  });
};
