/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Chart primitives: plain SVG drawn at the container's real width. */
const KX = {cards: {}};
/* The query's FOR EACH table, for axis titles and legends. */
let ENTITIES = 'rows';
const NS = 'http://www.w3.org/2000/svg';
const COLOR = {
  ink: 'var(--kx-ink)', muted: 'var(--kx-muted)', grid: 'var(--kx-grid)',
  axis: 'var(--kx-axis)', green: 'var(--kx-green)', deep: 'var(--kx-deep)',
  greenbg: 'var(--kx-greenbg)', hold: 'var(--kx-hold)', train: 'var(--kx-train)',
  val: 'var(--kx-val)', base: 'var(--kx-base)', panel: 'var(--kx-panel)',
  bg: 'var(--kx-bg)', soft: 'var(--kx-soft)',
};
/* Series roles: what was observed is near-black, what the model said is
   steel blue, a reference is light grey; green marks the one highlight. */
const ROLE = {actual: COLOR.hold, model: COLOR.train, base: COLOR.base,
  other: COLOR.val, highlight: COLOR.green};

function S(tag, attrs, parent) {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === undefined || value === null) continue;
    if (key === 'style') Object.assign(node.style, value);
    else if (key === 'text') node.textContent = value;
    else node.setAttribute(key, value);
  }
  if (parent) parent.appendChild(node);
  return node;
}

function H(tag, cls, parent, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  if (parent) parent.appendChild(node);
  return node;
}

function scale(d0, d1, r0, r1) {
  const k = (r1 - r0) / ((d1 - d0) || 1);
  return v => r0 + (v - d0) * k;
}

/* A 1, 2 or 5 step whose tick count lands closest to the one asked for, so an
   axis never runs far past its largest value. */
function niceStep(span, count) {
  const raw = span / Math.max(count, 1);
  const mag = Math.pow(10, Math.floor(Math.log10(raw || 1)));
  const norm = raw / mag;
  const step = norm >= Math.sqrt(50) ? 10 : norm >= Math.sqrt(10) ? 5 : norm >= Math.sqrt(2) ? 2 : 1;
  return step * mag;
}

function ticks(lo, hi, count) {
  if (!(hi > lo)) hi = lo + 1;
  const step = niceStep(hi - lo, count);
  const start = Math.floor(lo / step + 1e-9) * step;
  const end = Math.ceil(hi / step - 1e-9) * step;
  const out = [];
  for (let v = start; v <= end + step * 1e-6; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return {values: out, lo: start, hi: end, step};
}

/* Tick labels: thousands separators, trailing zeros trimmed, true minus. */
function tickText(v, step) {
  let decimals = 0;
  while (decimals < 6 && Math.abs(Math.round(step * 10 ** decimals) - step * 10 ** decimals) > 1e-6) decimals++;
  let text = Math.abs(v).toLocaleString('en-US', {minimumFractionDigits: decimals,
    maximumFractionDigits: decimals});
  return (v < 0 && Number(text.replace(/,/g, '')) !== 0 ? '−' : '') + text;
}

/* Rendered text width, estimated where the text is not laid out. */
function measure(node) {
  let width = 0;
  try { width = node.getComputedTextLength(); } catch (error) { /* not laid out */ }
  return width > 0 ? width : node.textContent.length * 6.6;
}

function T(parent, x, y, text, cls, anchor, extra) {
  return S('text', Object.assign({x, y, class: cls || null,
    'text-anchor': anchor && anchor !== 'start' ? anchor : null, text}, extra || {}), parent);
}

/* Occupied boxes for collision-aware callout placement. */
class Space {
  constructor(x0, x1, y0, y1) { this.bounds = {x0, x1, y0, y1}; this.boxes = []; }
  add(x, y, w, h) { this.boxes.push({x, y, w, h}); }
  free(x, y, w, h) {
    const b = this.bounds;
    if (x < b.x0 - 0.5 || x + w > b.x1 + 0.5 || y < b.y0 - 0.5 || y + h > b.y1 + 0.5) return false;
    return !this.boxes.some(o => x < o.x + o.w && x + w > o.x && y < o.y + o.h && y + h > o.y);
  }
}

/* Place a text at the first free candidate; candidates are {x, y, anchor}
   with y the text baseline. Falls back to the first candidate, clamped. */
function note(parent, space, text, candidates, cls, optional) {
  /* A two-line callout reads like the reference annotations: the event on
     the first line, its numbers on the second. */
  const lines = String(text).split('\n');
  const node = T(parent, 0, 0, '', cls || 'kx-t-note');
  const spans = lines.map((line, i) => S('tspan', {x: 0, dy: i ? 15 : 0, text: line,
    class: i ? 'kx-t-sub' : null}, node));
  const w = Math.max(...spans.map(measure));
  const h = 14 + 15 * (lines.length - 1);
  const box = c => {
    const left = c.anchor === 'end' ? c.x - w : c.anchor === 'middle' ? c.x - w / 2 : c.x;
    return {x: left, y: c.y - 11, w, h};
  };
  let chosen = null;
  for (const c of candidates) {
    const b = box(c);
    if (space.free(b.x - 2, b.y, b.w + 4, b.h)) { chosen = b; break; }
  }
  if (!chosen && optional) { node.remove(); return null; }
  if (!chosen) {
    /* No candidate is free: search outward from the first one for the
       nearest free place inside the chart, so a callout never lands on an
       axis label or on the data; clamp only when nothing at all is free. */
    const first = box(candidates[0]);
    const bd = space.bounds;
    let best = null, bestD = Infinity;
    for (let dy = -160; dy <= 160; dy += 8) for (let dx = -240; dx <= 240; dx += 12) {
      const b = {x: first.x + dx, y: first.y + dy, w, h};
      const d = dx * dx + dy * dy;
      if (d < bestD && space.free(b.x - 2, b.y, b.w + 4, b.h)) { best = b; bestD = d; }
    }
    chosen = best || first;
    chosen.x = Math.max(bd.x0, Math.min(chosen.x, bd.x1 - w));
    chosen.y = Math.max(bd.y0, Math.min(chosen.y, bd.y1 - h));
  }
  node.setAttribute('y', chosen.y + 11);
  for (const span of spans) span.setAttribute('x', chosen.x);
  space.add(chosen.x - 2, chosen.y, w + 4, h);
  return chosen;
}

/* A reference-line label on two lines: before and after its colon, or
   before its last word. */
function refSplit(text) {
  const colon = text.indexOf(': ');
  if (colon > 0) return [text.slice(0, colon + 1), text.slice(colon + 2)];
  const space = text.lastIndexOf(' ');
  return space > 0 ? [text.slice(0, space), text.slice(space + 1)] : [text];
}

/* Callout text split after its first clause onto two lines. */
function twoLines(text) {
  const i = text.indexOf(' \u00b7 ');
  return i < 0 ? text : text.slice(0, i) + '\n' + text.slice(i + 3);
}

/* Pattern ids carry the widget id: url(#id) resolves page-wide. */
let hatchCount = 0;
function hatch(svg, color) {
  const id = widgetId + '-hatch-' + (++hatchCount);
  const defs = S('defs', {}, svg);
  const pattern = S('pattern', {id, width: 6, height: 6, patternUnits: 'userSpaceOnUse',
    patternTransform: 'rotate(45)'}, defs);
  S('rect', {width: 6, height: 6, style: {fill: COLOR.soft}}, pattern);
  S('path', {d: 'M0 0V6', style: {stroke: color || COLOR.base, strokeWidth: 3}}, pattern);
  return 'url(#' + id + ')';
}

function fill(svg, series) {
  return series.hatch ? hatch(svg, series.color) : series.color;
}

/* A legend swatch: a line (dashed, or with a marker dot) or a rounded
   square. */
function swatch(parent, item) {
  const line = item.kind === 'line' || item.kind === 'dash';
  const w = item.w || (line ? 22 : 12), h = item.h || 12;
  const svg = S('svg', {width: w, height: h, 'aria-hidden': 'true'}, parent);
  if (line) {
    S('path', {d: `M1 ${h / 2}H${w - 1}`, style: {stroke: item.color, strokeWidth: item.width || 2.2,
      strokeDasharray: item.dash || (item.kind === 'dash' ? '4 3' : null), fill: 'none'}}, svg);
    if (item.marker) S('circle', {cx: 11, cy: 6, r: 3.2, style: {fill: item.color}}, svg);
  } else {
    S('rect', {width: w, height: h, rx: item.rx || 2.5, style: {fill: item.hatch ? hatch(svg, item.color) : item.color,
      opacity: item.opacity || null}}, svg);
  }
  return svg;
}

/* A legend row above the chart (HTML, so it wraps). */
function legend(parent, items) {
  const row = H('div', 'kx-legend', parent);
  for (const item of items) {
    const span = H('span', null, row);
    swatch(span, item);
    H('span', null, span, item.label);
  }
  return row;
}

/* A checkbox before its label text, with an optional mark between. */
function checkbox(parent, text, checked, mark) {
  const label = H('label', null, parent), box = H('input', null, label);
  box.type = 'checkbox';
  box.checked = checked;
  if (mark) mark(label);
  H('span', null, label, text);
  return box;
}

function mix(color, percent, base) {
  return `color-mix(in srgb, ${color} ${percent.toFixed(1)}%, ${base || COLOR.bg})`;
}

/* Chart host: draws at the current width and again whenever it changes. */
function chartHost(parent, draw, ctx) {
  const host = H('div', 'kx-chart', parent);
  let width = 0;
  const render = () => {
    const w = Math.floor(host.clientWidth);
    if (!w || w === width) return;
    width = w;
    host.replaceChildren();
    draw(host, w);
  };
  const observer = new ResizeObserver(() => window.requestAnimationFrame(render));
  observer.observe(host);
  ctx.onDispose(() => observer.disconnect());
  host.redraw = () => { width = 0; render(); };
  render();
  return host;
}

function svgRoot(host, width, height, label) {
  return S('svg', {width, height, viewBox: `0 0 ${width} ${height}`, role: 'img',
    'aria-label': label || null}, host);
}

/* Horizontal gridlines with tick labels on the left. */
function yGrid(g, y, tk, x0, x1, fmt) {
  for (const v of tk.values) {
    const py = y(v);
    S('path', {d: `M${x0} ${py}H${x1}`, style: {stroke: v === 0 ? COLOR.axis : COLOR.grid,
      strokeWidth: 1}}, g);
    T(g, x0 - 8, py + 4, fmt ? fmt(v) : tickText(v, tk.step), 'kx-t-axis', 'end');
  }
}

function xGrid(g, x, tk, y0, y1, fmt) {
  for (const v of tk.values) {
    const px = x(v);
    S('path', {d: `M${px} ${y0}V${y1}`, style: {stroke: v === 0 && tk.lo < 0 && !tk.categorical ? COLOR.axis : COLOR.grid,
      strokeWidth: 1}}, g);
    T(g, px, y1 + 17, fmt ? fmt(v) : tickText(v, tk.step), 'kx-t-axis', 'middle');
  }
}

function yTitle(g, text, x, yMid) {
  T(g, x, yMid, text, 'kx-t-axis', 'middle', {transform: `rotate(-90 ${x} ${yMid})`});
}

function tickWidth(values, step) {
  return Math.max(...values.map(v => tickText(v, step).length)) * 7.2;
}

/* Vertical bars, one or two series per category, values printed on the
   marks. opts: categories [{label, sub, values, texts}], series [{name,
   color, hatch}], highlight index, ref {value, text}, notes [{index, text}],
   yTitle, xTitle, height, origin, signed. */
function columns(host, width, opts) {
  const n = opts.categories.length, m = opts.series.length;
  const all = opts.categories.flatMap(c => c.values).filter(v => v !== null && v !== undefined);
  const origin = opts.origin || 0;
  const lo = Math.min(origin, ...all, opts.ref ? opts.ref.value : origin);
  const hi = Math.max(origin, ...all, opts.ref ? opts.ref.value : origin);
  const tk = ticks(lo, hi, 4);
  /* A reference line is labelled directly, in a lane at its right end. */
  const refLines = opts.ref ? refSplit(opts.ref.text) : [];
  const right = opts.ref ? Math.max(...refLines.map(l => l.length)) * 6.6 + 14 : 8;
  const left = Math.max(34, tickWidth(tk.values, tk.step) + 12) + (opts.yTitle ? 20 : 0);
  const x0 = left, x1 = width - right;
  const band = (x1 - x0) / n;
  const inner = Math.min(band * 0.78, m * 46);
  const gap = m > 1 ? 3 : 0;
  const barW = (inner - gap * (m - 1)) / m;
  /* When two values of a category cannot sit side by side on their bars,
     they stack above the taller bar of the pair, coloured like their series. */
  const longest = Math.max(0, ...opts.categories.flatMap(c => (c.texts || []).map(t => (t || '').length)));
  const stack = m > 1 && longest * 6.4 > barW + gap + 2;
  const notesH = opts.notes ? 36 : 0;
  const head = 16;
  const top = 6 + notesH + head;
  const foot = lo < 0 ? 16 : 0;
  const subLines = opts.categories.some(c => c.sub) ? 1 : 0;
  const bottom = 26 + foot + subLines * 15 + (opts.xTitle ? 22 : 0);
  const height = opts.height || 280;
  const svg = svgRoot(host, width, height, opts.label);
  const y0 = top, y1 = height - bottom;
  const y = scale(tk.lo, tk.hi, y1, y0);
  const g = S('g', {}, svg);
  const space = new Space(x0 - 6, width - 2, 2, y1 + foot);
  if (opts.ref) {
    const ry = y(opts.ref.value), h = 16 + 15 * (refLines.length - 1);
    space.add(x1 + 4, ry - h / 2 - 2, right, h + 4);
  }
  yGrid(g, y, tk, x0, x1, opts.tickFmt);
  if (opts.yTitle) yTitle(g, opts.yTitle, 12, (y0 + y1) / 2);
  const fills = opts.series.map(s => fill(svg, s));
  if (opts.highlight !== undefined && opts.highlight !== null && opts.highlightBand) {
    const hx = x0 + band * opts.highlight + (band - inner) / 2 - 6;
    const by = y0 - 6;
    S('rect', {x: hx, y: by, width: inner + 12, height: y1 - by, rx: 6,
      style: {fill: COLOR.greenbg}}, g);
    g.insertBefore(g.lastChild, g.firstChild);
  }
  const labels = [];
  opts.categories.forEach((c, i) => {
    const gx = x0 + band * i + (band - inner) / 2;
    c.values.forEach((v, j) => {
      if (v === null || v === undefined) return;
      const bx = gx + j * (barW + gap);
      const yv = y(v), yo = y(origin);
      const top_ = Math.min(yv, yo), h = Math.max(Math.abs(yo - yv), 1);
      let color = fills[j];
      if (opts.highlight === i && opts.highlightBar === j) color = COLOR.green;
      S('rect', {x: bx, y: top_, width: barW, height: h, rx: Math.min(2.5, barW / 4),
        class: 'kx-bar', style: {fill: color}}, g);
      space.add(bx, top_, barW, h);
      if (c.texts && c.texts[j]) {
        const above = v >= origin;
        labels.push({x: bx + barW / 2, y: above ? top_ - 5 : top_ + h + 13, text: c.texts[j],
          i, j, above, v, top: top_, h});
      }
    });
    const lx = x0 + band * (i + 0.5);
    T(g, lx, y1 + foot + 18, c.label, 'kx-t-axis', 'middle');
    if (c.sub) T(g, lx, y1 + foot + 33, c.sub, 'kx-t-muted', 'middle', {style: {fontSize: '11.5px'}});
  });
  const valueLayer = S('g', {}, svg);
  /* Values never sit on the reference line. */
  if (opts.ref) space.add(x0, y(opts.ref.value) - 2, x1 - x0, 4);
  const textW = text => {
    const probe = T(valueLayer, 0, 0, text, 'kx-t-value', null, {style: {fontWeight: 600}});
    const w = measure(probe);
    probe.remove();
    return w;
  };
  if (stack) {
    /* A pair stacks above its bars as one block, the larger value on top,
       lifted together when anything is in the way. */
    const byCat = {};
    for (const l of labels) (byCat[l.i] = byCat[l.i] || []).push(l);
    for (const group of Object.values(byCat)) {
      group.sort((p, q) => q.v - p.v);
      const i = group[0].i, cx = x0 + band * (i + 0.5);
      const peak = Math.min(...group.map(l => l.above ? l.y : y(origin) - 5));
      const k = group.length;
      const widths = group.map(l => textW(l.text));
      const lift = [0, 1, 2, 3].map(q => 14 * k * q).find(dy => group.every((l, r) => {
        const base = peak - 14 * (k - 1 - r) - dy;
        return space.free(cx - widths[r] / 2 - 2, base - 11, widths[r] + 4, 14);
      }));
      group.forEach((l, r) => {
        const s = opts.series[l.j];
        const base = peak - 14 * (k - 1 - r) - (lift || 0);
        const placed = note(valueLayer, space, l.text, [{x: cx, y: base, anchor: 'middle'}], 'kx-t-value');
        const node = valueLayer.lastChild;
        if (placed && node) node.style.fill = s.hatch ? COLOR.muted : s.color;
        if (node) node.style.fontWeight = 600;
      });
    }
  } else {
    /* Each value sits on its own bar. When a reference line or another
       value is in the way it goes just inside the top of a tall enough bar,
       otherwise outward in 13 px steps, taller bars first. */
    labels.sort((p, q) => (p.above ? p.y - q.y : q.y - p.y));
    for (const l of labels) {
      const w = textW(l.text);
      const dir = l.above ? -13 : 13;
      const s = opts.series[l.j];
      if (l.above && !space.free(l.x - w / 2 - 2, l.y - 11, w + 4, 14) && l.h >= 20 && !s.hatch
          && w + 6 <= barW) {
        T(valueLayer, l.x, l.top + 14, l.text, 'kx-t-value', 'middle',
          {style: {fill: opts.highlight === l.i && opts.highlightBar === l.j ? 'var(--kx-on-green)'
            : 'var(--kx-panel)', stroke: 'none'}});
        continue;
      }
      note(valueLayer, space, l.text, [0, 1, 2, 3, 4].map(k => ({x: l.x, y: l.y + dir * k,
        anchor: 'middle'})), 'kx-t-value');
    }
  }
  const notes = S('g', {}, svg);
  if (opts.ref) {
    const ry = y(opts.ref.value);
    S('path', {d: `M${x0} ${ry}H${x1 + 4}`, style: {stroke: COLOR.ink, strokeWidth: 1.4,
      strokeDasharray: '5 4'}}, g);
    const node = T(notes, x1 + 8, ry - (refLines.length - 1) * 7.5 + 4, '', 'kx-t-note-plain');
    refLines.forEach((line, k) => S('tspan', {x: x1 + 8, dy: k ? 15 : 0, text: line}, node));
  }
  for (const nt of opts.notes || []) {
    const cx = x0 + band * (nt.index + 0.5);
    const vals = opts.categories[nt.index].values.filter(v => v !== null);
    const peak = Math.min(...vals.map(v => y(Math.max(v, origin)))) - 18;
    const low = Math.max(...vals.map(v => y(Math.min(v, origin)))) + 24;
    const text = twoLines(nt.text), up = text.includes('\n') ? 15 : 0;
    note(notes, space, text, ([
      {x: cx, y: peak - 4 - up, anchor: 'middle'}, {x: cx, y: peak - 17 - up, anchor: 'middle'},
      {x: cx + inner / 2 + 6, y: peak + 8 - up}, {x: cx - inner / 2 - 6, y: peak + 8 - up, anchor: 'end'},
      {x: cx + inner / 2 + 6, y: peak + 30 - up}, {x: cx - inner / 2 - 6, y: peak + 30 - up, anchor: 'end'},
      {x: cx, y: peak - 30 - up, anchor: 'middle'}, {x: cx, y: 17, anchor: 'middle'},
      {x: cx, y: 17, anchor: 'end'}, {x: cx, y: 17}, {x: cx, y: low, anchor: 'middle'}]));
  }
  if (opts.xTitle) T(g, (x0 + x1) / 2, height - 6, opts.xTitle, 'kx-t-axis', 'middle');
  return svg;
}

/* Tenths of a score, prediction or forecast: two series against their
   overall average, where only the lowest and highest tenth carry values and
   the callout carries the tenth it names; or one share per tenth against an
   equal share, the highest tenth highlighted. */
function tenthsView(body, view, title, ctx) {
  const pair = view.tenths[0].values.length > 1, last = view.tenths.length - 1;
  if (pair) legend(body, [{label: view.series[0], color: ROLE.model}, {label: view.series[1], color: ROLE.actual},
    {label: 'all ' + ENTITIES, kind: 'dash', color: COLOR.ink, width: 1.4}]);
  return chartHost(body, (host, width) => columns(host, width, {
    categories: view.tenths.map((t, i) => ({...t,
      texts: pair && (i % last || i === view.highlight) ? [] : t.texts})),
    series: pair ? [{color: ROLE.model}, {color: ROLE.actual}] : [{color: ROLE.actual}],
    highlight: view.highlight, highlightBand: pair, highlightBar: pair ? null : 0,
    ref: {value: view.reference, text: view.reference_callout}, notes: [{index: view.highlight, text: view.callout}],
    tickFmt: pair ? null : v => Math.round(v * 100) + '%',
    yTitle: view.y, xTitle: view.x, height: 300, label: title}), ctx);
}

/* Horizontal bars with labels on the left. rows: [{label, sub, values,
   texts, color, hatch, highlight, note, key}]; series as in columns. */
function rowsChart(host, width, opts) {
  const rows = opts.rows, m = opts.series ? opts.series.length : 1;
  const barH = m > 1 ? 9 : 16, gap = m > 1 ? 2 : 0;
  const rowH = Math.max(m * barH + (m - 1) * gap + 12, rows.some(r => r.sub) ? 36 : 26);
  const svg = svgRoot(host, width, 10, opts.label);
  const probe = T(svg, 0, 0, '', 'kx-t-axis', null, {style: {fontWeight: 700}});
  let labelW = 0;
  for (const r of rows) { probe.textContent = r.label; labelW = Math.max(labelW, measure(probe)); }
  probe.style.fontWeight = 400; probe.style.fontSize = '11.5px';
  for (const r of rows) if (r.sub) { probe.textContent = r.sub; labelW = Math.max(labelW, measure(probe)); }
  probe.remove();
  labelW = Math.min(labelW + 4, width * 0.42);
  const all = rows.flatMap(r => r.values).filter(v => v !== null && v !== undefined);
  const origin = opts.origin || 0;
  const lo = Math.min(origin, ...all, opts.domain ? opts.domain[0] : origin);
  const hi = Math.max(origin, ...all, opts.domain ? opts.domain[1] : origin);
  let tk = null;
  if (opts.axis) {
    const t = ticks(lo, hi, Math.max(3, Math.floor((width - labelW) / 90)));
    tk = {values: t.values.filter(v => v >= lo - 1e-9 && v <= hi + 1e-9), lo, hi, step: t.step};
  }
  const top = opts.top || 6;
  const left = labelW + 18 + (opts.marker ? 14 : 0);
  const right = opts.right || 140;
  const x0 = left, x1 = Math.max(left + 60, width - right);
  const height = top + rows.length * rowH + (tk ? 34 : 6) + (opts.xTitle ? 20 : 0);
  svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const x = tk ? scale(tk.lo, tk.hi, x0, x1) : scale(0, hi || 1, x0, x1);
  const g = S('g', {}, svg);
  const space = new Space(0, width, 0, height);
  const yEnd = top + rows.length * rowH;
  if (tk) xGrid(g, x, tk, top - 4, yEnd + 2);
  const fills = (opts.series || []).map(s => fill(svg, s));
  rows.forEach((r, i) => {
    const ry = top + i * rowH;
    const holder = r.onClick ? S('g', {class: 'kx-clickable', tabindex: 0, role: 'button',
      'aria-label': r.aria || r.label}, g) : g;
    if (r.onClick) {
      S('rect', {x: 0, y: ry, width: width, height: rowH, class: 'kx-focus',
        style: {fill: 'transparent'}, rx: 6}, holder);
      holder.addEventListener('click', r.onClick, {signal: opts.signal});
      holder.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); r.onClick(); } }, {signal: opts.signal});
    }
    if (r.band) S('rect', {x: 2, y: ry + 1, width: width - 4, height: rowH - 2, rx: 6,
      style: {fill: COLOR.greenbg}}, holder);
    const mid = ry + rowH / 2;
    if (opts.marker && r.selected) S('circle', {cx: 6, cy: (r.sub ? ry + 13 : mid), r: 4, style: {fill: COLOR.green}}, holder);
    const labelX = opts.marker ? 16 : 0;
    const lab = T(holder, labelX, (r.sub ? ry + 17 : mid + 4), r.label, 'kx-t-axis', null,
      {style: {fill: COLOR.ink, fontWeight: r.selected || r.strong ? 700 : 500}});
    if (measure(lab) > labelW + 2) {
      let s = r.label;
      while (s.length > 4 && measure(lab) > labelW) { s = s.slice(0, -2); lab.textContent = s + '…'; }
    }
    if (r.sub) T(holder, labelX, ry + 31, r.sub, 'kx-t-muted', null, {style: {fontSize: '11.5px'}});
    const blockH = m * barH + (m - 1) * gap;
    let endX = x0;
    r.values.forEach((v, j) => {
      if (v === null || v === undefined) return;
      const by = mid - blockH / 2 + j * (barH + gap);
      const a = x(Math.min(v, origin)), b = x(Math.max(v, origin));
      const color = r.color ? (r.hatch ? hatch(svg, r.color) : r.color) : fills[j];
      S('rect', {x: a, y: by, width: Math.max(b - a, 1.5), height: barH, rx: 2,
        class: 'kx-bar', style: {fill: color}}, holder);
      space.add(a, by, Math.max(b - a, 1.5), barH);
      const text = r.texts && r.texts[j];
      if (text) {
        const right_ = v >= origin;
        const tx = right_ ? b + 5 : a - 5;
        const node = T(holder, tx, by + barH / 2 + 4, text, 'kx-t-value', right_ ? null : 'end',
          {style: {fontWeight: r.strong && j === 0 ? 700 : null}});
        const w = measure(node);
        space.add(right_ ? tx : tx - w, by - 2, w, barH + 4);
        endX = Math.max(endX, right_ ? tx + w : b);
      } else endX = Math.max(endX, b);
    });
    if (r.note) T(holder, endX + 10, mid + 4, r.note, 'kx-t-note-plain');
  });
  if (opts.xTitle) T(g, (x0 + x1) / 2, height - 4, opts.xTitle, 'kx-t-axis', 'middle');
  return {svg, x, rowY: i => top + i * rowH, rowH, x0, x1, space, g};
}

/* Line chart on numeric axes: lines [{points, color, width, dash}],
   marker {x, y}, notes placed by the caller through the returned space. */
function curves(host, width, opts) {
  const height = opts.height || 300;
  const left = 52 + (opts.yTitle ? 8 : 0), bottom = 46, top = 26;
  const plotH = height - top - bottom;
  const plotW = Math.min(width - left - 12 - (opts.extraRight || 0), opts.maxRatio ? plotH * opts.maxRatio : 1e9);
  const svg = svgRoot(host, Math.min(width, left + plotW + 12 + (opts.extraRight || 0)), height, opts.label);
  const x0 = left, x1 = left + plotW, y0 = top, y1 = height - bottom;
  const xd = opts.xDomain || [0, 1], yd = opts.yDomain || [0, 1];
  const tkx = opts.xTicks ? {values: opts.xTicks, lo: xd[0], hi: xd[1], step: 1, categorical: true}
    : ticks(xd[0], xd[1], 5);
  const tky = opts.yTicks ? {values: opts.yTicks, lo: yd[0], hi: yd[1], step: 1} : ticks(yd[0], yd[1], 5);
  const x = scale(tkx.lo, tkx.hi, x0, x1), y = scale(tky.lo, tky.hi, y1, y0);
  const g = S('g', {}, svg);
  yGrid(g, y, tky, x0, x1, opts.yFmt);
  xGrid(g, x, tkx, y0, y1, opts.xFmt);
  if (opts.yTitle) yTitle(g, opts.yTitle, 12, (y0 + y1) / 2);
  if (opts.xTitle) T(g, (x0 + x1) / 2, height - 6, opts.xTitle, 'kx-t-axis', 'middle');
  const space = new Space(x0, x1 + (opts.extraRight || 0), 2, y1);
  for (const line of opts.lines) {
    const d = line.points.map((p, i) => (i ? 'L' : 'M') + x(p[0]).toFixed(1) + ' ' + y(p[1]).toFixed(1)).join('');
    S('path', {d, style: {fill: 'none', stroke: line.color, strokeWidth: line.width || 2.2,
      strokeDasharray: line.dash || null, strokeLinejoin: 'round', strokeLinecap: 'round'}}, g);
    /* Every segment is reserved along its whole length, so no callout or
       label is ever placed over a line. */
    for (let i = 0; opts.reserveSpace !== false && i < line.points.length; i++) {
      const px = x(line.points[i][0]), py = y(line.points[i][1]);
      if (!i) { space.add(px - 3, py - 3, 6, 6); continue; }
      const qx = x(line.points[i - 1][0]), qy = y(line.points[i - 1][1]);
      const steps = Math.ceil(Math.hypot(px - qx, py - qy) / 5);
      for (let k = 1; k <= steps; k++) {
        space.add(qx + (px - qx) * k / steps - 3, qy + (py - qy) * k / steps - 3, 6, 6);
      }
    }
    if (line.markers) for (const p of line.points) {
      const dot = S('circle', {cx: x(p[0]), cy: y(p[1]), r: 3.6, style: {fill: line.color}}, g);
      if (line.pointTitle) S('title', {text: line.pointTitle(p)}, dot);
    }
  }
  return {svg, g, x, y, x0, x1, y0, y1, space, notes: S('g', {}, svg)};
}

/* A thin leader from a mark to its callout when the callout had to move
   away from it. */
function leader(g, px, py, box, gapPx) {
  if (!box) return;
  const qx = Math.max(box.x, Math.min(px, box.x + box.w));
  const qy = Math.max(box.y, Math.min(py, box.y + box.h));
  const d = Math.hypot(qx - px, qy - py);
  if (d < (gapPx || 22)) return;
  const k = 9 / d;
  S('path', {d: `M${px + (qx - px) * k} ${py + (qy - py) * k}L${qx} ${qy}`,
    style: {stroke: COLOR.muted, strokeWidth: 1}}, g);
}

/* Mark the cutoff point on a curve: a green dot with a halo. */
function marker(g, px, py) {
  S('circle', {cx: px, cy: py, r: 7, style: {fill: COLOR.bg, opacity: 0.9}}, g);
  S('circle', {cx: px, cy: py, r: 5.2, style: {fill: COLOR.green}}, g);
}

/* A labelled native select. */
function picker(parent, text, kind) {
  const field = H('label', 'kx-sg-field kx-sg-' + kind, parent);
  H('span', null, field, text);
  const select = H('select', null, field);
  S('path', {d: 'm4 6 4 4 4-4'}, S('svg', {viewBox: '0 0 16 16', 'aria-hidden': 'true'}, field));
  return select;
}
