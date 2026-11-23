/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Ranking: one metric family at every cutoff K, a line per split. */

const SPLIT_COLORS = {holdout: ROLE.actual, val: ROLE.other, train: ROLE.model};

KX.cards.ranking = (parent, card, ctx) => cardFrame(parent, card, ctx, card.views, (body, view) => {
  legend(body, view.series.map(s => ({label: s.label, kind: 'line', marker: true, color: SPLIT_COLORS[s.id]})));
  chartHost(body, (host, width) => byKView(host, width, view, card.title), ctx);
});

function byKView(host, width, view, title) {
  const first = view.ks[0], last = view.ks[view.ks.length - 1], pad = (last - first) * 0.06;
  const points = s => view.ks.map((k, i) => [k, s.values[i]]).filter(p => p[1] !== null);
  const all = view.series.flatMap(s => points(s).map(p => p[1]));
  const hi = Math.max(...all);
  const spacing = (String(last).length + 2) * 7 + 12;
  let previous = -Infinity;
  const ticks = view.ks.filter(k => {
    const px = (k - first) / (last - first + 2 * pad) * Math.max(1, width - 80);
    if (px - previous < spacing) return false;
    previous = px;
    return true;
  });
  const c = curves(host, width, {height: 300, xDomain: [first - pad, last + pad], xTicks: ticks,
    yDomain: [0, hi <= 1 ? Math.min(1, hi * 1.2) : hi * 1.1],
    lines: view.series.map(s => ({points: points(s), color: SPLIT_COLORS[s.id], width: 2.4, markers: true,
      pointTitle: p => `${s.label} · K=${p[0]} · ${s.texts[view.ks.indexOf(p[0])]}`})),
    xTitle: view.x, yTitle: view.y, label: title, xFmt: v => 'K=' + v});
  const {x, y, space, notes} = c;
  /* Values sit above their dots; where they would collide, below. The
     holdout is always labelled, the other splits where there is room. */
  view.series.forEach((s, n) => view.ks.forEach((k, i) => {
    if (s.values[i] === null) return;
    const px = x(k), py = y(s.values[i]);
    const placed = note(notes, space, s.texts[i], [{x: px, y: py - 10, anchor: 'middle'}, {x: px, y: py + 20, anchor: 'middle'},
      {x: px + 10, y: py + 4}, {x: px - 10, y: py + 4, anchor: 'end'}], 'kx-t-value', n > 0);
    if (placed) notes.lastChild.style.fill = SPLIT_COLORS[s.id];
  }));
}
