/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Forecasting: Forecast against actual. */

KX.cards.forecast = (parent, card, ctx) => cardFrame(parent, card, ctx, card.views, (body, view) => {
  if (view.id === 'horizon') {
    legend(body, [{label: view.series[0], color: ROLE.model},
      {label: view.series[1], color: ROLE.base, hatch: true}]);
    chartHost(body, (host, width) => columns(host, width, {
      categories: view.steps.map(s => ({label: s.label, sub: s.bias_callout,
        values: [s.model, s.zero], texts: [s.model_text, s.zero_text]})),
      series: [{color: ROLE.model}, {color: ROLE.base, hatch: true}],
      yTitle: view.y, xTitle: view.x, height: 290, label: card.title}), ctx);
  } else if (view.id === 'totals') {
    legend(body, [{label: view.series[0], kind: 'line', marker: true, color: ROLE.actual},
      {label: view.series[1], kind: 'line', marker: true, color: ROLE.model}]);
    chartHost(body, (host, width) => totalsView(host, width, view), ctx);
  } else {
    chartHost(body, (host, width) => columns(host, width, {
      categories: view.tenths.map(t => ({label: t.label, values: [t.share], texts: [t.share_text]})),
      series: [{color: ROLE.actual}], highlight: view.tenths.length - 1, highlightBar: 0,
      ref: {value: view.reference, text: view.reference_callout},
      notes: [{index: view.tenths.length - 1, text: view.top_callout}],
      tickFmt: v => Math.round(v * 100) + '%',
      yTitle: view.y, xTitle: view.x, height: 300, label: card.title}), ctx);
  }
});

function totalsView(host, width, view) {
  const n = view.steps.length;
  const vals = view.steps.flatMap(s => [s.actual, s.forecast]);
  const c = curves(host, width, {height: 290, xDomain: [-0.35, n - 0.65], xTicks: view.steps.map((s, i) => i), yDomain: [Math.min(0, ...vals) * 1.15, Math.max(0, ...vals) * 1.15],
    lines: [
      {points: view.steps.map((s, i) => [i, s.actual]), color: ROLE.actual, width: 2.6, markers: true},
      {points: view.steps.map((s, i) => [i, s.forecast]), color: ROLE.model, width: 2.6, markers: true},
    ],
    xTitle: view.x, yTitle: view.y, label: 'Weekly totals',
    xFmt: v => (Number.isInteger(v) && view.steps[v] ? view.steps[v].label : ''),
    yFmt: v => v.toLocaleString('en-US')});
  const {x, y, space, notes} = c;
  view.steps.forEach((s, i) => {
    const px = x(i);
    note(notes, space, s.actual_text, [{x: px, y: y(s.actual) - 10, anchor: 'middle'},
      {x: px, y: y(s.actual) + 20, anchor: 'middle'}], 'kx-t-value');
    note(notes, space, s.forecast_text, [{x: px, y: y(s.forecast) - 10, anchor: 'middle'},
      {x: px, y: y(s.forecast) + 20, anchor: 'middle'}], 'kx-t-value');
  });
  const last = view.steps[n - 1];
  note(notes, space, view.total_callout, [
    {x: x(n - 1), y: y(last.forecast) - 30, anchor: 'end'},
    {x: x(n - 1), y: y(last.forecast) + 36, anchor: 'end'},
    {x: x(n - 1) - 10, y: y(last.forecast) - 12, anchor: 'end'}]);
}
