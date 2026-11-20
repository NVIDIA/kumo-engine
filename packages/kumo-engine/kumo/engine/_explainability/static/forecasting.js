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
      categories: view.steps,
      series: [{color: ROLE.model}, {color: ROLE.base, hatch: true}],
      yTitle: view.y, xTitle: view.x, height: 290, label: card.title}), ctx);
  } else if (view.id === 'totals') {
    legend(body, [{label: view.series[0], kind: 'line', marker: true, color: ROLE.actual},
      {label: view.series[1], kind: 'line', marker: true, color: ROLE.model}]);
    chartHost(body, (host, width) => totalsView(host, width, view), ctx);
  } else tenthsView(body, view, card.title, ctx);
});

function totalsView(host, width, view) {
  const n = view.steps.length;
  const vals = view.steps.flatMap(s => s.values);
  const c = curves(host, width, {height: 290, xDomain: [-0.35, n - 0.65], xTicks: view.steps.map((s, i) => i), yDomain: [Math.min(0, ...vals) * 1.15, Math.max(0, ...vals) * 1.15],
    lines: [
      {points: view.steps.map((s, i) => [i, s.values[0]]), color: ROLE.actual, width: 2.6, markers: true},
      {points: view.steps.map((s, i) => [i, s.values[1]]), color: ROLE.model, width: 2.6, markers: true},
    ],
    xTitle: view.x, yTitle: view.y, label: view.label,
    xFmt: v => (Number.isInteger(v) && view.steps[v] ? view.steps[v].label : ''),
    yFmt: v => v.toLocaleString('en-US')});
  const {x, y, space, notes} = c;
  view.steps.forEach((s, i) => s.values.forEach((v, k) => note(notes, space, s.texts[k],
    [{x: x(i), y: y(v) - 10, anchor: 'middle'}, {x: x(i), y: y(v) + 20, anchor: 'middle'}], 'kx-t-value')));
  const end = y(view.steps[n - 1].values[1]);
  note(notes, space, view.total_callout, [{x: x(n - 1), y: end - 30, anchor: 'end'},
    {x: x(n - 1), y: end + 36, anchor: 'end'}, {x: x(n - 1) - 10, y: end - 12, anchor: 'end'}]);
}
