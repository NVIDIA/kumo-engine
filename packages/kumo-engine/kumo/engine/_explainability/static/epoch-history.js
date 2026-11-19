/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* The training UI compares search experiments on separate, equally scaled
   training and validation plots. The published model's refit is a later run. */
function epochHistoryView(body, view, ctx, state) {
  const controls = H('div', 'kx-sg-controls kx-history-controls', body);
  const picker = subgraphPicker(controls, 'Metric', 'history-metric', ctx);
  for (const metric of view.metrics) H('option', null, picker.select, metric.toUpperCase()).value = metric;
  if (view.metrics.includes(state.metric)) picker.select.value = state.metric;
  picker.sync();
  H('span', 'kx-coverage', controls, 'Training experiments before model selection');
  const plots = H('div', 'kx-history-plots', body);
  const legend = H('div', 'kx-history-legend', body);
  legend.setAttribute('role', 'group');
  legend.setAttribute('aria-label', 'Visible experiments');
  const colors = [COLOR.deep, COLOR.train, COLOR.val, COLOR.hold];
  const dashes = ['', '6 3', '2 3'];
  const experiments = view.experiments.map((series, i) => ({
    series, label: 'Experiment ' + (i + 1), color: colors[i % colors.length],
    dash: dashes[Math.floor(i / colors.length) % dashes.length],
  }));

  function checkbox(label) {
    const box = H('input', null, label);
    box.type = 'checkbox';
    box.checked = true;
    return box;
  }
  const allLabel = H('label', null, legend);
  const all = checkbox(allLabel);
  H('span', null, allLabel, 'Select all');
  for (const [i, experiment] of experiments.entries()) {
    const label = H('label', null, legend);
    experiment.box = checkbox(label);
    experiment.box.checked = state.experiments?.[i] ?? true;
    const swatch = S('svg', {width: 19, height: 9, 'aria-hidden': true}, label);
    S('path', {d: 'M1 4.5H18', style: {stroke: experiment.color,
      strokeWidth: 2, strokeDasharray: experiment.dash}}, swatch);
    H('span', null, label, experiment.label);
  }

  all.checked = experiments.every(e => e.box.checked);
  all.indeterminate = !all.checked && experiments.some(e => e.box.checked);

  let metric, yDomain, exponent, unit;
  function setMetric() {
    metric = state.metric = picker.select.value;
    let low = Infinity, high = -Infinity;
    for (const {series} of experiments) {
      for (const split of ['train', 'val']) {
        for (const [, value] of series[split + '_' + metric] || []) {
          if (value === null) continue;
          low = Math.min(low, value);
          high = Math.max(high, value);
        }
      }
    }
    // Keep the chart's tick step representable for very large scores.
    const magnitude = Math.max(Math.abs(low), Math.abs(high));
    exponent = Number.isFinite(magnitude) && magnitude >= 1e12 ? Math.floor(Math.log10(magnitude)) : 0;
    unit = 10 ** exponent;
    yDomain = [low / unit - 0.02, high / unit + 0.02];
  }
  setMetric();

  const charts = ['train', 'val'].map(split => {
    const pane = H('div', null, plots);
    const heading = H('h4', null, pane);
    return chartHost(pane, (host, width) => {
      const splitName = split === 'train' ? 'Training' : 'Validation';
      heading.textContent = splitName + ' ' + metric.toUpperCase() + (exponent ? ' (×10^' + exponent + ')' : '');
      const lines = [];
      const epochs = new Set();
      for (const experiment of experiments) {
        if (!experiment.box.checked) continue;
        let last = null, line;
        for (const [epoch, value] of experiment.series[split + '_' + metric] || []) {
          epochs.add(epoch);
          if (value === null) { last = null; continue; }
          if (last === null || epoch !== last + 1) {
            line = {points: [], color: experiment.color, dash: experiment.dash,
              markers: true, pointTitle: ([e, , value]) => experiment.label + ' · Epoch ' + e + ' · ' + value};
            lines.push(line);
          }
          line.points.push([epoch, value / unit, value]);
          last = epoch;
        }
      }
      if (!lines.length) {
        const message = experiments.some(e => e.box.checked)
          ? 'No ' + splitName.toLowerCase() + ' values recorded for the selected experiments.'
          : 'Select an experiment to show its history.';
        H('p', 'kx-empty', host, message);
        return;
      }
      const xs = Array.from(epochs).sort((a, b) => a - b);
      const step = Math.max(1, Math.ceil(xs.length / Math.max(2, (width - 70) / 34)));
      const xTicks = xs.filter((epoch, index) => index % step === 0 || index === xs.length - 1);
      const xDomain = xs.length === 1 ? [xs[0] - 0.5, xs[0] + 0.5] : [xs[0], xs[xs.length - 1]];
      curves(host, width, {lines, xDomain, xTicks, yDomain, xTitle: 'Epoch', height: 210,
        label: heading.textContent + ' by epoch', reserveSpace: false});
    }, ctx);
  });
  function redraw() {
    state.experiments = experiments.map(e => e.box.checked);
    const count = experiments.filter(e => e.box.checked).length;
    all.checked = count === experiments.length;
    all.indeterminate = count > 0 && !all.checked;
    charts.forEach(chart => chart.redraw());
  }
  all.addEventListener('change', () => {
    for (const experiment of experiments) experiment.box.checked = all.checked;
  }, {signal: ctx.signal});
  legend.addEventListener('change', redraw, {signal: ctx.signal});
  picker.select.addEventListener('change', () => { setMetric(); redraw(); }, {signal: ctx.signal});
}
