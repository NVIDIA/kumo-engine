// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
function numericLabels(labels) {
  return labels.map(text => text.trim())
    .filter(text => /^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(text))
    .map(Number).filter(Number.isFinite);
}

function historyPoint(rendered, step, expectedValue, labels) {
  const tolerance = 1e-5 * Math.max(1, Math.abs(expectedValue));
  if (!Number.isFinite(rendered.step) || Math.abs(rendered.step - step) > 1e-3 ||
      !Number.isFinite(rendered.value) || Math.abs(rendered.value - expectedValue) > tolerance ||
      !numericLabels(labels).some(value => Math.abs(value - expectedValue) <= tolerance)) {
    throw new Error('Rendered epoch coordinates or tooltip differ from tracked history');
  }
  return { step: rendered.step, value: rendered.value, expected_step: step,
    expected_value: expectedValue, displayed: labels };
}

function renderedCoordinates(point) {
  const plot = point.closest('.js-plotly-plot');
  function screen(element) {
    const coordinate = element.ownerSVGElement.createSVGPoint();
    return coordinate.matrixTransform(element.getScreenCTM());
  }
  const center = screen(point);
  function interpolate(axis, selector) {
    const ticks = [...plot.querySelectorAll(selector)]
      .map(tick => ({ value: Number(tick.textContent.trim().replace('−', '-')), pixel: screen(tick)[axis] }))
      .filter(tick => Number.isFinite(tick.value));
    if (ticks.length < 2) throw new Error('Missing rendered axis ticks');
    const a = ticks[0], b = ticks.find(tick => tick.value !== a.value && tick.pixel !== a.pixel);
    if (!b) throw new Error('Missing distinct rendered axis ticks');
    return a.value + (center[axis] - a.pixel) * (b.value - a.value) / (b.pixel - a.pixel);
  }
  return { step: interpolate('x', '.xaxislayer-above .xtick text'),
    value: interpolate('y', '.yaxislayer-above .ytick text') };
}

module.exports = { numericLabels, historyPoint, renderedCoordinates };
