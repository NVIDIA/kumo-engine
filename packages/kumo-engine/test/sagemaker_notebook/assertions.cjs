// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0
const { VerificationError } = require('../snowflake_notebook/verification.cjs');

function verificationMessage(error) {
  return error instanceof VerificationError ? error.message :
    'Browser verification failed; inspect the completed check records.';
}

function captureAllowed(actualUrl, expectedUrl, identityVisible) {
  try {
    const actual = new URL(actualUrl), expected = new URL(expectedUrl);
    return identityVisible && actual.protocol === 'https:' && actual.search === '' &&
      expected.search === '' && actual.origin === expected.origin &&
      actual.pathname === expected.pathname && actual.hash === expected.hash;
  } catch {
    return false;
  }
}

function safeSnapshot(text, secrets = []) {
  for (const secret of secrets.filter(Boolean)) {
    for (const form of [secret, encodeURIComponent(secret), secret.replaceAll('&', '&amp;')]) {
      text = text.split(form).join('[authentication URL]');
    }
  }
  return text.replace(/https?:\/\/[^\s"'<>]+/gi, '[URL]')
    .replace(/([?&][A-Za-z0-9_.%-]+=)[^\s"'<>]+/g, '$1[redacted]')
    .replace(/((?:authorization|token|password|signature)\s*[:=])[^\n]+/gi, '$1[redacted]');
}

function apiFailure(url, trustedBase, status, body, secrets = []) {
  const actual = new URL(url), base = new URL(trustedBase);
  const endpoint = /\/(?:ajax-api|api)\/2\.0\/mlflow\/(metrics\/get-history(?:-bulk(?:-interval(?:-from-steps)?)?)?|experiments\/get|runs\/get)$/.exec(actual.pathname)?.[1];
  if (actual.origin !== base.origin || !endpoint || status < 400) return undefined;
  const report = { endpoint, status };
  if (['INTERNAL_ERROR', 'PERMISSION_DENIED', 'UNAUTHENTICATED', 'INVALID_PARAMETER_VALUE',
    'RESOURCE_DOES_NOT_EXIST', 'NOT_IMPLEMENTED', 'BAD_REQUEST'].includes(body?.error_code)) report.error_code = body.error_code;
  const message = body?.message ?? body?.Message;
  if (typeof message === 'string') report.message = safeSnapshot(message, secrets).slice(0, 500);
  return report;
}

function frontendFailure(error, stage) {
  const message = String(error.message);
  const category = /webgl|shader|plotly|gl context/i.test(message) ? 'graphics' :
    /react error #\d+/i.test(message) ? 'react' : 'frontend';
  const name = /^(Type|Range|Reference|Syntax)?Error$/.test(error.name) ? error.name : 'Error';
  const failure = { stage, type: 'pageerror', name, category };
  if (/^Cannot (?:read|set) properties of (?:undefined|null)(?: \(reading '[A-Za-z_][A-Za-z0-9_]{0,40}'\))?$/.test(message)) {
    failure.message = message;
  }
  const templates = {
    dynamic_import: /Failed to fetch dynamically imported module|Importing a module script failed|error loading dynamically imported module/i,
    chunk_load: /Loading (?:CSS )?chunk [\w-]+ failed|ChunkLoadError/i,
    script_load: /Failed to load (?:script|module)|Script error/i,
    resource_fetch: /Failed to fetch|NetworkError when attempting to fetch resource/i,
    resize_observer: /ResizeObserver loop/i,
    invalid_hook: /Invalid hook call/i,
    kernel: /\bkernel\b/i,
    widget: /\bwidget(?:s)?\b/i,
    comm: /\bcomm\b/i,
    disposed: /\bdispos(?:ed|al|e)\b/i,
    disposed_model: /^Model is disposed$/,
    disposed_cache: /^This cache has already been disposed$/,
    cancellation: /\b(?:cancel(?:ed|led|lation)?|abort(?:ed)?)\b/i,
    websocket: /\bwebsocket\b/i,
    session: /\bsession\b/i,
    extension: /\b(?:plugin|extension)\b/i,
    schema: /\bschema\b/i,
    module: /\bmodule\b/i,
    connection: /\bconnect(?:ion|ed|ing)?\b/i,
    missing: /not found|does not exist|no such/i,
    forbidden: /\b(?:forbidden|403)\b/i,
  };
  failure.signatures = Object.entries(templates).filter(([, pattern]) => pattern.test(message)).map(([name]) => name);
  const react = message.match(/react error #(\d+)/i);
  if (react) failure.react_code = Number(react[1]);
  return failure;
}

function numericLabels(labels) {
  return labels.map(text => text.trim())
    .filter(text => /^[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?$/i.test(text))
    .map(Number).filter(Number.isFinite);
}

function historyPoint(rendered, step, expectedValue, labels) {
  const tolerance = 1e-5 * Math.max(1, Math.abs(expectedValue));
  const stepTolerance = rendered.step_tolerance ?? 1e-3;
  if (!Number.isFinite(stepTolerance) || stepTolerance > 0.02 || stepTolerance <= 0 ||
      !Number.isFinite(rendered.step) || Math.abs(rendered.step - step) > stepTolerance ||
      !Number.isFinite(rendered.value) || Math.abs(rendered.value - expectedValue) > tolerance ||
      !numericLabels(labels).some(value => Math.abs(value - expectedValue) <= tolerance)) {
    throw new VerificationError('Rendered epoch coordinates or tooltip differ from tracked history');
  }
  return { step: rendered.step, value: rendered.value, expected_step: step,
    expected_value: expectedValue, displayed: labels };
}

function renderedGeometry(element) {
  const plot = element.closest('.js-plotly-plot');
  function screen(element) {
    const coordinate = element.ownerSVGElement.createSVGPoint();
    const point = coordinate.matrixTransform(element.getScreenCTM());
    return { x: point.x, y: point.y };
  }
  function ticksFor(axis, selector) {
    const ticks = [...plot.querySelectorAll(selector)]
      .map(tick => ({ value: Number(tick.textContent.trim().replace('−', '-')), pixel: screen(tick)[axis] }))
      .filter(tick => Number.isFinite(tick.value));
    if (ticks.length < 2) throw new Error('Missing rendered axis ticks');
    const a = ticks[0], b = ticks.reverse().find(tick => tick.value !== a.value && tick.pixel !== a.pixel);
    if (!b) throw new Error('Missing distinct rendered axis ticks');
    return [a, b];
  }
  return { x: ticksFor('x', '.xaxislayer-above .xtick text'),
    y: ticksFor('y', '.yaxislayer-above .ytick text'),
    origin: element.ownerSVGElement ? screen(element) : null };
}

function targetPosition(axes, step, value) {
  function pixel([a, b], value) {
    return a.pixel + (value - a.value) * (b.pixel - a.pixel) / (b.value - a.value);
  }
  return { x: pixel(axes.x, step), y: pixel(axes.y, value) };
}

function observedCoordinates(geometry) {
  function value([a, b], pixel) {
    return a.value + (pixel - a.pixel) * (b.value - a.value) / (b.pixel - a.pixel);
  }
  const [a, b] = geometry.x;
  return { step: value(geometry.x, geometry.origin.x),
    value: value(geometry.y, geometry.origin.y),
    step_tolerance: Math.min(0.02, 2 * Math.abs((b.value - a.value) / (b.pixel - a.pixel))) };
}

function historyCsv(content, run, metric, expected) {
  const [header, ...lines] = content.trim().split(/\r?\n/).map(line => line.split(','));
  const columns = Object.fromEntries(header.map((name, index) => [name, index]));
  if (!['run_id', 'key', 'step', 'value'].every(name => Object.hasOwn(columns, name))) {
    throw new VerificationError('UI history download has an unexpected schema');
  }
  const actual = lines.map(row => {
    if (row.length !== header.length ||
        numericLabels([row[columns.step], row[columns.value]]).length !== 2) {
      throw new VerificationError('UI history download has an invalid numeric row');
    }
    if (row[columns.run_id] !== run || row[columns.key] !== metric) {
      throw new VerificationError('UI history download identifies another run or metric');
    }
    return [Number(row[columns.step]), Number(row[columns.value])];
  });
  const ordered = values => [...values].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const wanted = ordered(expected), got = ordered(actual);
  if (got.length !== wanted.length || got.some(([step, value], index) =>
    !Number.isFinite(step) || !Number.isFinite(value) || step !== wanted[index][0] ||
    Math.abs(value - wanted[index][1]) > 1e-9 * Math.max(1, Math.abs(wanted[index][1])))) {
    throw new VerificationError('UI history download differs from recorded epochs');
  }
  return actual;
}

module.exports = { apiFailure, safeSnapshot, frontendFailure, numericLabels, historyPoint, renderedGeometry, targetPosition,
  observedCoordinates, historyCsv, VerificationError, verificationMessage, captureAllowed };
