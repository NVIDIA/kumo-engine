/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* Saved-example subgraph explorer. */
const labels = {
  saved: 'Saved examples', random: 'Random examples', unsure: 'Uncertain predictions',
  true_positive: 'True positives', true_negative: 'True negatives',
  false_positive: 'False positives', false_negative: 'False negatives',
  small_error: 'Small error', large_error: 'Large error',
  small_percent_error: 'Small percentage error', large_percent_error: 'Large percentage error',
  low_precision: 'Low precision', high_precision: 'High precision',
  old_entity: 'Older entities', recent_entity: 'Recent entities',
};
function fmt(value) {
  if (value == null) return 'Not recorded';
  if (typeof value === 'number' && !Number.isInteger(value)) return value.toLocaleString('en-US', {maximumFractionDigits: 3});
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}
const short = (v, n = 20) => {
  v = String(v);
  return v.length > n ? v.slice(0, Math.ceil((n - 1) / 2)) + '…' + v.slice(-Math.floor((n - 1) / 2)) : v;
};
const tierLabels = ['Lowest 25%', '25–50%', '50–75%', 'Top 25%'];
const tierRadii = [7, 10, 13, 16];
function scoreNodes(nodes) {
  const records = new Map(), scored = [];
  for (const node of nodes) {
    const features = node.count == null ? node.features.filter(f => Number.isFinite(f.score)) : [];
    const score = features.length ? features.reduce((sum, feature) => sum + feature.score, 0) : null;
    records.set(node.id, {score, tier: null, radius: 10});
    if (score !== null) scored.push(score);
  }
  scored.sort((a, b) => a - b);
  const equal = scored.length > 0 && scored[0] === scored.at(-1);
  const cuts = [0.25, 0.5, 0.75].map(q => scored[Math.floor(scored.length * q)]);
  for (const record of records.values()) {
    if (record.score === null) continue;
    if (equal) { record.tier = 'equal'; continue; }
    record.tier = cuts.filter(cut => record.score >= cut).length;
    record.radius = tierRadii[record.tier];
  }
  return {equal, count: scored.length, nodes: records};
}

/* Previous and Next with an "n of m" position; go(index) moves. Returns
   update(index, count). */
function pager(parent, go, signal, name) {
  const box = H('div', 'kx-sg-pager', parent);
  const prev = H('button', null, box, 'Previous'), next = H('button', null, box, 'Next');
  const position = H('span', 'kx-note-plain', box);
  prev.type = next.type = 'button';
  if (name) {
    prev.setAttribute('aria-label', 'Previous page of ' + name);
    next.setAttribute('aria-label', 'Next page of ' + name);
  }
  let at = 0;
  prev.addEventListener('click', () => go(at - 1), {signal});
  next.addEventListener('click', () => go(at + 1), {signal});
  return (index, count) => {
    at = index;
    position.textContent = index + 1 + ' of ' + count;
    prev.disabled = index === 0;
    next.disabled = index === count - 1;
  };
}

function classOutcome(parent, title, values, signal) {
  const section = H('div', 'kx-sg-class-outcome', parent);
  const table = H('table', 'kx-table', section);
  H('caption', null, table, title);
  const columns = H('tr', 'kx-col', H('thead', null, table));
  for (const name of ['Label', 'Value']) H('th', null, columns, name).scope = 'col';
  const body = H('tbody', null, table);
  const rows = values && typeof values === 'object' && !Array.isArray(values) ? Object.entries(values) : [];
  if (!rows.length) {
    H('td', null, H('tr', null, body), 'Not recorded').colSpan = 2;
    return;
  }
  const pages = Math.ceil(rows.length / 10);
  const show = page => {
    body.replaceChildren();
    for (const [label, value] of rows.slice(page * 10, (page + 1) * 10)) {
      const row = H('tr', null, body);
      H('td', null, row, label);
      H('td', null, row, Number.isFinite(value) ? value.toFixed(2) : 'Not recorded');
    }
    move?.(page, pages);
  };
  const move = pages > 1 ? pager(section, show, signal, title.toLowerCase()) : null;
  show(0);
}

KX.cards.subgraphs = (panel, payload, ctx) => {
  const options = {signal: ctx.signal}, entity = payload.entity_label || 'Entity';
  /* The selection, pinned record and filters, kept across exports. */
  const ui = ctx.state.ui(payload.id, {});
  const card = H('section', 'kx-card kx-sg', panel);
  card.dataset.card = payload.id;
  const top = H('div', 'kx-card-top', card);
  H('h3', 'kx-title', top, payload.title);
  const count = payload.samples.length;
  const coverage = count + (count === 1 ? ' saved example' : ' saved examples');
  H('span', 'kx-note-plain', top, payload.unavailable_count ? coverage + ' · ' + payload.unavailable_count + ' unavailable' : coverage);
  const controls = H('div', 'kx-sg-controls', card);
  const cat = picker(controls, 'Selection', 'category');
  const savedCategories = new Set(payload.samples.flatMap(s => s.categories));
  for (const c of new Set([...(payload.categories || []), ...savedCategories])) {
    if (savedCategories.has(c)) H('option', null, cat, Object.hasOwn(labels, c) ? labels[c] : c.replaceAll('_', ' ')).value = c;
  }
  const select = picker(controls, entity, 'entity');
  const move = pager(controls, index => { select.selectedIndex = index; draw(); }, ctx.signal);
  const outcome = H('div', 'kx-sg-outcome', card);
  const legend = H('div', 'kx-sg-legend', card);
  for (const [text, cls] of [['Prediction entity', 'root'], ['Related record', ''], ['Grouped records', 'aggregate']]) {
    const item = H('span', '', legend);
    H('i', cls, item);
    item.append(text);
  }
  const sizeLegend = H('div', 'kx-sg-size-legend', card);
  const sizeNote = H('p', 'kx-sg-size-note', card);
  const tools = H('div', 'kx-sg-tools', card);
  const filters = H('details', 'kx-sg-filters', tools);
  H('summary', null, filters, 'Filters');
  const filterFields = H('div', 'kx-sg-filter-fields', filters);
  const expand = H('button', 'kx-chipbtn', tools, 'Expand all');
  expand.type = 'button';
  const visibility = H('span', 'kx-note-plain', tools);
  visibility.setAttribute('role', 'status');
  const main = H('div', 'kx-sg-main', card);
  const canvas = H('div', 'kx-sg-canvas', main);
  const detail = H('aside', 'kx-sg-detail', main);
  detail.setAttribute('aria-label', 'Selected record details');
  H('p', 'kx-sg-note', card, 'Hover or focus a record to see its path to the prediction entity. Select a record to keep its details open.');
  const note = H('p', 'kx-sg-note', card);
  let list = [], sample, built, hovered = null, nodeMap, groups, paths, scoring, nodeEvents, parents, allTables, statuses;
  const recordLabel = n => n.count != null ? fmt(n.count) + ' grouped records'
    : n.value != null ? (n.key || 'ID') + ' ' + fmt(n.value) : n.table + ' record ' + n.id;
  const tierText = score => score.tier === null ? 'Unscored' : score.tier === 'equal'
    ? scoring.count === 1 ? 'Only scored record' : 'All scored records are equal' : tierLabels[score.tier];

  function showDetail(id, preview = false) {
    const n = nodeMap.get(id);
    if (!n) return;
    detail.replaceChildren();
    H('div', 'kx-sg-pin', detail, preview ? 'Preview' : 'Selected record');
    H('h4', 'kx-sg-detail-title', detail, n.table);
    H('div', 'kx-sg-id', detail, recordLabel(n));
    const score = scoring.nodes.get(id);
    const scoreLine = H('div', 'kx-sg-score', detail);
    H('span', '', scoreLine, 'Summed local sensitivity');
    H('strong', '', scoreLine, score.score === null ? 'Not recorded' : String(Number(score.score.toPrecision(6))))
      .title = score.score === null ? 'Not recorded' : String(score.score);
    H('span', 'kx-sg-score-tier', scoreLine, tierText(score));
    if (n.count != null) {
      H('p', 'kx-sg-note', detail, fmt(n.count) + ' additional sampled records are grouped here. Individual feature values were not saved for this group.');
      if (n.start || n.end) H('p', 'kx-sg-note', detail, [n.start, n.end].filter(Boolean).join(' to '));
      return;
    }
    const features = n.features;
    if (!features.length) {
      H('p', 'kx-sg-note', detail, 'No feature values were saved for this record.');
      return;
    }
    const addRows = (rows, parent) => {
      const dl = H('dl', '', parent);
      for (const f of rows) {
        H('dt', '', dl, f.name.replaceAll('_', ' '));
        H('dd', '', dl, fmt(f.value));
      }
    };
    addRows(features.slice(0, 7), detail);
    if (features.length > 7) {
      const more = H('details', '', detail);
      H('summary', '', more, 'All ' + features.length + ' features');
      addRows(features.slice(7), more);
    }
    const scored = features.filter(f => typeof f.score === 'number' && Number.isFinite(f.score)).sort((a, b) => b.score - a.score);
    if (scored.length) {
      const sensitivity = H('details', '', detail);
      H('summary', '', sensitivity, 'Local sensitivity');
      addRows(scored.map(f => ({name: f.name, value: f.score})), sensitivity);
      H('p', 'kx-sg-note', sensitivity, 'Unsigned gradient scores, scaled within this example. These do not indicate direction or causality.');
    }
  }
  function activate(id, preview = false) {
    if (id == null) {
      detail.replaceChildren();
      H('p', 'kx-sg-note', detail, 'No matching record selected.');
      return;
    }
    const path = new Set();
    for (let current = id; current != null; current = parents.get(current)) path.add(current);
    for (const [g, n] of groups) {
      g.dataset.active = String(n.id === id);
      g.dataset.muted = String(hovered !== null && !path.has(n.id));
      g.setAttribute('aria-pressed', String(n.id === ui.pinned));
    }
    for (const [p, e] of paths) {
      const on = hovered !== null && path.has(e.source) && path.has(e.target);
      p.dataset.active = String(on);
      p.dataset.moving = String(on);
      p.dataset.muted = String(hovered !== null && !on);
    }
    showDetail(id, preview);
  }
  function outcomeValue(label, value) {
    H('strong', '', H('span', '', outcome, label), value);
  }
  function buildFilters() {
    filterFields.replaceChildren();
    const fieldset = (heading, choices, chosen) => {
      const field = H('fieldset', null, filterFields);
      H('legend', null, field, heading);
      for (const item of choices) {
        const box = checkbox(field, item, chosen.includes(item));
        box.value = item;
        box.dataset.filter = heading;
      }
    };
    fieldset('Tables', allTables, ui.tables);
    const sizes = scoring.equal ? ['Same score'] : [...tierLabels].reverse();
    if ([...scoring.nodes.values()].some(n => n.score === null)) sizes.push('Not recorded');
    fieldset('Sensitivity size', sizes, ui.sizes);
  }
  filterFields.addEventListener('change', ({target: box}) => {
    const chosen = box.dataset.filter === 'Tables' ? ui.tables : ui.sizes;
    if (box.checked) chosen.push(box.value); else chosen.splice(chosen.indexOf(box.value), 1);
    draw();
  }, options);
  expand.addEventListener('click', () => {
    ui.fullDepth = !ui.fullDepth;
    ui.expanded = [];
    draw();
  }, options);

  ctx.onDispose(() => nodeEvents?.abort());
  function draw() {
    nodeEvents?.abort();
    nodeEvents = new AbortController();
    const nodeOptions = {signal: nodeEvents.signal};
    sample = list.find(s => s.id === select.value) || list[0];
    if (!sample) return;
    select.value = sample.id;
    hovered = null;
    nodeMap = new Map(sample.nodes.map(n => [n.id, n]));
    scoring = scoreNodes(sample.nodes);
    allTables = [...new Set(sample.nodes.map(n => n.table))];
    if (ui.sample !== sample.id) {
      Object.assign(ui, {sample: sample.id, pinned: sample.root, tables: [...allTables], sizes: [], expanded: [], fullDepth: false});
    }
    if (built !== sample) buildFilters();
    built = sample;
    sizeLegend.replaceChildren();
    const items = scoring.equal ? [['Same score', 10]] : scoring.count ? tierLabels.map((label, i) => [label, tierRadii[i]]) : [];
    if (scoring.equal && scoring.count === 1) items[0][0] = 'Only scored record';
    items.push(['Not recorded', 10]);
    for (const [label, r] of items) {
      const item = H('span', '', sizeLegend);
      const svg = S('svg', {width: 36, height: 36, viewBox: '0 0 36 36', 'aria-hidden': 'true'}, item);
      S('circle', {cx: 18, cy: 18, r, class: label === 'Not recorded' ? 'kx-sg-size-unscored' : 'kx-sg-size-scored'}, svg);
      item.append(label);
    }
    sizeNote.textContent = !scoring.count ? 'No local sensitivity scores were recorded for this example.'
      : !scoring.equal ? 'Size shows the within-example sensitivity tier, not proportional contribution. Ties share a size.'
      : scoring.count === 1 ? 'One scored record; no relative size ranking.'
      : 'All scored records have the same summed sensitivity; no relative size ranking.';
    move(list.indexOf(sample), list.length);
    outcome.replaceChildren();
    if (['multiclass_classification', 'multilabel_classification'].includes(payload.task_type)) {
      classOutcome(outcome, 'Predictions', sample.prediction, nodeOptions.signal);
      classOutcome(outcome, 'Ground truth', sample.target, nodeOptions.signal);
    } else if (ctx.data.model_type === 'classification') {
      outcomeValue(payload.prediction_label || 'Predicted probability',
        typeof sample.prediction === 'number' ? (sample.prediction * 100).toFixed(1) + '%' : fmt(sample.prediction));
      outcomeValue(payload.target_label || 'Observed outcome', payload.class_labels?.[sample.target] ?? fmt(sample.target));
    } else if (ctx.data.model_type === 'regression') {
      outcomeValue(payload.prediction_label || 'Predicted value', fmt(sample.prediction));
      outcomeValue(payload.target_label || 'Observed value', fmt(sample.target));
      outcomeValue('Prediction − observed', typeof sample.prediction === 'number' && typeof sample.target === 'number'
        ? fmt(sample.prediction - sample.target) : 'Not recorded');
    } else {
      const pred = sample.prediction, actual = sample.target;
      outcomeValue(payload.prediction_label || 'Recommended records', fmt(pred?.length));
      outcomeValue(payload.target_label || 'Observed records', fmt(actual?.length));
      const targets = Array.isArray(actual) ? new Set(actual.map(String)) : null;
      outcomeValue('Matches in recommendations', fmt(targets && Array.isArray(pred) ? pred.filter(x => targets.has(String(x))).length : null));
    }
    if (sample.time != null) outcomeValue('Prediction date', String(sample.time).slice(0, 10));
    const neighbors = new Map(sample.nodes.map(node => [node.id, []]));
    for (const edge of sample.edges) {
      neighbors.get(edge.source)?.push(edge.target);
      neighbors.get(edge.target)?.push(edge.source);
    }
    parents = new Map();
    const depth = new Map([[sample.root, 0]]), queue = [sample.root];
    for (let i = 0; i < queue.length; i++) {
      for (const neighbor of neighbors.get(queue[i]) || []) {
        if (depth.has(neighbor)) continue;
        depth.set(neighbor, depth.get(queue[i]) + 1);
        parents.set(neighbor, queue[i]);
        queue.push(neighbor);
      }
    }
    const tableFilter = ui.tables.length > 0 && ui.tables.length < allTables.length;
    const sizeFilter = ui.sizes.length > 0;
    const filtering = tableFilter || sizeFilter;
    const children = new Map();
    for (const [id, parent] of parents) {
      if (!children.has(parent)) children.set(parent, []);
      children.get(parent).push(id);
    }
    statuses = new Map();
    if (filtering) {
      for (const node of sample.nodes) {
        const {tier} = scoring.nodes.get(node.id);
        const size = tier === null ? 'Not recorded' : tier === 'equal' ? 'Same score' : tierLabels[tier];
        if (sizeFilter && !ui.sizes.includes(size)) continue;
        if (!tableFilter || ui.tables.includes(node.table)) statuses.set(node.id, 'active');
        else if (sizeFilter) statuses.set(node.id, 'table-excluded');
      }
      for (const id of [...statuses.keys()]) {
        for (let parent = parents.get(id); parent != null && !statuses.has(parent); parent = parents.get(parent)) statuses.set(parent, 'path-only');
      }
    } else {
      for (const id of queue) {
        const parent = parents.get(id);
        if (ui.fullDepth || depth.get(id) <= 2 || (statuses.has(parent) && ui.expanded.includes(parent))) statuses.set(id, 'active');
      }
      for (const node of sample.nodes) if (!depth.has(node.id)) statuses.set(node.id, 'active');
    }
    const activeCount = [...statuses.values()].filter(status => status === 'active').length;
    const excludedCount = [...statuses.values()].filter(status => status === 'table-excluded').length;
    const pathCount = statuses.size - activeCount - excludedCount;
    if (statuses.get(ui.pinned) !== 'active') ui.pinned = sample.nodes.find(n => statuses.get(n.id) === 'active')?.id;
    const hasDeeperNodes = [...depth.values()].some(hops => hops > 2);
    expand.hidden = filtering || !hasDeeperNodes;
    expand.textContent = ui.fullDepth ? 'Collapse to 2 hops' : 'Expand all';
    visibility.textContent = filtering ? [activeCount + ' matching', ...(pathCount ? [pathCount + ' on connecting paths'] : []),
      ...(excludedCount ? [excludedCount + ' outside selected tables'] : [])].join(' · ')
      : hasDeeperNodes ? statuses.size + ' of ' + sample.nodes.length + ' records' : '';
    const connected = sample.nodes.filter(n => depth.has(n.id) && statuses.has(n.id));
    const detached = sample.nodes.filter(n => !depth.has(n.id) && statuses.has(n.id));
    const maxDepth = Math.max(0, ...connected.map(n => depth.get(n.id)));
    const lanes = Array.from({length: maxDepth + 1}, (_, i) => connected.filter(n => depth.get(n.id) === i));
    const cols = Math.max(3, lanes.length);
    const W = Math.max(720, 150 + (cols - 1) * 180), pad = 75, step = (W - 2 * pad) / (cols - 1);
    const maxRows = Math.max(1, ...lanes.map(a => a.length));
    const height = Math.max(300, 90 + maxRows * 43);
    const positions = new Map();
    lanes.forEach((nodes, i) => nodes.forEach((n, j) => positions.set(n.id, {x: pad + i * step,
      y: nodes.length === 1 ? height / 2 : 75 + (j * (height - 125)) / Math.max(1, nodes.length - 1)})));
    canvas.replaceChildren();
    const svg = S('svg', {viewBox: `0 0 ${W} ${height}`, role: 'group',
      'aria-label': 'Saved graph for ' + recordLabel(nodeMap.get(sample.root))}, canvas);
    if (W > 720) svg.style.minWidth = W + 'px';
    lanes.forEach((nodes, i) => {
      const tables = [...new Set(nodes.map(n => n.table))];
      S('text', {x: pad + i * step, y: 28, 'text-anchor': 'middle', class: 'kx-sg-lane', text: i === 0 ? entity
        : tables.length === 1 ? tables[0].charAt(0).toUpperCase() + tables[0].slice(1) : 'Related records'}, svg);
    });
    paths = [];
    groups = [];
    function addEdges(target, points) {
      for (const e of sample.edges) {
        const a = points.get(e.source), b = points.get(e.target);
        if (!a || !b) continue;
        const mid = (a.x + b.x) / 2;
        const p = S('path', {d: `M${a.x},${a.y} C${mid},${a.y} ${mid},${b.y} ${b.x},${b.y}`, class: 'kx-sg-edge'}, target);
        S('title', {text: e.relation}, p);
        paths.push([p, e]);
      }
    }
    addEdges(svg, positions);
    function addNode(n, {x, y}, parent) {
      const score = scoring.nodes.get(n.id), status = statuses.get(n.id);
      const scoreText = 'Summed local sensitivity ' + (score.score === null ? 'not recorded' : String(score.score))
        + (score.tier === null ? '' : score.tier === 'equal' ? ', no relative ranking' : ', ' + tierLabels[score.tier]);
      const g = S('g', {class: 'kx-sg-node', transform: `translate(${x},${y})`, role: 'button',
        tabindex: status === 'active' ? '0' : '-1', 'data-status': status, 'aria-disabled': String(status !== 'active'),
        'aria-label': n.table + ', ' + recordLabel(n) + '. ' + scoreText, 'aria-pressed': 'false',
        'data-root': String(n.id === sample.root), 'data-id': n.id,
        'data-score': score.score === null ? 'unscored' : score.score,
        'data-tier': score.tier === null ? 'unscored' : score.tier}, parent);
      S('circle', {r: n.count != null ? 20 : score.radius + 5, class: 'kx-sg-ring'}, g);
      if (n.count != null) {
        S('rect', {x: -12, y: -12, width: 24, height: 24, rx: 5, class: 'kx-sg-dot'}, g);
        S('text', {x: 0, y: 4, 'text-anchor': 'middle', class: 'kx-sg-plus', text: '+'}, g);
      } else S('circle', {r: score.radius, class: 'kx-sg-dot'}, g);
      const label = n.count != null ? '+' + n.count + ' records' : n.value != null ? short(n.value, 18) : 'Record ' + n.id;
      const side = lanes[depth.get(n.id)]?.length > 3;
      const last = depth.get(n.id) === maxDepth && maxDepth > 0;
      S('text', side ? {x: last ? -24 : 24, y: 4, 'text-anchor': last ? 'end' : 'start', text: label}
        : {x: 0, y: 35, 'text-anchor': 'middle', text: label}, g);
      S('title', {text: n.table + ' · ' + recordLabel(n) + '. ' + scoreText}, g);
      groups.push([g, n]);
      if (status !== 'active') return;
      const descendants = children.get(n.id) || [];
      if (!filtering && !ui.fullDepth && depth.get(n.id) >= 2 && descendants.length) {
        const open = ui.expanded.includes(n.id);
        const chip = S('g', {class: 'kx-sg-branch', role: 'button', tabindex: '0', transform: `translate(${x + 20},${y - 20})`,
          'data-branch': n.id, 'aria-expanded': String(open),
          'aria-label': (open ? 'Collapse' : 'Expand') + ' related records for ' + recordLabel(n)}, parent);
        S('rect', {x: -14, y: -9, width: 28, height: 18, rx: 8}, chip);
        S('text', {x: 0, y: 4, 'text-anchor': 'middle', text: open ? '−' : '+' + descendants.length}, chip);
        const toggle = () => {
          ui.expanded = open ? ui.expanded.filter(id => id !== n.id) : [...ui.expanded, n.id];
          draw();
          [...canvas.querySelectorAll('.kx-sg-branch')].find(el => el.dataset.branch === n.id)?.focus();
        };
        chip.addEventListener('click', toggle, nodeOptions);
        chip.addEventListener('keydown', event => {
          if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); toggle(); }
        }, nodeOptions);
      }
      const preview = () => { hovered = n.id; activate(n.id, n.id !== ui.pinned); };
      const restore = () => { hovered = null; activate(ui.pinned); };
      const pin = () => activate(ui.pinned = n.id);
      g.addEventListener('mouseenter', preview, nodeOptions);
      g.addEventListener('mouseleave', restore, nodeOptions);
      g.addEventListener('focus', preview, nodeOptions);
      g.addEventListener('blur', restore, nodeOptions);
      g.addEventListener('click', pin, nodeOptions);
      g.addEventListener('keydown', e => {
        if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pin(); }
        if (e.key === 'Escape') { ui.pinned = sample.root; restore(); }
      }, nodeOptions);
    }
    connected.forEach(n => addNode(n, positions.get(n.id), svg));
    if (!statuses.size) H('p', 'kx-sg-no-matches', canvas, 'No records match these filters.');
    else if (!sample.edges.length) {
      S('text', {x: W / 2, y: height - 24, 'text-anchor': 'middle', class: 'kx-sg-lane', text: 'No related context records were saved.'}, svg);
    }
    if (detached.length) {
      const detachedBox = H('div', 'kx-sg-detached', canvas);
      detachedBox.style.padding = '8px 18px 12px';
      H('div', 'kx-note-plain', detachedBox, 'Records outside the prediction entity’s neighborhood');
      const h = Math.ceil(detached.length / 5) * 82 + 20;
      const detachedSvg = S('svg', {viewBox: `0 0 720 ${h}`, class: 'kx-sg-other-svg', role: 'group', 'aria-label': 'Other saved records'}, detachedBox);
      const detachedPositions = new Map(detached.map((n, i) => [n.id, {x: 72 + (i % 5) * 144, y: 30 + Math.floor(i / 5) * 82}]));
      addEdges(detachedSvg, detachedPositions);
      detached.forEach(n => addNode(n, detachedPositions.get(n.id), detachedSvg));
    }
    note.textContent = 'Saved, sampled context from the test explanation pass. '
      + (sample.nodes.some(n => n.count != null) ? 'Grouped records summarize additional sampled records. ' : '')
      + (detached.length ? 'Records shown separately have no saved path to the prediction entity.' : '');
    activate(ui.pinned);
  }
  function filter(restore) {
    ui.category = cat.value;
    list = payload.samples.filter(s => s.categories.includes(cat.value));
    if (payload.category_samples && Object.hasOwn(payload.category_samples, cat.value)) {
      const order = new Map(payload.category_samples[cat.value].map((id, index) => [String(id), index]));
      list.sort((a, b) => (order.get(a.id) ?? Infinity) - (order.get(b.id) ?? Infinity));
    }
    select.replaceChildren();
    for (const s of list) {
      const root = s.nodes.find(n => n.id === s.root);
      H('option', '', select, fmt(s.original_ids?.[cat.value] ?? root?.value ?? s.id) + ' · ' + s.nodes.length
        + (s.nodes.length === 1 ? ' record' : ' records')).value = s.id;
    }
    select.value = restore && list.some(s => s.id === ui.sample) ? ui.sample : list[0].id;
    draw();
  }
  cat.addEventListener('change', () => filter(), options);
  select.addEventListener('change', draw, options);
  if (savedCategories.has(ui.category)) cat.value = ui.category;
  filter(true);
};
