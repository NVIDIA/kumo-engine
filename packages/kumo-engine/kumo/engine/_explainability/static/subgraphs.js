/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
function svgEl(tag, attrs, parent, text) {
  return S(tag, text === undefined ? attrs : { ...attrs, text }, parent);
}
const labels = {
  saved: 'Saved examples',
  random: 'Random examples',
  unsure: 'Uncertain predictions',
  true_positive: 'True positives',
  true_negative: 'True negatives',
  false_positive: 'False positives',
  false_negative: 'False negatives',
  small_error: 'Small error',
  large_error: 'Large error',
  small_percent_error: 'Small percentage error',
  large_percent_error: 'Large percentage error',
  low_precision: 'Low precision',
  high_precision: 'High precision',
  old_entity: 'Older entities',
  recent_entity: 'Recent entities',
};
function fmt(value) {
  if (value == null) return 'Not recorded';
  if (typeof value === 'number' && !Number.isInteger(value))
    return value.toLocaleString('en-US', {maximumFractionDigits: 3});
  return typeof value === 'object' ? JSON.stringify(value) : String(value);
}
const short = (v, n = 20) => {
  v = String(v);
  return v.length > n
    ? v.slice(0, Math.ceil((n - 1) / 2)) + '…' + v.slice(-Math.floor((n - 1) / 2))
    : v;
};
const tierLabels = ['Lowest 25%', '25–50%', '50–75%', 'Top 25%'];
const tierRadii = [7, 10, 13, 16];
function scoreNodes(nodes) {
  const records = new Map();
  const scored = [];
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
function classOutcome(parent, title, values, options) {
  const section = H('div', 'kx-sg-class-outcome', parent);
  const table = H('table', 'kx-table', section);
  H('caption', null, table, title);
  const columns = H('tr', 'kx-col', H('thead', null, table));
  for (const name of ['Label', 'Value']) H('th', null, columns, name).scope = 'col';
  const body = H('tbody', null, table);
  const rows = values && typeof values === 'object' && !Array.isArray(values)
    ? Object.entries(values) : [];
  if (!rows.length) {
    const missing = H('td', null, H('tr', null, body), 'Not recorded');
    missing.colSpan = 2;
    return;
  }
  const pageSize = 10;
  const pages = Math.ceil(rows.length / pageSize);
  let page = 0, prev, next, position;
  function show() {
    body.replaceChildren();
    for (const [label, value] of rows.slice(page * pageSize, (page + 1) * pageSize)) {
      const row = H('tr', null, body);
      H('td', null, row, label);
      H('td', null, row, Number.isFinite(value) ? value.toFixed(2) : 'Not recorded');
    }
    if (pages > 1) {
      position.textContent = (page + 1) + ' of ' + pages;
      prev.disabled = page === 0;
      next.disabled = page === pages - 1;
    }
  }
  if (pages > 1) {
    const pager = H('div', 'kx-sg-pager', section);
    prev = H('button', null, pager, 'Previous');
    next = H('button', null, pager, 'Next');
    prev.type = next.type = 'button';
    prev.setAttribute('aria-label', 'Previous page of ' + title.toLowerCase());
    next.setAttribute('aria-label', 'Next page of ' + title.toLowerCase());
    position = H('span', 'kx-note-plain', pager);
    prev.addEventListener('click', () => {page--; show();}, options);
    next.addEventListener('click', () => {page++; show();}, options);
  }
  show();
}

KX.cards.subgraphs = (panel, payload, ctx) => {
  const options = { signal: ctx.signal };
  const entity = payload.entity_label || 'Entity';
  const card = H('section', 'kx-card kx-sg', panel);
  card.dataset.card = payload.id;
  const top = H('div', 'kx-card-top', card);
  H('h3', 'kx-title', top, payload.title);
  const count = payload.samples.length;
  const coverage = count + (count === 1 ? ' saved example' : ' saved examples');
  H('span', 'kx-note-plain', top, payload.unavailable_count
    ? coverage + ' · ' + payload.unavailable_count + ' unavailable' : coverage);
  const controls = H('div', 'kx-sg-controls', card);
  const categoryPicker = subgraphPicker(controls, 'Selection', 'category', ctx);
  const cat = categoryPicker.select;
  const savedCategories = new Set(payload.samples.flatMap(s => s.categories));
  const cats = [...new Set([...(payload.categories || []), ...savedCategories])].filter(c => savedCategories.has(c));
  for (const c of cats) {
    const o = H('option', '', cat, Object.hasOwn(labels, c) ? labels[c] : c.replaceAll('_', ' '));
    o.value = c;
  }
  const entityPicker = subgraphPicker(controls, entity, 'entity', ctx);
  const select = entityPicker.select;
  const pager = H('div', 'kx-sg-pager', controls);
  const prev = H('button', '', pager, 'Previous');
  const next = H('button', '', pager, 'Next');
  prev.type = next.type = 'button';
  const position = H('span', 'kx-note-plain', pager);
  const outcome = H('div', 'kx-sg-outcome', card);
  const legend = H('div', 'kx-sg-legend', card);
  for (const [t, c] of [
    ['Prediction entity', 'root'],
    ['Related record', ''],
    ['Grouped records', 'aggregate'],
  ]) {
    const l = H('span', '', legend);
    H('i', c, l);
    l.append(document.createTextNode(t));
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
  H(
    'p',
    'kx-sg-note',
    card,
    'Hover or focus a record to see its path to the prediction entity. Select a record to keep its details open.',
  );
  const note = H('p', 'kx-sg-note', card);
  let list = [],
    sample,
    pinned,
    hovered = null,
    nodeMap,
    groups,
    paths,
    svg,
    scoring;
  function recordLabel(n) {
    return n.count != null
      ? fmt(n.count) + ' grouped records'
      : n.value != null
        ? (n.key || 'ID') + ' ' + fmt(n.value)
        : n.table + ' record ' + n.id;
  }
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
    const scoreValue = H(
      'strong',
      '',
      scoreLine,
      score.score === null ? 'Not recorded' : String(Number(score.score.toPrecision(6))),
    );
    scoreValue.title = score.score === null ? 'Not recorded' : String(score.score);
    H(
      'span',
      'kx-sg-score-tier',
      scoreLine,
      score.tier === null
        ? 'Unscored'
        : score.tier === 'equal'
          ? scoring.count === 1
            ? 'Only scored record'
            : 'All scored records are equal'
          : tierLabels[score.tier],
    );
    if (n.count != null) {
      H(
        'p',
        'kx-sg-note',
        detail,
        fmt(n.count) +
          ' additional sampled records are grouped here. Individual feature values were not saved for this group.',
      );
      if (n.start || n.end)
        H('p', 'kx-sg-note', detail, [n.start, n.end].filter(Boolean).join(' to '));
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
    const scored = features
      .filter((f) => typeof f.score === 'number' && Number.isFinite(f.score))
      .sort((a, b) => b.score - a.score);
    if (scored.length) {
      const sensitivity = H('details', '', detail);
      H('summary', '', sensitivity, 'Local sensitivity');
      addRows(
        scored.map((f) => ({ name: f.name, value: f.score })),
        sensitivity,
      );
      H(
        'p',
        'kx-sg-note',
        sensitivity,
        'Unsigned gradient scores, scaled within this example. These do not indicate direction or causality.',
      );
    }
  }
  function activate(id, preview = false) {
    const path = new Set();
    if (id == null) {
      detail.replaceChildren();
      H('p', 'kx-sg-note', detail, 'No matching record selected.');
      return;
    }
    for (let current = id; current != null; current = parents.get(current)) path.add(current);
    for (const [g, n] of groups) {
      g.dataset.active = String(n.id === id);
      g.dataset.muted = String(hovered !== null && !path.has(n.id));
      g.setAttribute('aria-pressed', String(n.id === pinned));
    }
    for (const [p, e] of paths) {
      const on = hovered !== null && path.has(e.source) && path.has(e.target);
      p.dataset.active = String(on);
      p.dataset.moving = String(on && hovered !== null);
      p.dataset.muted = String(hovered !== null && !on);
    }
    showDetail(id, preview);
  }
  function outcomeValue(label, value) {
    const s = H('span', '', outcome, label);
    H('strong', '', s, value);
  }
  let nodeEvents, parents, selectedTables, selectedSizes, allTables, statuses;
  let expanded = new Set(), fullDepth = false;
  function resetFilters() {
    allTables = [...new Set(sample.nodes.map(n => n.table))];
    selectedTables = new Set(allTables);
    selectedSizes = new Set();
    expanded = new Set();
    fullDepth = false;
    filterFields.replaceChildren();
    function checkboxes(heading, choices, chosen) {
      const field = H('fieldset', null, filterFields);
      H('legend', null, field, heading);
      for (const item of choices) {
        const label = H('label', null, field);
        const box = H('input', null, label);
        box.type = 'checkbox';
        box.value = item;
        box.dataset.filter = heading;
        box.checked = chosen.has(item);
        label.append(document.createTextNode(item));
      }
    }
    checkboxes('Tables', allTables, selectedTables);
    const sizes = scoring.equal ? ['Same score'] : [...tierLabels].reverse();
    if ([...scoring.nodes.values()].some(n => n.score === null)) sizes.push('Not recorded');
    checkboxes('Sensitivity size', sizes, selectedSizes);
  }
  filterFields.addEventListener('change', event => {
    const box = event.target;
    const chosen = box.dataset.filter === 'Tables' ? selectedTables : selectedSizes;
    if (box.checked) chosen.add(box.value); else chosen.delete(box.value);
    draw();
  }, options);
  expand.addEventListener('click', () => {
    fullDepth = !fullDepth;
    expanded.clear();
    draw();
  }, options);

  ctx.onDispose(() => nodeEvents?.abort());
  function draw() {
    nodeEvents?.abort();
    nodeEvents = new AbortController();
    const nodeOptions = { signal: nodeEvents.signal };
    const previous = sample;
    sample = list.find((s) => s.id === select.value) || list[0];
    if (!sample) return;
    select.value = sample.id;
    categoryPicker.sync();
    entityPicker.sync();
    if (previous !== sample) pinned = sample.root;
    hovered = null;
    nodeMap = new Map(sample.nodes.map((n) => [n.id, n]));
    scoring = scoreNodes(sample.nodes);
    if (previous !== sample) resetFilters();
    sizeLegend.replaceChildren();
    const items = scoring.equal
      ? [['Same score', 10]]
      : scoring.count
        ? tierLabels.map((label, i) => [label, tierRadii[i]])
        : [];
    if (scoring.equal && scoring.count === 1) items[0][0] = 'Only scored record';
    items.push(['Not recorded', 10]);
    for (const [label, r] of items) {
      const item = H('span', '', sizeLegend);
      const swatch = svgEl(
        'svg',
        { width: 36, height: 36, viewBox: '0 0 36 36', 'aria-hidden': 'true' },
        item,
      );
      svgEl(
        'circle',
        {
          cx: 18,
          cy: 18,
          r,
          class: label === 'Not recorded' ? 'kx-sg-size-unscored' : 'kx-sg-size-scored',
        },
        swatch,
      );
      item.append(document.createTextNode(label));
    }
    sizeNote.textContent = !scoring.count
      ? 'No local sensitivity scores were recorded for this example.'
      : scoring.equal
        ? scoring.count === 1
          ? 'One scored record; no relative size ranking.'
          : 'All scored records have the same summed sensitivity; no relative size ranking.'
        : 'Size shows the within-example sensitivity tier, not proportional contribution. Ties share a size.';
    const index = list.indexOf(sample);
    position.textContent = index + 1 + ' of ' + list.length;
    prev.disabled = index === 0;
    next.disabled = index === list.length - 1;
    outcome.replaceChildren();
    if (['multiclass_classification', 'multilabel_classification'].includes(payload.task_type)) {
      classOutcome(outcome, 'Predictions', sample.prediction, nodeOptions);
      classOutcome(outcome, 'Ground truth', sample.target, nodeOptions);
    } else if (ctx.data.model_type === 'classification') {
      outcomeValue(
        payload.prediction_label || 'Predicted probability',
        typeof sample.prediction === 'number'
          ? (sample.prediction * 100).toFixed(1) + '%'
          : fmt(sample.prediction),
      );
      outcomeValue(
        payload.target_label || 'Observed outcome',
        payload.class_labels?.[sample.target] ?? fmt(sample.target),
      );
    } else if (ctx.data.model_type === 'regression') {
      outcomeValue(payload.prediction_label || 'Predicted value', fmt(sample.prediction));
      outcomeValue(payload.target_label || 'Observed value', fmt(sample.target));
      outcomeValue(
        'Prediction − observed',
        typeof sample.prediction === 'number' && typeof sample.target === 'number'
          ? fmt(sample.prediction - sample.target)
          : 'Not recorded',
      );
    } else {
      const pred = sample.prediction,
        actual = sample.target;
      outcomeValue(payload.prediction_label || 'Recommended records', fmt(pred?.length));
      outcomeValue(payload.target_label || 'Observed records', fmt(actual?.length));
      const targets = Array.isArray(actual) ? new Set(actual.map(String)) : null;
      outcomeValue(
        'Matches in recommendations',
        fmt(
          targets && Array.isArray(pred) ? pred.filter((x) => targets.has(String(x))).length : null,
        ),
      );
    }
    if (sample.time != null) outcomeValue('Prediction date', String(sample.time).slice(0, 10));
    const neighbors = new Map(sample.nodes.map((node) => [node.id, []]));
    for (const edge of sample.edges) {
      neighbors.get(edge.source)?.push(edge.target);
      neighbors.get(edge.target)?.push(edge.source);
    }
    parents = new Map();
    const depth = new Map([[sample.root, 0]]),
      queue = [sample.root];
    for (let i = 0; i < queue.length; i++) {
      const id = queue[i];
      for (const neighbor of neighbors.get(id) || []) {
        if (!depth.has(neighbor)) {
          depth.set(neighbor, depth.get(id) + 1);
          parents.set(neighbor, id);
          queue.push(neighbor);
        }
      }
    }
    const tableFilter = selectedTables.size > 0 && selectedTables.size < allTables.length;
    const sizeFilter = selectedSizes.size > 0;
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
        if (sizeFilter && !selectedSizes.has(size)) continue;
        if (!tableFilter || selectedTables.has(node.table)) statuses.set(node.id, 'active');
        else if (sizeFilter) statuses.set(node.id, 'table-excluded');
      }
      for (const id of [...statuses.keys()]) {
        for (let parent = parents.get(id); parent != null && !statuses.has(parent); parent = parents.get(parent))
          statuses.set(parent, 'path-only');
      }
    } else {
      for (const id of queue) {
        const parent = parents.get(id);
        if (fullDepth || depth.get(id) <= 2 || (statuses.has(parent) && expanded.has(parent)))
          statuses.set(id, 'active');
      }
      for (const node of sample.nodes) if (!depth.has(node.id)) statuses.set(node.id, 'active');
    }
    const activeCount = [...statuses.values()].filter(status => status === 'active').length;
    const excludedCount = [...statuses.values()].filter(status => status === 'table-excluded').length;
    const pathCount = statuses.size - activeCount - excludedCount;
    if (statuses.get(pinned) !== 'active') pinned = sample.nodes.find(n => statuses.get(n.id) === 'active')?.id;
    const hasDeeperNodes = [...depth.values()].some(hops => hops > 2);
    expand.hidden = filtering || !hasDeeperNodes;
    expand.textContent = fullDepth ? 'Collapse to 2 hops' : 'Expand all';
    visibility.textContent = filtering ? [activeCount + ' matching',
      ...(pathCount ? [pathCount + ' on connecting paths'] : []),
      ...(excludedCount ? [excludedCount + ' outside selected tables'] : [])].join(' · ')
      : hasDeeperNodes ? statuses.size + ' of ' + sample.nodes.length + ' records' : '';
    const connected = sample.nodes.filter(n => depth.has(n.id) && statuses.has(n.id));
    const detached = sample.nodes.filter(n => !depth.has(n.id) && statuses.has(n.id));
    const maxDepth = Math.max(0, ...connected.map(n => depth.get(n.id)));
    const lanes = Array.from({ length: maxDepth + 1 }, (_, i) =>
      connected.filter((n) => depth.get(n.id) === i),
    );
    const cols = Math.max(3, lanes.length);
    const W = Math.max(720, 150 + (cols - 1) * 180),
      pad = 75,
      step = (W - 2 * pad) / (cols - 1);
    const maxRows = Math.max(1, ...lanes.map((a) => a.length));
    const height = Math.max(300, 90 + maxRows * 43);
    const positions = new Map();
    lanes.forEach((nodes, i) =>
      nodes.forEach((n, j) =>
        positions.set(n.id, {
          x: pad + i * step,
          y:
            nodes.length === 1
              ? height / 2
              : 75 + (j * (height - 125)) / Math.max(1, nodes.length - 1),
        }),
      ),
    );
    canvas.replaceChildren();
    svg = svgEl(
      'svg',
      {
        viewBox: `0 0 ${W} ${height}`,
        role: 'group',
        'aria-label': 'Saved graph for ' + recordLabel(nodeMap.get(sample.root)),
      },
      canvas,
    );
    if (W > 720) svg.style.minWidth = W + 'px';
    lanes.forEach((nodes, i) => {
      const tables = [...new Set(nodes.map((n) => n.table))];
      svgEl(
        'text',
        { x: pad + i * step, y: 28, 'text-anchor': 'middle', class: 'kx-sg-lane' },
        svg,
        i === 0
          ? entity
          : tables.length === 1
            ? tables[0].charAt(0).toUpperCase() + tables[0].slice(1)
            : 'Related records',
      );
    });
    paths = [];
    groups = [];
    function addEdges(target, points) {
      for (const e of sample.edges) {
        const a = points.get(e.source),
          b = points.get(e.target);
        if (!a || !b) continue;
        const mid = (a.x + b.x) / 2;
        const p = svgEl(
          'path',
          { d: `M${a.x},${a.y} C${mid},${a.y} ${mid},${b.y} ${b.x},${b.y}`, class: 'kx-sg-edge' },
          target,
        );
        svgEl('title', {}, p, e.relation);
        paths.push([p, e]);
      }
    }
    addEdges(svg, positions);
    function addNode(n, position, parent) {
      const { x, y } = position;
      const score = scoring.nodes.get(n.id);
      const scoreText =
        'Summed local sensitivity ' +
        (score.score === null ? 'not recorded' : String(score.score)) +
        (score.tier === null
          ? ''
          : score.tier === 'equal'
            ? ', no relative ranking'
            : ', ' + tierLabels[score.tier]);
      const g = svgEl(
        'g',
        {
          class: 'kx-sg-node',
          transform: `translate(${x},${y})`,
          role: 'button',
          tabindex: statuses.get(n.id) === 'active' ? '0' : '-1',
          'data-status': statuses.get(n.id),
          'aria-disabled': String(statuses.get(n.id) !== 'active'),
          'aria-label': n.table + ', ' + recordLabel(n) + '. ' + scoreText,
          'aria-pressed': 'false',
          'data-root': String(n.id === sample.root),
          'data-id': n.id,
          'data-score': score.score === null ? 'unscored' : score.score,
          'data-tier': score.tier === null ? 'unscored' : score.tier,
          'data-lavish-action': 'true',
        },
        parent,
      );
      svgEl('circle', { r: n.count != null ? 20 : score.radius + 5, class: 'kx-sg-ring' }, g);
      svgEl(
        n.count != null ? 'rect' : 'circle',
        n.count != null
          ? { x: -12, y: -12, width: 24, height: 24, rx: 5, class: 'kx-sg-dot' }
          : { r: score.radius, class: 'kx-sg-dot' },
        g,
      );
      if (n.count != null)
        svgEl('text', { x: 0, y: 4, 'text-anchor': 'middle', class: 'kx-sg-plus' }, g, '+');
      const label =
        n.count != null
          ? '+' + n.count + ' records'
          : n.value != null
            ? short(n.value, 18)
            : 'Record ' + n.id;
      const side = lanes[depth.get(n.id)]?.length > 3;
      const last = depth.get(n.id) === maxDepth && maxDepth > 0;
      svgEl(
        'text',
        side
          ? { x: last ? -24 : 24, y: 4, 'text-anchor': last ? 'end' : 'start' }
          : { x: 0, y: 35, 'text-anchor': 'middle' },
        g,
        label,
      );
      svgEl('title', {}, g, n.table + ' · ' + recordLabel(n) + '. ' + scoreText);
      if (statuses.get(n.id) !== 'active') {
        groups.push([g, n]);
        return;
      }
      const descendants = children.get(n.id) || [];
      if (!filtering && !fullDepth && depth.get(n.id) >= 2 && descendants.length) {
        const chip = svgEl('g', {class:'kx-sg-branch', role:'button', tabindex:'0',
          transform:`translate(${x + 20},${y - 20})`, 'data-branch':n.id, 'data-lavish-action':'true',
          'aria-expanded':String(expanded.has(n.id)),
          'aria-label':(expanded.has(n.id) ? 'Collapse' : 'Expand') + ' related records for ' + recordLabel(n)}, parent);
        svgEl('rect', {x:-14,y:-9,width:28,height:18,rx:8}, chip);
        svgEl('text', {x:0,y:4,'text-anchor':'middle'}, chip, expanded.has(n.id) ? '−' : '+' + descendants.length);
        const toggle = () => {
          if (expanded.has(n.id)) expanded.delete(n.id); else expanded.add(n.id);
          draw();
          Array.from(canvas.querySelectorAll('.kx-sg-branch')).find(el => el.dataset.branch === n.id)?.focus();
        };
        chip.addEventListener('click', toggle, nodeOptions);
        chip.addEventListener('keydown', event => {
          if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); toggle();}
        }, nodeOptions);
      }
      const preview = () => {
        hovered = n.id;
        activate(n.id, n.id !== pinned);
      };
      const restore = () => {
        hovered = null;
        activate(pinned);
      };
      g.addEventListener('mouseenter', preview, nodeOptions);
      g.addEventListener('mouseleave', restore, nodeOptions);
      g.addEventListener('focus', preview, nodeOptions);
      g.addEventListener('blur', restore, nodeOptions);
      const pin = () => {
        pinned = n.id;
        activate(pinned);
      };
      g.addEventListener('click', pin, nodeOptions);
      g.addEventListener(
        'keydown',
        (e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            pin();
          }
          if (e.key === 'Escape') {
            pinned = sample.root;
            hovered = null;
            activate(pinned);
          }
        },
        nodeOptions,
      );
      groups.push([g, n]);
    }
    connected.forEach((n) => addNode(n, positions.get(n.id), svg));
    if (!statuses.size) H('p','kx-sg-no-matches',canvas,'No records match these filters.');
    else if (!sample.edges.length) {
      svgEl(
        'text',
        { x: W / 2, y: height - 24, 'text-anchor': 'middle', class: 'kx-sg-lane' },
        svg,
        'No related context records were saved.',
      );
    }
    if (detached.length) {
      const detachedBox = H('div', 'kx-sg-detached', canvas);
      detachedBox.style.padding = '8px 18px 12px';
      H(
        'div',
        'kx-note-plain',
        detachedBox,
        'Records outside the prediction entity’s neighborhood',
      );
      const rows = Math.ceil(detached.length / 5),
        h = rows * 82 + 20;
      const detachedSvg = svgEl(
        'svg',
        {
          viewBox: `0 0 720 ${h}`,
          class: 'kx-sg-other-svg',
          role: 'group',
          'aria-label': 'Other saved records',
        },
        detachedBox,
      );
      const detachedPositions = new Map(
        detached.map((n, i) => [n.id, { x: 72 + (i % 5) * 144, y: 30 + Math.floor(i / 5) * 82 }]),
      );
      addEdges(detachedSvg, detachedPositions);
      detached.forEach((n) => addNode(n, detachedPositions.get(n.id), detachedSvg));
    }
    note.textContent =
      'Saved, sampled context from the test explanation pass. ' +
      (sample.nodes.some((n) => n.count != null)
        ? 'Grouped records summarize additional sampled records. '
        : '') +
      (detached.length
        ? 'Records shown separately have no saved path to the prediction entity.'
        : '');
    activate(pinned);
  }
  function filter() {
    list = payload.samples.filter((s) => s.categories.includes(cat.value));
    if (payload.category_samples && Object.hasOwn(payload.category_samples, cat.value)) {
      const order = new Map(payload.category_samples[cat.value].map((id, index) => [String(id), index]));
      list.sort((a,b) => (order.get(a.id) ?? Infinity) - (order.get(b.id) ?? Infinity));
    }
    select.replaceChildren();
    list.forEach((s) => {
      const root = s.nodes.find((n) => n.id === s.root);
      const o = H(
        'option',
        '',
        select,
        fmt(s.original_ids?.[cat.value] ?? root?.value ?? s.id) + ' · ' + s.nodes.length + (s.nodes.length === 1 ? ' record' : ' records'),
      );
      o.value = s.id;
    });
    select.value = list[0].id;
    draw();
  }
  cat.addEventListener('change', () => filter(), options);
  select.addEventListener('change', draw, options);
  prev.addEventListener(
    'click',
    () => {
      select.selectedIndex = Math.max(0, select.selectedIndex - 1);
      draw();
    },
    options,
  );
  next.addEventListener(
    'click',
    () => {
      select.selectedIndex = Math.min(list.length - 1, select.selectedIndex + 1);
      draw();
    },
    options,
  );
  filter();
  card.exportGraphState = () => ({pinned, expanded: [...expanded], fullDepth});
  card.restoreGraphState = state => {
    pinned = nodeMap.has(state.pinned) ? state.pinned : sample.root;
    expanded = new Set(state.expanded.filter(id => nodeMap.has(id)));
    fullDepth = !!state.fullDepth;
    draw();
  };
};
