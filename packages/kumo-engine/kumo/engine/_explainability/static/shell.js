/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* The widget shell and the cards every model type shares. */

/* The widget's UI state, with listeners: the cutoff card publishes the
   current cutoff, Column importance the column Column groups shows, and each
   card keeps its own choices under its id. HTML export serializes the values
   and the exported widget starts from them. */
function store(signal, saved) {
  const values = {...saved}, listeners = {};
  return {
    values,
    get: key => values[key],
    /* A card's state object, filled with defaults where nothing is saved. */
    ui: (key, defaults) => Object.assign(values[key] ??= {}, {...defaults, ...values[key]}),
    set(key, value) { values[key] = value; for (const fn of listeners[key] || []) fn(value); },
    on(key, fn) { (listeners[key] = listeners[key] || []).push(fn);
      signal.addEventListener('abort', () => { listeners[key] = []; }); },
  };
}

/* A card: title, optional switch buttons, a body redrawn per view.
   renderView(body, view, ui) gets the card's state object. */
function cardFrame(parent, card, ctx, views, renderView, extraTop) {
  const ui = ctx.state.ui(card.id, {});
  const root = H('section', 'kx-card', parent);
  root.dataset.card = card.id;
  const top = H('div', 'kx-card-top', root);
  H('h3', 'kx-title', top, card.title);
  if (extraTop) extraTop(root);
  const switchParent = extraTop ? H('div', 'kx-switch-row', root) : top;
  const body = H('div', 'kx-card-body', root);
  if (!views?.some(v => v.id === ui.view)) ui.view = views?.[0]?.id ?? null;
  const show = id => {
    ui.view = id;
    for (const button of root.querySelectorAll('.kx-switch button')) {
      button.setAttribute('aria-pressed', String(button.dataset.view === id));
    }
    body.querySelectorAll('.kx-chart').forEach(chart => ctx.resizeObserver?.unobserve(chart));
    body.replaceChildren();
    renderView(body, views ? views.find(v => v.id === id) : null, ui);
  };
  if (views && views.length > 1) {
    const sw = H('div', 'kx-switch', switchParent);
    sw.setAttribute('role', 'group');
    sw.setAttribute('aria-label', card.title + ' view');
    for (const view of views) {
      const button = H('button', null, sw, view.label);
      button.type = 'button';
      button.dataset.view = view.id;
      button.addEventListener('click', () => show(view.id), {signal: ctx.signal});
    }
  }
  show(ui.view);
  if (card.coverage) H('p', 'kx-coverage', root, card.coverage);
  return {root, body, show};
}

/* ------------------------------------------------------------- Metrics */

function metricsView(body, view) {
  const pills = H('div', 'kx-pills', body);
  for (const pill of view.pills) {
    const box = H('div', 'kx-pill', pills);
    H('div', 'kx-pill-name', box, pill.name);
    H('div', 'kx-pill-val', box, pill.holdout_text);
  }
  reportTable(body, view.table, 'Metric', 'kx-metrics');
}

function reportTable(body, t, heading, className, format = text => text) {
  const wrap = H('div', 'kx-table-wrap', body);
  const table = H('table', 'kx-table ' + className, wrap);
  const cr = H('tr', 'kx-col', H('thead', null, table));
  H('th', null, cr, heading);
  for (const col of t.columns) H('th', null, cr, col.label);
  const tb = H('tbody', null, table);
  let first = true;
  for (const row of t.rows) {
    const tr = H('tr', row.baseline ? 'kx-baseline' : row.tune ? 'kx-tune' : null, tb);
    if (row.baseline && first) { tr.classList.add('kx-baseline-first'); first = false; }
    H('td', null, tr, row.name);
    for (const text of row.texts) H('td', null, tr, text === null ? '' : format(text));
  }
}

KX.cards.metrics = (parent, card, ctx) => {
  cardFrame(parent, {...card, coverage: null}, ctx, card.views, (body, view) => {
    metricsView(body, view);
    for (const note of [view.coverage, card.coverage]) if (note) H('p', 'kx-coverage', body, note);
  });
};

/* ---------------------------------------------------- Column importance */

const TABLE_COLORS = [COLOR.hold, COLOR.train, COLOR.val];

/* A table keeps its colour in every widget of the same graph: when every
   table has a colour, colours go to names in alphabetical order; with more
   tables than colours, the top-ranked ones get them. */
function tableColor(tables, name) {
  const order = tables.length <= TABLE_COLORS.length ? [...tables].sort() : tables;
  const i = order.indexOf(name);
  return i >= 0 && i < TABLE_COLORS.length ? {color: TABLE_COLORS[i]} : {color: COLOR.base, hatch: true};
}

KX.cards.importance = (parent, card, ctx) => {
  const linked = !!ctx.cards.groups;
  const expanded = ctx.state.ui(card.id, {expanded: {}}).expanded;
  let chart = null;
  ctx.state.on('column', () => { if (chart && chart.isConnected) chart.redraw(); });
  cardFrame(parent, card, ctx, card.views, (body, view) => {
    const order = (card.views.find(v => v.legend) || view).tables.map(t => t.table);
    legend(body, view.legend
      ? view.tables.map(t => Object.assign({label: t.text}, tableColor(order, t.table)))
      : order.map(name => Object.assign({label: name}, tableColor(order, name))));
    const maxShare = Math.max(...view.bars.map(b => b.share),
      view.remainder ? view.remainder.share : 0);
    chart = chartHost(body, (host, width) => {
      const open = expanded[view.id];
      const shown = open ? view.bars : view.bars.slice(0, view.top);
      const pick = ctx.state.get('column');
      const selected = pick ? pick.key : ctx.cards.groups?.default;
      const rows = shown.map((bar, i) => Object.assign({
        label: bar.label, values: [bar.share],
        texts: [i === 0 && view.top_callout ? view.top_callout : bar.text],
        strong: i === 0, note: bar.callout || null,
        selected: linked && !!bar.group && bar.group === selected,
        aria: bar.label + ' ' + bar.text,
        onClick: linked ? () => ctx.state.set('column', {key: bar.group, label: bar.label}) : null,
      }, tableColor(order, bar.table)));
      if (view.remainder && view.bars.length > view.top) {
        rows.push(open
          ? {label: 'Top ' + view.top + ' only \u25B4', values: [null], texts: [],
            aria: 'Show the top ' + view.top + ' only',
            onClick: () => { expanded[view.id] = false; host.redraw(); }}
          : {label: view.remainder.label + ' \u25BE', values: [view.remainder.share],
            texts: [view.remainder.text], color: COLOR.base, hatch: true,
            aria: 'Show all ' + view.remainder.label,
            onClick: () => { expanded[view.id] = true; host.redraw(); }});
      }
      rowsChart(host, width, {rows, origin: 0, domain: [0, maxShare], marker: linked,
        right: 150, signal: ctx.signal, label: card.title});
    }, ctx);
  });
};

/* ------------------------------------------------------- Column groups */

KX.cards.groups = (parent, card, ctx) => {
  let columnEvents;
  ctx.onDispose(() => columnEvents?.abort());
  const ui = ctx.state.ui(card.id, {cls: 0});
  const frame = cardFrame(parent, card, ctx, null, body => {
    columnEvents?.abort();
    columnEvents = new AbortController();
    const columnCtx = {...ctx, signal: columnEvents.signal,
      onDispose: cleanup => columnEvents.signal.addEventListener('abort', cleanup, {once: true})};
    const pick = ctx.state.get('column');
    const key = pick ? pick.key : card.default;
    const breakdown = key ? card.columns[key] : null;
    let col = breakdown?.by_class ? breakdown.by_class[ui.cls] || breakdown.by_class[0] : breakdown;
    if (!col) {
      if (pick) H('p', 'kx-colhead', body, pick.label);
      H('p', 'kx-empty', body, 'No group breakdown written for this column');
      return;
    }
    if (card.selectable) {
      const select = picker(H('div', 'kx-sg-controls', body), 'Column', 'cohort-column');
      for (const [id, view] of Object.entries(card.columns)) {
        H('option', null, select, (view.by_class?.[0] || view).header.split(' · ').slice(0, 2).join(' · ')).value = id;
      }
      select.value = key;
      select.addEventListener('change', () => ctx.state.set('column', {key: select.value}), {signal: columnCtx.signal});
    }
    const head = H('p', 'kx-colhead', body);
    function labelColumn() {
      const parts = col.header.split(' · ');
      head.textContent = parts[0];
      if (parts.length > 1) H('span', null, head, ' · ' + parts.slice(1).join(' · '));
    }
    labelColumn();
    if (breakdown.by_class?.length > 1) {
      const select = picker(H('div', 'kx-sg-controls', body), 'Class label', 'group-class');
      for (const view of breakdown.by_class) H('option', null, select, view.label);
      select.selectedIndex = breakdown.by_class.indexOf(col);
      select.addEventListener('change', () => {
        col = breakdown.by_class[ui.cls = select.selectedIndex];
        labelColumn();
        chart.redraw();
      }, {signal: columnCtx.signal});
    }
    /* A batch records no actual values, so its groups have one series. */
    const series = [['actual', ROLE.actual], ['predicted', ROLE.model]].filter(([key]) => col['average_' + key] != null);
    legend(body, series.map(([key, color]) => ({label: 'average ' + key, color})));
    const chart = chartHost(body, (host, width) => {
      const rows = col.groups.map((grp, i) => ({
        label: grp.label, sub: grp.share_text,
        values: series.map(([key]) => grp[key]),
        /* The gap callout carries this group's two values, so its bars
           are not labelled a second time. */
        texts: col.gap_callout && col.gap_callout.index === i ? [] : series.map(([key]) => grp[key + '_text']),
        band: col.gap_callout && col.gap_callout.index === i,
      }));
      const vals = rows.flatMap(row => row.values)
        .concat(series.map(([key]) => col['average_' + key])).filter(v => v !== null && v !== undefined);
      const lo = Math.min(0, ...vals), hi = Math.max(0, ...vals);
      const gap = col.gap_callout ? twoLines(col.gap_callout.text) : '';
      const lane = gap ? Math.max(...gap.split('\n').map(l => l.length)) * 6.8 + 24 : 0;
      /* Bars start at zero; each average is its own dashed line in its
         series' colour, labelled at its top. */
      const res = rowsChart(host, width, {rows, series: series.map(([, color]) => ({color})),
        origin: 0, domain: [lo, hi * 1.04], axis: true, right: 56 + lane,
        top: 44, xTitle: col.axis, label: card.title});
      const notes = S('g', {}, res.svg);
      const averages = (col.average_callout || '').split(' \u00b7 ');
      series.forEach(([key, color], k) => {
        const v = col['average_' + key];
        if (!averages[k]) return;
        const px = res.x(v);
        /* Drawn under the bars and their values, like a reference line. */
        const line = S('path', {d: `M${px} 36V${res.rowY(rows.length)}`, style: {stroke: color,
          strokeWidth: 1.4, strokeDasharray: '5 4'}});
        res.g.insertBefore(line, res.g.firstChild);
        res.space.add(px - 2, 36, 4, res.rowY(rows.length) - 36);
        const placed = note(notes, res.space, averages[k], [{x: px, y: 30, anchor: 'middle'},
          {x: px, y: 14, anchor: 'middle'}, {x: px + 6, y: 30}, {x: px - 6, y: 30, anchor: 'end'},
          {x: px + 6, y: 14}, {x: px - 6, y: 14, anchor: 'end'}], 'kx-t-note-plain');
        const node = notes.lastChild;
        if (placed && node) node.style.fill = color;
      });
      if (gap) {
        const i = col.gap_callout.index;
        const ry = res.rowY(i) + res.rowH / 2 - 3;
        const grp = col.groups[i];
        const end = res.x(Math.max(grp.actual, grp.predicted, col.average_actual)) + 8;
        const start = res.x(Math.min(grp.actual, grp.predicted, col.average_actual)) - 8;
        const right = [], leftSide = [];
        for (let dx = 0; dx < width; dx += 8) {
          right.push({x: end + dx, y: ry});
          leftSide.push({x: start - dx, y: ry, anchor: 'end'});
        }
        note(notes, res.space, gap, right.concat(leftSide));
      }
    }, columnCtx);
  });
  ctx.state.on('column', () => {
    ui.cls = 0;
    frame.show(null);
    if (frame.root.scrollIntoView) frame.root.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  });
  return frame;
};

KX.cards.status = (parent, card, ctx) => cardFrame(parent, card, ctx, null, body => {
  for (const message of card.messages) H('p', 'kx-coverage', body, message);
});

/* ------------------------------------------------------------- Shell */

/* `saved` is the UI state the initializer was given (see _render.py): a
   restored export or the copy being printed. */
function mount(host, data, signal, onDispose) {
  const root = H('div', 'kx', host);
  const cards = Object.fromEntries(data.tabs.flatMap(t => t.cards).map(c => [c.id, c]));
  const ctx = {signal, onDispose, state: store(signal, saved), data, host, cards};
  root.state = ctx.state;
  ENTITIES = data.entities || 'rows';
  if (data.header) {
    const head = H('div', 'kx-head', root);
    if (data.header.query) H('div', 'kx-query', head, data.header.query);
    for (const [key, value] of Object.entries(data.header)) {
      if (key === 'identity') {
        const list = H('dl', 'kx-identity', head);
        for (const [label, text] of value) {
          const item = H('div', null, list);
          H('dt', null, item, label);
          H('dd', null, item, text);
        }
      } else if (key !== 'query' && value) H('div', 'kx-sub', head, value);
    }
  }

  const tabs = H('div', 'kx-tabs', root);
  tabs.setAttribute('role', 'tablist');
  const panels = [];
  data.tabs.forEach((tab, i) => {
    const button = H('button', 'kx-tab', tabs, tab.label);
    button.type = 'button';
    button.id = host.id + '-tab-' + tab.id;
    button.setAttribute('role', 'tab');
    const panel = H('div', 'kx-panel', root);
    panel.id = host.id + '-panel-' + tab.id;
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', button.id);
    button.setAttribute('aria-controls', panel.id);
    panels.push({tab, button, panel, built: false});
  });
  const select = index => {
    ctx.state.values.tab = index;
    panels.forEach((p, i) => {
      const on = i === index;
      p.button.setAttribute('aria-selected', String(on));
      p.button.tabIndex = on ? 0 : -1;
      p.panel.hidden = !on;
      if (on && !p.built) {
        p.built = true;
        for (const card of p.tab.cards) {
          const render = KX.cards[card.id];
          if (render) render(p.panel, card, ctx);
        }
      }
    });
  };
  panels.forEach((p, i) => {
    p.button.addEventListener('click', () => select(i), {signal});
    p.button.addEventListener('keydown', e => {
      const step = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (!step) return;
      e.preventDefault();
      const next = (i + step + panels.length) % panels.length;
      select(next);
      panels[next].button.focus();
    }, {signal});
  });
  /* One tab needs no tab strip. */
  if (panels.length === 1) {
    tabs.remove();
    panels[0].panel.setAttribute('role', 'region');
    panels[0].panel.setAttribute('aria-label', panels[0].tab.label);
  }
  select(panels[ctx.state.values.tab] ? ctx.state.values.tab : 0);
  return root;
}

KX.mount = mount;
