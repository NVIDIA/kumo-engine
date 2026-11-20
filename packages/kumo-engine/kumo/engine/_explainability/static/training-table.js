/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* The training table's splits: rows, entities, target balance and time range. */
/* Keep a date on one line when a cell wraps. */
const keepDates = text => String(text).replace(/(\d{4})-(\d{2})-(\d{2})/g, '$1\u2011$2\u2011$3');

KX.cards.training_table = (parent, card, ctx) => {
  cardFrame(parent, card, ctx, null, body => {
    const wrap = H('div', 'kx-table-wrap', body);
    const table = H('table', 'kx-table kx-splits', wrap);
    const head = H('tr', 'kx-col', H('thead', null, table));
    H('th', null, head, 'Split');
    for (const label of card.columns) H('th', null, head, label);
    const tb = H('tbody', null, table);
    for (const row of card.rows) {
      const tr = H('tr', null, tb);
      H('td', null, tr, row.name);
      for (const text of row.texts) H('td', null, tr, keepDates(text));
    }
  });
};
