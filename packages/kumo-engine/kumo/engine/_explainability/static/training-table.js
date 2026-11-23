/* SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* The training table's splits: rows, entities, target balance and time range. */
/* Keep a date on one line when a cell wraps. */
const keepDates = text => String(text).replace(/(\d{4})-(\d{2})-(\d{2})/g, '$1\u2011$2\u2011$3');

KX.cards.training_table = (parent, card, ctx) => {
  cardFrame(parent, card, ctx, null, body => {
    reportTable(body, {columns: card.columns.map(label => ({label})), rows: card.rows},
      'Split', 'kx-splits', keepDates);
  });
};
