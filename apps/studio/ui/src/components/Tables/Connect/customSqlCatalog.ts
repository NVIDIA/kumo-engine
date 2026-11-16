// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { SQLNamespace } from "@codemirror/lang-sql";

/**
 * Connector catalog used to seed the SQL editor's autocomplete.
 *
 * Warehouse-authoritative validation and column discovery run against the
 * backend (`resolve_sql` / `get_columns?query`); this catalog only feeds
 * lang-sql's `schema` option so table/column names complete as the user types.
 * The catalog is still stubbed until connector metadata is threaded through — a
 * separate, autocomplete-only concern from the wired authoring endpoints.
 */

/** Connector catalog for autocomplete: table name → column names. Assignable to
 * lang-sql's `SQLNamespace`, so it can be passed straight to the editor. */
export type SqlCatalog = Record<string, string[]>;

export const mockSqlCatalog: SqlCatalog = {
    customers: ["customer_id", "name", "region", "created_at"],
    orders: ["order_id", "customer_id", "amount", "status", "ordered_at"]
};

/** A catalog is already a valid `SQLNamespace`; this keeps the cast in one place. */
export const catalogToSchema = (catalog: SqlCatalog): SQLNamespace => catalog;
