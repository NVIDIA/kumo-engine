// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Meta, StoryObj } from "@storybook/react-vite";
import { expect, userEvent, within } from "storybook/test";
import { useState } from "react";

import {
    HopGraph,
    HopGraphEdge,
    HopGraphNode
} from "@/components/blocks/visual-graph/hop-graph";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";

const meta: Meta<typeof HopGraph> = {
    title: "Blocks/HopGraph",
    component: HopGraph,
    tags: ["autodocs"],
    parameters: {
        layout: "padded",
        docs: { extractComponentDescription: () => "" }
    },
    argTypes: {
        nodes: { control: "object" },
        edges: { control: "object" }
    }
};

export default meta;
type Story = StoryObj<typeof HopGraph>;

// ─── Sample data ──────────────────────────────────────────────────────────────
// neighborCount is omitted on entity (hop 0) nodes — they have no sampling
// budget. All other nodes use neighborCount: 12 as a uniform default.

const simpleNodes: HopGraphNode[] = [
    { id: "account", table: "account", hop: 0 },
    { id: "transaction", table: "transaction", hop: 1, neighborCount: 12 },
    { id: "card", table: "card", hop: 1, neighborCount: 12 },
    { id: "merchant", table: "merchant", hop: 2, neighborCount: 12 }
];

const simpleEdges: HopGraphEdge[] = [
    { id: "e1", sourceId: "account", targetId: "transaction" },
    { id: "e2", sourceId: "account", targetId: "card" },
    { id: "e3", sourceId: "transaction", targetId: "merchant" }
];

const multiRowNodes: HopGraphNode[] = [
    { id: "customer", table: "customer", hop: 0 },
    { id: "order", table: "order", hop: 1, neighborCount: 12 },
    { id: "review", table: "review", hop: 1, neighborCount: 12 },
    { id: "wishlist", table: "wishlist", hop: 1, neighborCount: 12 },
    { id: "product", table: "product", hop: 2, neighborCount: 12 },
    { id: "category", table: "category", hop: 2, neighborCount: 12 },
    { id: "supplier", table: "supplier", hop: 2, neighborCount: 12 }
];

const multiRowEdges: HopGraphEdge[] = [
    { id: "e1", sourceId: "customer", targetId: "order" },
    { id: "e2", sourceId: "customer", targetId: "review" },
    { id: "e3", sourceId: "customer", targetId: "wishlist" },
    { id: "e4", sourceId: "order", targetId: "product" },
    { id: "e5", sourceId: "review", targetId: "product" },
    { id: "e6", sourceId: "review", targetId: "category" },
    { id: "e7", sourceId: "wishlist", targetId: "supplier" }
];

// 6-hop e-commerce schema: customer entity → orders/sessions/… → products →
// categories/suppliers → subcategories/contracts → attributes → values.
// Some hops intentionally have many rows to stress-test vertical layout.
const largeNodes: HopGraphNode[] = [
    // Hop 0 — entity (no neighborCount)
    { id: "customer", table: "customer", hop: 0 },

    // Hop 1 — 6 direct neighbours
    { id: "order", table: "order", hop: 1, neighborCount: 12 },
    { id: "session", table: "session", hop: 1, neighborCount: 12 },
    {
        id: "support_ticket",
        table: "support_ticket",
        hop: 1,
        neighborCount: 12
    },
    { id: "address", table: "address", hop: 1, neighborCount: 12 },
    {
        id: "payment_method",
        table: "payment_method",
        hop: 1,
        neighborCount: 12
    },
    { id: "review", table: "review", hop: 1, neighborCount: 12 },

    // Hop 2 — 7 nodes (largest hop)
    { id: "order_item", table: "order_item", hop: 2, neighborCount: 12 },
    { id: "shipment", table: "shipment", hop: 2, neighborCount: 12 },
    { id: "invoice", table: "invoice", hop: 2, neighborCount: 12 },
    { id: "campaign", table: "campaign", hop: 2, neighborCount: 12 },
    { id: "device", table: "device", hop: 2, neighborCount: 12 },
    { id: "product", table: "product", hop: 2, neighborCount: 12 },
    { id: "agent", table: "agent", hop: 2, neighborCount: 12 },

    // Hop 3 — 6 nodes
    { id: "category", table: "category", hop: 3, neighborCount: 12 },
    { id: "supplier", table: "supplier", hop: 3, neighborCount: 12 },
    { id: "brand", table: "brand", hop: 3, neighborCount: 12 },
    { id: "warehouse", table: "warehouse", hop: 3, neighborCount: 12 },
    { id: "return_item", table: "return_item", hop: 3, neighborCount: 12 },
    { id: "promotion", table: "promotion", hop: 3, neighborCount: 12 },

    // Hop 4 — 5 nodes
    { id: "subcategory", table: "subcategory", hop: 4, neighborCount: 12 },
    {
        id: "supplier_contract",
        table: "supplier_contract",
        hop: 4,
        neighborCount: 12
    },
    {
        id: "warehouse_zone",
        table: "warehouse_zone",
        hop: 4,
        neighborCount: 12
    },
    { id: "promo_rule", table: "promo_rule", hop: 4, neighborCount: 12 },
    { id: "brand_owner", table: "brand_owner", hop: 4, neighborCount: 12 },

    // Hop 5 — 4 nodes
    { id: "attribute", table: "attribute", hop: 5, neighborCount: 12 },
    { id: "certification", table: "certification", hop: 5, neighborCount: 12 },
    { id: "zone_stock", table: "zone_stock", hop: 5, neighborCount: 12 },
    { id: "brand_metric", table: "brand_metric", hop: 5, neighborCount: 12 },

    // Hop 6 — 3 nodes
    { id: "attribute_val", table: "attribute_val", hop: 6, neighborCount: 12 },
    { id: "cert_body", table: "cert_body", hop: 6, neighborCount: 12 },
    { id: "metric_log", table: "metric_log", hop: 6, neighborCount: 12 }
];

const largeEdges: HopGraphEdge[] = [
    // hop 0 → hop 1
    { id: "le1", sourceId: "customer", targetId: "order" },
    { id: "le2", sourceId: "customer", targetId: "session" },
    { id: "le3", sourceId: "customer", targetId: "support_ticket" },
    { id: "le4", sourceId: "customer", targetId: "address" },
    { id: "le5", sourceId: "customer", targetId: "payment_method" },
    { id: "le6", sourceId: "customer", targetId: "review" },

    // hop 1 → hop 2
    { id: "le7", sourceId: "order", targetId: "order_item" },
    { id: "le8", sourceId: "order", targetId: "shipment" },
    { id: "le9", sourceId: "order", targetId: "invoice" },
    { id: "le10", sourceId: "session", targetId: "campaign" },
    { id: "le11", sourceId: "session", targetId: "device" },
    { id: "le12", sourceId: "support_ticket", targetId: "agent" },
    { id: "le13", sourceId: "review", targetId: "product" },
    { id: "le14", sourceId: "payment_method", targetId: "invoice" },

    // hop 2 → hop 3
    { id: "le15", sourceId: "product", targetId: "category" },
    { id: "le16", sourceId: "product", targetId: "supplier" },
    { id: "le17", sourceId: "product", targetId: "brand" },
    { id: "le18", sourceId: "shipment", targetId: "warehouse" },
    { id: "le19", sourceId: "order_item", targetId: "return_item" },
    { id: "le20", sourceId: "campaign", targetId: "promotion" },

    // hop 3 → hop 4
    { id: "le21", sourceId: "category", targetId: "subcategory" },
    { id: "le22", sourceId: "supplier", targetId: "supplier_contract" },
    { id: "le23", sourceId: "warehouse", targetId: "warehouse_zone" },
    { id: "le24", sourceId: "promotion", targetId: "promo_rule" },
    { id: "le25", sourceId: "brand", targetId: "brand_owner" },

    // hop 4 → hop 5
    { id: "le26", sourceId: "subcategory", targetId: "attribute" },
    { id: "le27", sourceId: "supplier_contract", targetId: "certification" },
    { id: "le28", sourceId: "warehouse_zone", targetId: "zone_stock" },
    { id: "le29", sourceId: "brand_owner", targetId: "brand_metric" },

    // hop 5 → hop 6
    { id: "le30", sourceId: "attribute", targetId: "attribute_val" },
    { id: "le31", sourceId: "certification", targetId: "cert_body" },
    { id: "le32", sourceId: "zone_stock", targetId: "metric_log" },
    { id: "le33", sourceId: "brand_metric", targetId: "metric_log" }
];

// Varying neighborCount to demonstrate node sizing:
//   transactions (30) → large,  card (10) → medium,  district (3) → small.

// ─── Stories ──────────────────────────────────────────────────────────────────

export const LargeHopGraph: Story = {
    render: function Render() {
        const [open, setOpen] = useState(false);
        return (
            <>
                <div className="h-150 w-full overflow-y-auto rounded border">
                    <HopGraph
                        nodes={largeNodes}
                        edges={largeEdges}
                        onOpenFullscreen={() => setOpen(true)}
                    />
                </div>
                <Sheet open={open} onOpenChange={setOpen}>
                    <SheetContent
                        className="w-[99%] rounded"
                        style={{ height: "98vh" }}
                        side="top">
                        <SheetTitle className="sr-only">
                            Graph fullscreen view
                        </SheetTitle>
                        <div className="flex h-full w-full flex-col">
                            <HopGraph
                                nodes={largeNodes}
                                edges={largeEdges}
                                onCloseFullscreen={() => setOpen(false)}
                            />
                        </div>
                    </SheetContent>
                </Sheet>
            </>
        );
    }
};

export const Simple: Story = {
    render: function Render() {
        return (
            <div className="h-100 w-full overflow-y-auto rounded border">
                <HopGraph nodes={simpleNodes} edges={simpleEdges} />
            </div>
        );
    }
};

export const MultiRow: Story = {
    render: function Render() {
        return (
            <div className="h-125 w-full overflow-y-auto rounded border">
                <HopGraph nodes={multiRowNodes} edges={multiRowEdges} />
            </div>
        );
    }
};

export const WithFullscreen: Story = {
    render: function Render() {
        const [open, setOpen] = useState(false);
        return (
            <>
                <div className="h-100 w-full overflow-y-auto rounded border">
                    <HopGraph
                        nodes={multiRowNodes}
                        edges={multiRowEdges}
                        onOpenFullscreen={() => setOpen(true)}
                    />
                </div>
                <Sheet open={open} onOpenChange={setOpen}>
                    <SheetContent
                        className="w-[99%] rounded"
                        style={{ height: "98vh" }}
                        side="top">
                        <SheetTitle className="sr-only">
                            Graph fullscreen view
                        </SheetTitle>
                        <div className="flex h-full w-full flex-col">
                            <HopGraph
                                nodes={multiRowNodes}
                                edges={multiRowEdges}
                                onCloseFullscreen={() => setOpen(false)}
                            />
                        </div>
                    </SheetContent>
                </Sheet>
            </>
        );
    },
    play: async ({ canvasElement }) => {
        const canvas = within(canvasElement);
        await userEvent.click(
            canvas.getByRole("button", { name: /open full screen/i })
        );
        await expect(
            within(document.body).getByRole("button", { name: /close/i })
        ).toBeVisible();
    }
};

export const SingleNode: Story = {
    render: function Render() {
        return (
            <div className="h-75 w-full overflow-y-auto rounded border">
                <HopGraph
                    nodes={[{ id: "entity", table: "entity", hop: 0 }]}
                    edges={[]}
                />
            </div>
        );
    }
};
