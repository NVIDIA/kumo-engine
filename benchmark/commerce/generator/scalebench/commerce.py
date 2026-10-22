# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Graph 1: commerce marketplace (link prediction).

Hidden state (never written to the tables, only to _ground_truth):
- each customer has a taste cluster (category, price tier, tags), may drift to a
  second cluster mid-window, and may be a "browser" (many views, few purchases);
- each merchant has a cluster and may be a cold-start merchant created in the
  last weeks, which can be bought only in the last `cold_purchase_days` and
  only through a recent view (the multi-hop path).

Purchases follow a funnel: a purchase repeats one of the customer's earlier
merchants (calibrated to the profile's is_first_purchase rate), or buys a
merchant viewed in the previous `funnel_days`, or picks by taste. A shard is a
market with a home region; markets come in blocks that trade with each other
(base.GraphGen.target_shards).
"""
from functools import lru_cache

import numpy as np
import pyarrow as pa
from scalebench import core
from scalebench import profile as P
from scalebench.base import EVENT_STRIDE, GraphGen, correlated, owners
from scipy.special import ndtr, ndtri

CUSTOMER_EDGES = {"purchases": "purchases.customer_id->customers",
                  "merchant_views": "merchant_views.customer_id->customers",
                  "searches": "searches.customer_id->customers",
                  "favorites": "favorites.customer_id->customers"}
MERCHANT_EDGES = {"purchases": "purchases.merchant_id->merchants",
                  "merchant_views": "merchant_views.merchant_id->merchants"}
SEARCH_TERMS_PER_CLUSTER = 40


class Commerce(GraphGen):
    graph = "commerce"
    main = "customers"
    families = {"customers": ["customers", "_truth_customers"],
                "merchants": ["merchants", "_truth_merchants"],
                "activity": ["merchant_views", "purchases", "favorites", "reviews"],
                "searches": ["searches"]}
    balance = {"merchants": ("purchases.customer_id->customers", "purchases.merchant_id->merchants")}

    def __init__(self, prof, gcfg, seed):
        super().__init__(prof, gcfg, seed)
        mk = self.mk = prof["marketplace"]
        self.K = int(mk["n_clusters"])
        # reviews per purchase, from the per-customer means (keeps both edges consistent)
        self.review_rate = min(1.0, P.degree_mean(self.edges["reviews.customer_id->customers"]["in_degree"])
                               / P.degree_mean(self.edges[CUSTOMER_EDGES["purchases"]]["in_degree"]))
        self.signup_latest = self.time.end - 7 * core.DAY
        self.cust_deg = {e: self.tenure_degree(edge, "customers", self.signup_latest)
                         for e, edge in CUSTOMER_EDGES.items()}
        self.merch_deg = {e: self.degree(edge) for e, edge in MERCHANT_EDGES.items()}
        b = float(mk["browser_frac"])
        bp, bv = float(mk["browser_purchase_mult"]), float(mk["browser_view_mult"])
        # browsers buy less and view more; the others compensate so the means hold
        self.mult = {"purchases": (bp, (1 - b * bp) / (1 - b)),
                     "merchant_views": (bv, max(1 - b * bv, 0.05) / (1 - b))}
        # share of purchases that repeat an earlier merchant of the same customer
        repeat = mk.get("repeat_rate_orders")
        if isinstance(repeat, dict):
            repeat = repeat.get("mean")
        pdeg = self.edges[CUSTOMER_EDGES["purchases"]]["in_degree"]
        first_share = (1 - float(pdeg.get("frac_zero", 0))) / max(P.degree_mean(pdeg), 1.0)
        self.p_repeat = float(np.clip(float(repeat) / max(1 - first_share, 1e-6), 0.0, 0.95))

    def expected_rows_per_shard(self):
        c = self.shard_main
        ev = {e: c * sampler.mean() for e, (sampler, _) in self.cust_deg.items()}
        ev["reviews"] = ev["purchases"] * self.review_rate
        return {"customers": c, "merchants": self.n("merchants"), **ev}

    # ---- latent state ---------------------------------------------------------

    @lru_cache(maxsize=1)
    def clusters(self):
        r = self.rng("clusters")
        w = (np.arange(self.K) + 1.0) ** -float(self.mk["cluster_zipf_s"])
        w = r.permutation(w / w.sum())
        mc = self.tables["merchants"]["columns"]
        q_card = int(self.tables["searches"]["columns"]["query_term"]["categorical"]["cardinality"])
        return {
            "cdf": np.cumsum(w),
            "category": P.zipf_rank(r.random(self.K), mc["primary_category"]["categorical"]["cardinality"],
                                    mc["primary_category"]["categorical"]["zipf_s"]),
            "tier": r.integers(0, mc["price_tier"]["categorical"]["cardinality"], self.K),
            "tags": r.integers(0, mc["tags"]["multicat"]["token_cardinality"], (self.K, 3)),
            # each cluster favours a random subset of search terms (overlapping across clusters)
            "terms": np.stack([r.choice(q_card, min(SEARCH_TERMS_PER_CLUSTER, q_card), replace=False)
                               for _ in range(self.K)]),
        }

    def draw_cluster(self, r, n):
        return np.minimum(np.searchsorted(self.clusters()["cdf"], r.random(n), side="right"), self.K - 1)

    def home_region(self, cols, r, shard, n, p_home):
        """Region column: the market's region with probability p_home, else any region.

        Markets take regions from a low-discrepancy (golden-ratio) sequence, so any
        prefix of shards, i.e. any rung, has close to the profiled region mix."""
        spec = self.tables["customers"]["columns"]["region"]["categorical"]
        u = np.array([((shard + 1) * 0.6180339887498949) % 1.0])
        home = P.zipf_rank(u, cols.cardinality("region"), float(spec.get("zipf_s") or 1.0))[0]
        return np.where(r.random(n) < p_home, home, cols.cat_codes(r, "region", n))

    @lru_cache(maxsize=16)
    def merchants(self, shard):
        r = self.rng("merchants", shard)
        n = self.n("merchants")
        cl = self.draw_cluster(r, n)
        cold = r.random(n) < float(self.mk["cold_start_frac"])
        cold_s = int(float(self.mk["cold_start_days"]) * core.DAY)
        created = self.time.created(r, n, self.prof["time"]["created"]["merchants"], self.time.end - cold_s)
        newest = self.time.end - core.DAY - (r.random(n) * (cold_s - core.DAY)).astype(np.int64)
        created = np.where(cold, newest, created)
        z = ndtri(r.random(n).clip(1e-12, 1 - 1e-12))
        order = np.argsort(cl, kind="stable")
        bounds = np.searchsorted(cl[order], np.arange(self.K + 1))
        cum = {}
        for e, d in self.merch_deg.items():
            w = d(correlated(r, z, 0.9))
            if e == "purchases":
                w = np.where(cold, 0.0, w)  # cold-start merchants are bought only via a recent view
            cum[e] = np.concatenate([[0.0], np.cumsum(w[order])])
        return {"n": n, "cluster": cl, "cold": cold, "created": created, "order": order,
                "bounds": bounds, "cum": cum}

    def pick_merchant(self, r, tshard, cluster, in_cluster, edge):
        """Chung-Lu pick: P(merchant) proportional to its target in-degree, within the
        customer's current cluster with probability p_in_cluster."""
        n = len(tshard)
        gid = np.empty(n, np.int64)
        created = np.empty(n, np.int64)
        u = r.random(n)
        for s in np.unique(tshard):
            sel = np.nonzero(tshard == s)[0]
            m = self.merchants(int(s))
            cum, b = m["cum"][edge], m["bounds"]
            k = cluster[sel]
            a, z = cum[b[k]], cum[b[k + 1]]
            local = in_cluster[sel] & (z > a)
            x = np.where(local, a + u[sel] * (z - a), u[sel] * cum[-1])
            pos = np.clip(np.searchsorted(cum, x, side="right") - 1, 0, m["n"] - 1)
            idx = m["order"][pos]
            gid[sel] = s * m["n"] + idx
            created[sel] = m["created"][idx]
        return gid, created

    def merchant_info(self, gid):
        """(created, cold) for global merchant ids."""
        n = self.n("merchants")
        created = np.empty(len(gid), np.int64)
        cold = np.empty(len(gid), bool)
        shard = gid // n
        for s in np.unique(shard):
            sel = shard == s
            m = self.merchants(int(s))
            created[sel], cold[sel] = m["created"][gid[sel] % n], m["cold"][gid[sel] % n]
        return created, cold

    @lru_cache(maxsize=4)
    def customers(self, shard):
        r = self.rng("customers", shard)
        n = self.shard_main
        mk = self.mk
        k1 = self.draw_cluster(r, n)
        drift = r.random(n) < float(mk["drift_frac"])
        k2 = np.where(drift, self.draw_cluster(r, n), -1)
        signup = self.time.created(r, n, self.prof["time"]["created"]["customers"], self.signup_latest)
        t_drift = signup + (r.random(n) * (self.time.end - signup)).astype(np.int64)
        browser = r.random(n) < float(mk["browser_frac"])
        z_view = ndtri(r.random(n).clip(1e-12, 1 - 1e-12))
        u = {"merchant_views": ndtr(z_view),
             "purchases": correlated(r, z_view, float(mk["rho"])),
             "searches": correlated(r, z_view, 0.5),
             "favorites": correlated(r, z_view, 0.9)}  # favorites come from views: heavy viewers favorite
        counts = {}
        for e, (sampler, mean_tenure) in self.cust_deg.items():
            scale = self.tenure_scale(signup, mean_tenure)  # events per day independent of signup time
            if e in self.mult:
                scale = scale * np.where(browser, *self.mult[e])
            counts[e] = P.stochastic_round(sampler(u[e]) * scale, r.random(n))
        return {"n": n, "k1": k1, "k2": k2, "t_drift": t_drift, "browser": browser,
                "signup": signup, "counts": counts}

    # ---- dimension tables -------------------------------------------------------

    def gen_customers(self, shard):
        c = self.customers(shard)
        n = c["n"]
        r = self.rng("customers.cols", shard)
        cols = self.columns("customers", shard)
        ids = shard * n + np.arange(n, dtype=np.int64)
        region = self.home_region(cols, r, shard, n, float(self.mk["p_home_region"]))
        table = pa.table({
            "customer_id": ids, "signup_time": core.ts(c["signup"]),
            "region": cols.categorical(r, "region", n, codes=region),
            "city": cols.categorical(r, "city", n),
            "membership": cols.categorical(r, "membership", n),
            "platform": cols.categorical(r, "platform", n),
            "marketing_opt_in": cols.boolean(r, "marketing_opt_in", n)})
        truth = pa.table({"customer_id": ids, "cluster": c["k1"], "drift_cluster": c["k2"],
                          "drift_time": core.ts(c["t_drift"]), "browser": c["browser"]})
        return {"customers": table, "_truth_customers": truth}

    def gen_merchants(self, shard):
        m = self.merchants(shard)
        n, cl = m["n"], m["cluster"]
        r = self.rng("merchants.cols", shard)
        cols = self.columns("merchants", shard)
        cs = self.clusters()
        mk = self.mk

        def reveal(name, key, p):
            return np.where(r.random(n) < float(mk[p]), cs[key][cl], cols.cat_codes(r, name, n))

        def tag_override(rows, codes):
            hit = r.random(len(rows)) < float(mk["tag_reveal"])
            return np.where(hit, cs["tags"][cl[rows], r.integers(0, 3, len(rows))], codes)

        ids = shard * n + np.arange(n, dtype=np.int64)
        table = pa.table({
            "merchant_id": ids, "created_time": core.ts(m["created"]),
            "region": cols.categorical(r, "region", n,
                                       codes=self.home_region(cols, r, shard, n, float(mk["p_home_region"]))),
            "primary_category": cols.categorical(r, "primary_category", n,
                                                 codes=reveal("primary_category", "category", "category_reveal")),
            "tags": cols.multicat(r, "tags", n, override=tag_override),
            "price_tier": cols.categorical(r, "price_tier", n, codes=reveal("price_tier", "tier", "tier_reveal")),
            "is_partner": cols.boolean(r, "is_partner", n),
            "avg_rating": cols.numeric(r, "avg_rating", n)})
        truth = pa.table({"merchant_id": ids, "cluster": cl, "cold_start": m["cold"]})
        return {"merchants": table, "_truth_merchants": truth}

    # ---- events -----------------------------------------------------------------

    def _timeline(self, r, shard, b0, b1, table):
        """Owner, time and current taste cluster for one block of customers."""
        c = self.customers(shard)
        owner = owners(c["counts"][table][b0:b1]) + b0
        n = len(owner)
        t = self.time.bursty(r, owner, c["signup"][owner], np.full(n, self.time.last),
                             self.prof["time"]["events"][table])
        drifted = (c["k2"][owner] >= 0) & (t > c["t_drift"][owner]) & (r.random(n) < float(self.mk["p_drift"]))
        cluster = np.where(drifted, c["k2"][owner], c["k1"][owner])
        return owner, t, cluster

    def _taste_pick(self, r, shard, table, owner, t, cluster):
        """Merchants by taste that exist at the event time (8 redraws, then the time
        moves after the merchant's creation)."""
        n = len(t)
        signup = self.customers(shard)["signup"][owner]
        in_cl = r.random(n) < float(self.mk["p_in_cluster"][table])
        gid = np.empty(n, np.int64)
        lo = np.empty(n, np.int64)
        todo = np.arange(n)
        for _ in range(8):
            g, created = self.pick_merchant(r, self.target_shards(r, shard, len(todo)),
                                            cluster[todo], in_cl[todo], table)
            gid[todo], lo[todo] = g, np.maximum(signup[todo], created)
            todo = todo[t[todo] < lo[todo]]
            if len(todo) == 0:
                break
        if len(todo):
            t[todo] = self.time.seasonal(r, lo[todo], np.full(len(todo), self.time.last))
        return gid

    def _views(self, r, shard, b0, b1):
        owner, t, cluster = self._timeline(r, shard, b0, b1, "merchant_views")
        merchant = self._taste_pick(r, shard, "merchant_views", owner, t, cluster)
        o = np.lexsort((t, owner))
        return owner[o], t[o], merchant[o]

    def _purchases(self, r, shard, b0, b1, views):
        """Purchases sorted by (customer, time): repeat an earlier merchant, else buy a
        merchant viewed in the last funnel_days, else pick by taste."""
        owner, t, cluster = self._timeline(r, shard, b0, b1, "purchases")
        merchant = self._taste_pick(r, shard, "purchases", owner, t, cluster)
        o = np.lexsort((t, owner))
        owner, t, cluster, merchant = owner[o], t[o], cluster[o], merchant[o]
        n = len(t)
        rank = np.arange(n) - np.searchsorted(owner, owner, side="left")
        u = r.random(n)
        repeat = (rank > 0) & (u < self.p_repeat)
        funnel = ~repeat & (u < self.p_repeat + (1 - self.p_repeat) * float(self.mk["p_view_funnel"]))
        # funnel: a view by the same customer in [t - funnel_days, t)
        v_owner, v_t, v_m = views
        key = (v_owner.astype(np.int64) << 32) | (v_t - self.time.start)
        hi = np.searchsorted(key, (owner.astype(np.int64) << 32) | (t - self.time.start), side="left")
        lo = np.searchsorted(key, (owner.astype(np.int64) << 32)
                             | np.maximum(t - int(float(self.mk["funnel_days"]) * core.DAY) - self.time.start, 0))
        has = funnel & (hi > lo)
        pick = lo + (r.random(n) * (hi - lo)).astype(np.int64)
        cand = v_m[np.minimum(pick, len(v_m) - 1)] if len(v_m) else merchant
        _, cold = self.merchant_info(cand)
        late = self.time.end - int(float(self.mk["cold_purchase_days"]) * core.DAY)
        ok = has & (~cold | (t >= late))  # cold-start merchants only in the last cold_purchase_days
        merchant = np.where(ok, cand, merchant)
        # repeat: copy the merchant of a uniformly chosen earlier purchase (pointer jumping
        # resolves repeats of repeats)
        ptr = np.arange(n)
        ptr[repeat] = (np.arange(n) - 1 - (r.random(n) * np.maximum(rank, 1)).astype(np.int64))[repeat]
        for _ in range(64):
            nxt = ptr[ptr]
            if np.array_equal(nxt, ptr):
                break
            ptr = nxt
        merchant = merchant[ptr]
        return owner, t, cluster, merchant

    def gen_activity(self, shard):
        cols = self.columns("purchases", shard)
        vcols = self.columns("merchant_views", shard)
        rcols = self.columns("reviews", shard)
        c = self.customers(shard)
        base = shard * self.shard_main
        tier_card = self.tables["merchants"]["columns"]["price_tier"]["categorical"]["cardinality"]
        effect = float(self.mk["tier_spend_effect"])
        lag_q = P.numeric_sampler(self.prof["time"]["events"]["favorites"]["burst_gap_seconds"])
        review_lag = P.numeric_sampler(self.prof["time"]["review_lag_seconds"])
        parts, counter = [], 0
        for bi, (b0, b1) in enumerate(self.blocks()):
            r = self.rng("activity", shard, bi)
            v_owner, v_t, v_m = self._views(r, shard, b0, b1)
            views = pa.table({"view_time": core.ts(v_t), "customer_id": base + v_owner, "merchant_id": v_m,
                              "source": vcols.categorical(r, "source", len(v_t))})
            fav = self._favorites(r, (v_owner, v_t, v_m), c["counts"]["favorites"], lag_q, base)
            owner, t, cluster, merchant = self._purchases(r, shard, b0, b1, (v_owner, v_t, v_m))
            n = len(t)
            cid = base + owner
            first = np.ones(n, bool)
            o = np.lexsort((np.arange(n), merchant, cid))  # rows are in time order already
            first[o[1:]] = ~((cid[o][1:] == cid[o][:-1]) & (merchant[o][1:] == merchant[o][:-1]))
            tier = self.clusters()["tier"][cluster] / max(tier_card - 1, 1)
            spend_u = r.random(n) ** np.exp(-effect * (tier - 0.5) * 2)
            purchases = pa.table({
                "purchase_id": shard * EVENT_STRIDE + counter + np.arange(n, dtype=np.int64),
                "purchase_time": core.ts(t), "customer_id": cid, "merchant_id": merchant,
                "channel": cols.categorical(r, "channel", n),
                "fulfillment": cols.categorical(r, "fulfillment", n),
                "item_count": cols.numeric(r, "item_count", n),
                "subtotal": cols.numeric(r, "subtotal", n, u=spend_u),
                "fees": cols.numeric(r, "fees", n),
                "tip": cols.numeric(r, "tip", n),
                "is_first_purchase": pa.array(first),
                "rating": cols.numeric(r, "rating", n)})  # purchase-time rating, independent of reviews
            counter += n
            sel = np.nonzero(r.random(n) < self.review_rate)[0]
            pt = t[sel]
            rt = pt + review_lag(r.random(len(sel))).astype(np.int64)
            over = rt > self.time.last  # lag past the window end: a uniform time before it instead
            rt[over] = pt[over] + (r.random(over.sum()) * (self.time.last - pt[over])).astype(np.int64)
            reviews = pa.table({"review_time": core.ts(rt), "customer_id": cid[sel], "merchant_id": merchant[sel],
                                "stars": rcols.numeric(r, "stars", len(sel))})
            parts.append({"merchant_views": views, "purchases": purchases, "favorites": fav, "reviews": reviews})
        return {k: pa.concat_tables([p[k] for p in parts]) for k in parts[0]}

    def _favorites(self, r, views, target, lag_q, id_base):
        """Each customer favorites up to its target count of distinct merchants it has
        viewed, some time after the first view of that merchant."""
        v_owner, v_t, v_m = views
        o = np.lexsort((v_t, v_m, v_owner))
        owner, merch, t = v_owner[o], v_m[o], v_t[o]
        first = np.ones(len(o), bool)
        first[1:] = (owner[1:] != owner[:-1]) | (merch[1:] != merch[:-1])
        owner, merch, t = owner[first], merch[first], t[first]
        o = np.lexsort((r.random(len(owner)), owner))
        owner, merch, t = owner[o], merch[o], t[o]
        keep = np.arange(len(owner)) - np.searchsorted(owner, owner, side="left") < target[owner]
        owner, merch, t = owner[keep], merch[keep], t[keep]
        ft = t + lag_q(r.random(len(t))).astype(np.int64)
        over = ft > self.time.last
        ft[over] = t[over] + (r.random(over.sum()) * (self.time.last - t[over])).astype(np.int64)
        o = np.lexsort((ft, owner))
        return pa.table({"created_time": core.ts(ft[o]), "customer_id": id_base + owner[o],
                         "merchant_id": merch[o]})

    def gen_searches(self, shard):
        cols = self.columns("searches", shard)
        cs = self.clusters()
        p_in = float(self.mk["p_in_cluster"]["searches"])
        p_cat = float(self.mk["search_category_reveal"])
        s = float(self.tables["searches"]["columns"]["query_term"]["categorical"]["zipf_s"])
        parts = []
        for bi, (b0, b1) in enumerate(self.blocks()):
            r = self.rng("searches", shard, bi)
            owner, t, cluster = self._timeline(r, shard, b0, b1, "searches")
            o = np.lexsort((t, owner))
            owner, t, cluster = owner[o], t[o], cluster[o]
            n = len(t)
            own = cs["terms"][cluster, P.zipf_rank(r.random(n), cs["terms"].shape[1], s)]
            q = np.where(r.random(n) < p_in, own, cols.cat_codes(r, "query_term", n))
            cat = np.where(r.random(n) < p_cat, cs["category"][cluster], cols.cat_codes(r, "category", n))
            parts.append(pa.table({
                "search_time": core.ts(t), "customer_id": shard * self.shard_main + owner,
                "query_term": cols.categorical(r, "query_term", n, codes=q),
                "category": cols.categorical(r, "category", n, codes=cat),
                "result_page": cols.numeric(r, "result_page", n)}))
        return {"searches": pa.concat_tables(parts)}
