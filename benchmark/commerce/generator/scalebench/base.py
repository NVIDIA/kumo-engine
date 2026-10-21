"""Shared shard-level machinery for both graphs."""
import numpy as np
from scipy.special import ndtr

from scalebench import core, schema
from scalebench import profile as P

EVENT_STRIDE = 1 << 32  # event PK = shard * EVENT_STRIDE + counter within shard


class GraphGen:
    graph = None
    main = None
    families = {}  # family -> list of output tables (first one sizes the files)

    def __init__(self, prof, gcfg, seed):
        self.prof = prof
        self.seed = seed
        self.tables = prof["tables"]
        self.edges = prof["edges"]
        self.time = core.Timeline(prof["time"])
        self.shard_main = int(gcfg["shard_entities"])
        self.block = int(gcfg["block_entities"])
        self.scale = self.shard_main / float(self.tables[self.main]["rows"])
        m = prof.get("marketplace", {})
        self.cross_frac = float(m.get("cross_shard_frac", 0.0))
        self.block_shards = int(gcfg.get("block_shards", 1))

    def rng(self, *key):
        return core.rng(self.seed, self.graph, *key)

    balance = {}  # table -> (edge whose mean is per main entity, edge whose mean is per table row)

    def n(self, table):
        """Fixed per-shard row count of a dimension table (same in every shard).

        Tables in `balance` are sized so that out- and in-degree means agree:
        rows = main rows * mean(per-main degree) / mean(per-row degree).
        """
        if table == self.main:
            return self.shard_main
        if table in self.balance:
            out_edge, in_edge = self.balance[table]
            out_side = "out_degree" if "~" in out_edge else "in_degree"
            ratio = P.degree_mean(self.edges[out_edge][out_side]) / P.degree_mean(self.edges[in_edge]["in_degree"])
            return max(1, int(round(self.shard_main * ratio)))
        return max(1, int(round(self.tables[table]["rows"] * self.scale)))

    def degree(self, edge, side="in_degree"):
        """Degree sampler. Degrees of the main entity are per-node properties, so its
        top anchor is the global max; in-degrees of other parents (hubs) keep their
        share of a market (profile.degree_sampler)."""
        if "~" in edge:
            parent = self.main if side == "out_degree" else None
        else:
            parent = edge.split("->")[1]
        if parent is None:
            return P.degree_sampler(self.edges[edge][side])
        if parent == self.main:
            return P.degree_sampler(self.edges[edge][side], nodes=self.shard_main)
        return P.degree_sampler(self.edges[edge][side], self.tables[parent]["rows"], self.n(parent))

    def columns(self, table, shard):
        return core.Columns(self.tables[table], shard, self.scale,
                            schema.nullable(self.graph, table), schema.BOUNDS[self.graph].get(table, {}),
                            schema.int_columns(self.graph, table))

    def target_shards(self, r, shard, n):
        """Markets trade within fixed blocks of `block_shards` consecutive shards: a
        fraction cross_shard_frac of picks goes to one of the other shards of the block,
        uniformly. Every shard sends and receives the same share in every rung, and a
        rung is a whole number of blocks, so FKs stay inside the rung."""
        m = self.block_shards
        if m <= 1:
            return np.full(n, shard)
        first = shard - shard % m
        other = first + (shard % m + 1 + r.integers(0, m - 1, n)) % m
        return np.where(r.random(n) < self.cross_frac, other, shard)

    def tenure_factor_sample(self, table, latest):
        """(tenure^g / mean, mean tenure^g) for a fixed sample of creation times, with
        g = marketplace.tenure_exponent (default 1: events per day independent of tenure)."""
        g = float(self.prof.get("marketplace", {}).get("tenure_exponent", 1.0))
        t = self.time.created(self.rng("tenure-sample", table), 256, self.prof["time"]["created"][table], latest)
        ten = (self.time.end - t).astype(float) ** g
        return np.sort(ten / ten.mean()), ten.mean()

    def tenure_degree(self, edge, table, latest):
        """(sampler, mean tenure^g): counts = sampler(u) * tenure^g / mean tenure^g."""
        f, mean = self.tenure_factor_sample(table, latest)
        return P.TenureScaled(self.degree(edge), f), mean

    def tenure_scale(self, created, mean):
        g = float(self.prof.get("marketplace", {}).get("tenure_exponent", 1.0))
        return (self.time.end - created).astype(float) ** g / mean

    def blocks(self):
        return [(b, min(b + self.block, self.shard_main))
                for b in range(0, self.shard_main, self.block)]


def correlated(r, z, rho):
    """Uniform marginal whose normal score has correlation rho with z (Gaussian copula)."""
    rho = np.asarray(rho, float)
    return ndtr(rho * z + np.sqrt(1.0 - rho * rho) * r.standard_normal(len(z)))


def owners(counts):
    return np.repeat(np.arange(len(counts)), counts)
