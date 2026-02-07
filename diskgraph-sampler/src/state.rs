use parking_lot::Mutex;
use std::sync::Arc;
use gxhash::{HashMap as GxHashMap, HashMapExt};

/// Unique identifier for a node instance in the traversal.
/// If disjoint=false, seed_index is always 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LookupKey {
    pub seed_index: u32,
    pub global_id: u64,
}

/// Metadata recorded for every sampled node.
#[derive(Debug, Clone, Copy)]
pub struct NodeRecord {
    pub global_id: u64,
    pub seed_index: u32,
    pub local_id: u32,
}

/// An item in the active frontier for the next hop.
#[derive(Debug, Clone, Copy)]
pub struct FrontierItem {
    pub global_id: u64,
    pub seed_index: u32,
    pub local_id: u32,
    // We might add timestamp here if needed for temporal propagation
}

/// State specific to a single Node Type (table).
pub struct TypeState {
    /// Maps (Seed, Global) -> LocalID
    pub index_map: GxHashMap<LookupKey, u32>,

    /// Ordered list of nodes discovered (Index = LocalID)
    pub nodes: Vec<NodeRecord>,

    /// Nodes to be processed in the next hop for this type
    pub next_frontier: Vec<FrontierItem>,
}

impl TypeState {
    fn new() -> Self {
        Self {
            index_map: GxHashMap::with_capacity(1024),
            nodes: Vec::with_capacity(1024),
            next_frontier: Vec::with_capacity(1024),
        }
    }

    fn reset(&mut self) {
        self.index_map.clear();
        self.nodes.clear();
        self.next_frontier.clear();
    }
}

/// Metadata recorded for every sampled edge.
#[derive(Debug, Clone, Copy)]
pub struct EdgeRecord {
    pub src_local_id: u32,
    pub dst_local_id: u32,
    pub rel_id: u16,
}

/// The main context for a single sampling request.
/// Pooled to avoid allocation churn.
pub struct SamplingState {
    /// Registry of state per node type.
    pub type_states: GxHashMap<String, TypeState>,
    
    /// Sampled edges across all relations.
    pub edges: Vec<EdgeRecord>,
    
    // Scratch buffers for EdgeUnpacker (u64 fast path)
    pub dst_scratch: Vec<u64>,
    pub key_scratch: Vec<u64>,
}

impl SamplingState {
    fn new() -> Self {
        Self {
            type_states: GxHashMap::default(),
            edges: Vec::with_capacity(4096),
            dst_scratch: Vec::with_capacity(1024),
            key_scratch: Vec::with_capacity(1024),
        }
    }

    fn reset(&mut self) {
        // Recurse into type states
        for state in self.type_states.values_mut() {
            state.reset();
        }
        self.edges.clear();
        self.dst_scratch.clear();
        self.key_scratch.clear();
    }
    /// Access or create the state for a specific node type.
    pub fn get_type_state_mut(&mut self, node_type: &str) -> &mut TypeState {
        self.type_states
            .entry(node_type.to_string())
            .or_insert_with(TypeState::new)
    }
}

/// A thread-safe pool of SamplingState objects.
#[derive(Clone)]
pub struct StatePool {
    pool: Arc<Mutex<Vec<Box<SamplingState>>>>,
}

impl StatePool {
    pub fn new() -> Self {
        Self {
            pool: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn acquire(&self) -> PooledState {
        let mut inner = self.pool.lock();
        let state = if let Some(s) = inner.pop() {
            s
        } else {
            Box::new(SamplingState::new())
        };
        PooledState {
            state: Some(state),
            pool: self.clone(),
        }
    }
}

/// RAII Guard for a SamplingState. Returns the state to the pool on Drop.
pub struct PooledState {
    state: Option<Box<SamplingState>>,
    pool: StatePool,
}

impl std::ops::Deref for PooledState {
    type Target = SamplingState;
    fn deref(&self) -> &Self::Target {
        self.state.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for PooledState {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.state.as_mut().unwrap()
    }
}

impl Drop for PooledState {
    fn drop(&mut self) {
        if let Some(mut state) = self.state.take() {
            // Reset state before returning to pool
            state.reset();
            let mut inner = self.pool.pool.lock();
            inner.push(state);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_recycling() {
        let pool = StatePool::new();

        // 1. Acquire and Modify
        {
            let mut state = pool.acquire();
            let ts = state.get_type_state_mut("user");

            // Push some data to grow capacity
            ts.nodes.reserve(1000);
            ts.nodes.push(NodeRecord { global_id: 1, seed_index: 0, local_id: 0 });
            ts.index_map.insert(LookupKey { seed_index: 0, global_id: 1 }, 0);

            assert!(ts.nodes.capacity() >= 1000);
            assert_eq!(ts.nodes.len(), 1);
            assert_eq!(ts.index_map.len(), 1);
        } // state dropped here

        // 2. Re-Acquire
        {
            let mut state = pool.acquire();
            let ts = state.get_type_state_mut("user");

            // Should be logically empty
            assert_eq!(ts.nodes.len(), 0);
            assert_eq!(ts.index_map.len(), 0);

            // Should retain capacity (Zero Allocation check)
            assert!(ts.nodes.capacity() >= 1000);

            // "user" entry should still exist in type_states (map not cleared, values reset)
            assert!(state.type_states.contains_key("user"));
        }
    }

    #[test]
    fn test_pool_concurrency() {
        let pool = StatePool::new();
        let mut guards = Vec::new();

        // Acquire 10 states
        for _ in 0..10 {
            guards.push(pool.acquire());
        }

        // Drop them all
        drop(guards);

        // Acquire 1 (should reuse one of the 10)
        let _s = pool.acquire();
        let inner = pool.pool.lock();
        assert_eq!(inner.len(), 9); // 1 taken, 9 remaining
    }
}
