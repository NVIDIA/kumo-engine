use arrow_array::RecordBatch;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use uuid::Uuid;

/// A handle to a feature stream channel receiver.
pub type FeatureReceiver = mpsc::Receiver<RecordBatch>;
/// A handle to a feature stream channel sender.
pub type FeatureSender = mpsc::Sender<RecordBatch>;

struct RegistryState {
    channels: HashMap<String, (FeatureReceiver, Instant)>,
    last_gc: Instant,
}

impl Default for RegistryState {
    fn default() -> Self {
        Self {
            channels: HashMap::new(),
            last_gc: Instant::now(),
        }
    }
}

/// Registry for managing side-channel feature tickets.
#[derive(Clone, Default)]
pub struct TicketRegistry {
    // Map of ticket_id -> (receiver, created_at)
    inner: Arc<RwLock<RegistryState>>,
}

// Configants
const TICKET_TTL: Duration = Duration::from_secs(60);
const GC_INTERVAL: Duration = Duration::from_secs(10);

impl TicketRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new receiver and return its ticket ID.
    /// Performs lazy garbage collection of expired tickets.
    pub fn register(&self, receiver: FeatureReceiver) -> String {
        let ticket = Uuid::new_v4().to_string();
        let now = Instant::now();
        
        let mut guard = self.inner.write();
        
        // Lazy GC
        if now.duration_since(guard.last_gc) > GC_INTERVAL {
            let initial_len = guard.channels.len();
            guard.channels.retain(|_, (_, created_at)| {
                now.duration_since(*created_at) < TICKET_TTL
            });
            guard.last_gc = now;
            let dropped = initial_len - guard.channels.len();
            if dropped > 0 {
                // Log or metric could go here
                // println!("GC: Dropped {} expired tickets", dropped);
            }
        }

        guard.channels.insert(ticket.clone(), (receiver, now));
        ticket
    }

    /// Retrieve and remove a receiver by its ticket ID.
    pub fn take(&self, ticket: &str) -> Option<FeatureReceiver> {
        let mut guard = self.inner.write();
        guard.channels.remove(ticket).map(|(rx, _)| rx)
    }
}
