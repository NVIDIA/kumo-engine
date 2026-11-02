// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native replica ownership and bounded round-robin epoch prefetch.
use crate::{SamplingClient, SamplingStream, TaskRequest};
use anyhow::{anyhow, ensure};
use diskgraph_runtime::ArrowBatch;
use parking_lot::Mutex;
use std::{
    collections::{HashSet, VecDeque},
    path::Path,
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc, Weak,
    },
    thread::JoinHandle,
};
use tonic::Status;

type ReplicaResult = anyhow::Result<Option<ArrowBatch>>;
type Pending = VecDeque<(usize, mpsc::Receiver<ReplicaResult>)>;

/// One epoch, with exactly one demanded minibatch per replica at a time.
pub struct RemoteBatchIterator {
    streams: Vec<Arc<SamplingStream>>,
    commands: Vec<mpsc::SyncSender<()>>,
    pending: Mutex<Pending>,
    threads: Mutex<Vec<JoinHandle<()>>>,
    stopped: Arc<AtomicBool>,
    pub num_samples: u64,
}
impl RemoteBatchIterator {
    fn new(streams: Vec<Arc<SamplingStream>>) -> anyhow::Result<Arc<Self>> {
        let num_samples = streams
            .iter()
            .try_fold(0u64, |total, stream| total.checked_add(stream.num_samples))
            .ok_or_else(|| anyhow!("Remote sample count overflow"))?;
        let mut commands = Vec::with_capacity(streams.len());
        let mut pending = VecDeque::with_capacity(streams.len());
        let mut receivers = Vec::with_capacity(streams.len());
        for index in 0..streams.len() {
            let (command, receiver) = mpsc::sync_channel(1);
            let (output, result) = mpsc::sync_channel(1);
            commands.push(command);
            pending.push_back((index, result));
            receivers.push((receiver, output));
        }
        let iterator = Arc::new(Self {
            streams,
            commands,
            pending: Mutex::new(pending),
            threads: Mutex::new(Vec::new()),
            stopped: Arc::new(AtomicBool::new(false)),
            num_samples,
        });
        for (index, (receiver, output)) in receivers.into_iter().enumerate() {
            let stream = iterator.streams[index].clone();
            let stopped = iterator.stopped.clone();
            let thread = std::thread::Builder::new()
                .name(format!("sampling-replica-{index}"))
                .spawn(move || {
                    while receiver.recv().is_ok() && !stopped.load(Ordering::Acquire) {
                        let result = stream.next_batch();
                        let complete = !matches!(&result, Ok(Some(_)));
                        if output.send(result).is_err() || complete {
                            break;
                        }
                    }
                })?;
            iterator.threads.lock().push(thread);
            iterator.commands[index].send(())?;
        }
        Ok(iterator)
    }

    /// Return the next replica's complete batch, preserving deterministic order.
    pub fn next_batch(&self) -> anyhow::Result<Option<ArrowBatch>> {
        if self.stopped.load(Ordering::Acquire) {
            return Ok(None);
        }
        let result = (|| {
            let mut pending = self.pending.lock();
            while let Some((index, receiver)) = pending.pop_front() {
                let result = receiver.recv().map_err(|_| {
                    if self.stopped.load(Ordering::Acquire) {
                        anyhow!(Status::cancelled("Remote epoch closed"))
                    } else {
                        anyhow!("Sampling replica reader stopped unexpectedly")
                    }
                })??;
                if self.stopped.load(Ordering::Acquire) {
                    return Err(Status::cancelled("Remote epoch closed").into());
                }
                if let Some(batch) = result {
                    self.commands[index]
                        .send(())
                        .map_err(|_| anyhow!("Sampling replica reader stopped unexpectedly"))?;
                    pending.push_back((index, receiver));
                    return Ok(Some(batch));
                }
            }
            Ok(None)
        })();
        if !matches!(&result, Ok(Some(_))) {
            self.close()?;
        }
        result
    }
    fn cancel(&self) {
        self.stopped.store(true, Ordering::Release);
        for stream in &self.streams {
            stream.close();
        }
        for command in &self.commands {
            let _ = command.try_send(());
        }
        if let Some(mut pending) = self.pending.try_lock() {
            pending.clear();
        }
    }
    /// Cancel pending reads and join all native prefetch threads; safe to repeat.
    pub fn close(&self) -> anyhow::Result<()> {
        self.cancel();
        let mut failed = false;
        for thread in self.threads.lock().drain(..) {
            failed |= thread.join().is_err();
        }
        self.pending.lock().clear();
        ensure!(!failed, "A sampling replica reader panicked");
        Ok(())
    }
}
impl Drop for RemoteBatchIterator {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[derive(Default)]
struct State {
    clients: Vec<Arc<SamplingClient>>,
    epochs: Vec<Weak<RemoteBatchIterator>>,
}

/// Cached native replica connections shared across stages and epoch iterators.
pub struct RemoteRuntimeIndex {
    endpoints: Vec<String>,
    state: Mutex<State>,
    closed: AtomicBool,
}
impl RemoteRuntimeIndex {
    /// Validate replica identities without opening a graph, socket or thread.
    pub fn new(endpoints: Vec<String>) -> anyhow::Result<Self> {
        ensure!(
            !endpoints.is_empty() && endpoints.iter().all(|value| !value.is_empty()),
            "At least one nonempty sampling endpoint is required"
        );
        ensure!(
            endpoints.iter().collect::<HashSet<_>>().len() == endpoints.len(),
            "Sampling endpoints must be unique"
        );
        Ok(Self {
            endpoints,
            state: Mutex::new(State::default()),
            closed: AtomicBool::new(false),
        })
    }
    fn clients(&self, count: usize) -> anyhow::Result<Vec<Arc<SamplingClient>>> {
        let mut state = self.state.lock();
        ensure!(
            !self.closed.load(Ordering::Acquire),
            Status::cancelled("Remote runtime closed")
        );
        while state.clients.len() < count {
            let endpoint = &self.endpoints[state.clients.len()];
            state.clients.push(Arc::new(SamplingClient::new(endpoint)?));
        }
        Ok(state.clients[..count].to_vec())
    }
    /// Partition one native epoch across replicas and prefetch complete batches.
    pub fn iter(
        &self,
        request: &TaskRequest,
        graph_root: &Path,
    ) -> anyhow::Result<Arc<RemoteBatchIterator>> {
        let mut request = request.clone();
        relativize_weights(&mut request, graph_root)?;
        let replicas = if request.options.rhs_seed_node_type.is_some() {
            1
        } else {
            self.endpoints.len()
        };
        let clients = self.clients(replicas)?;
        let workers = request.options.num_workers;
        let worker = request.options.worker_id;
        request.options.num_workers = workers
            .checked_mul(replicas as u64)
            .ok_or_else(|| anyhow!("Sampling shard count overflow"))?;
        let mut streams = Vec::with_capacity(replicas);
        for (replica, client) in clients.iter().take(replicas).enumerate() {
            request.options.worker_id = worker
                .checked_add(
                    workers
                        .checked_mul(replica as u64)
                        .ok_or_else(|| anyhow!("Sampling worker ID overflow"))?,
                )
                .ok_or_else(|| anyhow!("Sampling worker ID overflow"))?;
            streams.push(Arc::new(client.open(&request)?));
        }
        let epoch = RemoteBatchIterator::new(streams)?;
        let mut state = self.state.lock();
        if self.closed.load(Ordering::Acquire) {
            epoch.close()?;
            return Err(Status::cancelled("Remote runtime closed").into());
        }
        state.epochs.retain(|epoch| {
            epoch
                .upgrade()
                .is_some_and(|epoch| !epoch.stopped.load(Ordering::Acquire))
        });
        state.epochs.push(Arc::downgrade(&epoch));
        Ok(epoch)
    }
    fn cancel(&self) -> Vec<Arc<RemoteBatchIterator>> {
        self.closed.store(true, Ordering::Release);
        let mut state = self.state.lock();
        for client in state.clients.drain(..) {
            client.close();
        }
        let epochs = state
            .epochs
            .drain(..)
            .filter_map(|epoch| epoch.upgrade())
            .collect::<Vec<_>>();
        for epoch in &epochs {
            epoch.cancel();
        }
        epochs
    }
    /// Cancel every active epoch and release all replica connections.
    pub fn close(&self) -> anyhow::Result<()> {
        let mut error = None;
        for epoch in self.cancel() {
            if let Err(failure) = epoch.close() {
                error.get_or_insert(failure);
            }
        }
        if let Some(error) = error {
            return Err(error);
        }
        Ok(())
    }
}
impl Drop for RemoteRuntimeIndex {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn relativize_weights(request: &mut TaskRequest, graph_root: &Path) -> anyhow::Result<()> {
    if let Some(weighted) = &mut request.options.weighted_seed_sampling_json {
        let mut value: serde_json::Value = serde_json::from_str(weighted)?;
        let root = graph_root.join(&request.extension_ref);
        let buckets = value["buckets"]
            .as_array_mut()
            .ok_or_else(|| anyhow!("Weighted sampling requires buckets"))?;
        for bucket in buckets {
            for key in ["node_ids_path", "cdf_path"] {
                if let Some(path) = bucket[key].as_str() {
                    let relative = Path::new(path).strip_prefix(&root)?;
                    bucket[key] =
                        serde_json::Value::String(relative.to_string_lossy().into_owned());
                }
            }
        }
        *weighted = serde_json::to_string(&value)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::tests::{batch, frames, layout, reader};
    use futures::{stream, StreamExt};
    use std::{
        sync::atomic::AtomicUsize,
        time::{Duration, Instant},
    };

    fn epoch_stream(id: u64, count: u64) -> Arc<SamplingStream> {
        let mut wire = Vec::new();
        for number in 0..count {
            let mut layout = layout();
            layout["task"]["replica"] = id.into();
            wire.extend(frames(number, 0, 1, &batch(), Some(layout)));
        }
        Arc::new(reader(stream::iter(wire), count))
    }
    fn wait_for_count(count: &AtomicUsize, expected: usize) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while count.load(Ordering::Acquire) < expected {
            assert!(
                Instant::now() < deadline,
                "Native prefetch did not progress"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }
    #[test]
    fn replicas_merge_in_round_robin_order_with_uneven_and_empty_shards() {
        let iterator = RemoteBatchIterator::new(vec![
            epoch_stream(0, 2),
            epoch_stream(1, 1),
            epoch_stream(2, 0),
        ])
        .unwrap();
        assert_eq!(iterator.num_samples, 6);
        let mut replicas = Vec::new();
        while let Some(batch) = iterator.next_batch().unwrap() {
            replicas.push(batch.layout["task"]["replica"].as_u64().unwrap());
        }
        assert_eq!(replicas, vec![0, 1, 0]);
        iterator.close().unwrap();
    }
    #[test]
    fn prefetch_reads_only_one_batch_until_the_consumer_requests_more() {
        let count = Arc::new(AtomicUsize::new(0));
        let observed = count.clone();
        let wire = (0..3)
            .flat_map(|id| frames(id, 0, 1, &batch(), Some(layout())))
            .collect::<Vec<_>>();
        let stream = stream::iter(wire).inspect(move |_| {
            observed.fetch_add(1, Ordering::Release);
        });
        let iterator = RemoteBatchIterator::new(vec![Arc::new(reader(stream, 3))]).unwrap();
        wait_for_count(&count, 3);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(count.load(Ordering::Acquire), 3);
        assert!(iterator.next_batch().unwrap().is_some());
        wait_for_count(&count, 6);
        std::thread::sleep(Duration::from_millis(30));
        assert_eq!(count.load(Ordering::Acquire), 6);
        iterator.close().unwrap();
    }
    #[test]
    fn shutdown_wakes_blocked_readers_and_joins_every_prefetch_thread() {
        let backend = RemoteRuntimeIndex::new(vec!["grpc://127.0.0.1:1".into()]).unwrap();
        let epoch = RemoteBatchIterator::new(vec![Arc::new(reader(stream::pending(), 1))]).unwrap();
        backend.state.lock().epochs.push(Arc::downgrade(&epoch));
        let active = epoch.clone();
        let (tx, rx) = mpsc::channel();
        let thread = std::thread::spawn(move || {
            tx.send(active.next_batch()).unwrap();
        });
        backend.close().unwrap();
        let result = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(result.is_err() || result.unwrap().is_none());
        thread.join().unwrap();
        assert!(epoch.threads.lock().is_empty());
        backend.close().unwrap();
    }
    #[test]
    fn replica_errors_cancel_peers_and_are_not_retried() {
        let failed = Arc::new(reader(
            stream::iter(vec![Err(Status::unavailable("replica failed").into())]),
            1,
        ));
        let blocked = Arc::new(reader(stream::pending(), 1));
        let epoch = RemoteBatchIterator::new(vec![failed, blocked]).unwrap();
        let error = epoch.next_batch().err().unwrap();
        assert_eq!(
            error.downcast_ref::<Status>().unwrap().code(),
            tonic::Code::Unavailable
        );
        assert!(epoch.threads.lock().is_empty());
        assert!(epoch.next_batch().unwrap().is_none());
    }
    #[test]
    fn connections_are_cached_and_closed_backends_cannot_reopen() {
        let backend = RemoteRuntimeIndex::new(vec![
            "grpc://127.0.0.1:1".into(),
            "grpc://127.0.0.1:2".into(),
        ])
        .unwrap();
        let first = backend.clients(1).unwrap();
        assert_eq!(backend.state.lock().clients.len(), 1);
        let second = backend.clients(2).unwrap();
        assert!(Arc::ptr_eq(&first[0], &second[0]));
        let first = backend.clients(2).unwrap();
        assert!(first.iter().zip(&second).all(|(a, b)| Arc::ptr_eq(a, b)));
        backend.close().unwrap();
        assert!(backend.clients(1).is_err());
        assert!(backend.state.lock().clients.is_empty());
    }
    #[test]
    fn invalid_replica_sets_fail_without_native_resources() {
        for endpoints in [
            vec![],
            vec!["".into()],
            vec!["grpc://a:1".into(), "grpc://a:1".into()],
        ] {
            assert!(RemoteRuntimeIndex::new(endpoints).is_err());
        }
    }
    #[test]
    fn shard_count_overflow_fails_before_any_rpc() {
        let backend = RemoteRuntimeIndex::new(vec![
            "grpc://127.0.0.1:1".into(),
            "grpc://127.0.0.1:2".into(),
        ])
        .unwrap();
        let options = serde_json::from_value(serde_json::json!({"stage_table":"train_seeds","sampling_plan_json":"{}","batch_size":1,"num_workers":u64::MAX,"worker_id":0,"epoch":0,"shuffle":false})).unwrap();
        let request = TaskRequest::new("train_tables/task".into(), options, None, 1).unwrap();
        let error = backend.iter(&request, Path::new("/unused")).err().unwrap();
        assert!(error.to_string().contains("shard count overflow"));
        backend.close().unwrap();
    }

    #[test]
    fn weighted_sidecars_become_replica_relative_without_trainer_files() {
        let options=serde_json::from_value(serde_json::json!({"stage_table":"train_seeds","sampling_plan_json":"{}","batch_size":2,"num_workers":1,"worker_id":0,"epoch":0,"shuffle":false,
            "weighted_seed_sampling_json":"{\"buckets\":[{\"node_ids_path\":\"/trainer/graph/train_tables/task/train_seeds/ids.bin\",\"cdf_path\":null}]}"})).unwrap();
        let mut request = TaskRequest::new("train_tables/task".into(), options, None, 1).unwrap();
        relativize_weights(&mut request, Path::new("/trainer/graph")).unwrap();
        let value: serde_json::Value = serde_json::from_str(
            request
                .options
                .weighted_seed_sampling_json
                .as_ref()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(value["buckets"][0]["node_ids_path"], "train_seeds/ids.bin");
        request.options.weighted_seed_sampling_json =
            Some("{\"buckets\":[{\"node_ids_path\":\"/outside/ids.bin\"}]}".into());
        assert!(relativize_weights(&mut request, Path::new("/trainer/graph")).is_err());
    }
}
