// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Caches AWS SDK credentials for the `object_store` S3 client.
//!
//! `object_store` asks custom providers for credentials on every request, while
//! the provider exposed by the AWS SDK config is uncached. This adapter caches
//! credentials and shares refresh work between callers.

use std::fmt;
use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, SystemTime};

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use aws_credential_types::provider::{ProvideCredentials, SharedCredentialsProvider};
use object_store::aws::AwsCredential;
use object_store::{CredentialProvider, Error as ObjectStoreError, Result as ObjectStoreResult};
use tokio::runtime::{Handle, Runtime};
use tokio::sync::{Mutex, RwLock};

// Prefer renewal outside object_store's 180-second request retry window.
const RENEWAL_LEAD: Duration = Duration::from_secs(5 * 60);
const UNKNOWN_EXPIRY_TTL: Duration = Duration::from_secs(15 * 60);
const MIN_RENEWAL_INTERVAL: Duration = Duration::from_secs(30);
const RENEWAL_RETRY_INTERVAL: Duration = Duration::from_secs(30);

/// AWS SDK clients need a runtime that outlives callers' short-lived runtimes.
/// Multi-threading lets synchronous callers block while SDK tasks continue.
fn aws_runtime() -> Result<&'static Runtime> {
    static RUNTIME: OnceLock<std::io::Result<Runtime>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .thread_name("diskgraph-aws")
                .build()
        })
        .as_ref()
        .map_err(|err| anyhow!("failed to start the AWS credential runtime: {err}"))
}

pub(super) fn aws_runtime_handle() -> Result<Handle> {
    Ok(aws_runtime()?.handle().clone())
}

pub(super) fn block_on_aws<F, T>(future: F) -> Result<T>
where
    F: Future<Output = Result<T>>,
{
    aws_runtime()?.block_on(future)
}

pub(super) async fn spawn_on_aws<F, T>(future: F) -> Result<T>
where
    F: Future<Output = Result<T>> + Send + 'static,
    T: Send + 'static,
{
    aws_runtime()?
        .handle()
        .spawn(future)
        .await
        .map_err(|err| anyhow!("AWS credential runtime task failed: {err}"))?
}

struct CachedCredential {
    credential: Arc<AwsCredential>,
    renew_at: SystemTime,
    expires_at: Option<SystemTime>,
}

struct Backoff {
    retry_at: SystemTime,
    reason: String,
}

fn unauthenticated(err: anyhow::Error) -> ObjectStoreError {
    ObjectStoreError::Unauthenticated {
        path: "AWS credentials".to_string(),
        source: err.into(),
    }
}

pub(super) struct RefreshingCredentials {
    provider: SharedCredentialsProvider,
    runtime: Handle,
    cached: RwLock<Option<CachedCredential>>,
    refresh_lock: Mutex<()>,
    backoff: RwLock<Option<Backoff>>,
}

#[derive(Clone, Copy)]
enum Freshness {
    Fresh,
    Valid,
}

impl fmt::Debug for RefreshingCredentials {
    // Do not expose cached credentials through Debug.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RefreshingCredentials")
            .finish_non_exhaustive()
    }
}

impl RefreshingCredentials {
    pub(super) fn new(provider: SharedCredentialsProvider, runtime: Handle) -> Self {
        Self {
            provider,
            runtime,
            cached: RwLock::new(None),
            refresh_lock: Mutex::new(()),
            backoff: RwLock::new(None),
        }
    }

    async fn cached_credential(&self, freshness: Freshness) -> Option<Arc<AwsCredential>> {
        let now = SystemTime::now();
        let slot = self.cached.read().await;
        let cached = slot.as_ref()?;
        let valid = cached.expires_at.map_or(true, |expiry| now < expiry);
        let acceptable = match freshness {
            Freshness::Fresh => valid && now < cached.renew_at,
            Freshness::Valid => valid,
        };
        acceptable.then(|| Arc::clone(&cached.credential))
    }

    async fn backing_off(&self) -> Option<String> {
        let now = SystemTime::now();
        let backoff = self.backoff.read().await;
        backoff
            .as_ref()
            .filter(|backoff| now < backoff.retry_at)
            .map(|backoff| backoff.reason.clone())
    }

    pub(super) async fn prime(&self) -> Result<()> {
        self.get_credential()
            .await
            .map_err(|err| anyhow!("Failed to load AWS credentials: {err}"))?;
        Ok(())
    }

    async fn resolve(&self) -> Result<CachedCredential> {
        let provider = self.provider.clone();
        let resolved = self
            .runtime
            .spawn(async move { provider.provide_credentials().await })
            .await
            .map_err(|err| anyhow!("AWS credential runtime task failed: {err}"))?
            .map_err(anyhow::Error::new)?;

        let now = SystemTime::now();
        let expires_at = resolved.expiry();
        if expires_at.is_some_and(|expiry| expiry <= now) {
            return Err(anyhow!(
                "AWS credential provider returned an expired credential"
            ));
        }
        let renew_at = expires_at
            .and_then(|expiry| expiry.checked_sub(RENEWAL_LEAD))
            .unwrap_or_else(|| now + UNKNOWN_EXPIRY_TTL)
            .max(now + MIN_RENEWAL_INTERVAL);

        Ok(CachedCredential {
            credential: Arc::new(AwsCredential {
                key_id: resolved.access_key_id().to_string(),
                secret_key: resolved.secret_access_key().to_string(),
                token: resolved.session_token().map(str::to_string),
            }),
            renew_at,
            expires_at,
        })
    }
}

#[async_trait]
impl CredentialProvider for RefreshingCredentials {
    type Credential = AwsCredential;

    async fn get_credential(&self) -> ObjectStoreResult<Arc<AwsCredential>> {
        if let Some(fresh) = self.cached_credential(Freshness::Fresh).await {
            return Ok(fresh);
        }

        let _refresh = match self.refresh_lock.try_lock() {
            Ok(guard) => guard,
            // Only the refresh winner waits when the cached credential is still valid.
            Err(_) => match self.cached_credential(Freshness::Valid).await {
                Some(valid) => return Ok(valid),
                None => self.refresh_lock.lock().await,
            },
        };

        if let Some(fresh) = self.cached_credential(Freshness::Fresh).await {
            return Ok(fresh);
        }

        // Backoff also applies after expiry, preventing one provider call per S3 request.
        if let Some(reason) = self.backing_off().await {
            return match self.cached_credential(Freshness::Valid).await {
                Some(valid) => Ok(valid),
                None => Err(unauthenticated(anyhow!(reason))),
            };
        }

        match self.resolve().await {
            Ok(fresh) => {
                let expires_in = fresh
                    .expires_at
                    .and_then(|expiry| expiry.duration_since(SystemTime::now()).ok())
                    .map(|remaining| remaining.as_secs());
                tracing::info!(
                    expires_in_secs = expires_in,
                    key_id = %fresh.credential.key_id,
                    "refreshed AWS credentials for S3"
                );
                let credential = Arc::clone(&fresh.credential);
                *self.cached.write().await = Some(fresh);
                *self.backoff.write().await = None;
                Ok(credential)
            }
            Err(err) => {
                let reason = format!("{err:#}");
                *self.backoff.write().await = Some(Backoff {
                    retry_at: SystemTime::now() + RENEWAL_RETRY_INTERVAL,
                    reason: reason.clone(),
                });
                // A due credential remains usable until its actual expiration.
                match self.cached_credential(Freshness::Valid).await {
                    Some(valid) => {
                        tracing::warn!(
                            error = %reason,
                            "could not refresh AWS credentials; reusing the current one until it expires"
                        );
                        Ok(valid)
                    }
                    None => Err(unauthenticated(err)),
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use aws_credential_types::provider::future;
    use aws_credential_types::Credentials;

    use super::*;

    /// object_store reuses one signature for up to its `retry_timeout`, so a
    /// credential handed out now has to still sign requests this far into the
    /// future. This is the number the renewal lead exists to clear.
    const REQUEST_RETRY_WINDOW: Duration = Duration::from_secs(180);

    /// Counts calls so a test can tell a cache hit from a fetch, and hands out
    /// a distinguishable credential each time so a test can tell *which* one
    /// it got.
    #[derive(Debug)]
    struct CountingProvider {
        calls: Arc<AtomicUsize>,
        ttl: Option<Duration>,
        fail_after: usize,
        born_expired: bool,
    }

    impl CountingProvider {
        fn new(ttl: Option<Duration>) -> (Self, Arc<AtomicUsize>) {
            let calls = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    calls: Arc::clone(&calls),
                    ttl,
                    fail_after: usize::MAX,
                    born_expired: false,
                },
                calls,
            )
        }

        fn failing_after(ttl: Option<Duration>, fail_after: usize) -> (Self, Arc<AtomicUsize>) {
            let (mut provider, calls) = Self::new(ttl);
            provider.fail_after = fail_after;
            (provider, calls)
        }

        /// Hands back a credential that is already dead on arrival.
        fn born_expired() -> (Self, Arc<AtomicUsize>) {
            let (mut provider, calls) = Self::new(Some(Duration::from_secs(1)));
            provider.born_expired = true;
            (provider, calls)
        }
    }

    impl ProvideCredentials for CountingProvider {
        fn provide_credentials<'a>(&'a self) -> future::ProvideCredentials<'a>
        where
            Self: 'a,
        {
            let nth = self.calls.fetch_add(1, Ordering::SeqCst);
            let ttl = self.ttl;
            let fail = nth >= self.fail_after;
            let born_expired = self.born_expired;
            future::ProvideCredentials::new(async move {
                if fail {
                    return Err(
                        aws_credential_types::provider::error::CredentialsError::not_loaded(
                            "no credentials in this test",
                        ),
                    );
                }
                let expiry = if born_expired {
                    Some(SystemTime::now() - Duration::from_secs(1))
                } else {
                    ttl.map(|ttl| SystemTime::now() + ttl)
                };
                Ok(Credentials::new(
                    format!("AKIAKEY{nth}"),
                    format!("secret-{nth}"),
                    Some(format!("token-{nth}")),
                    expiry,
                    "counting-provider",
                ))
            })
        }
    }

    fn provider_with(inner: CountingProvider) -> RefreshingCredentials {
        RefreshingCredentials::new(
            SharedCredentialsProvider::new(inner),
            Handle::try_current().expect("tests run on a tokio runtime"),
        )
    }

    /// The sync builders block on the shared runtime while the future they
    /// drive spawns onto that same runtime and awaits it. With far more callers
    /// than the runtime has worker threads, that shape has to not deadlock.
    #[test]
    fn concurrent_blocking_callers_do_not_deadlock() {
        let threads: Vec<_> = (0..16u64)
            .map(|i| {
                std::thread::spawn(move || {
                    block_on_aws(async move { spawn_on_aws(async move { Ok(i * 2) }).await })
                        .unwrap()
                })
            })
            .collect();
        for (i, thread) in threads.into_iter().enumerate() {
            assert_eq!(thread.join().unwrap(), i as u64 * 2);
        }
    }

    #[tokio::test]
    async fn a_live_credential_is_served_from_cache() {
        let (inner, calls) = CountingProvider::new(Some(Duration::from_secs(3600)));
        let creds = provider_with(inner);

        for _ in 0..50 {
            let got = creds.get_credential().await.unwrap();
            // All three fields, because all three are SigV4 inputs and the key
            // id alone would not notice a dropped secret or session token.
            assert_eq!(got.key_id, "AKIAKEY0");
            assert_eq!(got.secret_key, "secret-0");
            assert_eq!(got.token.as_deref(), Some("token-0"));
        }

        // The whole point of the cache: object_store asks per request, and each
        // miss would be an AssumeRoleWithWebIdentity round trip.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn renewal_is_scheduled_a_fixed_lead_ahead_of_expiry() {
        // This is the fix in one assertion. Renewing at or after expiry is the
        // original production bug, and renewing too close to it leaves a window
        // where object_store is still retrying a signature the credential can
        // no longer back.
        //
        // Several lifetimes, because the lead has to be read off the *stated*
        // expiry. A renewal point computed from anything else - a fixed TTL, a
        // constant offset from now - can look right at one lifetime and put
        // renewal past expiry at another.
        for lifetime in [3600, 900, 600].map(Duration::from_secs) {
            let (inner, _calls) = CountingProvider::new(Some(lifetime));
            let creds = provider_with(inner);
            creds.get_credential().await.unwrap();

            let slot = creds.cached.read().await;
            let cached = slot.as_ref().unwrap();
            let expires_at = cached.expires_at.expect("the provider stated an expiry");
            assert!(
                cached.renew_at < expires_at,
                "renewal must be scheduled before the session expires, but for a \
                 {lifetime:?} credential it was not"
            );

            // Exact, not a bound: renew_at is derived from expires_at, so any
            // lifetime long enough to clear MIN_RENEWAL_INTERVAL must land the
            // renewal point exactly RENEWAL_LEAD before expiry.
            let lead = expires_at.duration_since(cached.renew_at).unwrap();
            assert_eq!(
                lead, RENEWAL_LEAD,
                "a {lifetime:?} credential should renew exactly {RENEWAL_LEAD:?} \
                 before it expires"
            );
            assert!(
                lead >= REQUEST_RETRY_WINDOW,
                "renewal lead {lead:?} must clear object_store's retry_timeout, \
                 or a request can still be retrying a signature past expiry"
            );
        }
    }

    #[tokio::test]
    async fn a_credential_inside_the_renewal_lead_is_floored_not_refetched() {
        // Expires sooner than RENEWAL_LEAD, so it is due for renewal at once.
        let (inner, calls) = CountingProvider::new(Some(Duration::from_secs(60)));
        let creds = provider_with(inner);

        let first = creds.get_credential().await.unwrap();
        assert_eq!(first.key_id, "AKIAKEY0");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // MIN_RENEWAL_INTERVAL keeps this from re-fetching immediately. Without
        // the floor, object_store's per-request get_credential turns a
        // short-lived credential into one STS call per S3 request.
        let second = creds.get_credential().await.unwrap();
        assert_eq!(second.key_id, "AKIAKEY0");
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // ...but once that floor passes, the next call does refresh.
        creds.cached.write().await.as_mut().unwrap().renew_at =
            SystemTime::now() - Duration::from_secs(1);
        let third = creds.get_credential().await.unwrap();
        assert_eq!(third.key_id, "AKIAKEY1");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_cached_credential_is_replaced_once_it_falls_due_for_renewal_or_expires() {
        let (inner, calls) = CountingProvider::new(Some(Duration::from_secs(3600)));
        let creds = provider_with(inner);

        let first = creds.get_credential().await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // A fresh credential is served from the cache, so the provider is not
        // consulted again.
        let cached = creds.get_credential().await.unwrap();
        assert_eq!(cached.key_id, first.key_id);
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Reaching the renewal point is the only thing that changes here; the
        // credential is still far from expiry.
        creds.cached.write().await.as_mut().unwrap().renew_at =
            SystemTime::now() - Duration::from_secs(1);

        let renewed = creds.get_credential().await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        assert_eq!(
            renewed.key_id, "AKIAKEY1",
            "renewal must yield a new credential rather than the cached copy"
        );
        assert_eq!(renewed.secret_key, "secret-1");
        assert_eq!(renewed.token.as_deref(), Some("token-1"));

        // Past its expiry a credential cannot sign anything, so it is not
        // served even though its renewal point is still in the future.
        creds.cached.write().await.as_mut().unwrap().expires_at =
            Some(SystemTime::now() - Duration::from_secs(1));

        let replaced = creds.get_credential().await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 3);
        assert_eq!(replaced.key_id, "AKIAKEY2");
    }

    #[tokio::test]
    async fn an_expired_credential_is_never_served_as_current() {
        // A credential shorter-lived than RENEWAL_LEAD gets a renew_at pushed
        // out by MIN_RENEWAL_INTERVAL, so renew_at can land AFTER expires_at.
        // Freshness alone would then keep serving it for the gap.
        let (inner, calls) = CountingProvider::new(Some(Duration::from_secs(3600)));
        let creds = provider_with(inner);

        creds.get_credential().await.unwrap();
        {
            let mut slot = creds.cached.write().await;
            let cached = slot.as_mut().unwrap();
            cached.expires_at = Some(SystemTime::now() - Duration::from_secs(1));
            cached.renew_at = SystemTime::now() + Duration::from_secs(60);
        }

        let got = creds.get_credential().await.unwrap();
        assert_eq!(
            got.key_id, "AKIAKEY1",
            "an expired credential must be replaced, not served because its \
             renewal time has not arrived"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn an_expired_credential_is_not_served_while_a_refresh_is_in_flight() {
        // The contention path hands out a credential rather than queueing, but
        // only while that credential still signs requests.
        let (inner, _calls) = CountingProvider::new(Some(Duration::from_secs(3600)));
        let creds = Arc::new(provider_with(inner));
        creds.get_credential().await.unwrap();
        {
            let mut slot = creds.cached.write().await;
            let cached = slot.as_mut().unwrap();
            cached.renew_at = SystemTime::now() - Duration::from_secs(1);
            cached.expires_at = Some(SystemTime::now() - Duration::from_secs(1));
        }

        let held = creds.refresh_lock.lock().await;
        let waiter = tokio::spawn({
            let creds = Arc::clone(&creds);
            async move { creds.get_credential().await.unwrap().key_id.clone() }
        });
        // Let the waiter reach the lock before releasing it.
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
        drop(held);

        assert_eq!(
            waiter.await.unwrap(),
            "AKIAKEY1",
            "a dead credential must not be handed out just because someone \
             else is already fetching"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_requests_share_one_refresh() {
        let (inner, calls) = CountingProvider::new(Some(Duration::from_secs(3600)));
        let creds = Arc::new(provider_with(inner));

        let mut tasks = Vec::new();
        for _ in 0..64 {
            let creds = Arc::clone(&creds);
            tasks.push(tokio::spawn(async move {
                creds.get_credential().await.unwrap()
            }));
        }
        for task in tasks {
            assert_eq!(task.await.unwrap().key_id, "AKIAKEY0");
        }

        // Without single-flight this is a thundering herd onto STS.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_refresh_in_flight_does_not_stall_other_requests() {
        let (inner, calls) = CountingProvider::new(Some(Duration::from_secs(3600)));
        let creds = provider_with(inner);

        creds.get_credential().await.unwrap();
        creds.cached.write().await.as_mut().unwrap().renew_at =
            SystemTime::now() - Duration::from_secs(1);

        // Stand in for another task that is already fetching.
        let held = creds.refresh_lock.lock().await;
        // Bounded, because the regression this guards against is a caller that
        // waits forever rather than one that answers wrongly. Without the
        // timeout, that regression hangs CI instead of failing it.
        let got = tokio::time::timeout(Duration::from_secs(5), creds.get_credential())
            .await
            .expect("a valid credential must be served without waiting for a refresh")
            .unwrap();
        drop(held);

        // Due for renewal is not the same as expired. Uploads carry on against
        // the credential we have rather than queueing behind an STS round trip
        // - if this regressed, every request would block on every refresh.
        assert_eq!(got.key_id, "AKIAKEY0");
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_still_valid_credential_survives_a_failing_provider() {
        // Good for an hour, but due for renewal immediately.
        let (inner, calls) = CountingProvider::failing_after(Some(Duration::from_secs(3600)), 1);
        let creds = provider_with(inner);

        let first = creds.get_credential().await.unwrap();
        assert_eq!(first.key_id, "AKIAKEY0");

        creds.cached.write().await.as_mut().unwrap().renew_at =
            SystemTime::now() - Duration::from_secs(1);

        // The refresh fails, but the cached credential has not expired, so the
        // upload should proceed rather than fail permanently. A credential Err
        // bypasses object_store's retry layer entirely.
        let second = creds.get_credential().await.unwrap();
        assert_eq!(second.key_id, "AKIAKEY0");
        assert_eq!(second.secret_key, "secret-0");
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_failing_provider_is_not_retried_once_per_request() {
        let (inner, calls) = CountingProvider::failing_after(Some(Duration::from_secs(3600)), 1);
        let creds = provider_with(inner);

        creds.get_credential().await.unwrap();
        creds.cached.write().await.as_mut().unwrap().renew_at =
            SystemTime::now() - Duration::from_secs(1);

        for _ in 0..20 {
            creds.get_credential().await.unwrap();
        }

        // The first fetch, then one failed renewal - not one per request.
        // object_store resolves credentials outside its retry layer, so the
        // alternative is 20 STS calls for 20 uploads.
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn a_failing_provider_with_nothing_cached_still_backs_off() {
        // The credential has genuinely expired and STS is down. Every request
        // fails either way, but they must not each drive their own STS call.
        let (inner, calls) = CountingProvider::failing_after(Some(Duration::from_secs(3600)), 0);
        let creds = provider_with(inner);

        for _ in 0..20 {
            let err = creds.get_credential().await.unwrap_err();
            assert!(matches!(err, ObjectStoreError::Unauthenticated { .. }));
        }

        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "a failing provider with no usable credential must be retried on a \
             timer, not once per request"
        );
    }

    #[tokio::test]
    async fn an_expired_credential_with_a_failing_provider_is_an_error() {
        let (inner, _calls) = CountingProvider::failing_after(Some(Duration::from_secs(3600)), 1);
        let creds = provider_with(inner);

        creds.get_credential().await.unwrap();
        {
            let mut slot = creds.cached.write().await;
            let cached = slot.as_mut().unwrap();
            cached.renew_at = SystemTime::now() - Duration::from_secs(1);
            cached.expires_at = Some(SystemTime::now() - Duration::from_secs(1));
        }

        let err = creds.get_credential().await.unwrap_err();
        assert!(
            matches!(err, ObjectStoreError::Unauthenticated { .. }),
            "expected Unauthenticated, got {err:?}"
        );
    }

    #[tokio::test]
    async fn a_credential_that_arrives_already_expired_is_rejected() {
        // Caching one would be worse than failing: every later request would
        // re-resolve, with no backoff, because resolve() returned Ok.
        let (inner, calls) = CountingProvider::born_expired();
        let creds = provider_with(inner);

        let err = creds.get_credential().await.unwrap_err();
        assert!(
            matches!(err, ObjectStoreError::Unauthenticated { .. }),
            "expected Unauthenticated, got {err:?}"
        );
        assert!(
            creds.cached.read().await.is_none(),
            "a dead credential must not be cached"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn a_credential_without_an_expiry_is_still_renewed_eventually() {
        let (inner, calls) = CountingProvider::new(None);
        let creds = provider_with(inner);

        creds.get_credential().await.unwrap();
        let renew_at = creds.cached.read().await.as_ref().unwrap().renew_at;
        let horizon = SystemTime::now() + UNKNOWN_EXPIRY_TTL;
        assert!(
            renew_at <= horizon,
            "a credential with no stated expiry must still be renewed"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn debug_output_does_not_carry_the_secret() {
        let (inner, _calls) = CountingProvider::new(Some(Duration::from_secs(3600)));
        let creds = provider_with(inner);
        creds.get_credential().await.unwrap();

        // The incident that motivated this module leaked a live session token
        // into a pod log, so the type that holds one must stay quiet.
        let rendered = format!("{creds:?}");
        assert!(!rendered.contains("secret-0"), "{rendered}");
        assert!(!rendered.contains("token-0"), "{rendered}");
    }
}
