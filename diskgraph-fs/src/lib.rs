use std::sync::Arc;

use anyhow::{anyhow, Result};
use aws_config::{self, BehaviorVersion};
use aws_credential_types::provider::ProvideCredentials;
use aws_types::region::Region;
use object_store::aws::AmazonS3Builder;
use object_store::azure::MicrosoftAzureBuilder;
use object_store::gcp::GoogleCloudStorageBuilder;
use object_store::ObjectStore;
use tokio::runtime::Runtime;

/// Parse a bool environment flag (e.g., "1", "true").
fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name)
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
}

#[derive(Clone, Debug)]
pub struct S3Config {
    pub bucket: String,
    pub region: Option<String>,
    pub endpoint: Option<String>,
    pub allow_http: Option<bool>,
    pub virtual_hosted: Option<bool>,
    pub profile: Option<String>,
    pub access_key_id: Option<String>,
    pub secret_access_key: Option<String>,
    pub session_token: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedS3 {
    pub bucket: String,
    pub region: String,
    pub endpoint: Option<String>,
    pub allow_http: bool,
    pub virtual_hosted: bool,
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}

#[derive(Clone, Debug)]
pub struct S3Client {
    pub store: Arc<dyn ObjectStore>,
    pub resolved: ResolvedS3,
}

pub fn build_s3_client(cfg: S3Config) -> Result<S3Client> {
    let allow_http = cfg
        .allow_http
        .or_else(|| env_flag("S3_ALLOW_HTTP"))
        .or_else(|| env_flag("AWS_ALLOW_HTTP"))
        .unwrap_or(false);
    let endpoint = cfg.endpoint.clone().or_else(|| std::env::var("S3_ENDPOINT").ok());
    let profile = cfg
        .profile
        .clone()
        .or_else(|| std::env::var("S3_PROFILE").ok())
        .or_else(|| std::env::var("AWS_PROFILE").ok());
    let mut loader = aws_config::defaults(BehaviorVersion::latest());
    if let Some(p) = profile.clone() {
        loader = loader.profile_name(p);
    }
    if let Some(region) = cfg.region.clone().or_else(|| std::env::var("S3_REGION").ok()) {
        loader = loader.region(Region::new(region));
    }
    let rt = Runtime::new()?;
    let aws_conf = rt.block_on(loader.load());
    let creds_provider = aws_conf
        .credentials_provider()
        .ok_or_else(|| anyhow!("No AWS credentials provider found; ensure AWS_PROFILE/SSO is set"))?;
    let creds = rt.block_on(async {
        creds_provider
            .provide_credentials()
            .await
            .map_err(|e| anyhow!("Failed to load AWS credentials: {e}"))
    })?;

    let region = cfg
        .region
        .clone()
        .or_else(|| std::env::var("S3_REGION").ok())
        .or_else(|| aws_conf.region().map(|r| r.as_ref().to_string()))
        .unwrap_or_else(|| "us-east-1".to_string());
    let virtual_hosted = cfg
        .virtual_hosted
        .or_else(|| env_flag("S3_VIRTUAL_HOSTED"))
        .unwrap_or_else(|| endpoint.as_ref().map_or(true, |ep| ep.contains("amazonaws.com")));
    let access_key_id = cfg
        .access_key_id
        .clone()
        .or_else(|| std::env::var("S3_ACCESS_KEY").ok())
        .or_else(|| std::env::var("AWS_ACCESS_KEY_ID").ok())
        .unwrap_or_else(|| creds.access_key_id().to_string());
    let secret_access_key = cfg
        .secret_access_key
        .clone()
        .or_else(|| std::env::var("S3_SECRET_KEY").ok())
        .or_else(|| std::env::var("AWS_SECRET_ACCESS_KEY").ok())
        .unwrap_or_else(|| creds.secret_access_key().to_string());
    let session_token = cfg
        .session_token
        .clone()
        .or_else(|| std::env::var("S3_SESSION_TOKEN").ok())
        .or_else(|| creds.session_token().map(|t| t.to_string()));

    let mut builder = AmazonS3Builder::new()
        .with_bucket_name(&cfg.bucket)
        .with_region(&region)
        .with_virtual_hosted_style_request(virtual_hosted)
        .with_allow_http(allow_http)
        .with_access_key_id(access_key_id.clone())
        .with_secret_access_key(secret_access_key.clone());
    if let Some(ep) = &endpoint {
        builder = builder.with_endpoint(ep);
    }
    if let Some(token) = &session_token {
        builder = builder.with_token(token);
    }
    let store = Arc::new(builder.build()?);
    Ok(S3Client {
        store,
        resolved: ResolvedS3 {
            bucket: cfg.bucket,
            region,
            endpoint,
            allow_http,
            virtual_hosted,
            access_key_id,
            secret_access_key,
            session_token,
        },
    })
}

#[derive(Clone, Debug)]
pub struct GcsConfig {
    pub bucket: String,
}

#[derive(Clone, Debug)]
pub struct GcsClient {
    pub store: Arc<dyn ObjectStore>,
    pub bucket: String,
}

pub fn build_gcs_client(cfg: GcsConfig) -> Result<GcsClient> {
    let builder = GoogleCloudStorageBuilder::from_env().with_bucket_name(&cfg.bucket);
    Ok(GcsClient {
        store: Arc::new(builder.build()?),
        bucket: cfg.bucket,
    })
}

#[derive(Clone, Debug)]
pub struct AdlsConfig {
    pub account: String,
    pub container: String,
    pub endpoint: Option<String>,
    pub allow_http: Option<bool>,
    pub access_key: Option<String>,
    pub sas_key: Option<String>,
    pub bearer_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub tenant_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ResolvedAdls {
    pub account: String,
    pub container: String,
    pub endpoint: Option<String>,
    pub allow_http: bool,
    pub access_key: Option<String>,
    pub sas_key: Option<String>,
    pub bearer_token: Option<String>,
    pub client_id: Option<String>,
    pub client_secret: Option<String>,
    pub tenant_id: Option<String>,
}

#[derive(Clone, Debug)]
pub struct AdlsClient {
    pub store: Arc<dyn ObjectStore>,
    pub resolved: ResolvedAdls,
}

pub fn build_adls_client(cfg: AdlsConfig) -> Result<AdlsClient> {
    let allow_http = cfg
        .allow_http
        .or_else(|| env_flag("AZURE_ALLOW_HTTP"))
        .unwrap_or(false);
    let mut builder = MicrosoftAzureBuilder::new()
        .with_account(&cfg.account)
        .with_container_name(&cfg.container)
        .with_allow_http(allow_http);
    if let Some(ep) = cfg.endpoint.clone().or_else(|| std::env::var("AZURE_ENDPOINT").ok()) {
        builder = builder.with_endpoint(ep);
    }
    if let Some(access_key) = cfg
        .access_key
        .clone()
        .or_else(|| std::env::var("AZURE_ACCESS_KEY").ok())
    {
        builder = builder.with_access_key(access_key);
    }
    if let Some(sas) = cfg
        .sas_key
        .clone()
        .or_else(|| std::env::var("AZURE_SAS_KEY").ok())
    {
        builder = builder.with_config(object_store::azure::AzureConfigKey::SasKey, sas);
    }
    if let Some(bearer) = cfg
        .bearer_token
        .clone()
        .or_else(|| std::env::var("AZURE_BEARER_TOKEN").ok())
    {
        builder = builder.with_bearer_token_authorization(bearer);
    }
    if let Some(client_id) = cfg
        .client_id
        .clone()
        .or_else(|| std::env::var("AZURE_CLIENT_ID").ok())
    {
        builder = builder.with_client_id(client_id);
    }
    if let Some(client_secret) = cfg
        .client_secret
        .clone()
        .or_else(|| std::env::var("AZURE_CLIENT_SECRET").ok())
    {
        builder = builder.with_client_secret(client_secret);
    }
    if let Some(tenant_id) = cfg
        .tenant_id
        .clone()
        .or_else(|| std::env::var("AZURE_TENANT_ID").ok())
    {
        builder = builder.with_tenant_id(tenant_id);
    }
    Ok(AdlsClient {
        store: Arc::new(builder.build()?),
        resolved: ResolvedAdls {
            account: cfg.account,
            container: cfg.container,
            endpoint: cfg.endpoint,
            allow_http,
            access_key: cfg.access_key,
            sas_key: cfg.sas_key,
            bearer_token: cfg.bearer_token,
            client_id: cfg.client_id,
            client_secret: cfg.client_secret,
            tenant_id: cfg.tenant_id,
        },
    })
}
