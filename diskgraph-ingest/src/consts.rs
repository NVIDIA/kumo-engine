pub const NODE_BATCH_SIZE: usize = 100_000;
pub const MAX_KEY_SAMPLE_LEN: usize = 64;
pub const TEMP_DIR_NAME: &str = "_tmp";
pub const TEMP_META_MAGIC: [u8; 4] = *b"DGTP";
pub const TEMP_META_VERSION: u8 = 2;
pub const SAMPLE_ROW_LIMIT: usize = 100_000;
pub const MAX_NODE_DATA_LEN: usize = 64 * 1024 * 1024;
pub const PROGRESS_ROW_BATCH: u64 = 1_000_000;
pub const MAX_VAR_KEY_LEN: usize = u16::MAX as usize;
pub const GXHASH_SHARD_SEED: u64 = 0x6a09e667f3bcc909;
pub const PK_MAP_MEM_BUDGET_FRAC: f64 = 0.9;
pub const PK_MAP_LOAD_FACTOR_NUM: u64 = 8;
pub const PK_MAP_LOAD_FACTOR_DEN: u64 = 7;
pub const PK_MAP_MPHF_FACTOR_NUM: u64 = 125;
pub const PK_MAP_MPHF_FACTOR_DEN: u64 = 100;
pub const PK_MAP_ESTIMATE_PAD_PCT: u64 = 25;
pub const INVALID_LOCAL_ID: u32 = u32::MAX;
pub const DEFAULT_INDEX_MEM_BUDGET_FRAC: f64 = 0.7;
/// Column name whose value seeds TT-NG temporal-LP timestamp grouping. Matches
/// the reserved `TIMESTAMP` rowpack column ttgen emits for seed stages.
pub const TIMESTAMP_GROUP_COL: &str = "TIMESTAMP";
