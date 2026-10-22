# ADLS implementation and resource controls

The provider adapts Azure SDK operations to shared Arrow source and sink
machinery in `kumo-core`. ETags guard range/full-file reads and source versions;
account-qualified paths distinguish accounts and clouds. Azure Identity refreshes
tokens, and identity cleanup closes cached clients. Sessions are cached for the
process lifetime, without automatic eviction; identity cleanup or
`kumo_adls.session.close()` releases them. Construction runs under a global lock
to prevent duplicate clients, so concurrent first use of different accounts can
serialize. Arrow/fsspec wrappers are lightweight and created per access.

Identity probing checks Python token acquisition only, not native output compatibility. It does not verify storage
existence, RBAC/ACL permissions, or connectivity; list or read the configured
source to verify those.

Bulk downloads use one conditional SDK stream per file, with 16 MiB initial
and subsequent responses and 16 MiB HTTP transport blocks. Files download
concurrently; each file holds one transfer slot and uses no per-file executor
or queued chunk futures. The SDK writes each response to the local temporary
file before requesting the next chunk. The initial response remains buffered
for the stream's lifetime; this is not zero-copy network-to-disk streaming.
Large projected Arrow reads
can use up to four SDK workers, bounded by the shared transfer slots. A 64 MiB
range uses up to three workers; ranges above 64 MiB can use four. The initial
response plus four 16 MiB chunks can account for about 80 MiB before transport
buffers, SDK copies, and the complete requested range returned by `readall()`.
Projected reads are not covered by the 64 MiB bulk-copy reservation.

Transfers across files and accounts share a CPU-derived concurrency ceiling
(at most 128, lowered by `KUMO_SOURCE_COPY_WORKERS`). The HTTP connection pool
matches that ceiling. Metadata bypasses transfer slots and shares the HTTP
transport. Source bulk copies use the core copy reservation. These SDK
limits and reservations do not apply to native Arrow uploads. These reservations
are accounting estimates, not caps on total process RSS. Larger responses
reduce requests and fragment assembly while increasing memory per active file.
Plan roughly 48–64 MiB of payload buffers per active bulk download, with
additional headroom for SDK copies and other allocations. At 128 active files,
that estimate alone is 6–8 GiB; the reservation does not guarantee that this
fits a host's memory limit. Actual memory depends on allocation lifetimes.

For memory-constrained hosts, set `KUMO_ADLS_TRANSFER_WORKERS=4` before
importing the provider to limit ADLS transfer requests and its HTTP pool
without lowering other providers' copy limits. This does not limit native writes. The value must be a positive
integer and cannot raise the shared copy/CPU ceiling. The limit is process-wide
across ADLS accounts and readers; it is read once at import. Even with one slot,
files use the same serial SDK stream. Arrow buffers, cached bytes, SDK copies,
loaded libraries, and Python overhead still consume memory. Reduced ceilings
are covered by correctness tests; a fixed-RAM host requires separate validation.
An invalid value fails at the provider's first import. Set these variables in
the launching environment and restart an existing process after changing them.
Catchable allocation failures inside ADLS SDK operations retain their original
exception and traceback, with a note suggesting these controls if the failure
occurred during a transfer. They may not help with listing or session setup.
An operating-system OOM kill cannot provide a Python error message.
Reducing the shared copy ceiling
also limits waiting file workers and their buffers; the ADLS-only limit does not.

Projected Arrow reads disable fsspec read-ahead, avoiding unrelated column data.
Successful small ranges can reuse immutable bytes under the opened file's raw
ETag. The process-wide LRU holds at most 64 MiB of payload/1,024 entries, admits
ranges up to 128 KiB, and isolates filesystem/principal namespaces. Opening a
file still performs fresh properties checks; metadata is never cached here.
Existing opened streams may reuse their pinned version, while ingest's final
identity checks still reject source changes. Filesystem cleanup releases its
cached ranges; Python container/key overhead is additional. Each source access
creates a new filesystem namespace. This cache avoids repeated footer reads
within one reader, not across separate reads or connector operations. The
64 MiB limit bounds aggregate retained payload across concurrent readers; it
is not a persistent connector cache or a preallocated buffer.
Recursive discovery uses Azure's paginated recursive listing. Version ETag
checks use fresh bounded listings when complete ETags are available, falling
back to concurrent HEADs for missing metadata, narrow permissions, and exact
single files. ADLS uses the shared object-store `prepare`: discover files once,
stamp the version from that same selected list, and read those files. A supplied
version is stamped without refreshing it, matching S3/GCS. Shared per-file
identity guards check selected objects before and after reading; files added
after discovery belong to a subsequent read. Identity checks may still perform
listings or HEADs, so one discovery does not mean one metadata request.
ETag quote and directory-path normalization can invalidate existing version
hashes once, requiring restaging.
Prepared exports, generated Parquet/CSV output, and model bundles use native
Arrow output streams. The shared sink copies prepared bytes in 8 MiB blocks,
with up to eight files per directory export; model bundles upload files
sequentially. Arrow owns per-file buffering and background upload scheduling.
The ADLS SDK transfer semaphore and 64 MiB copy reservation do not constrain
these native buffers. Copy block size is not an upload-memory limit. Consumers
must wait for export completion; streaming writes and export directories are
not transactional. Arrow close after a caller failure may publish partial
output, as with the shared S3/GCS sink. The SDK writer's conditional cleanup
and final-flush guarantees do not apply to native output.

Native output is bound lazily on first write, leaving SDK metadata, source
reads, model downloads, and their account-qualified paths unchanged. Only
output streams drop the virtual account qualifier to use Arrow's container/key
path. Existing root containment checks run before filesystem access.
A native filesystem is reused within one sink access, with thread-safe first
construction; it is not cached across accesses or identities. The provider
requires Arrow 24, the version whose native credential/URI APIs are validated.

Without an explicit identity, writes use Arrow's C++ ambient credential chain;
reads use Python Azure Identity. These chains can select different ambient
credentials. Explicit workload writes require process `AZURE_TENANT_ID` and
`AZURE_CLIENT_ID` to equal the selected identity, plus
`AZURE_FEDERATED_TOKEN_FILE`. They select only Arrow's workload credential,
without ambient fallback. Arrow captures these settings at construction and
rereads projected token contents when obtaining a fresh assertion. A different
per-connector identity remains supported for reads, but its native writes fail
with a configuration error. The provider never changes process environment or
mints delegation SAS to bridge credentials.

Native writes validate `AZURE_AUTHORITY_HOST` against the selected cloud. Public
Azure defaults to `https://login.microsoftonline.com`; US Government needs
`https://login.microsoftonline.us`, and China needs
`https://login.chinacloudapi.cn`. Both native Blob and DFS endpoints use the
selected storage suffix. Offline tests construct real native filesystems for
all three clouds and check identity/authority rejection and endpoint mapping.
They do not establish live federation, token renewal, sovereign connectivity,
or RBAC/ACL propagation; those require suitable live environments.

Downloaded files remain owner-private, matching the S3 staging path. Local
engine ingestion uses the same OS user; sharing scratch files across users
requires explicitly managed permissions. Temporary files are closed before
promotion with `os.replace`, including on failure cleanup paths. This avoids
requiring POSIX open-file replacement; it does not establish full Windows
engine support.
