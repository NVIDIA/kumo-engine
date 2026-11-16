# GPU timing reference

These 60 implementation modules were copied from Arena revision
`f8ffd61bbe280d0cd515a6387387e9502f5d911d`, under
`gpu-backend/kumo_pytorch_gpu/`. The only source change is replacing the
`kumo_pytorch_gpu` namespace with `test.kumo_pytorch.reference_engine.gpu`.
The reference has no Arena runtime dependency and is not installed as part of
`kumo.pytorch`. Upstream tests, data, saved results and environment scripts were
not copied. Historical Arena results are not validation of this port.

The backend exposes `prepare(contract, dataset, ...)`, returning its own plan
and task with Kumo's contracts, fitted state and artifact format. It loads
Parquet and delivers CUDA-resident batches using cuDF, CuPy and cuGraph. It
retains Arena's direct/split routing and batching behavior. Stored-column
vocabularies use GPU counts; numerical fitting and tokenizer training retain
host paths. Unsupported contracts fail explicitly.

The nightly uses it only for loader timing. The
[adapter hooks](../../benchmarking/gpu_reference.py) measure preparation and
fitting, with synchronization around GPU operations. The runner synchronizes
batch delivery before reporting latency. There is no model or optimizer.
The installed package, correctness reference and existing training campaigns
are unchanged by this addition.

Install only the [GPU library pins](../gpu-requirements.txt) into the existing
Engine CUDA environment. The nightly's CUDA runner owns live validation; CPU
jobs validate import isolation and timing bookkeeping without claiming GPU
execution. See [configuration and reports](../../docs/benchmarking.md#gpu-loader-timing-reference).
