# Reusable packages

Directory groups describe ownership, not installation names. Each project keeps
its own manifest, tests and documentation.

| Group | Responsibility |
| --- | --- |
| `contracts/` | API schemas, PQL and settings |
| `data/` | DiskGraph storage, ingest, sampling and training-table execution |
| `runtime/` | Shared training/prediction runtime and existing GNN code |
| `experimental/` | Experimental standalone packages, including the PyTorch data API |
| `models/` | Model artifacts and model-specific adapters |
| `integrations/` | Source providers and platform-specific integrations |
| `clients/` | Local engine SDK and remote Studio SDK |
| `ui/` | Reusable web components |

The published engine wheel includes only the Python projects in
[engine-bundle.txt](engine-bundle.txt), plus the separately built native DiskGraph
wheel. A workspace project is not automatically a release input.

See the [repository design](../docs/designs/repository-layout.md) for placement
rules and the foundation-model batch inference example.
