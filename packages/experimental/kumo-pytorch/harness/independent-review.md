# Independent review

The reviewer step inside the self-review loop ([self-review.md](self-review.md)).
Run it as a separate agent with a fresh context, using the brief below.

## Independence

- **Fresh context.** The reviewer sees the diff, the relevant guarantees and
  golden cases, [rules.md](rules.md) and the draft self-review notes. It does
  **not** see the author's conversation or reasoning.
- **Adversarial brief.** Its job is to find a rule violation, or an input that
  breaks the change, not to confirm the change works.
- **Ideally a different model** from the author's.
- **The brief is versioned here.** Changing it needs a harness owner's
  approval, so local runs and any future CI run use the same version.

## Brief

> You are reviewing a kumo-pytorch change you did not write. Find what is
> wrong with it. Read the diff, the guarantees and golden cases it touches or
> relies on, and the rules in
> `packages/experimental/kumo-pytorch/harness/rules.md`. Then:
>
> 1. Walk the checklist below against the diff.
> 2. Look for an input that breaks the change: boundary values, empty and
>    single-row tables, ties at the cutoff, nulls, rows dated after the
>    cutoff at every hop, on DiskGraph and in the differential comparison
>    with the in-memory engine.
> 3. Check that docs, examples and guarantee text still describe the new
>    behavior (C5), and that the draft self-review notes are accurate.
>
> Report findings only. Each finding has: severity (blocking or
> non-blocking), location (file and line), the rule ID or guarantee it
> concerns, and, where possible, a failing input. Style preferences are not
> findings. If nothing is wrong, say so and list what was checked.

## Checklist

Generated from the rules enforced by **Review** in [rules.md](rules.md):

- **Touches time** → future-row cases at every hop (S3).
- **Unsupported input** → refused at `check()` with path and suggestion; no
  silent approximation (S2); a DiskGraph refusal comes from an entry of its
  unsupported list, raised by `check_contract()` (S5).
- **Output, defaults or public API change** → a harness owner approves, and
  compatibility is explained (S6, C4); docs and examples updated (C5).
- **Engine differences** → DiskGraph's output varies only where a guarantee
  leaves room (S4); a discrepancy with the in-memory engine is triaged against
  the rule, recorded in the right list (gap, divergence or named tolerance),
  and never settled by copying either engine's output (D1–D4).
- **Bug fix** → a case failing on the merge base, plus a neighboring boundary
  case (V1, V2).
- **Goldens and tolerances** → calculated from the rule, not engine output;
  tolerances from the spec (V6, V7).
- **Nondeterminism** → seeded; no retries (V5).
- **New mechanism** → no existing one does the same thing (C2); replaced code
  deleted (C3).
- **Errors** → name the stage, resource or path (C6).
- **New dependency** → stated need and install check (X2).
- **Scope** → one coherent goal (C1).
