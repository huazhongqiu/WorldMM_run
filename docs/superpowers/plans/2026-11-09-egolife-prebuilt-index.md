# EgoLife prebuilt index plan

Goal: reuse base OpenIE and time-scoped multiscale HippoRAG snapshots.

1. Test matching 30sec reuse, missing coarse records, snapshot isolation and stale identity rejection.
2. Add timestamp-specific snapshot save/load to EpisodicMemory; copy snapshots into worker caches, validate source/model identity and file hashes.
3. Extend existing evaluator with query-day selection, OpenIE-only and index-only preparation, and strict snapshot reuse. No new executable launcher.
4. Run regression tests, real-data offline reuse checks and a bounded single-GPU integration validation. Store persistent assets under /myworkspace; temporary checks under /workspace.

## Validation and recovery update

Implemented timestamp snapshots, base OpenIE reuse, query-day selection, and separate OpenIE/index preparation modes in the existing evaluator and launcher. Persistent artifacts default to worldmm_needed/egolife/prebuilt/A1_JAKE.

A real hourly caption failed because five 4096-token outputs were truncated. Increasing to 8192 with schema/compact formatting was insufficient. The recovery uses 1200-character paragraph/word segments, schema-constrained OpenIE, adaptive re-splitting on truncation, and per-caption checkpointing. Output remains attached to the original caption, so retrieval granularity is preserved while extraction context changes.

Real regression on the exact failed caption passed at the original 4096-token cap: 17 segments, 194 deduplicated triples, about 35 seconds. Tests cover schema forwarding, fenced JSON followed by prose, truncation without identical retries, adaptive splitting, and preservation of successful batch outputs.
