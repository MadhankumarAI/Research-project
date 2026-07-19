# Notation

Shared across both workstreams. Update this file via PR, not silently — τ and ξ interfaces
are frozen and consumed cross-workstream.

## Graph
- `G = (V, E)` — the full heterogeneous graph
- `V` — nodes (users, and for TwiBot-22 also tweets/lists/hashtags, though our GNN operates on the user-projected graph)
- `E` — edges, each `e = (u, v, r, t)`:
  - `u, v` — source/target user (node) ids
  - `r` — relation type, `r ∈ {follow, mention, retweet, reply, quote}` (extend as needed per dataset)
  - `t` — timestamp of the edge event. **See timestamp reliability note below.**

## Node features
- `x_u` — raw input feature vector for node/user `u` (metadata + text embedding, pre-GNN)
- `h_u` — learned node embedding for user `u` (post-GNN, any layer)
- `h_u^(0) = x_u` by convention

## Trust score (Pillar 1 — frozen interface)
- `τ(u, v, r, t) ∈ [0, 1]` — trust score for edge `e = (u,v,r,t)`
  - 0 = fully distrusted (e.g. one-way follow with no reciprocity/behavioral similarity)
  - 1 = fully trusted
  - Consumed by: heterophily-aware aggregation, explanation head, ξ_u(t) synergy module

## Evasion-event signal (Pillar 2 — consumed by synergy module ξ_u(t))
- `ξ_u(t)` — evasion-event score for user `u` at time `t` (owned/implemented by teammate; τ is an input to it)
- `E_t ⊆ E` — subset of edges with **reliable, observed** per-event timestamps.
  Currently: `r ∈ {mention, retweet, reply, quote}` (timestamp inherited from underlying tweet's `created_at`)
- `t_u0` — account creation timestamp for user `u` (used as the temporal anchor for evasion-event windows)

## Timestamp reliability note (IMPORTANT — see pillar2_evasion_event_revision_draft.md)
**UPDATED, now verified against real reference code** (github.com/LuoUndergradXJTU/TwiBot-22/tree/master/src/BotRGCN), not just documentation:
`edge.csv` has exactly 3 columns (source_id, relation, target_id) with **no timestamp column at all**, confirmed for TwiBot-20, TwiBot-22, and Cresci-2015. Additionally — a stronger finding than we had before — **the reference pipeline doesn't even construct mention/retweet/reply/quote as graph edges**; tweets are used purely as a text source for embeddings, never as edges. So the earlier plan (Section 2 of the evasion-event revision doc) to fall back on a "timestamped-relation subset E_t" of mention/retweet/reply/quote edges has **no edges to draw from** in this reference pipeline — that subset is currently empty. If we want real per-event timing for the evasion-event signal, we will need to construct our own tweet-derived edges directly from raw tweet objects (which may carry usable timestamps in their own metadata), separately from edge.csv. This needs a team discussion — flagging here rather than deciding unilaterally.

Convention used throughout the codebase:
- `e.t_observed: bool` — currently always False for every edge we load, since edge.csv provides no observed timestamps for any relation type.
- Follow edges get a **proxy** timestamp (see `pipeline.time_proxy.estimate_follow_time_proxy`); currently this proxy is weak (falls back to account creation time in most cases) since there are no timestamped interaction edges to anchor it to.

## Relation vocabulary — CONFIRMED, differs by dataset (do not assume uniformity!)
- TwiBot-20 / Cresci-2015 `edge.csv`: relation values include `'friend'` (→ FOLLOW) and `'post'` (user→tweet, excluded from the social graph). At least one more relation string exists in the reference code's "else" branch for these two datasets — exact string **not yet confirmed**, check against real files.
- TwiBot-22 `edge.csv`: relation values are `'followers'` and `'following'` (both → FOLLOW) — a **completely different vocabulary** from TB20/Cresci. Do not reuse a single RELATION_MAP across datasets (see `loaders/twibot.py`'s `RELATION_MAP_BY_DATASET`).

## Node/user fields — CONFIRMED via reference code, corrects earlier wrong assumptions
Real fields use a Twitter API v2-style shape, not the v1.1-style shape originally assumed:
- `id` (not `id_str`), `username` (not `screen_name`)
- `public_metrics.followers_count` / `.following_count` / `.listed_count` — **nested**, not top-level. Note the reference code's "statuses" variable is actually populated from `listed_count`, not a statuses field — likely an upstream naming artifact, replicated faithfully rather than silently corrected.
- `created_at` is a **Unix timestamp** (seconds), not a Twitter date string
- `verified`/`protected`: boolean in TwiBot-22, but the **string** `'True '` (trailing space) in TwiBot-20 — parse differently per dataset
- `default_profile_image` is **derived** by string-matching `profile_image_url` against known default-avatar URLs — it is not a raw field
- Cresci-2015 has a thinner feature set: no verified/protected extracted by the reference pipeline (categorical features = default_profile_image only)



## Dataset-specific ids
- Each loader normalizes to global user ids as strings prefixed by dataset: `"tb20:<id>"`, `"tb22:<id>"`, `"cresci15:<id>"`
  to avoid collisions if datasets are ever combined.
