# Showdown exposure/count fix — report (Oct 27, 2026)

**Status: DEPLOYED to droplet staging and VERIFIED LIVE; engine
committed LOCALLY in dfs-lab, NOT pushed (Chris's one-time-PAT flow).**

## What was broken (Oct 26 DAL@PHI MNF, live beta API)

Chris set exposure rules on Showdown builds and the delivered sets
broke them, silently:

- Pierce Max Exposure 50 → delivered 70%. A 30 max landed 38%.
- McKee Max 50 → 67% (single-entry run). Schoonmaker Max 10 → 25%.
- Barkley Min Exposure 50 → 3/20. (A hard Lock held, 20/20.)
- Runs silently returned 9, 13, 18, 14, 12, 10 of 20 lineups. No
  warnings in any payload. Chris hand-audited and hand-assembled
  both of his sets.

What already held hard: Exclude, CPT Eligible, cpt_qb_passcatchers.

## Diagnosis (two layers, both fixed)

1. **The generator's caps were a lean, not a wall.**
   `generate_showdown_lineups` used a "hard-ish" accept check: the
   first 8 lineups were exempt from caps, later lineups got +3
   slack, and usage was measured against the PARTIAL count built so
   far (at lineup 12, a 50% cap allowed 9 appearances — 75% of the
   final 20). Min Exposure was enforced only by tail-forcing
   (`need >= remaining`), with no repair when forcing couldn't
   satisfy the solve.
2. **The API enforced against the wrong count.** On the plain
   (no-story) path, `_run_showdown_build` generated ~3× candidates
   (85 for a 20 request), let the generator apply the soft caps
   against 85, then returned the top 20 by Rating Score. Caps
   honored at candidate scale say nothing about the delivered
   slice — that truncation is exactly how Pierce's 50 cap became
   70% delivered, and how shortfalls (9/20 etc.) shipped without a
   word: the payload had no warnings channel filled for Showdown.

Pre-fix reproduction on the frozen Oct 26 rules-build capture pool:
generator asked for 85, returned **27 silently**; the old top-20
slice had Pierce 60% vs Max 50 and McKee 65% vs Max 60.

## The change

### Engine — `dfs_lab/showdown.py` (Showdown only)

- **`_showdown_exposure_targets(df, strategy_map, count)`** — one
  exact accounting pass shared by generation and selection:
  Max = `floor(pct × count / 100)`, Min = `ceil(...)`, same for CPT
  Min/Max. Before any solve it names every promise that cannot be
  kept (excluded/inactive minimums, CPT Min on a non-eligible
  player, Min > Max contradictions, Lock/CPT-Lock overriding a cap,
  and aggregate capacity — caps whose total room is below the
  6 × count slots a build needs). Unhonorable minimums are zeroed
  in the targets so forcing can never make a solve infeasible;
  locks are exempt from caps (a lock IS an appearance promise).
- **`solve_showdown_one(..., blocked_ids, cpt_blocked_ids)`** —
  players at their exact max get upper bound 0 (all slots /
  captain slot). Caps now hold **by construction** in the MILP,
  not by post-hoc acceptance.
- **`generate_showdown_lineups`** — the soft accept block is gone;
  acceptance is an exact check against the targets. After the main
  loop, fill rounds relax ONLY `min_unique` (stepwise, announced)
  and a **swap repair** closes unmet minimums (flex swap-in,
  captain promotion, swap+promote) validating salary, team split,
  K/DST caps, QB-exact, captain-pairing rules, uniqueness and other
  players' min/max. Repair is skipped when relationship rules are
  enabled (a swap could break a QB-stack correlation the audit
  can't see). Every deviation lands in
  `out.attrs["showdown_notes"]`.
- **`select_showdown_portfolio(candidates, df, strategy_map, count,
  min_unique=2)` (new)** — the delivered-set enforcer: Phase A
  reserves minimums (largest deficit first), Phase B fills in
  rating order under the exact caps for the DELIVERED count,
  Phase C repairs unmet minimums by whole-candidate swaps,
  Phase D steps uniqueness down (announced). Max caps are NEVER
  relaxed at any phase. With no exposure settings set it returns
  exactly the old top-N truncation (regression-safe by design).

### API — `aytia-web/api/app.py` (deployed by file sync)

- `_generate_showdown_rows` propagates the generator's named notes
  on the returned frame's attrs.
- The plain path generates the full candidate pool, then selects
  with `select_showdown_portfolio` against `lineup_count`, with up
  to **2 top-up rounds** (fresh seeds, merged candidates, re-select)
  before conceding a shortfall. Notes accumulate (deduped);
  candidate-level shortfall notes ("built X of 85") are filtered —
  the user-facing shortfall is the selection's, against the count
  actually requested.
- `_execute_build` spreads the notes into the payload's top-level
  **`"warnings"`** — the channel the client already renders — only
  when non-empty. Default builds gain no key and identical outputs.
- The story path is deliberately untouched (its portfolio block
  already reports per-core built vs requested); eligibility stays a
  pool concept only — a merely CPT-eligible player is never labeled
  "core" anywhere in the payload (verified in acceptance (d)).

### Classic

Untouched and unaudited in this pass, per spec. No Classic file
changed; the only shared surface is the payload `"warnings"` key,
added for Showdown payloads only when notes exist.

## Exact warning texts now shipped (payload `warnings`)

- `Min Exposure for X (50%) cannot be honored — X is excluded by
  your rules, so the achievable range is 0 of 20 lineups.` (Variants:
  "is not active for this build"; CPT Min: "is not
  captain-eligible in your rules, so the achievable captain range
  is 0 of 20 lineups.")
- `Min Exposure for X is 60% but Max Exposure is 50% — building to
  the max; the achievable range is 0–10 of 20 lineups.`
- `X is locked into every lineup (Lock), which overrides its Max
  Exposure of 50% (cap 10 of 20).`
- `Exposure caps leave room for only 108 of the 120 player slots
  this build needs — a shortfall is unavoidable under these caps.
  Binding caps: …`
- `Min Exposure: X targeted 50% (10 of 20 lineups), reached N — the
  legal-lineup supply under your other rules ran out.` (CPT Min
  equivalent; selection variant ends "— no generated candidate
  could close the gap without breaking another rule.")
- `Shortfall: assembled 7 of 20 requested lineups from the
  generated candidates — exposure caps bind on 17 players
  (tightest: CeeDee Lamb at 20%) and the candidate supply under
  those caps ran out.` (verbatim from the live (c) run)
- `Uniqueness relaxed from 2 to 1 to reach the requested 20
  lineups.`

## Verification

**Engine tests** — new `tests/test_showdown_exposure_enforcement.py`
(11 test functions, 26 checks, synthetic pool + the frozen Oct 26
capture pool): hard Max, hard CPT Max, Min 50 landing exactly
10/20 without a lock, CPT Min 25 landing exactly 5, excluded-min /
ineligible-CPT-min / min>max / lock-beats-cap all named, aggregate
-capacity shortfall named, selection-layer enforcement against the
delivered count, no-settings defaults → 20 lineups and zero notes.
Full engine suite A/B vs the pre-change baseline: **175 passed /
0 failed / 14 errors** vs 164/0/14 — identical except the new file
(the 14 errors are the pre-existing trio:
test_availability_freshness 7, test_lab_agent_llm 3,
test_results_command_center 4).

**API suites** (fresh local servers, final code):
`api_test_v11.py` **49/51** — exactly the recorded baseline drift
pair (CPT Maye 99.0 $49,700; Lab Maye Opt 67.4 / CPT 20.2, live
nflverse data moving under golden numbers pinned Oct 21);
`api_test_v11_adopt.py` **13/13**; `api_test_v28.py` **15/15**.
(One mid-verification run showed 48/51 with a spurious
"B list empty" — the scratch data dir had been reused across runs;
fresh-dir rerun scored baseline 49/51. Recorded, not smoothed.)

**Acceptance — replayed LOCALLY, then LIVE on droplet staging**
(slate `34f3f184eff1fe6f`, DAL@PHI, workspace
`andy-mnf-build-20261026`, Oct 26 rule shapes):

- (a) Oct 26 20-max shape (Barkley Max 30 / Pierce Max 50 /
  McKee Max 60, sub-$3k out except Spann-Ford/Schoonmaker ≤2,
  captains restricted to Lamb/Dak/Pickens/Javonte, Dak CPT = 2+
  DAL pass catchers): **20/20 lineups**, Pierce **10/20** (was
  70%), Barkley 6/20, McKee 12/20, both TEs ≤2, captains Dak 9 /
  Lamb 11, pairing rule held, no warnings.
- (b) Barkley Min Exposure 50, no lock: **exactly 10/20**, no
  warnings.
- (c) Every usable player Max 20: **7/20 returned WITH the named
  shortfall** quoted above (caps bind on 17 players, tightest
  CeeDee Lamb at 20%).
- (d) Javonte CPT Min 25: **exactly 5 captaincies**; payload
  contains no "core" label for any merely-eligible player
  (plain-path lineups carry no Core key; portfolio is null).

## Deploy state

- Droplet files md5-verified both sides: engine
  `dfs_lab/showdown.py` `68eedb2306bdf0a0f7e313feab4bb45c`,
  `api/app.py` `dbb6495cdcced7bbaf35d6edaf4a1422`, new test file
  `93f2f8bae2f29203da1077afd0ce4c7c`. Pre-deploy droplet files were
  exactly the committed baseline (engine `bfc031cd…` = git HEAD;
  app.py `150335bc…` = the WS23-recorded file).
- `docker compose up -d --build` → container recreated, healthy,
  `/health` ok (prod, 11 slates, builds reset 0). Slate store
  byte-identical: aggregate md5
  `2be4f32ee93f6cc07729f3780339f4d6` before and after. No shell
  bump (backend-only change; sw.js untouched).
- dfs-lab commit: LOCAL ONLY, not pushed (see hash in the commit
  log / delivery note). Streamlit parity follows on Chris's next
  PAT push, same as d41b2cb.
