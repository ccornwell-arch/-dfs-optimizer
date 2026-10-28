"""Aytia self-audit layer (WS26 — trust release, Piece 2).

Chris's mandate (Oct 28, 2026): the app must think the way a builder
thinks on ALL slates, so he can run a build and enter the lineups in
DraftKings as-is. Catching an incoherent lineup himself is a bug in
the process, not the QC plan. The no-QB lineup that triggered this
was, in his words, "more of a metaphor": the general question the
audit asks of EVERY delivered lineup is —

    "In one sentence: what happens in this game for this lineup to
     be the right lineup, and does every player in it belong in
     that sentence?"

The audit runs inside the build, after selection and before the set
is served or captured, and argues against the app's own lineups in
three voices:

  * coherence — the lineup contradicts football (the slate-agnostic
    inventory C1–C9 below; C9 is the generalized story-coherence
    evaluation: a lineup's own best worlds are decomposed and its
    members' required performances tested for mutual contradiction,
    by class, never by naming one day's shape);
  * domination — a best-legal WHOLE construction beats the lineup
    decisively in the worlds where the lineup performs best (its
    home worlds). Challengers are re-solved constructions — the
    sacrifice taken wherever the thesis says, across teams and
    positions — never one-swap mutations of the lineup;
  * fidelity — the delivered set does not do what the user asked
    (his hard settings verified against the delivered counts; his
    chosen story's share and fit).

Disposition (Chris's decision, Oct 28): a dominated lineup is
REPLACED by its challenger by default when the replacement is legal
and preserves his settings; it is flagged by name with the numbers
when replacement would break a setting, duplicate another lineup,
or the shape is his own construction. His settings outrank the audit
everywhere, and every audit action taken against the pull of a user
setting is named. The audit budget is fixed and always-on (Tier 2:
at most 6 lineups, 20 s wall) — no toggle.

Failure contract: unlike the learning capture (measurement, fails
silent by design), the audit changes what is delivered, so it fails
LOUD. Missing sim worlds produce an "Audit: not run" trust line with
the reason; a Tier-2 budget expiry names the lineups that were not
deep-audited. It never produces a clean-looking set by default.

Spec: ~/workspace/aytia-web/hidden_files/trust-release-spec.md
"""

import time
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from dfs_lab.showdown import (
    _showdown_exposure_targets,
    _showdown_linkage_breaches,
    _showdown_no_role_players,
    _showdown_qb_linkage_exemptions,
    _showdown_team_qb_name,
    add_showdown_ratings,
    audit_showdown_portfolio,
    captain_pool_ids,
    showdown_aggression,
    showdown_align_worlds,
    showdown_candidate_row,
    showdown_lineup_totals,
    showdown_lineup_world_metrics,
)

SLOT_ID_COLS = ["CPT_ID"] + [f"FLEX{j}_ID" for j in range(1, 6)]
SLOT_NAME_COLS = ["CPT"] + [f"FLEX{j}" for j in range(1, 6)]

#: Tier-2 deep-audit budget (spec §2.2): absolute, set-size
#: independent, always on. Named constants, not hidden behavior.
AUDIT_TIER2_CAP = 6
AUDIT_TIER2_BUDGET_S = 20.0

#: Domination decisiveness (spec §2.2, fixed in advance from the
#: Oct 28 diagnostic's method): the challenger must beat the lineup
#: in at least this share of its home worlds, by at least this mean
#: margin, without being a portfolio downgrade on overall sim mean.
DOM_WIN_SHARE = 0.60
DOM_MARGIN = 2.0
DOM_MEAN_TOLERANCE = 1.0

#: C9 story-coherence thresholds: two load-bearing members whose
#: top-quartile performances co-occur at less than this fraction of
#: the independent rate are in contradictory requirements (they spike
#: in different worlds); below the severe line the lineup is also
#: queued for the deep audit. Calibrated conservatively — a strange
#: lineup with ONE coherent story must never trip these.
C9_LIFT_FLAG = 0.50
C9_LIFT_SEVERE = 0.35
C9_LOAD_SHARE = 0.10


def audit_not_run(reason, n=0):
    """The named not-run audit object (§2.5): the build is still
    served, but the trust line says the set was never argued with."""
    return {
        "ran": False,
        "reason": reason,
        "line": f"Audit: not run — {reason} (lineups unaudited)",
        "counts": {"lineups": int(n), "replaced": 0, "flagged": 0,
                   "penalized": 0, "overrides": 0, "unaudited": int(n),
                   "clean": 0},
        "details": [f"Audit not run: {reason}."],
        "per_lineup": [],
        "set_notes": [],
        "warnings": [f"Audit: not run — {reason}"],
        "wall_s": 0.0,
    }


def _ids_of(row):
    return [str(row[c]) for c in SLOT_ID_COLS if str(row.get(c, ""))]


def _shape_text(df_by_id, cpt_id, ids):
    names = [df_by_id[p]["Name"] for p in ids]
    cpt_name = df_by_id[cpt_id]["Name"]
    flex = [nm for p, nm in zip(ids, names) if p != cpt_id]
    return f"CPT {cpt_name} + " + ", ".join(flex)


def _story_fit(members, cpt_id, ids):
    """The API's Story Fit formula, restated for audit use: for core
    C with k members and m of them in the lineup: m == k -> 100 when
    the captain is in C else 94; else round(90 * m / k)."""
    if not members:
        return None
    member_set = set(members)
    m = sum(1 for p in ids if p in member_set)
    k = len(members)
    if m == k:
        return 100 if cpt_id in member_set else 94
    return round(90 * m / k)


def _world_families(W, d):
    """Classify every simulated world into a script family from the
    world's own aggregates (team skill totals, pass/run shares,
    scoring environment). Returns (family per world, env terciles) —
    the decomposition C9 uses to ask whether a lineup's best worlds
    tell one story or two opposed ones. Family names mirror
    GAME_WORLDS': shootout / pass_ceiling / rb_control / comeback /
    low / balanced."""
    teams = [str(t) for t in d["Team"].dropna().unique() if str(t)]
    if len(teams) != 2 or W.shape[0] == 0:
        return np.array(["balanced"] * W.shape[0]), None
    team_arr = d["Team"].astype(str).to_numpy()
    skill = ~(d["is_DST"].to_numpy(bool) | d["is_K"].to_numpy(bool)) \
        if "is_DST" in d.columns else np.ones(len(d), bool)
    pass_mask = ((d["is_QB"] | d["is_passcatcher"]).to_numpy(bool)
                 if "is_QB" in d.columns else np.zeros(len(d), bool))
    rb_mask = d["is_RB"].to_numpy(bool) if "is_RB" in d.columns \
        else np.zeros(len(d), bool)
    env = W.sum(axis=1)
    lo, hi = np.percentile(env, [33.3, 66.7])
    totals, passes, runs = {}, {}, {}
    for t in teams:
        m = (team_arr == t) & skill
        totals[t] = W[:, m].sum(axis=1)
        passes[t] = W[:, m & pass_mask].sum(axis=1)
        runs[t] = W[:, m & rb_mask].sum(axis=1)
    a, b = teams
    fams = np.full(W.shape[0], "balanced", dtype=object)
    pass_share = {t: np.divide(passes[t], np.maximum(totals[t], 1e-9))
                  for t in teams}
    rb_share = {t: np.divide(runs[t], np.maximum(totals[t], 1e-9))
                for t in teams}
    lead = np.where(totals[a] >= totals[b], a, b)
    trail = np.where(totals[a] >= totals[b], b, a)
    for w in range(W.shape[0]):
        if env[w] <= lo:
            fams[w] = "low"
        elif env[w] >= hi:
            fams[w] = "shootout"
        elif pass_share[lead[w]][w] >= 0.62:
            fams[w] = "pass_ceiling"
        elif rb_share[lead[w]][w] >= 0.42:
            fams[w] = "rb_control"
        elif pass_share[trail[w]][w] >= 0.55:
            fams[w] = "comeback"
    return fams, (float(lo), float(hi))


#: Script families that cannot both be the story of one lineup.
_FAMILY_OPPOSED = {
    "shootout": {"low"},
    "low": {"shootout", "pass_ceiling"},
    "pass_ceiling": {"low", "rb_control"},
    "rb_control": {"pass_ceiling"},
    "comeback": {"low"},
    "balanced": set(),
}

_SCRIPT_FAMILY = {
    "Shootout": "shootout",
    "Pass-heavy shootout": "pass_ceiling",
    "Low-scoring game": "low",
    "Defensive / field-goal battle": "low",
    "Ground-and-pound": "rb_control",
    "Team plays from ahead": "rb_control",
    "Team passing comeback": "comeback",
}


def audit_showdown_set(df, strategy_map, result, worlds_df=None,
                       sim_frame=None, settings=None,
                       challengers=None, budget_s=AUDIT_TIER2_BUDGET_S,
                       tier2_cap=AUDIT_TIER2_CAP):
    """Audit one delivered showdown set. Returns (result, audit):
    the delivered frame — dominated lineups replaced where legal,
    survivors carrying their flags in an "Audit Flag" column — and
    the audit object (line / counts / details / per-lineup verdicts)
    the API puts on the build payload and into the capture.

    df is the build frame (post-scenario pool); worlds_df + sim_frame
    are the build's own sim bundle (the worlds the Lab showed).
    settings carries the build's rules (salary band, captain rules,
    min_unique, script/story/cores, contest shape). challengers is
    the build's undelivered candidate pool (DataFrame or (cpt, ids)
    pairs) feeding Tier 1."""
    t_start = time.monotonic()
    s = dict(settings or {})
    min_salary = int(s.get("min_salary", 47500))
    max_salary = int(s.get("max_salary", 50000))
    min_unique = int(s.get("min_unique", 1))
    if result is None or len(result) == 0:
        obj = audit_not_run("no lineups to audit", 0)
        obj["ran"] = True
        obj["line"] = "Audit: 0 lineups · 0 replaced · 0 flagged · 0 unaudited"
        return result, obj
    if worlds_df is None or sim_frame is None:
        return result, audit_not_run("sim unavailable", len(result))
    d = df.reset_index(drop=True)
    W = showdown_align_worlds(worlds_df, sim_frame, d)
    if W is None or W.shape[0] == 0:
        return result, audit_not_run("sim unavailable", len(result))

    result = result.reset_index(drop=True).copy()
    n = len(result)
    smap = strategy_map or {}
    by_id = {str(r["ID"]): r for _, r in d.iterrows()}
    pos_of = {pid: i for i, pid in
              enumerate(str(x) for x in d["ID"])}
    name_of = {pid: str(r["Name"]) for pid, r in by_id.items()}
    exempt = _showdown_qb_linkage_exemptions(d, smap)
    no_role = _showdown_no_role_players(d, smap)
    locked_ids = {pid for pid, st in smap.items()
                  if isinstance(st, dict)
                  and (bool(st.get("Lock")) or bool(st.get("CPT Lock")))
                  and not bool(st.get("Exclude"))}

    # ---- per-lineup structural profiles -----------------------------
    profiles = []
    for _, row in result.iterrows():
        ids = _ids_of(row)
        cpt_id = str(row["CPT_ID"])
        per_team = {}
        for pid in ids:
            r = by_id.get(pid)
            if r is None:
                continue
            cell = per_team.setdefault(str(r["Team"]),
                                       {"pc": 0, "qb": False})
            if bool(r["is_passcatcher"]):
                cell["pc"] += 1
            if bool(r["is_QB"]):
                cell["qb"] = True
        profiles.append({
            "ids": ids, "cpt_id": cpt_id, "per_team": per_team,
            "core": (str(row["Core"]) if "Core" in row
                     and not pd.isna(row.get("Core")) else None),
            "fit": (row.get("Story Fit") if "Story Fit" in row
                    else None),
            "game_world": (str(row["Game World"])
                           if "Game World" in row else None),
        })

    # Cores by label (thesis classes + C8): the story's own core is
    # labelled with the story name in the build's cores plan.
    cores_by_name = {}
    for c in s.get("cores") or []:
        if isinstance(c, dict) and c.get("name"):
            cores_by_name[str(c["name"])] = [
                str(m) for m in (c.get("members") or [])]
    story = s.get("story") if isinstance(s.get("story"), dict) else None
    story_core_ids = []
    if story:
        story_core_ids = [str(m) for m in (story.get("core") or [])]
        if story.get("name"):
            cores_by_name.setdefault(str(story["name"]), story_core_ids)

    def _core_ids_for(profile):
        if profile["core"] and profile["core"] in cores_by_name:
            return cores_by_name[profile["core"]]
        return []

    # ---- coherence inventory (per lineup) ---------------------------
    # checks[i] = [{id, kind: penalty|flag|override, text}]
    checks = [[] for _ in range(n)]
    deep_queue_flags = [False] * n  # C1/C2/C3/C8/severe-C9 -> Tier 2

    shootoutish = False
    _scripts = [str(s.get("script") or "")]
    if story:
        _scripts.append(str(story.get("script") or ""))
    _totals = [float(v) for v in (s.get("team_scores") or {}).values()]
    if story and story.get("team_scores"):
        _totals += [float(v) for v in story["team_scores"].values()]
    if any(x in ("Shootout", "Pass-heavy shootout") for x in _scripts) \
            or (_totals and sum(_totals) >= 50):
        shootoutish = True
    slugfest = any(x in ("Low-scoring game",
                         "Defensive / field-goal battle")
                   for x in _scripts)

    for i, p in enumerate(profiles):
        ids, cpt_id = p["ids"], p["cpt_id"]
        # C1 — linkage hard case: the guard owns it; the audit
        # verifies the delivered set and names any exempt delivery.
        for team, n_pc, qb_name in _showdown_linkage_breaches(
                d, ids, skip_teams=set(exempt)):
            checks[i].append({
                "id": "C1", "kind": "flag",
                "text": f"{n_pc} {team} pass catchers without "
                        f"{qb_name or 'their quarterback'} — the "
                        f"QB-linkage guard's shape, delivered anyway"})
            deep_queue_flags[i] = True
        for team, n_pc, qb_name in _showdown_linkage_breaches(d, ids):
            if team in exempt and n_pc >= 3:
                checks[i].append({
                    "id": "C1", "kind": "override",
                    "text": f"{n_pc} {team} pass catchers without "
                            f"{qb_name or 'their quarterback'} — "
                            f"delivered because {exempt[team]}"})
        # C2 — linkage soft case: exactly 2 catchers, no QB.
        for team, cell in sorted(p["per_team"].items()):
            if cell["pc"] == 2 and not cell["qb"]:
                qb_name = _showdown_team_qb_name(d, team)
                checks[i].append({
                    "id": "C2", "kind": "penalty",
                    "text": f"2 {team} pass catchers without "
                            f"{qb_name or 'their quarterback'}"})
                deep_queue_flags[i] = True
        cpt_row = by_id.get(cpt_id)
        # C3 — reverse linkage: a bare QB flex with no stack partner.
        for pid in ids:
            if pid == cpt_id:
                continue
            r = by_id.get(pid)
            if r is None or not bool(r["is_QB"]):
                continue
            team = str(r["Team"])
            if p["per_team"].get(team, {}).get("pc", 0) == 0:
                checks[i].append({
                    "id": "C3", "kind": "penalty",
                    "text": f"{name_of[pid]} rostered with no "
                            f"{team} pass catcher — a bare QB flex "
                            f"needs a rushing/TD path"})
                deep_queue_flags[i] = True
        # C4 — captain coherence (the captain-slot sense checks,
        # surfaced per lineup instead of buried in a flags string).
        if cpt_row is not None:
            cteam = str(cpt_row["Team"])
            cell = p["per_team"].get(cteam, {"pc": 0, "qb": False})
            if bool(cpt_row["is_QB"]) and cell["pc"] == 0:
                checks[i].append({
                    "id": "C4", "kind": "penalty",
                    "text": f"QB captain {name_of[cpt_id]} with no "
                            f"{cteam} pass catcher"})
                deep_queue_flags[i] = True
            elif bool(cpt_row["is_passcatcher"]) and not cell["qb"]:
                checks[i].append({
                    "id": "C4", "kind": "penalty",
                    "text": f"Captain {name_of[cpt_id]} without his "
                            f"quarterback"})
                deep_queue_flags[i] = True
        # C5 — DST/K vs the game script.
        n_dst = sum(1 for pid in ids
                    if pid in by_id and bool(by_id[pid]["is_DST"]))
        n_k = sum(1 for pid in ids
                  if pid in by_id and bool(by_id[pid]["is_K"]))
        if shootoutish and (n_dst >= 2 or n_k >= 2):
            checks[i].append({
                "id": "C5", "kind": "penalty",
                "text": "both defenses / two kickers in a "
                        "shootout-script build — the script and the "
                        "construction argue with each other"})
        # C6 — punt-role reality, re-verified on the delivered set.
        for pid in ids:
            if pid in no_role:
                checks[i].append({
                    "id": "C6", "kind": "flag",
                    "text": f"{no_role[pid]['name']} has no measurable "
                            f"role (punt floor breached)"})
        # C8 — story fidelity inside a story batch.
        if p["core"] and p["core"] != "Free" and p["fit"] is not None \
                and not pd.isna(p["fit"]) and float(p["fit"]) < 60:
            checks[i].append({
                "id": "C8", "kind": "penalty",
                "text": f"in the {p['core']} batch but Story Fit "
                        f"{float(p['fit']):.0f} — the lineup does "
                        f"not carry its own story's core"})
            deep_queue_flags[i] = True

    # ---- world totals for the delivered set --------------------------
    totals = showdown_lineup_totals(W, d, result)
    fams, _env = _world_families(W, d)

    # ---- C9 — generalized story coherence ----------------------------
    # For every lineup: do the performances it NEEDS come from one
    # coherent script, or does it need mutually exclusive things at
    # once? Two instruments, both read off the sim itself:
    #  (a) required-performance contradiction — among the lineup's
    #      load-bearing members (the players its best worlds lean
    #      on), a pair whose top-quartile games co-occur at less
    #      than half the independent rate spike in different worlds;
    #  (b) story-family mismatch — a story-batch lineup whose home
    #      worlds belong mostly to a family OPPOSED to the story it
    #      was built for.
    p75 = np.percentile(W, 75, axis=0)
    story_family = None
    if story:
        story_family = _SCRIPT_FAMILY.get(str(story.get("script") or ""))
    for i, p in enumerate(profiles):
        ids = p["ids"]
        home = totals[i] >= np.percentile(totals[i], 90)
        if home.sum() == 0:
            continue
        cols = [pos_of[pid] for pid in ids if pid in pos_of]
        if len(cols) != 6:
            continue
        contrib = np.array([
            (1.5 if pid == p["cpt_id"] else 1.0)
            * W[home][:, pos_of[pid]].mean() for pid in ids])
        shares = contrib / max(contrib.sum(), 1e-9)
        load = [(ids[k], shares[k]) for k in range(len(ids))
                if shares[k] >= C9_LOAD_SHARE]
        worst = None
        for x in range(len(load)):
            for y in range(x + 1, len(load)):
                pa, pb = pos_of[load[x][0]], pos_of[load[y][0]]
                joint = float(((W[:, pa] >= p75[pa])
                               & (W[:, pb] >= p75[pb])).mean())
                lift = joint / 0.0625
                if worst is None or lift < worst[0]:
                    worst = (lift, load[x][0], load[y][0], joint)
        fam_shares = {}
        home_fams = fams[home]
        for f in set(home_fams.tolist()):
            fam_shares[f] = float((home_fams == f).mean())
        p["family_shares"] = fam_shares
        if worst is not None and worst[0] < C9_LIFT_FLAG:
            lift, pa_id, pb_id, joint = worst
            fam_txt = ""
            top_fams = sorted(fam_shares.items(),
                              key=lambda kv: -kv[1])[:2]
            if len(top_fams) == 2:
                fam_txt = (f" Its best worlds split between "
                           f"{top_fams[0][0]} ({top_fams[0][1]:.0%}) "
                           f"and {top_fams[1][0]} "
                           f"({top_fams[1][1]:.0%}).")
            checks[i].append({
                "id": "C9", "kind": "penalty",
                "text": f"contradictory requirements: "
                        f"{name_of[pa_id]} and {name_of[pb_id]} "
                        f"rarely fire together — top-quartile games "
                        f"at the same time in {joint:.1%} of worlds "
                        f"vs 6.2% if they were independent, so the "
                        f"lineup needs two different games at once."
                        + fam_txt})
            if lift < C9_LIFT_SEVERE:
                deep_queue_flags[i] = True
        if story_family and p["core"] and p["core"] != "Free":
            opposed = _FAMILY_OPPOSED.get(story_family, set())
            opp_share = sum(v for f, v in fam_shares.items()
                            if f in opposed)
            own_share = fam_shares.get(story_family, 0.0)
            if opp_share >= 0.30 and opp_share > own_share:
                checks[i].append({
                    "id": "C9", "kind": "penalty",
                    "text": f"wins in the wrong story's worlds: its "
                            f"best worlds are {opp_share:.0%} "
                            f"{'/'.join(sorted(opposed & set(fam_shares)))} "
                            f"but it was built for a {story_family} "
                            f"story ({own_share:.0%})"})
                deep_queue_flags[i] = True


    # ---- set-level fidelity (C7 + C8 share + C5 portfolio) ----------
    set_notes = []
    if slugfest:
        n_dk_slots = sum(
            1 for p in profiles for pid in p["ids"]
            if pid in by_id and (bool(by_id[pid]["is_DST"])
                                 or bool(by_id[pid]["is_K"])))
        if n_dk_slots == 0:
            set_notes.append(
                "Story check: a slugfest script but the set carries "
                "no defense or kicker slots at all.")
    # C7 — the user's hard rules, verified against the delivered set.
    hard_violations = audit_showdown_portfolio(result, smap, name_of)
    _c7_seen = set()
    for v in hard_violations:
        set_notes.append(f"Settings check: {v}")
        # "Lineup N:" prefix -> flag that lineup (rank order == row
        # order here; the delivered frame arrives rank-sorted). One
        # C7 check per lineup, carrying the first violation's own
        # words; the rest stay in the set-level detail.
        try:
            rank_no = int(str(v).split(":", 1)[0].replace("Lineup", ""))
            if 1 <= rank_no <= n and rank_no not in _c7_seen:
                _c7_seen.add(rank_no)
                msg = str(v).split(":", 1)[1].strip() \
                    if ":" in str(v) else str(v)
                checks[rank_no - 1].append({
                    "id": "C7", "kind": "flag",
                    "text": f"violates one of your hard rules — {msg}"})
        except (ValueError, IndexError):
            pass
    targets, _tnotes = _showdown_exposure_targets(d, smap, n)
    counts = defaultdict(int)
    cpt_counts = defaultdict(int)
    for p in profiles:
        for pid in p["ids"]:
            counts[pid] += 1
        cpt_counts[p["cpt_id"]] += 1
    for pid, tgt in targets.items():
        if not tgt["usable"]:
            continue
        got, cgot = counts.get(pid, 0), cpt_counts.get(pid, 0)
        if got < tgt["min"] or got > tgt["max"] \
                or cgot < tgt["cpt_min"] or cgot > tgt["cpt_max"]:
            set_notes.append(
                f"Settings check: {tgt['name']} delivered {got} of "
                f"{n} lineups ({cgot} at captain) vs your targets "
                f"({tgt['min']}–{tgt['max']} lineups, "
                f"{tgt['cpt_min']}–{tgt['cpt_max']} captain).")
    qb_pc_rule = int(s.get("cpt_qb_passcatchers") or 0)
    if qb_pc_rule != 0:
        for i, p in enumerate(profiles):
            cpt_row = by_id.get(p["cpt_id"])
            if cpt_row is None or not bool(cpt_row["is_QB"]):
                continue
            team = str(cpt_row["Team"])
            n_flex_pc = sum(
                1 for pid in p["ids"] if pid != p["cpt_id"]
                and pid in by_id and str(by_id[pid]["Team"]) == team
                and bool(by_id[pid]["is_passcatcher"]))
            bad = (n_flex_pc < qb_pc_rule) if qb_pc_rule > 0 \
                else (n_flex_pc > abs(qb_pc_rule))
            if bad:
                checks[i].append({
                    "id": "C7", "kind": "flag",
                    "text": f"QB captain with {n_flex_pc} {team} pass "
                            f"catchers — your rule asks "
                            f"{qb_pc_rule:+d}"})
    _wrte_raw = s.get("wrte_cpt_qb_pair_pct")
    wrte_pct = float(_wrte_raw) if _wrte_raw is not None else 80.0
    wrte_rows = [(i, p) for i, p in enumerate(profiles)
                 if p["cpt_id"] in by_id
                 and bool(by_id[p["cpt_id"]]["is_passcatcher"])]
    if wrte_rows:
        paired = 0
        for i, p in wrte_rows:
            team = str(by_id[p["cpt_id"]]["Team"])
            has_qb = p["per_team"].get(team, {}).get("qb", False)
            if has_qb:
                paired += 1
            elif wrte_pct >= 99.5:
                checks[i].append({
                    "id": "C7", "kind": "flag",
                    "text": "WR/TE captain without his QB — your "
                            "pairing rule is set to Always"})
        share = 100.0 * paired / len(wrte_rows)
        if wrte_pct < 99.5 and share < wrte_pct - 100.0 / max(n, 1):
            set_notes.append(
                f"Settings check: WR/TE captains carry their QB in "
                f"{share:.0f}% of lineups vs your {wrte_pct:.0f}% "
                f"pairing target.")
    if min_unique > 0 and n >= 2:
        sets = [set(p["ids"]) for p in profiles]
        realized = min(6 - len(sets[a] & sets[b])
                       for a in range(n) for b in range(a + 1, n))
        if realized < min_unique:
            set_notes.append(
                f"Uniqueness check: you asked for at least "
                f"{min_unique} different players between lineups; "
                f"the delivered set's closest pair differs by "
                f"{realized}.")
    else:
        realized = min_unique
    # C8 set-level — the story's realized share vs its plan/target.
    # The share target binds only when the story's core was actually
    # engaged in the build (an active core in the plan). A story
    # chosen with its cores switched off drives the script only —
    # the audit must not invent a promise the build never made.
    if story and story_core_ids:
        realized_core = sum(
            1 for p in profiles
            if set(story_core_ids).issubset(set(p["ids"])))
        expected = None
        _cores_setting = s.get("cores")
        if _cores_setting is None:
            pass  # no plan supplied -> share fallback below
        else:
            expected = -1  # plan supplied; -1 = story core not engaged
            for c in _cores_setting:
                if isinstance(c, dict) and str(c.get("name")) == str(
                        story.get("name")):
                    try:
                        expected = int(c.get("lineups") or 0)
                    except (TypeError, ValueError):
                        expected = -1
        if expected == -1:
            expected = realized_core  # not engaged: nothing to verify
        if expected is None:
            if n == 1:
                share_t = 1.0
            elif str(s.get("entry_format")) == "Single Entry":
                share_t = 0.75
            else:
                share_t = 0.60
            expected = round(share_t * n)
        if abs(realized_core - expected) > 1:
            set_notes.append(
                f"Story check: the {story.get('name')} core landed "
                f"in {realized_core} of {n} lineups vs the "
                f"{expected} its contest share calls for.")

    # ---- challenger pool ----------------------------------------------
    chal_pairs = []
    seen_keys = set()
    delivered_keys = {tuple(sorted(p["ids"])) for p in profiles}
    for cpt, ids in _challenger_pairs(challengers):
        key = tuple(sorted(ids))
        if key in delivered_keys or key in seen_keys or len(set(ids)) != 6:
            continue
        seen_keys.add(key)
        sal = _salary_of(by_id, cpt, ids)
        if sal is None or sal < min_salary or sal > max_salary:
            continue
        chal_pairs.append((cpt, tuple(ids)))
    chal_totals = showdown_lineup_totals(W, d, chal_pairs) \
        if chal_pairs else np.zeros((0, W.shape[0]))

    def _home_mask(i, extra_qb_ids=()):
        tot = totals[i]
        mask = tot >= np.percentile(tot, 90)
        for qb_id in extra_qb_ids:
            if qb_id in pos_of:
                col = W[:, pos_of[qb_id]]
                mask = mask | (col >= np.percentile(col, 75))
        return mask

    def _linkage_qb_ids(p):
        out = []
        for team, cell in p["per_team"].items():
            if cell["pc"] >= 2 and not cell["qb"]:
                qbs = d[d["Team"].astype(str).eq(team) & d["is_QB"]]
                if len(qbs):
                    qbs = qbs.sort_values("My Proj", ascending=False,
                                          kind="mergesort")
                    out.append(str(qbs.iloc[0]["ID"]))
        return out

    def _decisive(chal_tot, i, mask):
        diff = chal_tot[mask] - totals[i][mask]
        win = float((diff > 0).mean()) if mask.any() else 0.0
        margin = float(diff.mean()) if mask.any() else 0.0
        mean_diff = float(chal_tot.mean() - totals[i].mean())
        return (win >= DOM_WIN_SHARE and margin >= DOM_MARGIN
                and mean_diff >= -DOM_MEAN_TOLERANCE), win, margin, mean_diff

    def _thesis_ok(p, cpt, ids):
        if cpt == p["cpt_id"]:
            return True
        core = _core_ids_for(p)
        return bool(core) and set(core).issubset(set(ids))

    # ---- Tier 1: pool challengers, vectorized --------------------------
    dominated = {}   # i -> {cpt, ids, win, margin, mean_diff, source}
    for i, p in enumerate(profiles):
        if not len(chal_pairs):
            break
        mask = _home_mask(i, _linkage_qb_ids(p))
        best = None
        for j, (cpt, ids) in enumerate(chal_pairs):
            if not _thesis_ok(p, cpt, ids):
                continue
            ok, win, margin, md = _decisive(chal_totals[j], i, mask)
            if ok and (best is None or (win, margin) > (best[1], best[2])):
                best = (j, win, margin, md)
        if best is not None:
            j, win, margin, md = best
            cpt, ids = chal_pairs[j]
            dominated[i] = {"cpt": cpt, "ids": list(ids), "win": win,
                            "margin": margin, "mean_diff": md,
                            "source": "tier1"}

    # ---- Tier 2: re-solved challengers for the flagged few -------------
    rating_key = []
    for i in range(n):
        row = result.iloc[i]
        rating_key.append(float(row.get("Rating Score", 0) or 0))
    queue = [i for i in range(n)
             if i in dominated or deep_queue_flags[i]]
    queue.sort(key=lambda i: -rating_key[i])
    queue = queue[:tier2_cap] if tier2_cap else []
    not_deep = [i for i in range(n)
                if (i in dominated or deep_queue_flags[i])
                and i not in queue]
    t2_start = time.monotonic()
    for i in queue:
        if time.monotonic() - t2_start > budget_s:
            not_deep.append(i)
            continue
        p = profiles[i]
        mask = _home_mask(i, _linkage_qb_ids(p))
        chal = _solve_challenger(
            d, W, mask, smap, p, settings=s, exempt=set(exempt),
            no_role=set(no_role), by_id=by_id,
            core_ids=_core_ids_for(p),
            captain_free=any(c["id"] == "C4" for c in checks[i]),
            time_left=max(0.5, budget_s - (time.monotonic() - t2_start)))
        if chal is None:
            continue
        cpt, ids = chal
        if tuple(sorted(ids)) == tuple(sorted(p["ids"])):
            continue
        ct = showdown_lineup_totals(W, d, [(cpt, tuple(ids))])[0]
        ok, win, margin, md = _decisive(ct, i, mask)
        if ok:
            dominated[i] = {"cpt": cpt, "ids": list(ids), "win": win,
                            "margin": margin, "mean_diff": md,
                            "source": "tier2"}
    deep_checked = len([i for i in queue if i not in not_deep])

    # ---- disposition: replace by default, flag with the reason ----
    verdicts = [None] * n
    details = []
    cur_ids = [list(p["ids"]) for p in profiles]
    cur_cpt = [p["cpt_id"] for p in profiles]
    eff_mu = min(min_unique, realized) if min_unique > 0 else 0
    replaced_rows = {}
    order = sorted(dominated, key=lambda i: -rating_key[i])
    for i in order:
        dom = dominated[i]
        p = profiles[i]
        ch_set = set(dom["ids"])
        reason = None
        own_construction = False
        # His own construction: an exempt shape, or the challenger
        # removes a player he locked. His settings win; named — and
        # counted as an override, never as a catch.
        if any(c["kind"] == "override" for c in checks[i]):
            reason = "the shape is your own construction (your " \
                     "settings outrank the audit)"
            own_construction = True
        else:
            removed_locked = [pid for pid in p["ids"]
                              if pid not in ch_set and pid in locked_ids]
            if removed_locked:
                reason = (f"replacing it would remove your locked "
                          f"{name_of[removed_locked[0]]}")
        if reason is None and any(
                tuple(sorted(cur_ids[k])) == tuple(sorted(dom["ids"]))
                for k in range(n) if k != i):
            reason = "the challenger duplicates another delivered lineup"
        if reason is None and eff_mu > 0:
            for k in range(n):
                if k == i:
                    continue
                if 6 - len(ch_set & set(cur_ids[k])) < eff_mu:
                    reason = ("the challenger would break the "
                              "portfolio's uniqueness rule")
                    break
        if reason is None:
            for pid in p["ids"]:
                tgt = targets.get(pid)
                if tgt is not None and counts[pid] - 1 < tgt["min"]:
                    reason = (f"replacing it breaks your "
                              f"{tgt['name']} Min Exposure")
                    break
            if reason is None and targets.get(p["cpt_id"]) is not None \
                    and cpt_counts[p["cpt_id"]] - 1 \
                    < targets[p["cpt_id"]]["cpt_min"]:
                reason = (f"replacing it breaks your "
                          f"{targets[p['cpt_id']]['name']} CPT Min")
        if reason is None:
            for pid in dom["ids"]:
                tgt = targets.get(pid)
                if tgt is not None and counts.get(pid, 0) + 1 > tgt["max"]:
                    reason = (f"the challenger breaks your "
                              f"{tgt['name']} Max Exposure")
                    break
            if reason is None and targets.get(dom["cpt"]) is not None \
                    and cpt_counts.get(dom["cpt"], 0) + 1 \
                    > targets[dom["cpt"]]["cpt_max"]:
                reason = (f"the challenger breaks your "
                          f"{targets[dom['cpt']]['name']} CPT Max")
        if reason is None:
            core = _core_ids_for(p)
            if core:
                fit = _story_fit(core, dom["cpt"], list(dom["ids"]))
                if fit is not None and fit < 60:
                    reason = ("the challenger falls out of this "
                              "lineup's story core (Story Fit "
                              f"{fit} < 60)")
        if reason is None:
            # The replacement may not itself be dominated by another
            # pool challenger in ITS home worlds (one pass, no chains).
            ct = showdown_lineup_totals(W, d, [(dom["cpt"],
                                                tuple(dom["ids"]))])[0]
            home = ct >= np.percentile(ct, 90)
            for j, (cpt2, ids2) in enumerate(chal_pairs):
                if tuple(sorted(ids2)) == tuple(sorted(dom["ids"])):
                    continue
                if not _thesis_ok({"cpt_id": dom["cpt"],
                                   "core": p["core"]}, cpt2, ids2):
                    continue
                diff = chal_totals[j][home] - ct[home]
                if float((diff > 0).mean()) >= DOM_WIN_SHARE \
                        and float(diff.mean()) >= DOM_MARGIN \
                        and float(chal_totals[j].mean() - ct.mean()) \
                        >= -DOM_MEAN_TOLERANCE:
                    reason = ("the challenger is itself beaten in "
                              "its own best worlds by another "
                              "candidate, so the set keeps the "
                              "original and flags it instead")
                    break
        if reason is not None:
            verdicts[i] = "override" if own_construction else "flagged"
            label = "kept" if own_construction else "flagged"
            details.append(
                f"Lineup {i + 1} {label}: beaten in its best worlds "
                f"by {_shape_text(by_id, dom['cpt'], dom['ids'])} "
                f"({dom['win']:.0%} of its best worlds · "
                f"+{dom['margin']:.1f}) — kept because {reason}.")
            continue
        # Build the replacement row; a coherence hard-reject at this
        # last gate (e.g. a DST conflict the solve did not model)
        # falls back to a flag — never a silent swap.
        flex_sorted = sorted((pid for pid in dom["ids"]
                              if pid != dom["cpt"]),
                             key=lambda pid: -float(
                                 by_id[pid]["FlexSalary"]))
        chosen = [("CPT", pos_of[dom["cpt"]])] + [
            (f"FLEX{j}", pos_of[pid])
            for j, pid in enumerate(flex_sorted, start=1)]
        new_row = showdown_candidate_row(
            d, chosen, smap, str(s.get("script") or "Neutral"),
            str(s.get("script_team") or ""),
            game_world=p["game_world"], exempt_teams=exempt)
        if new_row is None:
            verdicts[i] = "flagged"
            details.append(
                f"Lineup {i + 1} flagged: beaten in its best worlds "
                f"by {_shape_text(by_id, dom['cpt'], dom['ids'])} "
                f"({dom['win']:.0%} · +{dom['margin']:.1f}) — kept "
                f"because the challenger fails the coherence check "
                f"at delivery.")
            continue
        if p["core"]:
            new_row["Core"] = p["core"]
            core = _core_ids_for(p)
            new_row["Story Fit"] = _story_fit(
                core, dom["cpt"], list(dom["ids"])) if core else p["fit"]
        replaced_rows[i] = new_row
        for pid in p["ids"]:
            counts[pid] -= 1
        cpt_counts[p["cpt_id"]] -= 1
        for pid in dom["ids"]:
            counts[pid] += 1
        cpt_counts[dom["cpt"]] += 1
        cur_ids[i] = list(dom["ids"])
        cur_cpt[i] = dom["cpt"]
        verdicts[i] = "replaced"
        details.append(
            f"Lineup {i + 1} replaced: "
            f"{_shape_text(by_id, p['cpt_id'], p['ids'])} → "
            f"{_shape_text(by_id, dom['cpt'], dom['ids'])} — the "
            f"challenger wins {dom['win']:.0%} of the original's "
            f"best worlds by +{dom['margin']:.1f} on average.")

    # Surviving lineups: verdicts from the coherence inventory.
    for i in range(n):
        if verdicts[i] is not None:
            continue
        kinds = {c["kind"] for c in checks[i]}
        if "flag" in kinds:
            verdicts[i] = "flagged"
        elif "override" in kinds:
            verdicts[i] = "override"
        elif "penalty" in kinds:
            verdicts[i] = "penalized"
        else:
            verdicts[i] = "clean"
        for c in checks[i]:
            if c["kind"] == "override":
                details.append(
                    f"Lineup {i + 1}: {c['text']} ({c['id']}) — "
                    f"your settings outrank the audit.")
            else:
                details.append(f"Lineup {i + 1}: {c['text']} ({c['id']}).")
    details.extend(set_notes)
    if not_deep:
        names_txt = ", ".join(str(i + 1) for i in sorted(set(not_deep)))
        details.append(
            f"Lineups {names_txt} were not deep-audited — the "
            f"audit's fixed budget bound first; their Tier 1 "
            f"verdicts stand.")

    # ---- assemble the final frame -----------------------------------
    rows_out = []
    for i in range(n):
        if i in replaced_rows:
            rows_out.append(replaced_rows[i])
        else:
            rows_out.append(result.iloc[i].to_dict())
    final = pd.DataFrame(rows_out).reset_index(drop=True)
    # World metrics over the delivered set + challenger pool (the
    # same pool the delivered rows were scored against in Piece 1),
    # then the world-anchored rating.
    pool_pairs = [(cur_cpt[i], tuple(cur_ids[i])) for i in range(n)] \
        + chal_pairs
    metrics = showdown_lineup_world_metrics(W, d, pool_pairs)
    if len(metrics) == len(pool_pairs) and n > 0:
        for col in ("Sim Mean", "Ceiling P90", "Best-of-pool %",
                    "Top-decile %"):
            final[col] = metrics[col].to_numpy()[:n]
        aggr = showdown_aggression(
            int(s.get("field_size", 2000)),
            str(s.get("payout_style", "Top-heavy payouts")),
            str(s.get("entry_format", "Single Entry")))
        final = add_showdown_ratings(final, aggr,
                                     use_world_metrics=True)
        final = final.sort_values("Rating Score", ascending=False,
                                  kind="mergesort").reset_index(drop=True)
        final["Rank"] = range(1, len(final) + 1)
    flag_by_orig = {}
    for i in range(n):
        if verdicts[i] in ("flagged", "penalized"):
            first = next((c for c in checks[i]
                          if c["kind"] in ("flag", "penalty")), None)
            if i in dominated and verdicts[i] == "flagged":
                flag_by_orig[i] = "Flagged: beaten in its best worlds"
            elif first is not None:
                flag_by_orig[i] = f"Flagged: {first['text']}"
            else:
                flag_by_orig[i] = "Flagged"
        else:
            flag_by_orig[i] = ""
    # Map flags onto the (possibly re-sorted) final frame by player set.
    orig_sets = {tuple(sorted(cur_ids[i])): flag_by_orig[i]
                 for i in range(n)}
    if len(final):
        final["Audit Flag"] = [
            orig_sets.get(tuple(sorted(_ids_of(row))), "")
            for _, row in final.iterrows()]

    n_replaced = sum(1 for v in verdicts if v == "replaced")
    n_flagged = sum(1 for v in verdicts if v == "flagged")
    n_penalized = sum(1 for v in verdicts if v == "penalized")
    n_overrides = sum(1 for v in verdicts if v == "override")
    n_clean = sum(1 for v in verdicts if v == "clean")
    n_unaudited = len(set(not_deep))
    if n_unaudited:
        line = (f"Audit: partial — {n - n_unaudited} of {n} "
                f"deep-checked · {n_replaced} replaced · "
                f"{n_flagged} flagged")
    else:
        line = (f"Audit: {n} lineups · {n_replaced} replaced · "
                f"{n_flagged} flagged · 0 unaudited")
    if n_overrides:
        line += f" · {n_overrides} by your settings"
    per_lineup = []
    for i, p in enumerate(profiles):
        entry = {
            "lineup": i + 1,
            "captain": name_of.get(p["cpt_id"], p["cpt_id"]),
            "members": [name_of.get(pid, pid) for pid in p["ids"]],
            "verdict": verdicts[i],
            "checks": [c["id"] for c in checks[i]],
            "notes": [c["text"] for c in checks[i]],
            "deep_audited": i not in set(not_deep),
        }
        if i in dominated:
            dom = dominated[i]
            entry["domination"] = {
                "challenger": _shape_text(by_id, dom["cpt"],
                                          dom["ids"]),
                "win_share": round(dom["win"], 3),
                "margin": round(dom["margin"], 2),
                "source": dom["source"]}
        per_lineup.append(entry)
    audit_obj = {
        "ran": True,
        "line": line,
        "counts": {"lineups": n, "replaced": n_replaced,
                   "flagged": n_flagged, "penalized": n_penalized,
                   "overrides": n_overrides, "unaudited": n_unaudited,
                   "clean": n_clean},
        "details": details,
        "per_lineup": per_lineup,
        "set_notes": set_notes,
        "warnings": list(set_notes),
        "wall_s": round(time.monotonic() - t_start, 2),
        "tier2": {"eligible": len([i for i in range(n)
                                   if i in dominated
                                   or deep_queue_flags[i]]),
                  "deep_checked": deep_checked,
                  "budget_s": budget_s, "cap": tier2_cap},
    }
    return final, audit_obj


def _challenger_pairs(challengers):
    """Normalize the challenger pool to [(cpt_id, ids_tuple)]."""
    if challengers is None:
        return []
    if isinstance(challengers, pd.DataFrame):
        if "CPT_ID" not in challengers.columns:
            return []
        cols = SLOT_ID_COLS
        out = []
        for _, row in challengers.iterrows():
            ids = tuple(str(row[c]) for c in cols)
            out.append((ids[0], ids))
        return out
    out = []
    for item in challengers:
        if isinstance(item, dict):
            cpt = str(item["cpt_id"])
            ids = tuple(str(p) for p in item["player_ids"])
        else:
            cpt, ids = str(item[0]), tuple(str(p) for p in item[1])
        if cpt not in ids:
            ids = (cpt,) + ids
        out.append((cpt, ids))
    return out


def _salary_of(by_id, cpt_id, ids):
    try:
        total = int(by_id[cpt_id]["CaptainSalary"])
        total += sum(int(by_id[p]["FlexSalary"]) for p in ids
                     if p != cpt_id)
        return total
    except Exception:
        return None


def _solve_challenger(d, W, home_mask, smap, profile, settings, exempt,
                      no_role, by_id, core_ids, captain_free, time_left):
    """Re-solve the best-legal WHOLE construction against a lineup's
    home worlds (Tier 2): maximize the candidate's mean score over
    the home-world subset, under the build's legality — salary band,
    the user's excludes/locks/captain pool, the punt floor, the
    QB-linkage guard — with the lineup's thesis preserved (its core
    members stay; its captain stays unless the captain is itself the
    incoherence). The solver chooses the sacrifice anywhere the cap
    demands. Returns (cpt_id, ids) or None."""
    s = settings
    min_salary = int(s.get("min_salary", 47500))
    max_salary = int(s.get("max_salary", 50000))
    hv = W[home_mask].mean(axis=0) if home_mask.any() \
        else W.mean(axis=0)
    active_col = (d["ActiveForBuild"].to_numpy(bool)
                  if "ActiveForBuild" in d.columns
                  else np.ones(len(d), bool))

    def _excluded(pid):
        st = smap.get(str(pid), {}) or {}
        return bool(st.get("Exclude", False)) \
            or st.get("Priority") == "Exclude"

    usable = []
    for pos in range(len(d)):
        pid = str(d["ID"].iloc[pos])
        if not bool(active_col[pos]) or _excluded(pid) \
                or pid in no_role:
            continue
        usable.append(pos)
    if len(usable) < 6:
        return None
    sub = d.iloc[usable].reset_index(drop=True)
    n = len(sub)
    ids = [str(x) for x in sub["ID"]]
    pos_in_sub = {pid: i for i, pid in enumerate(ids)}
    flex_sal = pd.to_numeric(sub["FlexSalary"], errors="coerce") \
        .fillna(0).to_numpy(float)
    cpt_sal = pd.to_numeric(
        sub.get("CaptainSalary", sub["FlexSalary"] * 1.5),
        errors="coerce").fillna(0).to_numpy(float)
    pool = set(captain_pool_ids(d, smap))
    cpt_ok = np.array([pid in pool for pid in ids])
    if not cpt_ok.any():
        return None
    hv_sub = hv[usable]

    total = 2 * n
    rows, lows, highs = [], [], []
    rows.append({i: 1.0 for i in range(n)}); lows.append(5.0); highs.append(5.0)
    rows.append({n + i: 1.0 for i in range(n) if cpt_ok[i]})
    lows.append(1.0); highs.append(1.0)
    for i in range(n):
        rows.append({i: 1.0, n + i: 1.0}); lows.append(0.0); highs.append(1.0)
    rows.append({i: flex_sal[i] for i in range(n)}
                | {n + i: cpt_sal[i] for i in range(n)})
    lows.append(float(min_salary)); highs.append(float(max_salary))
    # The user's locks bind exactly as they bind generation.
    cpt_lock_id = None
    for i, pid in enumerate(ids):
        st = smap.get(pid, {}) or {}
        if bool(st.get("CPT Lock", False)) and not _excluded(pid):
            rows.append({n + i: 1.0}); lows.append(1.0); highs.append(1.0)
            cpt_lock_id = pid
        elif bool(st.get("Lock", False)) and not _excluded(pid):
            rows.append({i: 1.0, n + i: 1.0}); lows.append(1.0); highs.append(1.0)
    # Thesis: the story/core membership stays.
    for pid in core_ids or []:
        if pid in pos_in_sub:
            i = pos_in_sub[pid]
            rows.append({i: 1.0, n + i: 1.0}); lows.append(1.0); highs.append(1.0)
    # Thesis: the captain stays, unless the captain is the
    # incoherence (C4) or a CPT Lock names the captain instead.
    if cpt_lock_id is None and not captain_free:
        cpt_id = profile["cpt_id"]
        if cpt_id in pos_in_sub and cpt_ok[pos_in_sub[cpt_id]]:
            i = pos_in_sub[cpt_id]
            rows.append({n + i: 1.0}); lows.append(1.0); highs.append(1.0)
    teams = [str(t) for t in sub["Team"].dropna().unique() if str(t)]
    if len(teams) == 2:
        for t in teams:
            coeff = {}
            for i in np.where(sub["Team"].astype(str).eq(t).to_numpy())[0]:
                coeff[int(i)] = 1.0
                coeff[n + int(i)] = 1.0
            if coeff:
                rows.append(coeff); lows.append(1.0); highs.append(5.0)
    for mask_col, cap in (("is_K", s.get("max_k", 2)),
                          ("is_DST", s.get("max_dst", 1))):
        if cap is None or mask_col not in sub.columns:
            continue
        coeff = {}
        for i in np.where(sub[mask_col].to_numpy(bool))[0]:
            coeff[int(i)] = 1.0
            coeff[n + int(i)] = 1.0
        if coeff:
            rows.append(coeff); lows.append(0.0); highs.append(float(cap))
    if "is_passcatcher" in sub.columns and "is_QB" in sub.columns:
        for t in teams:
            if t in exempt:
                continue
            coeff = {}
            tm = sub["Team"].astype(str).eq(t).to_numpy()
            for i in np.where(tm & sub["is_passcatcher"].to_numpy(bool))[0]:
                coeff[int(i)] = coeff.get(int(i), 0.0) + 1.0
                coeff[n + int(i)] = coeff.get(n + int(i), 0.0) + 1.0
            for i in np.where(tm & sub["is_QB"].to_numpy(bool))[0]:
                coeff[int(i)] = coeff.get(int(i), 0.0) - 4.0
                coeff[n + int(i)] = coeff.get(n + int(i), 0.0) - 4.0
            if coeff:
                rows.append(coeff); lows.append(-np.inf); highs.append(2.0)
    A = lil_matrix((len(rows), total), dtype=float)
    for r, coeff in enumerate(rows):
        for col, val in coeff.items():
            A[r, col] = val
    cons = LinearConstraint(A.tocsr(), np.array(lows), np.array(highs))
    c = np.zeros(total)
    c[:n] = -hv_sub
    c[n:] = -1.5 * hv_sub
    # Captain variables outside the captain pool hard-zero (same
    # trap as the world-optimal solve: the captain-sum row spans
    # eligible columns only).
    ub = np.ones(total)
    ub[n:] = np.where(cpt_ok, 1.0, 0.0)
    try:
        res = milp(c=c, constraints=cons, integrality=np.ones(total),
                   bounds=Bounds(np.zeros(total), ub),
                   options={"time_limit": float(max(0.5, min(5.0, time_left))),
                            "presolve": True})
    except Exception:
        return None
    if res.status != 0 or res.x is None:
        return None
    x = res.x[:n] > 0.5
    cc = res.x[n:] > 0.5
    if x.sum() != 5 or cc.sum() != 1:
        return None
    chosen = [ids[i] for i in np.where(x | cc)[0]]
    return ids[int(np.where(cc)[0][0])], chosen
