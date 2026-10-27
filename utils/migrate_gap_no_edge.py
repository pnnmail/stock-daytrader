"""
One-off migration of signal history for the `gap_no_edge` outcome.

WHY. `_get_or_resolve_outcome` does not recompute records whose `outcome`
is already set (on purpose — otherwise every /stats would hit the network for the whole
history). So the new "the gap ate the R:R" check would apply only to FUTURE
signals, and already accumulated ones would keep an inflated outcome.

In §6.6 a similar case (`gap_invalidated`) was fixed by hand — there was exactly one
record. Here it is three out of four wins, and it is a recurring pattern, not
chance, hence a script.

WHAT IT DOES. For records with `entry_reanchored=true` it recomputes R:R from the
ACTUAL entry (`effective_entry_price`) and, if it is below MIN_ENTRY_RR,
rewrites the outcome to `gap_no_edge`. The check order is the same as in
`_resolve_signal_outcome`: first "opened beyond the stop" (that is already
`gap_invalidated`), then R:R.

Does not touch the network — all the needed numbers are already in the record.
Idempotent: a repeated run changes nothing.

    python -m utils.migrate_gap_no_edge --dry-run   # show only
    python -m utils.migrate_gap_no_edge             # apply (with a backup)
"""
import json
import shutil
import sys
from datetime import datetime

from config import SIGNAL_HISTORY_FILE
from utils.signals import MIN_ENTRY_RR

# Outcomes that could have been set incorrectly at all. `gap_invalidated`
# is not included — there the entry was beyond the stop, a stronger condition that
# is checked earlier and already worked correctly.
REVIEWABLE = ("target_1", "target_2", "stop", "none")


def actual_rr(record):
    """R:R from the actual entry. None if there is nothing to compute it from."""
    entry = record.get("effective_entry_price")
    stop = record.get("stop_loss")
    t1 = record.get("target_1")
    if entry is None or stop is None or t1 is None:
        return None
    risk = abs(entry - stop)
    if risk <= 0:
        return 0.0
    return abs(t1 - entry) / risk


def main():
    dry = "--dry-run" in sys.argv
    with open(SIGNAL_HISTORY_FILE, encoding="utf-8") as f:
        history = json.load(f)

    changes = []
    for ticker, records in history.items():
        for r in records:
            if not r.get("entry_reanchored"):
                continue
            if r.get("outcome") not in REVIEWABLE:
                continue
            rr = actual_rr(r)
            if rr is None or rr >= MIN_ENTRY_RR:
                continue
            changes.append((ticker, r, rr))

    if not changes:
        print("Nothing to migrate — no records with a failed R:R found.")
        return 0

    print(f"Records to change: {len(changes)} "
          f"(threshold MIN_ENTRY_RR = {MIN_ENTRY_RR:g})\n")
    print(f"{'ticker':<8}{'was':<12}{'R:R plan':<11}{'R:R actual':<11}"
          f"{'entry plan':<12}{'entry act.':<12}")
    print("-" * 68)
    for ticker, r, rr in changes:
        print(f"{ticker:<8}{r['outcome']:<12}{r.get('risk_reward_1', 0):<11.2f}"
              f"{rr:<11.2f}{r.get('entry_price', 0):<12.2f}"
              f"{r.get('effective_entry_price', 0):<12.2f}")

    if dry:
        print("\n--dry-run: nothing written.")
        return 0

    backup = f"{SIGNAL_HISTORY_FILE}.bak_{datetime.now():%Y%m%d_%H%M%S}"
    shutil.copy2(SIGNAL_HISTORY_FILE, backup)
    print(f"\nBackup: {backup}")

    for ticker, r, rr in changes:
        session = r.get("outcome_session") or ""
        r["outcome"] = "gap_no_edge"
        r["outcome_price"] = r.get("effective_entry_price")
        r["outcome_day"] = (f"the gap ate the potential at the open "
                            f"({session}, R:R became {rr:.2f})")
        # Provenance mark: shows the outcome was rewritten by the migration, not
        # computed by live analysis — this matters when investigating future oddities.
        r["migrated_gap_no_edge"] = datetime.now().isoformat(timespec="seconds")

    with open(SIGNAL_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Records updated: {len(changes)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
