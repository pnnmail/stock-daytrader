"""
One-off migration: recompute metadata["keywords"] for already stored facts.

Why: originally keywords were extracted only from programmable_markers/visual_markers,
and that is English pseudocode ("close > open", "none") in which the book's terms from
TERM_MAP are not found — 69% of facts got a single keyword="general", so
the route-topic bonus in knowledge/router.py could not tell chunks apart. Now keywords
are computed from the rule text + markers (see knowledge/loader.py).

Re-distillation is NOT needed for this: original_text and the markers are already in the database,
the recompute runs locally, without LLM calls and without changing embeddings.

RUN ONLY WITH THE BOT STOPPED (Chroma is not designed for two processes):
    docker compose stop stock-agent2
    docker compose run --rm stock-agent2 python knowledge/backfill_keywords.py
    docker compose up -d stock-agent2
"""
from knowledge.embeddings import collection, RAG_ENABLED
from knowledge.distiller import extract_keywords_from_chunk


def backfill():
    if not RAG_ENABLED or collection is None:
        print("❌ RAG is disabled or the collection is unavailable")
        return

    data = collection.get(include=["metadatas"])
    ids = data.get("ids", [])
    metadatas = data.get("metadatas", [])
    print(f"📚 Records found: {len(ids)}")

    upd_ids, upd_meta = [], []
    unchanged = 0

    for rec_id, meta in zip(ids, metadatas):
        if not meta:
            continue
        original_text = meta.get("original_text") or ""

        markers = []
        for key in ("programmable_markers", "visual_markers"):
            val = meta.get(key)
            if isinstance(val, str):
                markers.append(val)
            elif isinstance(val, (list, tuple)):
                markers.extend(str(v) for v in val)

        new_keywords = extract_keywords_from_chunk(" ".join([original_text] + markers))
        if not new_keywords:
            new_keywords = ["none"]

        old_val = meta.get("keywords")
        old_keywords = [old_val] if isinstance(old_val, str) else list(old_val or [])

        if sorted(old_keywords) == sorted(new_keywords):
            unchanged += 1
            continue

        new_meta = dict(meta)
        new_meta["keywords"] = new_keywords
        upd_ids.append(rec_id)
        upd_meta.append(new_meta)

        print(f"   {rec_id}: {old_keywords} -> {new_keywords}")

    if not upd_ids:
        print(f"✅ Nothing to update (all {unchanged} records are already up to date)")
        return

    # Update in batches — collection.update does not change embeddings, only metadata
    BATCH = 100
    for i in range(0, len(upd_ids), BATCH):
        collection.update(ids=upd_ids[i:i + BATCH], metadatas=upd_meta[i:i + BATCH])

    print(f"\n✅ Updated: {len(upd_ids)} | unchanged: {unchanged}")


if __name__ == "__main__":
    backfill()
