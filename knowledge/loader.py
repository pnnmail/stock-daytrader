# knowledge/loader.py
import os as os_module
import re
import time
import fitz
from config import KNOWLEDGE_PATH, RAG_MAX_CHUNKS, MAX_LINE_LOG
from knowledge.embeddings import collection, RAG_ENABLED, init_good_embeddings, classify_chunk
from knowledge.distiller import distill_paragraph, extract_keywords_from_chunk, ALL_TRADE_TERMS, _normalize_for_match
from utils.helpers import is_chapter_title, is_outline_or_junk, compute_quality_score


def load_pdfs_to_chroma():
    if not RAG_ENABLED:
        return
    init_good_embeddings()
    for pdf_file in [f for f in os_module.listdir(KNOWLEDGE_PATH) if f.endswith('.pdf')]:
        # Check: look for ANY chunk from this PDF by metadata
        try:
            existing = collection.get(
                where={"source": pdf_file},
                limit=1
            )
            if existing and existing.get('ids') and len(existing['ids']) > 0:
                all_chunks = collection.get(where={"source": pdf_file})
                total = len(all_chunks['ids']) if all_chunks else 0
                print(f"📚 RAG: {pdf_file} already loaded ({total} chunks), skipping")
                continue
        except Exception as e:
            print(f"   ⚠️ where query failed: {str(e)[:80]}, trying the old check")
            existing = collection.get(ids=[f"{pdf_file}_0"])
            if existing and existing.get('ids') and len(existing['ids']) > 0:
                print(f"📚 RAG: {pdf_file} already loaded, skipping")
                continue
        
        print(f"📚 RAG: loading {pdf_file}...")
        doc = fitz.open(os_module.path.join(KNOWLEDGE_PATH, pdf_file))
        full_text = ""
        for page in doc:
            full_text += page.get_text() + "\n"
        doc.close()
        lines = full_text.split('\n')
        chapters = []
        current_title = "Introduction"
        current_text = ""
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r'^(Table of Contents|Contents|ISBN|Copyright|All rights reserved)', stripped, re.IGNORECASE):
                continue
            if 'all rights reserved' in stripped.lower():
                continue
            if re.match(r'^https?://', stripped):
                continue
            if is_chapter_title(stripped):
                if current_text.strip():
                    chapters.append({"title": current_title, "text": current_text.strip()})
                current_title = stripped
                current_text = ""
            else:
                current_text += stripped + " "
        if current_text.strip():
            chapters.append({"title": current_title, "text": current_text.strip()})
        print(f"   📖 Chapters found: {len(chapters)}")

        # Count the total number of blocks for progress
        total_paragraphs = 0
        for chapter in chapters:
            chapter_text = chapter["text"]
            chapter_text = re.sub(r'Copyright\s*©[^.]*\.?\s*', '', chapter_text)
            chapter_text = re.sub(r'https?://\S+', '', chapter_text)
            chapter_text = re.sub(r'All rights reserved\.?\s*', '', chapter_text, flags=re.IGNORECASE)
            chapter_text = re.sub(r'\[Home\].*?\[Next\]\s*', '', chapter_text)
            lines = chapter_text.split('\n')
            merged_lines = []
            current_line = ""
            for line in lines:
                line = line.strip()
                if not line:
                    if current_line:
                        merged_lines.append(current_line)
                        current_line = ""
                    continue
                if current_line:
                    prev_end = current_line[-1] if current_line else ''
                    curr_start = line[0] if line else ''
                    if prev_end not in '.!?"' and curr_start.islower():
                        current_line += " " + line
                    else:
                        merged_lines.append(current_line)
                        current_line = line
                else:
                    current_line = line
            if current_line:
                merged_lines.append(current_line)
            chapter_text = '\n'.join(merged_lines)
            chapter_text = re.sub(r'\s{2,}', ' ', chapter_text)
            paragraphs = re.split(r'\n\s*\n', chapter_text)
            merged_blocks = []
            current_block = ""
            for p in paragraphs:
                p = p.strip()
                if not p:
                    continue
                if len(current_block) + len(p) < 1200:
                    current_block += "\n\n" + p if current_block else p
                else:
                    if current_block and len(current_block) > 100:
                        merged_blocks.append(current_block)
                    current_block = p
            if current_block and len(current_block) > 100:
                merged_blocks.append(current_block)
            total_paragraphs += sum(1 for b in merged_blocks if len(b.strip()) >= 100 and not is_outline_or_junk(b.strip()))
        print(f"   Total paragraphs to process: {total_paragraphs}")

        chunk_count = 0
        added_chunks = []
        category_stats = {}
        processed_paragraphs = 0
        start_time = time.time()
        last_time = start_time
        para_times = []

        for chapter in chapters:
            if chunk_count >= RAG_MAX_CHUNKS:
                break

            chapter_text = chapter["text"]
            chapter_title = chapter["title"]

            chapter_text = re.sub(r'Copyright\s*©[^.]*\.?\s*', '', chapter_text)
            chapter_text = re.sub(r'https?://\S+', '', chapter_text)
            chapter_text = re.sub(r'All rights reserved\.?\s*', '', chapter_text, flags=re.IGNORECASE)
            chapter_text = re.sub(r'\[Home\].*?\[Next\]\s*', '', chapter_text)

            lines = chapter_text.split('\n')
            merged_lines = []
            current_line = ""
            for line in lines:
                line = line.strip()
                if not line:
                    if current_line:
                        merged_lines.append(current_line)
                        current_line = ""
                    continue
                if current_line:
                    prev_end = current_line[-1] if current_line else ''
                    curr_start = line[0] if line else ''
                    if prev_end not in '.!?"' and curr_start.islower():
                        current_line += " " + line
                    else:
                        merged_lines.append(current_line)
                        current_line = line
                else:
                    current_line = line
            if current_line:
                merged_lines.append(current_line)

            chapter_text = '\n'.join(merged_lines)
            chapter_text = re.sub(r'\s{2,}', ' ', chapter_text)
            paragraphs = re.split(r'\n\s*\n', chapter_text)

            merged_blocks = []
            current_block = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current_block) + len(para) < 1200:
                    current_block += "\n\n" + para if current_block else para
                else:
                    if current_block and len(current_block) > 100:
                        merged_blocks.append(current_block)
                    current_block = para
            if current_block and len(current_block) > 100:
                merged_blocks.append(current_block)

            for para in merged_blocks:
                para = para.strip()
                if len(para) < 100 or is_outline_or_junk(para):
                    continue

                # Fast filter: skip paragraphs without trading keywords.
                # The term list comes from ALL_TRADE_TERMS (knowledge/distiller.py TERM_MAP) —
                # it covers the book's terminology ("support", "resistance", "volume"
                # etc.), rather than a short ad-hoc list of 15 words that
                # missed most of the book's vocabulary.
                # Lower-case both sides before matching.
                para_normalized = para.lower()
                if not any(kw.lower() in para_normalized for kw in ALL_TRADE_TERMS):
                    processed_paragraphs += 1
                    print(f"      ⏩ Skipped (no trading keywords)")
                    continue

                processed_paragraphs += 1
                distilled = distill_paragraph(para, chapter_title)

                if distilled:
                    # === SAVE ATOMIC FACTS (CONCEPTS) ===
                    fact_count = 0
                    for fact in distilled.get("facts", []):
                        fact_count += 1
                        if fact_count > 5:  # No more than 5 facts per paragraph
                            print(f"      ⚠️ Too many facts, the rest were skipped")
                            break
                        
                        concept = fact.get("concept", "").strip()
                        
                        # Skip facts with an empty concept
                        if not concept:
                            print(f"      ⚠️ Skipped an empty concept")
                            continue
                        
                        # 🔥 Guard against None and empty values (ALL fields BEFORE the check)
                        action = fact.get("action") or ""
                        if action is None:
                            action = ""
                        
                        market_phase = fact.get("market_phase", [])
                        if not market_phase or market_phase is None:
                            market_phase = ["any"]

                        category = fact.get("category")
                        # "context" was removed from the allowed categories (see distill_paragraph) —
                        # if the model still returned something outside {entry,exit,risk,confirmation},
                        # it is not an actionable fact; skip it rather than pull it into the base with a fake category.
                        if category not in ("entry", "exit", "risk", "confirmation"):
                            print(f"      ⚠️ Skipped a fact with an invalid category: {category!r} ({concept})")
                            continue

                        original_text = fact.get("original_text", "")
                        if original_text is None:
                            original_text = ""
                        original_lower = original_text.lower()

                        # 🔥 Anti-hallucination: checked AFTER original_text is declared.
                        # _normalize_for_match collapses hyphenated line breaks like "effec- tive"
                        # -> "effective" — otherwise the verbatim quote won't match the original
                        # from the PDF line by line, although in meaning it is the same word.
                        original_key = _normalize_for_match(original_text)[:100]
                        if len(original_key) > 20 and original_key not in _normalize_for_match(para):
                            print(f"      ⚠️ Skipped a hallucinated concept: {concept}")
                            continue

                        # References to figures/tables are useless without the image itself —
                        # a text-based system can do nothing with them, regardless of
                        # what the model decided.
                        if re.search(r'fig\.?\s*\d|figure\s*\d|table\s*\d|see fig', original_lower):
                            print(f"      ⚠️ Skipped a fact referring to a figure/table: {concept}")
                            continue

                        # A fragment starting mid-word/mid-sentence (e.g. "firmed..." instead of
                        # "confirmed..." — because the paragraph itself started at a page/paragraph
                        # break). A sentence should start with a capital
                        # letter, a digit or a quote — if not, it is almost certainly a fragment.
                        first_char = original_text.strip()[:1]
                        if first_char and not (first_char.isupper() or first_char.isdigit() or first_char in '"«'):
                            print(f"      ⚠️ Skipped a fact with a sentence fragment at the start: {concept}")
                            continue

                        quality = compute_quality_score(original_text)
                        if quality < 3:
                            print(f"      ⚠️ Skipped a fact with a low quality_score={quality}: {concept}")
                            continue

                        programmable_markers = fact.get("programmable_markers", [])
                        if not programmable_markers or programmable_markers is None:
                            programmable_markers = ["none"]
                        
                        visual_markers = fact.get("visual_markers", [])
                        if not visual_markers or visual_markers is None:
                            visual_markers = ["none"]
                        
                        confidence = fact.get("confidence", "derived")
                        if confidence is None:
                            confidence = "derived"
                        
                        requires_visual = fact.get("requires_visual", False)
                        if requires_visual is None:
                            requires_visual = False
                        
                        confirmation_needed = fact.get("confirmation_needed", False)
                        if confirmation_needed is None:
                            confirmation_needed = False

                        # Document text
                        doc_text = f"[{category.upper()}] {original_text}"

                        # Keywords — from the rule TEXT plus the markers.
                        # Previously they were taken ONLY from the markers, and the markers are English
                        # pseudocode ("close > open", "none") in which the book's terms
                        # from TERM_MAP are not found: 69% of facts got a single
                        # keyword="general", and the route-topic bonus in the router
                        # (knowledge/router.py) distinguished nothing.
                        all_markers = programmable_markers + visual_markers
                        keywords = extract_keywords_from_chunk(
                            " ".join([original_text] + all_markers)
                        )
                        if not keywords:
                            keywords = ["none"]

                        collection.add(
                            documents=[doc_text],
                            ids=[f"{pdf_file}_fact_{chunk_count}"],
                            metadatas=[{
                                "source": pdf_file,
                                "chapter": chapter_title,
                                "concept": concept,
                                "action": action,
                                "market_phase": market_phase,
                                "confidence": confidence,
                                "category": category,
                                "requires_visual": requires_visual,
                                "confirmation_needed": confirmation_needed,
                                "programmable_markers": programmable_markers,
                                "visual_markers": visual_markers,
                                "original_text": original_text,
                                "keywords": keywords,
                                "quality_score": quality,
                            }]
                        )
                        added_chunks.append(doc_text)
                        chunk_count += 1
                        print(f"      🧠 Concept {chunk_count}: {concept} | {action} | phase: {market_phase} | confidence: {confidence}")

                    category_stats["concepts"] = category_stats.get("concepts", 0) + min(fact_count, 5)
                else:
                    pass

                # Update progress
                # ✅ Time FOR THIS paragraph (previously this was time.time() - start_time —
                # the time accumulated since the start of the book, not for one paragraph, which made
                # the ETA grow like an avalanche and not reflect the real speed).
                now = time.time()
                para_times.append(now - last_time)
                last_time = now
                if len(para_times) > 10:
                    para_times.pop(0)
                avg_time = sum(para_times) / len(para_times) if para_times else 60
                remaining = total_paragraphs - processed_paragraphs
                eta_seconds = avg_time * remaining
                eta_minutes = eta_seconds / 60
                yield_pct = (chunk_count / processed_paragraphs * 100) if processed_paragraphs else 0
                print(f"   ⏳ Progress: {processed_paragraphs}/{total_paragraphs} paragraphs ({processed_paragraphs/total_paragraphs*100:.1f}%) | "
                      f"💾 Saved to the base: {chunk_count} facts (yield: {yield_pct:.1f}% per paragraph) | ETA: {eta_minutes:.1f} min")

                if chunk_count >= RAG_MAX_CHUNKS:
                    break

        print(f"📚 RAG: {pdf_file} loaded ({chunk_count} chunks, {len(chapters)} chapters)")
        print(f"   Categories: {dict(category_stats)}")