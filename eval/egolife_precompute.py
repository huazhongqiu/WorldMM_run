"""Persistent EgoLife OpenIE preparation and timestamp-specific index snapshots."""
import json
import os
import tempfile
import copy
import shutil
from itertools import chain
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from worldmm.memory.episodic.openie import OpenIE
from worldmm.memory.episodic.utils import compute_mdhash_id


def configure_memory(memory, args):
    root = args.episodic_artifacts_dir
    metadata = Path(args.metadata_dir or "output/metadata")
    candidates = ([Path(args.openie_file)] if getattr(args, "openie_file", None)
                  else list((metadata / "episodic_memory" / args.subject).glob("openie_results_*.json")))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one base OpenIE file: {candidates}")
    memory.load_precomputed_openie(str(candidates[0]), base_granularity="30sec", extract_coarse=True)
    prepared = getattr(args, "prepared_openie_dir", None)
    if prepared or root:
        memory.prepared_openie_dir = prepared or str(Path(root) / "openie")
    snapshots = getattr(args, "snapshot_dir", None)
    if snapshots or root:
        memory.configure_snapshots(snapshots or str(Path(root) / "snapshots"), load_only=not args.prepare_index_only)
        if args.prepare_index_only:
            Path(memory.worker_cache_root).mkdir(parents=True, exist_ok=True)
            memory.save_dir_root = tempfile.mkdtemp(prefix="build-", dir=memory.worker_cache_root)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False))
    os.replace(temporary, path)


def split_passage(passage, limit=1200):
    """Keep natural paragraphs intact where possible, without discarding any text."""
    parts = []
    for paragraph in passage.split("\n\n"):
        paragraph = paragraph.strip()
        while len(paragraph) > limit:
            boundary = paragraph.rfind(" ", 0, limit + 1)
            if boundary < limit // 2:
                boundary = limit
            parts.append(paragraph[:boundary])
            paragraph = paragraph[boundary:].lstrip()
        if paragraph:
            parts.append(paragraph)
    return parts


def extract_caption(extractor, key, passage):
    """Bound generation per segment, then attach merged facts to the original caption."""
    def smaller(text):
        if len(text) < 2:
            raise RuntimeError(f"OpenIE still truncated on minimal segment: {key}")
        return split_passage(text, max(1, len(text) // 2))

    def entities_for(text):
        result = extractor.ner(compute_mdhash_id(text, prefix="chunk-"), text)
        if result.metadata.get("output_truncated"):
            return sorted({entity for part in smaller(text) for entity in entities_for(part)})
        if result.metadata.get("error"):
            raise RuntimeError(f"NER failed for {key}: {result.metadata['error']}")
        return result.unique_entities

    entities = entities_for(passage)

    def triples_for(text):
        result = extractor.triple_extraction(compute_mdhash_id(text, prefix="chunk-"), text, entities)
        if result.metadata.get("output_truncated"):
            return [triple for part in smaller(text) for triple in triples_for(part)]
        if result.metadata.get("error"):
            raise RuntimeError(f"Triple extraction failed for {key}: {result.metadata['error']}")
        return result.triples

    segments = split_passage(passage)
    triples = sorted({tuple(t) for part in segments for t in triples_for(part)})
    return {"idx": key, "passage": passage, "extracted_entities": entities,
            "extracted_triples": [list(t) for t in triples],
            "extraction": {"policy": "bounded-openie-v1", "max_segment_chars": 1200, "segments": len(segments)}}


def prepare_openie(memory, directory, until_time, extractor=None):
    directory = Path(directory)
    extractor = extractor or OpenIE(memory.llm_model)
    provenance = {"llm": memory.llm_model.model_name, "base_openie": memory.base_openie_hash()}
    for granularity, entries in memory.captions.items():
        passages = {compute_mdhash_id(e.text, prefix="chunk-"): e.text for e in entries
                    if e.timestamp_int[1] <= until_time}
        if not passages:
            continue
        path = directory / f"{granularity}.json"
        previous = json.loads(path.read_text()) if path.exists() else {"provenance": provenance, "docs": []}
        if previous.get("provenance") != provenance:
            raise ValueError(f"OpenIE provenance mismatch: {path}")
        docs = {doc["idx"]: doc for doc in previous["docs"]}
        if granularity == memory.base_granularity:
            base = memory.precomputed_openie
            for key, text in passages.items():
                if key in base["ner_results"] and key in base["triple_results"]:
                    docs[key] = {"idx": key, "passage": text, "extracted_entities": base["ner_results"][key],
                                 "extracted_triples": base["triple_results"][key]}
        missing = [(key, text) for key, text in passages.items() if key not in docs]
        print(f"OpenIE {granularity}: reuse={len(passages)-len(missing)} missing={len(missing)}", flush=True)
        if not missing:
            if previous["docs"] != list(docs.values()) or not path.exists():
                write_json(path, {"provenance": provenance, "docs": list(docs.values())})
            continue

        with ThreadPoolExecutor(max_workers=4) as pool:
            for offset in range(0, len(missing), 16):
                futures = {pool.submit(extract_caption, extractor, key, text): key
                           for key, text in missing[offset:offset+16]}
                errors = []
                for future in as_completed(futures):
                    try:
                        doc = future.result()
                    except Exception as error:
                        errors.append(f"{futures[future]}: {error}")
                        continue
                    docs[doc["idx"]] = doc
                    # Preserve successful captions even if another request in the batch fails.
                    write_json(path, {"provenance": provenance, "docs": list(docs.values())})
                if errors:
                    raise RuntimeError("OpenIE incomplete; saved successful captions. " + "; ".join(errors))
                print(f"OpenIE {granularity}: extracted={min(offset+16,len(missing))}/{len(missing)}", flush=True)


def reextract_hourly(memory, directory, extractor=None):
    """Resume a separate all-week hourly rebuild and publish only when complete."""
    directory = Path(directory)
    entries = memory.captions.get("1h", [])
    if not entries:
        raise ValueError("No hourly captions loaded")
    workspace = directory.parent / "hourly_reextract"
    staging = workspace / "bounded-v1"
    hourly = copy.copy(memory)
    hourly.captions = {"1h": entries}
    expected = {compute_mdhash_id(e.text, prefix="chunk-"): e.text for e in entries}
    prepared = staging / "1h.json"
    if prepared.exists():
        for doc in json.loads(prepared.read_text())["docs"]:
            if (doc["idx"] not in expected or doc["passage"] != expected[doc["idx"]]
                    or doc.get("extraction", {}).get("policy") != "bounded-openie-v1"
                    or doc["extraction"].get("max_segment_chars") != 1200):
                raise ValueError(f"Incompatible hourly rebuild checkpoint: {prepared}")
    print(f"Hourly rebuild: all_week={len(entries)} unique={len(expected)} staging={staging}", flush=True)
    prepare_openie(hourly, staging, max(e.timestamp_int[1] for e in entries), extractor=extractor)
    result = json.loads(prepared.read_text())
    if {d["idx"] for d in result["docs"]} != set(expected):
        raise ValueError("Hourly rebuild incomplete; active OpenIE was not replaced")
    active = directory / "1h.json"
    backup = workspace / "legacy-1h.json"
    if active.exists() and not backup.exists():
        workspace.mkdir(parents=True, exist_ok=True)
        shutil.copy2(active, backup)
    # Atomic publication leaves the prior active artifact intact on extraction failure.
    write_json(active, result)
    print(f"Hourly rebuild complete: {len(result['docs'])}/{len(expected)} active={active} backup={backup}", flush=True)


def require_prepared_openie(memory, maximum):
    """Validate prepared OpenIE for every granularity up to maximum; indexing then makes no LLM calls."""
    expected_provenance = {"llm": memory.llm_model.model_name, "base_openie": memory.base_openie_hash()}
    for g, entries in memory.captions.items():
        expected = {compute_mdhash_id(e.text, prefix="chunk-") for e in entries if e.timestamp_int[1] <= maximum}
        if not expected:
            continue
        path = Path(memory.prepared_openie_dir) / f"{g}.json"
        prepared = json.loads(path.read_text())
        docs = {d["idx"] for d in prepared["docs"]}
        for doc in prepared["docs"]:
            if compute_mdhash_id(doc["passage"], prefix="chunk-") != doc["idx"]:
                raise ValueError(f"OpenIE passage hash mismatch: {path}")
        if prepared.get("provenance") != expected_provenance:
            raise ValueError(f"OpenIE provenance mismatch: {path}")
        if not expected <= docs:
            raise ValueError(f"Incomplete OpenIE: {path}; run --prepare-openie-only first")
    # A missing passage must be an error here, never an unexpected online model request.
    memory.require_precomputed_openie = True


def prepare_indexes(memory, times):
    # Validate all OpenIE coverage before loading an embedding model. No LLM calls in this phase.
    require_prepared_openie(memory, max(times))
    ordered = sorted(set(times))
    remaining = iter(ordered)
    last_ready = None
    for timestamp in remaining:
        if not (memory.snapshot_root / str(timestamp) / "manifest.json").is_file():
            break
        last_ready = timestamp
    else:
        memory._load_snapshot(ordered[-1])
        print(f"All {len(ordered)} snapshots already exist; latest read back at {ordered[-1]}", flush=True)
        return
    if last_ready is not None:
        memory._load_snapshot(last_ready, for_update=True)
        print(f"Snapshot resumed: {last_ready}", flush=True)
    for timestamp in chain((timestamp,), remaining):
        destination = memory.snapshot_root / str(timestamp)
        if destination.exists():
            memory._load_snapshot(timestamp, for_update=True)
            print(f"Snapshot reused: {timestamp}", flush=True)
            continue
        memory.index(timestamp)
        memory.save_snapshot(timestamp)
        print(f"Snapshot saved: {timestamp}", flush=True)
