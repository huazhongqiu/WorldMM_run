"""Build and verify WorldMM memories for a frozen experiment dataset."""

import argparse
import logging
import os
import pickle
import tempfile
from pathlib import Path

from preprocess.variant_inputs import (
    EGO_GRANULARITIES, GRANULARITIES, SUBJECT, digest,
    frame_indices, freeze_json, hhmmss, prepare_ego_captions,
    read_json, write_json,
)
from worldmm.embedding import EmbeddingModel
from worldmm.index_manifest import embedding_identity, ensure_manifest
from worldmm.llm import LLMModel, PromptTemplateManager
from worldmm.memory import WorldMemory
from worldmm.memory.episodic import EpisodicMemory
from worldmm.memory.episodic.gen_multiscale import gen_multiscale
from worldmm.memory.episodic.utils import compute_mdhash_id
from worldmm.retrieval_config import load_videospy_config

MODEL = "Qwen3.5-4B"
FINAL_TIME = 123595999


def root(name):
    return Path(os.environ[f"WORLDMM_{name}_ROOT"])


def caption_paths(dataset, unit):
    if dataset == "egolife":
        return {g: root("CAPTION") / f"EgoLifeCap/{unit}/{unit}_{g}.json"
                for g in EGO_GRANULARITIES}
    return {g: root("CAPTION") / f"{unit}/{g}.json" for g in GRANULARITIES}


def shared_paths(unit, dataset):
    shared = root("EXTRACTION")
    generator = "gpt-5-mini" if dataset == "egolife" else MODEL
    return (shared / f"episodic_memory/{unit}/openie_results_{generator}.json",
            shared / f"semantic_memory/{unit}/semantic_extraction_results_{generator}.json")


def prepared_openie_dir(dataset, unit):
    """Per-granularity OpenIE used to seed every episodic graph without online extraction."""
    return root("EXTRACTION") / "openie" if dataset == "egolife" else root("EXTRACTION") / "openie" / unit


def check_coverage(captions, openie):
    expected = {compute_mdhash_id(item["text"], prefix="chunk-") for item in captions}
    for field in ("ner_results", "triple_results"):
        if not expected <= openie[field].keys():
            raise ValueError(f"Incomplete shared OpenIE {field}: {len(expected - openie[field].keys())} missing")


def compat_copy(source, destination):
    if source.is_file():
        freeze_json(destination, read_json(source))


def extract_shared_openie(base, output, llm):
    from egolife_precompute import prepare_openie

    memory = EpisodicMemory(None, llm, PromptTemplateManager(), ["10sec"])
    memory.load_captions_from_data({"10sec": read_json(base)})
    memory.precomputed_openie = {"ner_results": {}, "triple_results": {}}
    checkpoint = output.parent / "prepared"
    prepare_openie(memory, checkpoint, FINAL_TIME)
    docs = read_json(checkpoint / "10sec.json")["docs"]
    write_json(output, {
        "ner_results": {doc["idx"]: doc["extracted_entities"] for doc in docs},
        "triple_results": {doc["idx"]: doc["extracted_triples"] for doc in docs},
    })


def prepare_shared(dataset, unit, video, paths, llm, clip_seconds, smoke_clips, cutoff):
    from preprocess.episodic_memory.generate_fine_caption import process_video
    from preprocess.semantic_memory.extract_semantic_triples import (
        group_captions_and_get_openie_triples, run_semantic_extraction,
    )

    legacy = root("DATASET")
    base = next(iter(paths.values()))
    openie, semantic = shared_paths(unit, dataset)
    provenance = root("EXTRACTION") / "provenance" / f"{unit}.json"
    if dataset == "egolife":
        compat_copy(legacy / f"episodic_memory/{unit}/{openie.name}", openie)
        raw_semantic = read_json(legacy / f"semantic_memory/{unit}/{semantic.name}")
        if cutoff:
            raw_semantic["semantic_triples"] = {
                k: v for k, v in raw_semantic["semantic_triples"].items() if int(k) <= cutoff
            }
        freeze_json(semantic, raw_semantic)
        for g in EGO_GRANULARITIES:
            target = root("EXTRACTION") / f"openie/{g}.json"
            if not target.exists():
                source = legacy / f"prebuilt/{unit}/openie/{g}.json"
                if source.is_file():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                        stream.write(source.read_bytes())
                    os.replace(stream.name, target)
        freeze_json(provenance, {"base_openie": "wgcyeo/WorldMM-EgoLife gpt-5-mini",
                                 "semantic_extraction": "wgcyeo/WorldMM-EgoLife gpt-5-mini",
                                 "multiscale_openie": str(legacy / f"prebuilt/{unit}/openie"),
                                 "base_sha256": digest(openie), "semantic_sha256": digest(semantic)})
        return

    imported = dataset == "lvbench" and clip_seconds == 10 and not smoke_clips and (legacy / f"caption/{unit}/10sec.json").is_file()
    if imported:
        for path in paths.values():
            compat_copy(legacy / f"caption/{unit}/{path.name}", path)
        compat_copy(legacy / f"episodic_memory/{unit}/{openie.name}", openie)
        compat_copy(legacy / f"semantic_memory/{unit}/{semantic.name}", semantic)
    if not base.exists():
        process_video(
            Path(video["video_path"]), Path(video["subtitle_path"]) if video["subtitle_path"] else None,
            base, llm, clip_seconds, max_segments=smoke_clips,
        )
    rows = read_json(base)
    if not rows or any(not r.get("text", "").strip() for r in rows):
        raise ValueError(f"Empty caption in {base}")
    if any(hhmmss(r["start_time"]) != i * clip_seconds for i, r in enumerate(rows)):
        raise ValueError(f"Caption cut points do not match clip_seconds={clip_seconds}: {base}")
    gen_multiscale(str(base), str(base.parent), llm)
    if not openie.exists():
        extract_shared_openie(base, openie, llm)
    check_coverage(rows, read_json(openie))
    if not semantic.exists():
        run_semantic_extraction(str(base), str(openie), str(semantic.parent), MODEL, llm)
    semantic_triples = read_json(semantic).get("semantic_triples")
    if not isinstance(semantic_triples, dict):
        raise ValueError(f"Invalid semantic extraction: {semantic}")
    expected = group_captions_and_get_openie_triples(rows, read_json(openie))
    if semantic_triples.keys() != expected.keys():
        raise ValueError(f"Incomplete semantic extraction: {semantic}")
    # Upstream WorldMM extracts triples at every episodic granularity, not only the base one.
    from egolife_precompute import prepare_openie

    memory = EpisodicMemory(None, llm, PromptTemplateManager(), list(paths))
    memory.load_captions_from_data({g: read_json(path) for g, path in paths.items()})
    memory.load_precomputed_openie(str(openie), base_granularity=GRANULARITIES[0], extract_coarse=True)
    prepare_openie(memory, prepared_openie_dir(dataset, unit), FINAL_TIME)
    freeze_json(provenance, {"caption_origin": str(legacy / f"caption/{unit}") if imported else "WorldMM caption generator",
                             "caption_sample_fps": 1, "clip_seconds": clip_seconds,
                             "subtitle_sha256": video.get("subtitle_sha256"), "generator": MODEL,
                             "caption_sha256": {g: digest(p) for g, p in paths.items()},
                             "base_sha256": digest(openie), "semantic_sha256": digest(semantic)})


def visual_index(base, destination, encoder, dataset, budget):
    from decord import VideoReader, cpu
    from PIL import Image

    ensure_manifest(destination, visual_identity(base, encoder, budget))
    output = destination / "visual_embeddings.pkl"
    records = {}
    if output.exists():
        with output.open("rb") as stream:
            records = pickle.load(stream)
    metadata = destination / "frames.json"
    samples = read_json(metadata) if metadata.exists() else {}
    rows = read_json(base)
    reader = None
    previous = None
    for row in rows:
        key = row["video_path"] if dataset == "egolife" else row["start_time"]
        if key in records and key in samples:
            continue
        source = row.get("source_video_path", row["video_path"])
        if source != previous:
            reader = VideoReader(source, ctx=cpu(0), num_threads=1)
            previous = source
        fps = float(reader.get_avg_fps())
        start = row.get("start_sec", hhmmss(row["start_time"]))
        end = min(row.get("end_sec", hhmmss(row["end_time"])), len(reader) / fps)
        indices = frame_indices(fps, len(reader), start, end, budget)
        images = [Image.fromarray(frame) for frame in reader.get_batch(indices).asnumpy()]
        try:
            if encoder.backend == "original":
                spec = {"video": images, "sample_fps": len(images) / (end - start)}
                record = encoder.encode_video([spec], nframes=len(images), batch_size=1)[0]
            else:
                spec = {"video": images, "frame_timestamps": [i / fps for i in indices],
                        "video_start": start, "video_end": end}
                if encoder.backend == "freeret":
                    record = encoder.encode_representations(
                        [spec], modality="video", nframes=len(images), batch_size=1)[0]
                else:
                    record = encoder.encode_video([spec], nframes=len(images), batch_size=1)[0]
        finally:
            for image in images:
                image.close()
        records[key] = record
        samples[key] = {"source": source, "start": start, "end": end, "fps": fps,
                        "frame_indices": indices, "nframes": len(indices),
                        "subtitle_cues": len(row.get("subtitles", []))}
        if len(samples) % 32 == 0:
            save_visual(output, records, metadata, samples)
    if len(records) != len(rows) or len(samples) != len(rows):
        raise ValueError(f"Incomplete visual index: {output}, records={len(records)}, clips={len(rows)}")
    if not output.exists() or len(samples) % 32:
        save_visual(output, records, metadata, samples)
    return output


def save_visual(output, records, metadata, samples):
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
        pickle.dump(records, stream)
    os.replace(stream.name, output)
    write_json(metadata, samples)


def consolidate(extraction, output, encoder, llm):
    from preprocess.semantic_memory.consolidate_semantic_memory import run_semantic_consolidation

    ensure_manifest(output, {"embedding": embedding_identity(encoder), "extraction": digest(extraction),
                             "inference": MODEL})
    target = output / f"semantic_consolidation_results_{MODEL}.json"
    if not target.exists():
        run_semantic_consolidation(str(extraction), str(output), MODEL, llm, encoder)
    if read_json(target).keys() != read_json(extraction)["semantic_triples"].keys():
        raise ValueError(f"Incomplete semantic consolidation: {target}")
    return target


def semantic_vectors(extraction, destination, encoder):
    from hipporag.embedding_store import EmbeddingStore

    identity = {"embedding": embedding_identity(encoder), "consolidation": digest(extraction)}
    ensure_manifest(destination, identity)
    store = EmbeddingStore(encoder, str(destination / "triples"), 4, "semantic")
    rows = read_json(extraction)
    texts = sorted({" ".join(triple[:3]) for row in rows.values()
                    for triple in row["consolidated_semantic_triples"] if len(triple) >= 3})
    for i in range(0, len(texts), 128):
        store.insert_strings(texts[i:i + 128], batch_size=4)
    return store


def cutoffs(dataset, rows):
    if dataset != "egolife":
        return [FINAL_TIME]
    return sorted({int(r["query_time"]["date"][-1] + r["query_time"]["time"].zfill(8)) for r in rows})


def record_legacy_assets(dataset):
    legacy = root("DATASET")
    if dataset == "egolife":
        snapshots = legacy / f"prebuilt/{SUBJECT}/snapshots"
        sample = next(iter(sorted(snapshots.glob("*/manifest.json"))), None)
        if sample:
            old = read_json(sample)["identity"]
            write_json(root("REPORT") / "legacy_assets.json",
                       {"source": str(snapshots), "count": len(list(snapshots.glob("*/manifest.json"))),
                        "legacy_model": old["embedding"], "legacy_inference": old["llm"],
                        "runtime_compatible": False, "reason": "legacy model-name identity and graph layout"})
    elif dataset == "lvbench":
        visual = legacy / "visual_memory"
        write_json(root("REPORT") / "legacy_assets.json",
                   {"source": str(visual), "count": len(list(visual.glob("*/visual_embeddings.pkl"))),
                    "runtime_compatible": False,
                    "reason": "legacy visual records omit exact frame indices and encoder fingerprint"})


def visual_identity(captions, encoder, budget):
    return {"embedding": embedding_identity(encoder), "captions": digest(captions),
            "frame_budget": budget, "sampler": "decord-uniform-half-open-v1",
            "subtitle_input": "caption-and-retrieval-metadata"}


def validate_visual_identity(directory, captions, encoder, budget):
    if not (directory / "index_manifest.json").is_file():
        raise FileNotFoundError(f"Missing visual index manifest: {directory}")
    ensure_manifest(directory, visual_identity(captions, encoder, budget))


def episodic_views(dataset, unit, paths, openie, encoder, llm, times, verify, previous_openie_dir=None):
    from egolife_precompute import prepare_indexes

    granularity = list(paths)
    index_dir = root("INDEX") / unit / "episodic_memory"
    if dataset == "egolife":
        identity = {"embedding": embedding_identity(encoder), "inference": MODEL}
        ensure_manifest(root("VIEW") / unit, identity)
    memory = EpisodicMemory(encoder, llm, PromptTemplateManager(), granularity, str(index_dir))
    memory.load_captions_from_data({g: read_json(path) for g, path in paths.items()})
    memory.load_precomputed_openie(str(openie), base_granularity=granularity[0], extract_coarse=True)
    memory.prepared_openie_dir = str(prepared_openie_dir(dataset, unit))
    if dataset == "egolife":
        snapshots = root("VIEW") / unit / "snapshots"
        memory.configure_snapshots(str(snapshots), load_only=verify)
        if previous_openie_dir:
            for timestamp in times:
                memory.upgrade_snapshot_identity(timestamp, previous_openie_dir)
        if not verify:
            index_dir.mkdir(parents=True, exist_ok=True)
            memory.save_dir_root = tempfile.mkdtemp(prefix="build-", dir=index_dir)
            prepare_indexes(memory, times)
    else:
        from egolife_precompute import require_prepared_openie

        check_coverage(read_json(paths[granularity[0]]), read_json(openie))
        require_prepared_openie(memory, times[-1])
        ensure_manifest(index_dir, {"embedding": embedding_identity(encoder), "inference": MODEL,
                                    "captions": {g: digest(p) for g, p in paths.items()},
                                    "openie": digest(openie),
                                    "prepared_openie": {g: digest(Path(memory.prepared_openie_dir) / f"{g}.json")
                                                        for g in granularity}})
        snapshots = root("VIEW") / unit / "snapshots"
        memory.configure_snapshots(str(snapshots), load_only=verify)
        if not verify and (snapshots / str(times[0]) / "manifest.json").is_file():
            memory._load_snapshot(times[0])
        elif not verify:
            memory.index(times[0])
            memory.save_snapshot(times[0])
    return memory


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("shared", "semantic", "semantic-index", "episodic", "visual", "verify", "audit"), required=True)
    parser.add_argument("--unit", action="append", help="Restrict to a manifest video ID or A1_JAKE")
    parser.add_argument("--legacy-only", action="store_true", help="Import existing LVBench shared inputs")
    parser.add_argument("--previous-openie-dir", help="Audit and upgrade snapshot identity from its previous OpenIE files")
    parser.add_argument("--smoke-clips", type=int, help="Generate the first N captions in an isolated smoke root")
    parser.add_argument("--smoke-questions", type=int, help="Use the first N time-ordered EgoLife questions")
    args = parser.parse_args()
    dataset = os.environ["WORLDMM_DATASET"]
    if any(value is not None and value <= 0 for value in (args.smoke_clips, args.smoke_questions)):
        parser.error("Smoke limits must be positive")
    if (args.smoke_clips or args.smoke_questions) and not os.environ.get("WORLDMM_SMOKE_ROOT"):
        parser.error("Smoke limits require WORLDMM_SMOKE_ROOT")
    if args.stage == "audit":
        record_legacy_assets(dataset)
        return
    if args.previous_openie_dir and (dataset != "egolife" or args.stage != "verify"):
        parser.error("--previous-openie-dir is for EgoLife verification")
    manifest = root("MANIFEST")
    rows = read_json(manifest / "questions.json")
    videos = read_json(manifest / "videos.json")
    if dataset == "egolife" and args.smoke_questions:
        rows = sorted(rows, key=lambda row: cutoffs(dataset, [row])[0])[:args.smoke_questions]
    units = args.unit or list(videos)
    if len(set(units)) != len(units) or set(units) - videos.keys():
        parser.error("Unit IDs must be distinct members of the frozen video manifest")
    if args.legacy_only:
        if dataset != "lvbench" or args.stage != "shared" or args.smoke_clips:
            parser.error("--legacy-only applies to the full LVBench shared import")
        units = [unit for unit in units if (root("DATASET") / f"caption/{unit}/10sec.json").is_file()]
    if args.stage == "shared" and os.environ.get("WORLDMM_SMOKE_ROOT"):
        if dataset == "egolife" and args.smoke_questions:
            freeze_json(root("SHARED") / "questions.json", rows)
        elif args.smoke_clips:
            freeze_json(root("SHARED") / "questions.json", [r for r in rows if r["video_id"] in units])
    encoder = None if args.stage == "shared" else EmbeddingModel()
    llm = LLMModel(model_name=MODEL)
    config = load_videospy_config()["tools"]["semantic_search"]
    clip_seconds = config.get("clip_seconds", 10)
    if type(clip_seconds) is not int or clip_seconds <= 0:
        raise ValueError("clip_seconds must be a positive whole number")
    budget = int(os.environ.get("WORLDMM_VISUAL_FRAME_BUDGET", "16"))
    if budget < 2 or budget % 2:
        raise ValueError("WORLDMM_VISUAL_FRAME_BUDGET must be a positive even number")
    if args.stage == "shared" and dataset == "egolife" and not next(iter(caption_paths(dataset, SUBJECT).values())).exists():
        prepare_ego_captions(root("DATASET"), root("CAPTION"), videos)
    for unit in units:
        paths = caption_paths(dataset, unit)
        base = next(iter(paths.values()))
        openie, extraction = shared_paths(unit, dataset)
        visual_dir = root("INDEX") / f"visual_memory/{unit}"
        semantic_dir = root("MEMORY") / f"semantic_memory/{unit}"
        semantic_path = semantic_dir / f"semantic_consolidation_results_{MODEL}.json"
        semantic_index_dir = root("INDEX") / f"semantic_memory/{unit}"
        times = cutoffs(dataset, rows)
        print(f"{dataset}/{unit} stage={args.stage} clip_seconds={clip_seconds} frame_budget={budget}", flush=True)
        visual_base = base
        if dataset == "egolife" and args.smoke_questions:
            visual_base = root("CAPTION") / "smoke_visual.json"
            entries = read_json(base)
            past = [row for row in entries if int(row["date"][-1] + row["end_time"].zfill(8)) <= times[-1]]
            future = [row for row in entries if int(row["date"][-1] + row["end_time"].zfill(8)) > times[-1]]
            if args.smoke_clips:
                past = past[:args.smoke_clips]
            if args.stage == "shared":
                freeze_json(visual_base, past + future[:1])
        if args.stage == "shared":
            prepare_shared(dataset, unit, videos[unit], paths, llm, clip_seconds, args.smoke_clips,
                           times[-1] if args.smoke_questions else None)
            if dataset == "egolife":
                from egolife_precompute import prepare_openie

                memory = EpisodicMemory(None, llm, PromptTemplateManager(), list(paths))
                memory.load_captions_from_data({g: read_json(path) for g, path in paths.items()})
                memory.load_precomputed_openie(str(openie), base_granularity="30sec", extract_coarse=True)
                prepare_openie(memory, root("EXTRACTION") / "openie", max(times))
        elif args.stage == "visual":
            visual_index(visual_base, visual_dir, encoder, dataset, budget)
        elif args.stage == "semantic":
            consolidate(extraction, semantic_dir, encoder, llm)
        elif args.stage == "semantic-index":
            semantic_vectors(semantic_path, semantic_index_dir, encoder)
        elif args.stage == "episodic":
            episodic_views(dataset, unit, paths, openie, encoder, llm, times, verify=False)
        else:
            memory = WorldMemory(encoder, llm, episodic_granularities=list(paths))
            memory.episodic_memory = episodic_views(
                dataset, unit, paths, openie, encoder, llm, times, verify=True,
                previous_openie_dir=args.previous_openie_dir)
            memory.load_semantic_triples(file_path=str(semantic_path))
            memory.semantic_memory.load_vector_store(semantic_index_dir, semantic_path)
            validate_visual_identity(visual_dir, visual_base, encoder, budget)
            memory.load_visual_clips(str(visual_dir / "visual_embeddings.pkl"), str(visual_base))
            verification_times = dict.fromkeys((times[-1], times[0])) if args.smoke_questions else times
            for cutoff in verification_times:
                memory.index(cutoff)
                counts = {g: len(e) for g, e in memory.episodic_memory.indexed_entries.items()}
                assert all(e.timestamp_int[1] <= cutoff for entries in memory.episodic_memory.indexed_entries.values() for e in entries)
                assert all(e.timestamp_int[1] <= cutoff for e in memory.visual_memory.indexed_entries)
                if not any(counts.values()) or not memory.visual_memory.indexed_entries:
                    raise ValueError("Smoke cutoff has no episodic or visual memory")
                for graph in memory.episodic_memory.hipporag.values():
                    graph.prepare_retrieval_objects()
                if memory.semantic_memory.indexed_timestamp > cutoff:
                    raise ValueError("Semantic memory contains a future timestamp")
                print(f"verified {unit} cutoff={cutoff} episodic={counts} visual={len(memory.visual_memory.indexed_entries)} "
                      f"semantic={len(memory.semantic_memory.indexed_entries)}", flush=True)
        if args.stage != "shared":
            write_json(root("REPORT") / unit / f"{args.stage}.json",
                       {"unit": unit, "dataset": dataset, "embedding": embedding_identity(encoder),
                        "inference": MODEL, "clip_seconds": None if dataset == "egolife" else clip_seconds,
                        "clip_policy": "caption-native" if dataset == "egolife" else "fixed-grid",
                        "frame_budget": budget, "caption_sha256": {g: digest(p) for g, p in paths.items()},
                        "context_sample_fps": config.get("sample_fps", 1),
                        "visual_embedding_subtitles": False,
                        "subtitle_sha256": videos[unit].get("subtitle_sha256"), "cutoffs": times})
    if args.stage == "verify":
        print(f"Read back {len(units)} {dataset} units", flush=True)


if __name__ == "__main__":
    main()
