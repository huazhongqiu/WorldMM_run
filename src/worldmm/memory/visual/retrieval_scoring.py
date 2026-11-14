"""Pure, candidate-pool C/S/R scoring. No corpus encoding or index mutation.

Calibration and temporal priors follow VideoSpy index/centering.py and
benchmark/{egolife,lvbench}/index.py; token BM25 uses binary set frequency.
"""
import os
import re
from collections import Counter
import numpy as np

CALIBRATION_QUERIES = (
    'a person', 'people talking', 'someone walking', 'an object on a table', 'a room',
    'a person holding something', 'someone using a phone', 'people eating', 'an outdoor scene',
    'a car on the road', 'a person speaking to the camera', 'text on a screen', 'an animal',
    'someone cooking', 'a group of people', 'what happens next',
)

def retrieval_config():
    options = {}
    if os.environ.get("WORLDMM_RETRIEVAL_CONFIG") or os.environ.get("WORLDMM_EMBEDDING_BACKEND") == "freeret":
        from worldmm.retrieval_config import visual_retrieval_options
        options = visual_retrieval_options(os.environ.get("WORLDMM_DATASET", "").lower())
    def flag(name, default=False):
        value = os.environ.get('WORLDMM_RETRIEVAL_' + name, str(int(default))).lower()
        if value not in ('0', '1', 'true', 'false'):
            raise ValueError(f'Invalid WORLDMM_RETRIEVAL_{name}: {value}')
        return value in ('1', 'true')
    config = dict(center=flag('CENTER',options.get('center_embeddings',False)), sparse=flag('SPARSE',options.get('keyword_search',False)), rerank=flag('RERANK',options.get('rerank',False)))
    if config['sparse'] and os.environ.get('WORLDMM_EMBEDDING_BACKEND') in ('original', 'wemm4b'):
        raise ValueError('Sparse retrieval requires the FreeRet backend')
    config['options'] = options
    dataset = os.environ.get('WORLDMM_DATASET', '').lower()
    if config['rerank'] and dataset not in ('egolife', 'lvbench'):
        raise ValueError('R supports only EgoLife/LVBench; Video-MME R is not applicable')
    return config, dataset

def minmax(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return values
    span = np.ptp(values)
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)

def dense_scores(query, vectors, *, center=False, query_mean=None, visible_vectors=None):
    query, vectors = np.asarray(query, dtype=np.float64).reshape(-1), np.asarray(vectors, dtype=np.float64)
    raw = vectors @ query / (np.maximum(np.linalg.norm(vectors, axis=1), 1e-12) * max(np.linalg.norm(query), 1e-12))
    if not center or len(visible_vectors) < 2:
        return raw, raw.copy()
    q = query - np.asarray(query_mean)
    v = vectors - np.asarray(visible_vectors, dtype=np.float64).mean(axis=0)
    return raw, v @ q / (np.maximum(np.linalg.norm(v, axis=1), 1e-6) * max(np.linalg.norm(q), 1e-12))

def bm25_counts(query, documents, k1=1.2, b=.75):
    terms = set(query)
    n = len(documents)
    if not n:
        return np.zeros(0)
    lengths = np.array([sum(doc.values()) for doc in documents])
    average = float(lengths.mean()) or 1.
    df = Counter(t for doc in documents for t in terms.intersection(doc))
    idf = {t: np.log1p((n - df[t] + .5) / (df[t] + .5)) for t in terms}
    return np.array([sum(idf[t] * doc[t] * (k1 + 1) / (doc[t] + k1 * (1-b + b * length / average)) for t in terms.intersection(doc)) for doc, length in zip(documents, lengths)])

def bm25_sets(query, documents, k1=1.2, b=.75):
    return bm25_counts(set(query), [Counter(set(doc)) for doc in documents],k1,b)

# The same light English stemming/stopword policy used by the reference subtitle branch.
STOPWORDS = set('a an the and or of to in on at for from with as by is are was were be been being i you he she it we they me him her us them my your his its our their this that these those what which who whom whose when where why how do does did doing done am about into have has had not no yes so but if then there here just now oh uh um ah hmm okay ok yeah right like will would can could should shall may might must let s t re ll ve d m don didn doesn isn wasn aren'.split())
def subtitle_terms(text):
    words = (w for w in re.findall('[a-z]+', text.lower()) if w not in STOPWORDS)
    return Counter(next((w[:-len(s)] for s in ('ing', 'ed', 'es', 's') if w.endswith(s) and len(w)-len(s) >= 3), w) for w in words)

def rerank_scores(scores, rows, *, dataset, question, query_time, duration=None, options=None):
    scores = np.asarray(scores, dtype=np.float64)
    options = options or {}
    if dataset == 'egolife':
        if query_time is None:
            raise ValueError('EgoLife R requires query_time in wall-clock seconds')
        subtitle = minmax(bm25_counts(subtitle_terms(question), [subtitle_terms(r.get('subtitle') or '') for r in rows]))
        ends = np.array([r['end'] for r in rows])
        if np.any(ends > query_time):
            raise ValueError('Future clip in EgoLife candidate pool')
        weight = options.get('subtitle_weight',.2)
        return (1-weight) * scores + weight * subtitle - options.get('recency_weight',.04) * np.log1p(np.maximum(query_time - ends, 0) / 3600.)
    if dataset != 'lvbench':
        raise ValueError('Video-MME R is not applicable')
    stem = re.split(r'\n\([A-D]\)\s*', question)[0]
    stamps = [sum(int(p) * 60**i for i,p in enumerate(reversed(s.split(':')))) for s in re.findall(r'\b\d+:\d{2}(?::\d{2})?\b', stem)] if options.get('time_weight',1.) else []
    opening = options.get('opening_weight',.5) and re.search(r'\b(beginning|opening|at the start|start of the video)\b', stem, re.I)
    if not stamps and not opening:
        return scores.copy()
    result = minmax(scores)
    if opening and (duration is None or duration <= 0):
        raise ValueError('LVBench opening prior requires video_duration')
    for i, row in enumerate(rows):
        if stamps:
            gap = min(max(row['start'] - t, t - row['end'], 0) for t in stamps)
            result[i] += options.get('time_weight',1.) * np.exp(-gap / options.get('time_tau',10.))
        if opening:
            result[i] += options.get('opening_weight',.5) * np.exp(-row['start'] / (.1 * duration))
    return result

def select_indices(scores, top_k, *, dataset='', rerank=False, ends=None, temporal_nms=60.):
    selected = []
    if top_k <= 0:
        return selected
    for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i)):
        if dataset == 'egolife' and rerank and any(temporal_nms > 0 and abs(ends[i]-ends[j]) <= temporal_nms for j in selected):
            continue
        selected.append(i)
        if len(selected) == top_k:
            break
    return selected

def wall_seconds(timestamp):
    value = str(int(timestamp)).zfill(9)
    day, clock = int(value[:-8]), value[-8:]
    return day * 86400 + int(clock[:2])*3600 + int(clock[2:4])*60 + int(clock[4:6]) + int(clock[6:])/100.
