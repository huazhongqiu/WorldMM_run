"""Read the authoritative VideoSpy YAML model + retrieval overlays."""
import copy
import json
import os
from pathlib import Path
import yaml

SOURCE_DIR = Path('/myworkspace/projects/videospy/config/videospy')

def deep_merge(target, overlay):
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_merge(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target

def load_videospy_config(dataset=None):
    raw = os.environ.get('WORLDMM_RETRIEVAL_CONFIG')
    dataset = dataset or os.environ.get('WORLDMM_DATASET', '')
    if raw:
        paths = json.loads(raw) if raw.lstrip().startswith('[') else [p.strip() for p in raw.split(',') if p.strip()]
    else:
        root = Path(os.environ.get('WORLDMM_VIDEOSPY_CONFIG_DIR', str(SOURCE_DIR)))
        name = 'videomme' if dataset in ('videomme_long','videomme') else dataset
        paths = [str(root/'lmdeploy.yml')]
        if name:
            paths.append(str(root/f'{name}_retrieval_best.yml'))
    if not isinstance(paths, list) or not paths or not all(isinstance(p,str) for p in paths):
        raise ValueError('WORLDMM_RETRIEVAL_CONFIG must be an ordered YAML path list')
    default = Path(paths[0]).parent/'default.yml'
    ordered = [default] + [Path(p) for p in paths if Path(p) != default]
    merged = {}
    for path in ordered:
        with path.open(encoding='utf-8') as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise ValueError(f'Expected YAML mapping: {path}')
        deep_merge(merged, value)
    return merged

def visual_retrieval_options(dataset):
    config = load_videospy_config(dataset)['tools']['semantic_search']
    if config.get('index_method', 'single') != 'single':
        raise ValueError('WorldMM visual adapter supports only VideoSpy single retrieval')
    single = config.get('single', {})
    supported = {'keyword_search','keyword_top_p','clip_token_top_k','query_token_top_k','dense_weight','center_embeddings','bm25_k1','bm25_b'}
    if set(single) - supported:
        raise ValueError(f'Unsupported single retrieval options: {sorted(set(single)-supported)}')
    if single.get('keyword_top_p') is not None:
        raise ValueError('WorldMM supports rank top-k token sets only; keyword_top_p must be null')
    for key in ('clip_token_top_k','query_token_top_k'):
        value = single.get(key,512)
        if not isinstance(value,int) or not 1 <= value <= 512:
            raise ValueError(f'{key} must be in [1,512]')
    dense_weight = float(single.get('dense_weight', .5))
    k1, b = float(single.get('bm25_k1',1.2)),float(single.get('bm25_b',.75))
    if not 0 <= dense_weight <= 1 or k1 < 0 or not 0 <= b <= 1:
        raise ValueError('Invalid dense_weight/bm25_k1/bm25_b')
    for name in ('keyword_search','center_embeddings'):
        if name in single and not isinstance(single[name],bool):
            raise ValueError(f'{name} must be a boolean')
    key = 'egolife_rerank' if dataset == 'egolife' else 'lvbench_rerank'
    rerank = config.get(key, {}) if dataset in ('egolife','lvbench') else {}
    defaults = {'subtitle_weight':0.,'recency_weight':0.,'temporal_nms':0.} if dataset == 'egolife' else {'time_weight':0.,'opening_weight':0.,'time_tau':10.}
    if not isinstance(rerank,dict) or set(rerank)-set(defaults):
        raise ValueError(f'Unsupported {key} options')
    options = {**defaults,**rerank}
    if any(float(v) < 0 for v in options.values()) or (dataset == 'egolife' and options['subtitle_weight'] > 1) or options.get('time_tau',1) <= 0:
        raise ValueError('Invalid rerank weights')
    enabled = any(options[k] for k in options if k != 'time_tau')
    return {**single,'sample_fps':config.get('sample_fps',1),'top_k':config.get('top_k'),'clip_seconds':config.get('clip_seconds',10),'dense_weight':dense_weight,'bm25_k1':k1,'bm25_b':b,'rerank_options':options,'rerank':enabled}
