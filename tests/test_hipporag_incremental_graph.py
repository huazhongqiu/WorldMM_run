from collections import Counter

import igraph as ig
from hipporag import HippoRAG
from hipporag.utils.misc_utils import compute_mdhash_id


def graph_store():
    h = HippoRAG.__new__(HippoRAG)
    h.graph = ig.Graph(directed=False)
    h.node_to_node_stats = {}
    h.ent_node_to_chunk_ids = {}
    return h


def add_nodes(h, names):
    known = set(h.graph.vs["name"]) if "name" in h.graph.vs.attribute_names() else set()
    new = [name for name in names if name not in known]
    h.graph.add_vertices(len(new), attributes={"name": new})


def test_incremental_updates_keep_one_edge_per_stats_entry():
    h = graph_store()
    add_nodes(h, ["a", "b", "c"])
    h.node_to_node_stats = {("a", "b"): 1.0, ("b", "a"): 1.0, ("a", "a"): 1.0}
    h.add_new_edges()
    h.node_to_node_stats[("b", "c")] = 0.9
    h.add_new_edges()
    h.add_new_edges()
    names = h.graph.vs["name"]
    edges = Counter((names[s], names[t]) for s, t in h.graph.get_edgelist())
    assert h.graph.ecount() == 3
    assert edges[("a", "b")] == 2 and edges[("b", "c")] == 1
    assert sorted(h.graph.es["weight"]) == [0.9, 1.0, 1.0]


def test_fact_edges_count_only_chunks_not_yet_in_graph():
    h = graph_store()
    chunk = compute_mdhash_id("caption", prefix="chunk-")
    jake = compute_mdhash_id("Jake", prefix="entity-")
    cup = compute_mdhash_id("cup", prefix="entity-")
    triples = [[("Jake", "holds", "cup")]]
    h.add_fact_edges([chunk], triples)
    add_nodes(h, [chunk, jake, cup])
    h.add_fact_edges([chunk], triples)
    assert h.node_to_node_stats == {(jake, cup): 1.0, (cup, jake): 1.0}
    assert h.ent_node_to_chunk_ids == {jake: {chunk}, cup: {chunk}}
