"""
GraphRAG Engine — lightweight graph-based retrieval structures.

Provides data structures and algorithms for the CollectorGraphRAG workflow:
  - Text chunking with overlap
  - Entity/relation graph construction and deduplication
  - Community detection (networkx optional, union-find fallback)
  - Local and global query helpers
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TextUnit:
    """A chunk of text from a fetched document."""

    id: str
    doc_url: str
    doc_title: str
    content: str
    offset: int  # character offset in the original document


@dataclass
class Entity:
    """A named entity extracted from text units."""

    id: str  # normalized lowercase key
    name: str  # display name
    type: str  # e.g. "PERSON", "ORG", "EVENT", "METRIC"
    description: str
    source_urls: set[str] = field(default_factory=set)
    text_unit_ids: set[str] = field(default_factory=set)


@dataclass
class Relation:
    """A directed relation between two entities."""

    head_id: str
    relation: str
    tail_id: str
    quote: str
    url: str
    text_unit_id: str


@dataclass
class Community:
    """A community of related entities."""

    id: str
    entity_ids: list[str]
    report: str = ""  # filled by LLM summarisation


@dataclass
class GraphIndex:
    """In-memory graph index built from extracted elements."""

    entities: dict[str, Entity] = field(default_factory=dict)
    relations: list[Relation] = field(default_factory=list)
    adjacency: dict[str, set[str]] = field(default_factory=dict)
    text_units: list[TextUnit] = field(default_factory=list)
    communities: list[Community] = field(default_factory=list)

    # ----- stats -----

    @property
    def stats(self) -> dict[str, int]:
        return {
            "docs": len({tu.doc_url for tu in self.text_units}),
            "text_units": len(self.text_units),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "communities": len(self.communities),
        }


# ---------------------------------------------------------------------------
# Text chunking
# ---------------------------------------------------------------------------

def chunk_text_units(
    text: str,
    url: str,
    title: str,
    chunk_size: int = 2500,
    overlap: int = 200,
) -> list[TextUnit]:
    """Split *text* into overlapping chunks and return TextUnit objects."""
    units: list[TextUnit] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            units.append(TextUnit(
                id=f"{_slug(url)}_{idx}",
                doc_url=url,
                doc_title=title,
                content=chunk,
                offset=start,
            ))
            idx += 1
        start += chunk_size - overlap
    return units


# ---------------------------------------------------------------------------
# Entity normalisation
# ---------------------------------------------------------------------------

def normalize_entity_name(name: str) -> str:
    """Lowercase, strip accents and non-alphanumeric chars."""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = re.sub(r"[^a-z0-9 ]", "", name.lower())
    return name.strip()


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------

def merge_elements_into_graph(
    graph: GraphIndex,
    elements: dict[str, Any],
    url: str,
    text_unit_id: str,
) -> None:
    """Merge LLM-extracted elements into the running graph index.

    *elements* is expected to have keys: entities, relations, claims.
    """
    # --- entities ---
    for ent in elements.get("entities", []):
        if not isinstance(ent, dict):
            continue
        name = str(ent.get("name", "")).strip()
        if not name:
            continue
        eid = normalize_entity_name(name)
        if not eid:
            continue
        if eid in graph.entities:
            existing = graph.entities[eid]
            existing.source_urls.add(url)
            existing.text_unit_ids.add(text_unit_id)
            # merge description if richer
            new_desc = str(ent.get("description", ""))
            if len(new_desc) > len(existing.description):
                existing.description = new_desc[:300]
        else:
            graph.entities[eid] = Entity(
                id=eid,
                name=name,
                type=str(ent.get("type", "UNKNOWN")).upper(),
                description=str(ent.get("description", ""))[:300],
                source_urls={url},
                text_unit_ids={text_unit_id},
            )
        graph.adjacency.setdefault(eid, set())

    # --- relations ---
    for rel in elements.get("relations", []):
        if not isinstance(rel, dict):
            continue
        head = normalize_entity_name(str(rel.get("head", "")))
        tail = normalize_entity_name(str(rel.get("tail", "")))
        if not head or not tail or head == tail:
            continue
        # auto-create missing entities
        for nid, raw_name in ((head, rel.get("head", "")), (tail, rel.get("tail", ""))):
            if nid not in graph.entities:
                graph.entities[nid] = Entity(
                    id=nid,
                    name=str(raw_name).strip(),
                    type="UNKNOWN",
                    description="",
                    source_urls={url},
                    text_unit_ids={text_unit_id},
                )
                graph.adjacency.setdefault(nid, set())

        graph.adjacency[head].add(tail)
        graph.adjacency[tail].add(head)

        graph.relations.append(Relation(
            head_id=head,
            relation=str(rel.get("relation", "RELATED_TO"))[:100],
            tail_id=tail,
            quote=str(rel.get("quote", ""))[:300],
            url=url,
            text_unit_id=text_unit_id,
        ))


# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

def detect_communities(graph: GraphIndex, min_size: int = 2) -> list[Community]:
    """Detect communities among entity nodes.

    Tries *networkx* greedy modularity first; falls back to connected
    components via union-find.
    """
    if len(graph.entities) < min_size:
        if graph.entities:
            graph.communities = [Community(
                id="c0",
                entity_ids=list(graph.entities.keys()),
            )]
        return graph.communities

    try:
        communities = _detect_networkx(graph, min_size)
    except Exception:
        communities = _detect_union_find(graph, min_size)

    graph.communities = communities
    return communities


def _detect_networkx(graph: GraphIndex, min_size: int) -> list[Community]:
    import networkx as nx  # type: ignore[import-untyped]

    G = nx.Graph()
    G.add_nodes_from(graph.entities.keys())
    for rel in graph.relations:
        if rel.head_id in graph.entities and rel.tail_id in graph.entities:
            G.add_edge(rel.head_id, rel.tail_id)

    raw = nx.community.greedy_modularity_communities(G)
    comms: list[Community] = []
    for i, members in enumerate(sorted(raw, key=len, reverse=True)):
        ids = list(members)
        if len(ids) >= min_size:
            comms.append(Community(id=f"c{i}", entity_ids=ids))
    return comms


def _detect_union_find(graph: GraphIndex, min_size: int) -> list[Community]:
    parent: dict[str, str] = {eid: eid for eid in graph.entities}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for rel in graph.relations:
        if rel.head_id in parent and rel.tail_id in parent:
            union(rel.head_id, rel.tail_id)

    groups: dict[str, list[str]] = {}
    for eid in graph.entities:
        root = find(eid)
        groups.setdefault(root, []).append(eid)

    comms: list[Community] = []
    for i, (_, members) in enumerate(
        sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ):
        if len(members) >= min_size:
            comms.append(Community(id=f"c{i}", entity_ids=members))
    return comms


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def build_local_context_pack(
    graph: GraphIndex,
    seed_entity_ids: list[str],
    max_chars: int = 4000,
) -> str:
    """Build a compact local context pack from seed entities + neighbors."""
    included: set[str] = set()
    for eid in seed_entity_ids:
        if eid in graph.entities:
            included.add(eid)
            for neighbour in graph.adjacency.get(eid, set()):
                included.add(neighbour)

    parts: list[str] = []

    # entities
    ent_lines: list[str] = []
    for eid in sorted(included):
        ent = graph.entities.get(eid)
        if ent:
            ent_lines.append(f"- {ent.name} ({ent.type}): {ent.description[:120]}")
    if ent_lines:
        parts.append("### ENTITIES\n" + "\n".join(ent_lines[:20]))

    # relations among included entities
    rel_lines: list[str] = []
    for rel in graph.relations:
        if rel.head_id in included and rel.tail_id in included:
            head_name = graph.entities.get(rel.head_id, Entity(id="", name=rel.head_id, type="", description="")).name
            tail_name = graph.entities.get(rel.tail_id, Entity(id="", name=rel.tail_id, type="", description="")).name
            rel_lines.append(f"- {head_name} --[{rel.relation}]--> {tail_name}")
    if rel_lines:
        parts.append("### RELATIONS\n" + "\n".join(rel_lines[:15]))

    # top quotes linked to included entities
    tu_ids: set[str] = set()
    for eid in included:
        ent = graph.entities.get(eid)
        if ent:
            tu_ids.update(ent.text_unit_ids)

    quote_lines: list[str] = []
    for tu in graph.text_units:
        if tu.id in tu_ids and len(quote_lines) < 5:
            snippet = tu.content[:300].replace("\n", " ")
            quote_lines.append(f"- [{tu.doc_url}] {snippet}")
    if quote_lines:
        parts.append("### SUPPORTING QUOTES\n" + "\n".join(quote_lines))

    result = "\n\n".join(parts)
    return result[:max_chars]


def rank_communities_by_query(
    communities: list[Community],
    query_terms: set[str],
    entities: dict[str, Entity],
    top_m: int = 3,
) -> list[Community]:
    """Rank communities by keyword overlap with query terms."""

    def score(c: Community) -> int:
        names: set[str] = set()
        for eid in c.entity_ids:
            ent = entities.get(eid)
            if ent:
                names.update(ent.name.lower().split())
                names.update(ent.description.lower().split())
        return len(names & query_terms)

    ranked = sorted(communities, key=score, reverse=True)
    return ranked[:top_m]


# ---------------------------------------------------------------------------
# Tier heuristics
# ---------------------------------------------------------------------------

_TIER1_DOMAINS = {
    ".gov", ".mil",
    "reuters.com", "apnews.com", "bloomberg.com",
}
_TIER2_DOMAINS = {
    "nytimes.com", "bbc.com", "bbc.co.uk", "wsj.com",
    "washingtonpost.com", "theguardian.com", "ft.com",
    "cnbc.com", "cnn.com", "foxnews.com", "npr.org",
    "politico.com", "axios.com",
}


def infer_credibility_tier(url: str) -> int:
    """Heuristic credibility tier from URL domain."""
    url_lower = url.lower()
    for domain in _TIER1_DOMAINS:
        if domain in url_lower:
            return 1
    if "press release" in url_lower or "press-release" in url_lower:
        return 1
    for domain in _TIER2_DOMAINS:
        if domain in url_lower:
            return 2
    return 3


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _slug(url: str) -> str:
    """Create a short filesystem-safe slug from a URL."""
    clean = re.sub(r"https?://", "", url)
    clean = re.sub(r"[^a-zA-Z0-9]", "_", clean)
    return clean[:40]
