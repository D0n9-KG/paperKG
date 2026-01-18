"""Neo4j graph mapper with Paper-centric model and evidence nodes."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from storage.neo4j.neo4j_utils import Neo4jConnection


def _hash_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _to_label(name: str) -> str:
    parts = [p for p in name.replace("-", "_").split("_") if p]
    return "".join([p[:1].upper() + p[1:] for p in parts]) or "Node"


def _normalize_doi(doi: Optional[str]) -> Optional[str]:
    if not doi:
        return None
    doi = doi.strip()
    if not doi:
        return None
    lowered = doi.lower()
    if "doi.org/" in lowered:
        doi = doi.split("doi.org/")[-1]
    return doi.strip().lower() or None


def _citation_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _normalize_keyword(keyword: str) -> str:
    return keyword.strip().lower()


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _collect_scalar_props(obj: Dict[str, Any]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for k, v in obj.items():
        if _is_scalar(v):
            props[k] = v
        elif isinstance(v, list) and all(_is_scalar(item) for item in v):
            props[k] = list(v)
    return props


def _collect_all_props(obj: Dict[str, Any]) -> Dict[str, Any]:
    props: Dict[str, Any] = {}
    for k, v in obj.items():
        if _is_scalar(v):
            props[k] = v
        elif isinstance(v, list):
            if all(_is_scalar(item) for item in v):
                props[k] = list(v)
            else:
                props[k] = json.dumps(v, ensure_ascii=False)
        elif isinstance(v, dict):
            props[k] = json.dumps(v, ensure_ascii=False)
        else:
            props[k] = str(v)
    return props


class GraphMapper:
    """Map PaperKG JSON into Neo4j with Paper nodes, evidence nodes, and citations."""

    def __init__(self, conn: Optional[Neo4jConnection] = None):
        self.conn = conn
        self.node_cache: Dict[str, str] = {}
        self.visited_paths: List[str] = []

    def map(self, data: Dict[str, Any], paper_uid: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        self.visited_paths = []
        self._collect_paths(data)

        paper_meta = data.get("paper_metadata", {}) if isinstance(data.get("paper_metadata"), dict) else {}
        key_type, key_value = self._paper_key_from_metadata(paper_meta, paper_uid)
        paper_props = self._build_paper_properties(paper_meta, key_type, key_value)

        paper_id = None
        if not dry_run:
            paper_id = self._merge_paper_node(key_type, key_value, paper_props)

        # Map metadata into independent, mergeable nodes
        if isinstance(paper_meta, dict):
            self._map_metadata_nodes(paper_meta, paper_id, dry_run)

        # Research narrative section
        research = data.get("research_narrative")
        if isinstance(research, dict):
            label = _to_label("research_narrative")
            uid = _hash_id(f"{key_value}:research_narrative")
            props = _collect_scalar_props(research)
            props["_path"] = "research_narrative"
            research_id = self._ensure_node(label, uid, props, dry_run)
            self._ensure_relationship(paper_id, research_id, "HAS_RESEARCH_NARRATIVE", dry_run)
            self._map_object(research, research_id, uid, "research_narrative", dry_run)

        # Multimedia content section
        multimedia = data.get("multimedia_content")
        if isinstance(multimedia, dict):
            label = _to_label("multimedia_content")
            uid = _hash_id(f"{key_value}:multimedia_content")
            props = _collect_scalar_props(multimedia)
            props["_path"] = "multimedia_content"
            multimedia_id = self._ensure_node(label, uid, props, dry_run)
            self._ensure_relationship(paper_id, multimedia_id, "HAS_MULTIMEDIA_CONTENT", dry_run)
            self._map_multimedia_content(multimedia, multimedia_id, uid, paper_id, key_value, dry_run)

        # Foundational works relationships
        self._map_foundational_relationships(data, paper_id, dry_run)

        return {
            "paper_uid": key_value,
            "visited_paths": self.visited_paths,
        }

    def _paper_key_from_metadata(self, metadata: Dict[str, Any], fallback_uid: Optional[str]) -> Tuple[str, str]:
        doi = _normalize_doi(metadata.get("doi") or metadata.get("url"))
        if doi:
            return "doi", doi

        title = str(metadata.get("title") or "").strip()
        year = str(metadata.get("publication_year") or "").strip()
        authors = metadata.get("authors") if isinstance(metadata.get("authors"), list) else []
        author_names = []
        for author in authors:
            if isinstance(author, dict):
                name = author.get("full_name") or author.get("family_name") or author.get("given_name")
                if name:
                    author_names.append(str(name))
        seed_parts = [part for part in [title, year, "|".join(author_names[:3])] if part]
        seed = "|".join(seed_parts) or (fallback_uid or "unknown-paper")
        return "citation_hash", _citation_hash(seed)

    def _paper_key_from_reference(self, reference: Dict[str, Any]) -> Tuple[str, str]:
        doi = _normalize_doi(reference.get("doi"))
        if doi:
            return "doi", doi
        citation = str(reference.get("citation") or "").strip()
        seed = citation or json.dumps(reference, ensure_ascii=False)
        return "citation_hash", _citation_hash(seed)

    def _build_paper_properties(self, metadata: Dict[str, Any], key_type: str, key_value: str) -> Dict[str, Any]:
        if key_type == "doi":
            return {"doi": key_value}
        return {"citation_hash": key_value}

    def _merge_paper_node(self, key_type: str, key_value: str, props: Dict[str, Any]) -> Optional[str]:
        if not self.conn:
            return None
        self.conn.merge_node("Paper", {key_type: key_value}, props)
        return self.conn.get_node_id("Paper", key_type, key_value)

    def _ensure_node(self, label: str, uid: str, props: Dict[str, Any], dry_run: bool) -> Optional[str]:
        if dry_run or not self.conn:
            return None
        props = dict(props)
        props["uid"] = uid
        self.conn.merge_node(label, {"uid": uid}, props)
        return self.conn.get_node_id(label, "uid", uid)

    def _ensure_relationship(self, from_id: Optional[str], to_id: Optional[str], rel_type: str, dry_run: bool,
                             props: Optional[Dict[str, Any]] = None):
        if dry_run or not self.conn:
            return
        if from_id is None or to_id is None:
            return
        self.conn.create_relationship(from_id, to_id, rel_type, props or {})

    def _merge_named_node(self, label: str, name: str, extra_props: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if not self.conn:
            return None
        name_norm = _normalize_text(name)
        props = {"name": name, "name_norm": name_norm}
        if extra_props:
            props.update(extra_props)
        self.conn.merge_node(label, {"name_norm": name_norm}, props)
        return self.conn.get_node_id(label, "name_norm", name_norm)

    def _merge_value_node(self, label: str, value: str, extra_props: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if not self.conn:
            return None
        value_norm = _normalize_text(value)
        props = {"value": value, "value_norm": value_norm}
        if extra_props:
            props.update(extra_props)
        self.conn.merge_node(label, {"value_norm": value_norm}, props)
        return self.conn.get_node_id(label, "value_norm", value_norm)

    def _merge_simple_node(self, label: str, key: str, value: Any, extra_props: Optional[Dict[str, Any]] = None) -> Optional[str]:
        if not self.conn:
            return None
        props = {key: value}
        if extra_props:
            props.update(extra_props)
        self.conn.merge_node(label, {key: value}, props)
        return self.conn.get_node_id(label, key, value)

    def _map_metadata_nodes(self, metadata: Dict[str, Any], paper_id: Optional[str], dry_run: bool):
        if not isinstance(metadata, dict):
            return

        title = metadata.get("title")
        if isinstance(title, str) and title.strip():
            node_id = self._merge_value_node("Title", title) if not dry_run else None
            self._ensure_relationship(paper_id, node_id, "HAS_TITLE", dry_run)

        abstract = metadata.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            node_id = self._merge_value_node("Abstract", abstract) if not dry_run else None
            self._ensure_relationship(paper_id, node_id, "HAS_ABSTRACT", dry_run)

        year = metadata.get("publication_year")
        if isinstance(year, str) and year.strip():
            node_id = self._merge_simple_node("PublicationYear", "value", year.strip()) if not dry_run else None
            self._ensure_relationship(paper_id, node_id, "PUBLISHED_IN_YEAR", dry_run)

        journal = metadata.get("journal_or_conference")
        if isinstance(journal, dict):
            journal_title = journal.get("title")
            if isinstance(journal_title, str) and journal_title.strip():
                journal_id = self._merge_named_node("Journal", journal_title) if not dry_run else None
                rel_props = {
                    "short_title": journal.get("short_title"),
                    "volume": journal.get("volume"),
                    "issue": journal.get("issue"),
                    "pages": journal.get("pages"),
                    "article_number": journal.get("article_number"),
                }
                rel_props = {k: v for k, v in rel_props.items() if v not in (None, "", [])}
                self._ensure_relationship(paper_id, journal_id, "PUBLISHED_IN", dry_run, rel_props)

        publisher = metadata.get("publisher")
        if isinstance(publisher, dict):
            publisher_name = publisher.get("name")
            if isinstance(publisher_name, str) and publisher_name.strip():
                pub_props = {
                    "location": publisher.get("location"),
                    "member_id": publisher.get("member_id"),
                }
                pub_props = {k: v for k, v in pub_props.items() if v not in (None, "", [])}
                publisher_id = self._merge_named_node("Publisher", publisher_name, pub_props) if not dry_run else None
                self._ensure_relationship(paper_id, publisher_id, "PUBLISHED_BY", dry_run)

        authors = metadata.get("authors")
        if isinstance(authors, list):
            for author in authors:
                if not isinstance(author, dict):
                    continue
                full_name = author.get("full_name") or " ".join(
                    [str(author.get("given_name") or "").strip(), str(author.get("family_name") or "").strip()]
                ).strip()
                if not full_name:
                    continue
                author_props = {
                    "given_name": author.get("given_name"),
                    "family_name": author.get("family_name"),
                    "orcid": author.get("orcid"),
                }
                author_props = {k: v for k, v in author_props.items() if v not in (None, "", [])}
                author_id = self._merge_named_node("Author", full_name, author_props) if not dry_run else None
                rel_props = {}
                if author.get("sequence"):
                    rel_props["sequence"] = author.get("sequence")
                self._ensure_relationship(paper_id, author_id, "AUTHORED_BY", dry_run, rel_props)

                affiliations = author.get("affiliations")
                if isinstance(affiliations, list):
                    for aff in affiliations:
                        if isinstance(aff, dict):
                            aff_name = aff.get("name")
                        else:
                            aff_name = aff
                        if not isinstance(aff_name, str) or not aff_name.strip():
                            continue
                        aff_id = self._merge_named_node("Affiliation", aff_name) if not dry_run else None
                        self._ensure_relationship(author_id, aff_id, "AFFILIATED_WITH", dry_run)

        identifiers = metadata.get("identifiers")
        if isinstance(identifiers, dict):
            for id_type in ("issn", "eissn"):
                value = identifiers.get(id_type)
                if isinstance(value, str) and value.strip():
                    id_node = None
                    if not dry_run and self.conn:
                        self.conn.merge_node("Identifier", {"id_type": id_type, "value": value}, {"id_type": id_type, "value": value})
                        id_node = self.conn.get_node_id("Identifier", "value", value)
                    self._ensure_relationship(paper_id, id_node, "HAS_IDENTIFIER", dry_run)
            isbn_list = identifiers.get("isbn")
            if isinstance(isbn_list, list):
                for isbn in isbn_list:
                    if not isinstance(isbn, str) or not isbn.strip():
                        continue
                    id_node = None
                    if not dry_run and self.conn:
                        self.conn.merge_node("Identifier", {"id_type": "isbn", "value": isbn}, {"id_type": "isbn", "value": isbn})
                        id_node = self.conn.get_node_id("Identifier", "value", isbn)
                    self._ensure_relationship(paper_id, id_node, "HAS_IDENTIFIER", dry_run)

        categories = metadata.get("categories")
        if isinstance(categories, dict):
            type_value = categories.get("type")
            if isinstance(type_value, str) and type_value.strip():
                type_id = self._merge_named_node("PublicationType", type_value) if not dry_run else None
                self._ensure_relationship(paper_id, type_id, "HAS_TYPE", dry_run)
            subtype_value = categories.get("subtype")
            if isinstance(subtype_value, str) and subtype_value.strip():
                subtype_id = self._merge_named_node("PublicationSubtype", subtype_value) if not dry_run else None
                self._ensure_relationship(paper_id, subtype_id, "HAS_SUBTYPE", dry_run)
            subjects = categories.get("subjects")
            if isinstance(subjects, list):
                for subject in subjects:
                    if isinstance(subject, str) and subject.strip():
                        node_id = self._merge_named_node("Subject", subject) if not dry_run else None
                        self._ensure_relationship(paper_id, node_id, "HAS_SUBJECT", dry_run)
            cats = categories.get("categories")
            if isinstance(cats, list):
                for cat in cats:
                    if isinstance(cat, str) and cat.strip():
                        node_id = self._merge_named_node("Category", cat) if not dry_run else None
                        self._ensure_relationship(paper_id, node_id, "HAS_CATEGORY", dry_run)

        # Keywords
        self._map_keywords(metadata, paper_id, dry_run)

        funding = metadata.get("funding")
        if isinstance(funding, list):
            for fund in funding:
                if not isinstance(fund, dict):
                    continue
                funder = fund.get("funder")
                if isinstance(funder, str) and funder.strip():
                    funder_id = self._merge_named_node("Funder", funder) if not dry_run else None
                    self._ensure_relationship(paper_id, funder_id, "FUNDED_BY", dry_run)
                    awards = fund.get("award")
                    if isinstance(awards, list):
                        for award in awards:
                            if isinstance(award, str) and award.strip():
                                award_id = self._merge_value_node("Award", award) if not dry_run else None
                                self._ensure_relationship(funder_id, award_id, "HAS_AWARD", dry_run)

    def _map_foundational_relationships(self, data: Dict[str, Any], paper_id: Optional[str], dry_run: bool):
        if not isinstance(data, dict):
            return
        background = data.get("research_narrative", {}).get("background", {})
        foundational = background.get("foundational_works", []) if isinstance(background, dict) else []
        references = data.get("multimedia_content", {}).get("references", {})
        ref_list = references.get("reference_list", []) if isinstance(references, dict) else []
        ref_lookup = {}
        if isinstance(ref_list, list):
            for ref in ref_list:
                if isinstance(ref, dict) and ref.get("id") is not None:
                    ref_lookup[str(ref.get("id")).strip()] = ref

        if not isinstance(foundational, list):
            return
        for item in foundational:
            if not isinstance(item, dict):
                continue
            citation = item.get("citation", {})
            ref_id = None
            if isinstance(citation, dict):
                ref_id = citation.get("value")
            if ref_id is not None:
                ref_id = str(ref_id).strip()

            ref_item = ref_lookup.get(ref_id) if ref_id else None
            if ref_item is None:
                seed = ref_id or json.dumps(item, ensure_ascii=False)
                ref_key_type, ref_key_value = ("citation_hash", _citation_hash(seed))
                ref_props = self._build_reference_properties({}, ref_key_type, ref_key_value)
            else:
                ref_key_type, ref_key_value = self._paper_key_from_reference(ref_item)
                ref_props = self._build_reference_properties(ref_item, ref_key_type, ref_key_value)

            ref_node = None
            if not dry_run:
                ref_node = self._merge_paper_node(ref_key_type, ref_key_value, ref_props)

            rel_props = {}
            if ref_id:
                rel_props["citation_id"] = ref_id
            if isinstance(citation, dict):
                rel_props["citation_value"] = citation.get("value")
                rel_props["citation_source_excerpt"] = citation.get("source_excerpt")
            contribution = item.get("contribution")
            if isinstance(contribution, dict):
                rel_props["contribution_value"] = contribution.get("value")
                rel_props["contribution_source_excerpt"] = contribution.get("source_excerpt")
            limitation = item.get("limitation_or_gap_identified")
            if isinstance(limitation, dict):
                rel_props["limitation_value"] = limitation.get("value")
                rel_props["limitation_source_excerpt"] = limitation.get("source_excerpt")
            rel_props = {k: v for k, v in rel_props.items() if v not in (None, "", [])}
            self._ensure_relationship(paper_id, ref_node, "FOUNDATIONAL", dry_run, rel_props)

    def _map_multimedia_content(self, multimedia: Dict[str, Any], multimedia_id: Optional[str], multimedia_uid: str,
                                paper_id: Optional[str], paper_uid: str, dry_run: bool):
        references = multimedia.get("references")
        if isinstance(references, dict):
            ref_list = references.get("reference_list")
            if isinstance(ref_list, list):
                for idx, item in enumerate(ref_list):
                    if not isinstance(item, dict):
                        continue
                    ref_key_type, ref_key_value = self._paper_key_from_reference(item)
                    ref_props = self._build_reference_properties(item, ref_key_type, ref_key_value)
                    ref_id = None
                    if not dry_run:
                        ref_id = self._merge_paper_node(ref_key_type, ref_key_value, ref_props)
                    rel_props = {"reference_index": idx}
                    if item.get("id") is not None:
                        rel_props["citation_id"] = str(item.get("id"))
                    if isinstance(item.get("citation"), str):
                        rel_props["citation_text"] = item.get("citation")
                    if isinstance(item.get("purpose"), str) and item.get("purpose").strip():
                        rel_props["purpose"] = item.get("purpose").strip()
                    self._ensure_relationship(paper_id, ref_id, "CITES", dry_run, rel_props)

        images = multimedia.get("images")
        if isinstance(images, dict):
            self._map_images(images, multimedia_id, multimedia_uid, dry_run)

        # Map remaining multimedia fields, skipping references list handled above
        self._map_object(
            multimedia,
            parent_id=multimedia_id,
            parent_uid=multimedia_uid,
            path_prefix="multimedia_content",
            dry_run=dry_run,
            skip_keys={"references", "images"},
        )

    def _build_reference_properties(self, reference: Dict[str, Any], key_type: str, key_value: str) -> Dict[str, Any]:
        if key_type == "doi":
            return {"doi": key_value}
        return {"citation_hash": key_value}

    def _map_images(self, images: Dict[str, Any], multimedia_id: Optional[str], multimedia_uid: str, dry_run: bool):
        if not isinstance(images, dict):
            return
        for fig_key, items in images.items():
            if not isinstance(items, list):
                continue
            fig_num = str(fig_key).strip()
            fig_uid = _hash_id(f"{multimedia_uid}:figure:{fig_num}")
            fig_props = {"figure_number": fig_num, "_path": f"multimedia_content.images.{fig_num}"}
            fig_id = self._ensure_node("Figure", fig_uid, fig_props, dry_run)
            self._ensure_relationship(multimedia_id, fig_id, "HAS_FIGURE", dry_run)

            for idx, item in enumerate(items):
                item_path = f"multimedia_content.images.{fig_num}[{idx}]"
                img_uid = _hash_id(f"{fig_uid}:image:{idx}")
                if isinstance(item, dict):
                    img_props = _collect_scalar_props(item)
                else:
                    img_props = {"value": item}
                img_props["index"] = idx
                img_props["_path"] = item_path
                img_id = self._ensure_node("Image", img_uid, img_props, dry_run)
                self._ensure_relationship(fig_id, img_id, "HAS_IMAGE", dry_run)

    def _map_keywords(self, metadata: Dict[str, Any], paper_id: Optional[str], dry_run: bool):
        keywords = metadata.get("keywords")
        if not isinstance(keywords, list):
            return
        for kw in keywords:
            if not isinstance(kw, str):
                continue
            kw_clean = kw.strip()
            if not kw_clean:
                continue
            kw_norm = _normalize_keyword(kw_clean)
            if not kw_norm:
                continue
            kw_id = None
            if not dry_run and self.conn:
                self.conn.merge_node("Keyword", {"name_norm": kw_norm}, {"name": kw_clean})
                kw_id = self.conn.get_node_id("Keyword", "name_norm", kw_norm)
            self._ensure_relationship(paper_id, kw_id, "HAS_KEYWORD", dry_run)

    def _map_object(self, obj: Dict[str, Any], parent_id: Optional[str], parent_uid: str, path_prefix: str,
                    dry_run: bool, skip_keys: Optional[Set[str]] = None):
        if not isinstance(obj, dict):
            return
        skip_keys = skip_keys or set()

        for key, value in obj.items():
            if key in skip_keys:
                continue

            path = f"{path_prefix}.{key}"

            if key == "source_excerpt" and isinstance(value, dict):
                source_uid = _hash_id(f"{parent_uid}:{path}")
                source_props = _collect_scalar_props(value)
                source_props["_path"] = path
                source_id = self._ensure_node("SourceExcerpt", source_uid, source_props, dry_run)
                self._ensure_relationship(parent_id, source_id, "HAS_SOURCE_EXCERPT", dry_run)

                segment_map = value.get("segment_map")
                if isinstance(segment_map, list):
                    for idx, segment in enumerate(segment_map):
                        if not isinstance(segment, dict):
                            continue
                        evidence_uid = _hash_id(f"{source_uid}:segment:{idx}")
                        evidence_props = _collect_scalar_props(segment)
                        evidence_props["index"] = idx
                        evidence_props["_path"] = f"{path}.segment_map[{idx}]"
                        evidence_id = self._ensure_node("Evidence", evidence_uid, evidence_props, dry_run)
                        self._ensure_relationship(source_id, evidence_id, "HAS_EVIDENCE", dry_run)
                continue

            if isinstance(value, dict):
                label = _to_label(key)
                uid = _hash_id(f"{parent_uid}:{path}")
                props = _collect_scalar_props(value)
                props["_path"] = path
                node_id = self._ensure_node(label, uid, props, dry_run)
                self._ensure_relationship(parent_id, node_id, f"HAS_{label.upper()}", dry_run)
                self._map_object(value, node_id, uid, path, dry_run)
                continue

            if isinstance(value, list):
                label = _to_label(key)
                container_uid = _hash_id(f"{parent_uid}:{path}:list")
                container_props = {"_path": path, "total_count": len(value)}
                container_id = self._ensure_node(label, container_uid, container_props, dry_run)
                self._ensure_relationship(parent_id, container_id, f"HAS_{label.upper()}", dry_run)

                for idx, item in enumerate(value):
                    item_path = f"{path}[{idx}]"
                    item_label = f"{label}Item"
                    item_uid = _hash_id(f"{parent_uid}:{item_path}")
                    if _is_scalar(item):
                        item_props = {"value": item, "index": idx, "_path": item_path}
                        item_id = self._ensure_node(item_label, item_uid, item_props, dry_run)
                        self._ensure_relationship(container_id, item_id, f"HAS_{item_label.upper()}", dry_run)
                    elif isinstance(item, dict):
                        item_props = _collect_scalar_props(item)
                        item_props["index"] = idx
                        item_props["_path"] = item_path
                        item_id = self._ensure_node(item_label, item_uid, item_props, dry_run)
                        self._ensure_relationship(container_id, item_id, f"HAS_{item_label.upper()}", dry_run)
                        self._map_object(item, item_id, item_uid, item_path, dry_run)
                continue

    def _collect_paths(self, obj: Any, prefix: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                self._collect_paths(v, path)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                path = f"{prefix}[{i}]"
                self._collect_paths(v, path)
        else:
            self.visited_paths.append(prefix)
