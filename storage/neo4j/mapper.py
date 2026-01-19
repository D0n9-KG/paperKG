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
        self.node_meta: Dict[str, str] = {}
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
            self._map_metadata_nodes(paper_meta, paper_id, key_value, dry_run)

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
        self.node_meta[uid] = label
        return self.conn.get_node_id(label, "uid", uid)

    def _update_node_props_by_uid(self, uid: str, props: Dict[str, Any], dry_run: bool) -> None:
        if dry_run or not self.conn:
            return
        label = self.node_meta.get(uid)
        if not label:
            return
        merged = dict(props)
        merged["uid"] = uid
        self.conn.merge_node(label, {"uid": uid}, merged)

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

    def _map_metadata_nodes(self, metadata: Dict[str, Any], paper_id: Optional[str], paper_uid: str, dry_run: bool):
        if not isinstance(metadata, dict):
            return

        def _pick_str(value: Any) -> Optional[str]:
            if isinstance(value, str) and value.strip():
                return value.strip()
            return None

        titles_payload: List[Tuple[str, str]] = []
        main_title = _pick_str(metadata.get("title"))
        if main_title:
            titles_payload.append(("title", main_title))
        subtitle = _pick_str(metadata.get("subtitle"))
        if subtitle:
            titles_payload.append(("subtitle", subtitle))
        short_title = _pick_str(metadata.get("short_title"))
        if short_title:
            titles_payload.append(("short_title", short_title))

        titles = metadata.get("titles")
        if isinstance(titles, dict):
            for kind, values in titles.items():
                if not isinstance(values, list):
                    continue
                for value in values:
                    value_str = _pick_str(value)
                    if value_str:
                        titles_payload.append((str(kind), value_str))

        titles_id = None
        if titles_payload:
            titles_uid = _hash_id(f"{paper_uid}:titles")
            titles_props = {"_path": "paper_metadata.titles"}
            titles_id = self._ensure_node("Titles", titles_uid, titles_props, dry_run)
            self._ensure_relationship(paper_id, titles_id, "HAS_TITLES", dry_run)

            seen_titles: Set[Tuple[str, str]] = set()
            for kind, value in titles_payload:
                key = (kind, value)
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                uid = _hash_id(f"title_variant:{kind}:{value}")
                props = {"value": value, "kind": kind}
                title_id = self._ensure_node("TitleVariant", uid, props, dry_run)
                self._ensure_relationship(titles_id, title_id, "HAS_TITLE_VARIANT", dry_run)

        abstract = metadata.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            node_id = self._merge_value_node("Abstract", abstract) if not dry_run else None
            self._ensure_relationship(paper_id, node_id, "HAS_ABSTRACT", dry_run)

        year = metadata.get("publication_year")
        if isinstance(year, str) and year.strip():
            node_id = self._merge_simple_node("PublicationYear", "value", year.strip()) if not dry_run else None
            self._ensure_relationship(paper_id, node_id, "PUBLISHED_IN_YEAR", dry_run)

        journal = metadata.get("journal_or_conference")
        journal_id = None
        journal_title = None
        journal_pages = None
        journal_norm = None
        if isinstance(journal, dict):
            journal_title = _pick_str(journal.get("title"))
            journal_short = _pick_str(journal.get("short_title"))
            journal_pages = _pick_str(journal.get("pages"))
            journal_norm = _normalize_text(journal_title or journal_short or "") if (journal_title or journal_short) else None

            if journal_title or journal_short:
                journal_props: Dict[str, Any] = {}
                if journal_short:
                    journal_props["short_title"] = journal_short

                identifiers = metadata.get("identifiers") if isinstance(metadata.get("identifiers"), dict) else {}
                issn_values: Set[str] = set()
                if isinstance(identifiers, dict):
                    for key in ("issn", "eissn"):
                        value = _pick_str(identifiers.get(key))
                        if value:
                            issn_values.add(value)
                            journal_props[key] = value
                    issn_list = identifiers.get("issn_list")
                    if isinstance(issn_list, list):
                        for issn in issn_list:
                            value = _pick_str(issn)
                            if value:
                                issn_values.add(value)

                issn_key = None
                if issn_values:
                    issn_sorted = sorted(issn_values)
                    issn_key = "|".join(issn_sorted)
                    journal_props["issn_list"] = issn_sorted
                    journal_props["issn_key"] = issn_key

                if not dry_run:
                    if issn_key and self.conn:
                        if journal_title:
                            journal_props["name"] = journal_title
                            journal_props["name_norm"] = _normalize_text(journal_title)
                        self.conn.merge_node("Journal", {"issn_key": issn_key}, journal_props)
                        journal_id = self.conn.get_node_id("Journal", "issn_key", issn_key)
                    else:
                        name_for_merge = journal_title or journal_short or ""
                        journal_id = self._merge_named_node("Journal", name_for_merge, journal_props)

                rel_props = {"pages": journal_pages} if journal_pages else {}
                self._ensure_relationship(paper_id, journal_id, "PUBLISHED_IN", dry_run, rel_props or None)

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
                name_norm = _normalize_text(full_name)
                orcid = author.get("orcid")
                if isinstance(orcid, str):
                    orcid = orcid.strip()
                if not orcid:
                    orcid = None

                affiliations = author.get("affiliations")
                affiliation_names: List[str] = []
                affiliation_norms: List[str] = []
                if isinstance(affiliations, list):
                    for aff in affiliations:
                        if isinstance(aff, dict):
                            aff_name = aff.get("name")
                        else:
                            aff_name = aff
                        if not isinstance(aff_name, str) or not aff_name.strip():
                            continue
                        aff_name = aff_name.strip()
                        affiliation_names.append(aff_name)
                        affiliation_norms.append(_normalize_text(aff_name))

                affiliation_norms = sorted({v for v in affiliation_norms if v})
                affiliation_key = "|".join(affiliation_norms) if affiliation_norms else None

                author_props = {
                    "name": full_name,
                    "name_norm": name_norm,
                    "given_name": author.get("given_name"),
                    "family_name": author.get("family_name"),
                }
                if orcid:
                    author_props["orcid"] = orcid
                if affiliation_norms:
                    author_props["affiliation_norms"] = affiliation_norms

                author_id = None
                if not dry_run and self.conn:
                    if orcid:
                        author_props["author_key"] = f"orcid:{orcid.lower()}"
                        self.conn.merge_node("Author", {"orcid": orcid.lower()}, author_props)
                        author_id = self.conn.get_node_id("Author", "orcid", orcid.lower())
                    elif affiliation_key:
                        author_props["author_key"] = f"name:{name_norm}|aff:{affiliation_key}"
                        self.conn.merge_node("Author", {"author_key": author_props["author_key"]}, author_props)
                        author_id = self.conn.get_node_id("Author", "author_key", author_props["author_key"])
                    else:
                        author_props["author_key"] = f"name:{name_norm}|paper:{paper_uid}"
                        author_props["ambiguous"] = True
                        self.conn.merge_node("Author", {"author_key": author_props["author_key"]}, author_props)
                        author_id = self.conn.get_node_id("Author", "author_key", author_props["author_key"])

                rel_props = {}
                if author.get("sequence"):
                    rel_props["sequence"] = author.get("sequence")
                self._ensure_relationship(paper_id, author_id, "AUTHORED_BY", dry_run, rel_props)

                for aff_name in affiliation_names:
                    aff_id = self._merge_named_node("Affiliation", aff_name) if not dry_run else None
                    self._ensure_relationship(author_id, aff_id, "AFFILIATED_WITH", dry_run)

        identifiers = metadata.get("identifiers")
        identifiers_id = None
        if isinstance(identifiers, dict):
            has_identifier_data = False
            for key in ("issn", "eissn"):
                if _pick_str(identifiers.get(key)):
                    has_identifier_data = True
                    break
            if not has_identifier_data:
                isbn_list = identifiers.get("isbn")
                issn_list = identifiers.get("issn_list")
                issn_type = identifiers.get("issn_type")
                has_identifier_data = bool(isbn_list or issn_list or issn_type)

            if has_identifier_data:
                identifiers_uid = _hash_id(f"{paper_uid}:identifiers")
                identifiers_props = {"_path": "paper_metadata.identifiers"}
                identifiers_id = self._ensure_node("Identifiers", identifiers_uid, identifiers_props, dry_run)
                self._ensure_relationship(paper_id, identifiers_id, "HAS_IDENTIFIERS", dry_run)

            for id_type in ("issn", "eissn"):
                value = _pick_str(identifiers.get(id_type))
                if value:
                    id_node = None
                    if not dry_run and self.conn:
                        self.conn.merge_node("Identifier", {"id_type": id_type, "value": value}, {"id_type": id_type, "value": value})
                        id_node = self.conn.get_node_id("Identifier", "value", value)
                    target_id = identifiers_id or paper_id
                    self._ensure_relationship(target_id, id_node, "HAS_IDENTIFIER", dry_run)
            isbn_list = identifiers.get("isbn")
            if isinstance(isbn_list, list):
                for isbn in isbn_list:
                    value = _pick_str(isbn)
                    if not value:
                        continue
                    id_node = None
                    if not dry_run and self.conn:
                        self.conn.merge_node("Identifier", {"id_type": "isbn", "value": value}, {"id_type": "isbn", "value": value})
                        id_node = self.conn.get_node_id("Identifier", "value", value)
                    target_id = identifiers_id or paper_id
                    self._ensure_relationship(target_id, id_node, "HAS_IDENTIFIER", dry_run)
            issn_list = identifiers.get("issn_list")
            if isinstance(issn_list, list):
                for issn in issn_list:
                    value = _pick_str(issn)
                    if not value:
                        continue
                    id_node = None
                    if not dry_run and self.conn:
                        self.conn.merge_node("Identifier", {"id_type": "issn", "value": value}, {"id_type": "issn", "value": value})
                        id_node = self.conn.get_node_id("Identifier", "value", value)
                    target_id = identifiers_id or paper_id
                    self._ensure_relationship(target_id, id_node, "HAS_IDENTIFIER", dry_run)
            issn_type = identifiers.get("issn_type")
            if isinstance(issn_type, list):
                for entry in issn_type:
                    if not isinstance(entry, dict):
                        continue
                    value = _pick_str(entry.get("value"))
                    if not value:
                        continue
                    uid = _hash_id(f"issn_type:{entry.get('type')}:{value}")
                    props = _collect_scalar_props(entry)
                    issn_type_id = self._ensure_node("IssnType", uid, props, dry_run)
                    target_id = identifiers_id or paper_id
                    self._ensure_relationship(target_id, issn_type_id, "HAS_ISSN_TYPE", dry_run)

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

        # Top-level subjects
        subjects_top = metadata.get("subject")
        if isinstance(subjects_top, list):
            for subject in subjects_top:
                if isinstance(subject, str) and subject.strip():
                    node_id = self._merge_named_node("Subject", subject) if not dry_run else None
                    self._ensure_relationship(paper_id, node_id, "HAS_SUBJECT", dry_run)

        # Crossref date objects (hierarchical container)
        dates = metadata.get("dates")
        if isinstance(dates, dict):
            dates_uid = _hash_id(f"{paper_uid}:dates")
            dates_props = {"_path": "paper_metadata.dates"}
            dates_id = self._ensure_node("Dates", dates_uid, dates_props, dry_run)
            self._ensure_relationship(paper_id, dates_id, "HAS_DATES", dry_run)
            for kind, date_obj in dates.items():
                if not isinstance(date_obj, dict):
                    continue
                props = _collect_all_props(date_obj)
                props["kind"] = kind
                uid = _hash_id(f"date:{kind}:{json.dumps(props, ensure_ascii=False, sort_keys=True)}")
                date_id = self._ensure_node("Date", uid, props, dry_run)
                self._ensure_relationship(dates_id, date_id, "HAS_DATE", dry_run)

        # Language
        language = metadata.get("language")
        if isinstance(language, str) and language.strip():
            lang_id = self._merge_named_node("Language", language) if not dry_run else None
            self._ensure_relationship(paper_id, lang_id, "HAS_LANGUAGE", dry_run)

        # Issue / Volume / Article number (journal children when possible)
        issue_value = _pick_str(journal.get("issue")) if isinstance(journal, dict) else None
        if not issue_value:
            issue_value = _pick_str(metadata.get("issue"))
        volume_value = _pick_str(journal.get("volume")) if isinstance(journal, dict) else None
        if not volume_value:
            volume_value = _pick_str(metadata.get("volume"))
        article_value = _pick_str(journal.get("article_number")) if isinstance(journal, dict) else None
        if not article_value:
            article_value = _pick_str(metadata.get("article_number"))

        def _link_journal_child(label: str, value: Optional[str], rel_type: str):
            if not value:
                return
            if journal_id and (journal_title or journal_norm):
                journal_norm_value = journal_norm or _normalize_text(journal_title or "")
                value_norm = _normalize_text(value)
                uid = _hash_id(f"journal:{journal_norm_value}:{label}:{value_norm}")
                props = {
                    "value": value,
                    "value_norm": value_norm,
                    "journal": journal_title,
                    "journal_norm": journal_norm_value,
                }
                child_id = self._ensure_node(label, uid, props, dry_run)
                self._ensure_relationship(journal_id, child_id, rel_type, dry_run)
            else:
                child_id = self._merge_simple_node(label, "value", value) if not dry_run else None
                self._ensure_relationship(paper_id, child_id, rel_type, dry_run)

        _link_journal_child("Issue", issue_value, "HAS_ISSUE")
        _link_journal_child("Volume", volume_value, "HAS_VOLUME")
        _link_journal_child("ArticleNumber", article_value, "HAS_ARTICLE_NUMBER")

        # Open access
        open_access = metadata.get("open_access")
        if isinstance(open_access, dict):
            props = _collect_scalar_props(open_access)
            if props:
                seed = metadata.get("doi") or metadata.get("url") or metadata.get("title") or "open_access"
                uid = _hash_id(f"open_access:{seed}")
                oa_id = self._ensure_node("OpenAccess", uid, props, dry_run)
                self._ensure_relationship(paper_id, oa_id, "HAS_OPEN_ACCESS", dry_run)

        # Keywords
        self._map_keywords(metadata, paper_id, dry_run)

        funding = metadata.get("funding")
        if isinstance(funding, list):
            for fund in funding:
                if not isinstance(fund, dict):
                    continue
                funder = fund.get("funder")
                funder_doi = fund.get("funder_doi")
                funder_name = funder.strip() if isinstance(funder, str) and funder.strip() else None
                funder_doi = funder_doi.strip() if isinstance(funder_doi, str) and funder_doi.strip() else None

                funder_id = None
                if not dry_run and self.conn and (funder_name or funder_doi):
                    funder_props = {}
                    if funder_name:
                        funder_props["name"] = funder_name
                        funder_props["name_norm"] = _normalize_text(funder_name)
                    if funder_doi:
                        funder_props["funder_doi"] = funder_doi

                    if funder_doi:
                        self.conn.merge_node("Funder", {"funder_doi": funder_doi}, funder_props)
                        funder_id = self.conn.get_node_id("Funder", "funder_doi", funder_doi)
                    elif funder_name:
                        funder_id = self._merge_named_node("Funder", funder_name, funder_props)

                self._ensure_relationship(paper_id, funder_id, "FUNDED_BY", dry_run)
                awards = fund.get("award")
                if isinstance(awards, list):
                    for award in awards:
                        if isinstance(award, str) and award.strip():
                            award_id = self._merge_value_node("Award", award) if not dry_run else None
                            self._ensure_relationship(funder_id, award_id, "HAS_AWARD", dry_run)

    def _parse_citation_ids(self, value: Optional[str]) -> List[str]:
        if not value:
            return []
        text = str(value).strip()
        if not text:
            return []
        text = text.replace("[", "").replace("]", "")
        tokens = text.replace(";", ",").replace("；", ",").split(",")
        ids: List[str] = []
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            handled = False
            for sep in ("-", "~", "–", "—"):
                if sep in token:
                    parts = [p.strip() for p in token.split(sep) if p.strip()]
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        start = int(parts[0])
                        end = int(parts[1])
                        if start <= end:
                            for i in range(start, end + 1):
                                ids.append(str(i))
                        else:
                            for i in range(start, end - 1, -1):
                                ids.append(str(i))
                        handled = True
                        break
            if handled:
                continue
            if token.isdigit():
                ids.append(token)
                continue
            import re
            for m in re.findall(r"\d+", token):
                ids.append(m)
        seen = set()
        out = []
        for cid in ids:
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
        return out

    def _extract_ids_from_text(self, text: Optional[str]) -> List[str]:
        if not text:
            return []
        import re
        ids: List[str] = []
        for match in re.finditer(r"\[(.*?)\]", text):
            ids.extend(self._parse_citation_ids(match.group(1)))
        if ids:
            return ids
        for match in re.finditer(r"\(([^)]*?)\)", text):
            group = match.group(1)
            if re.search(r"[A-Za-z]", group):
                continue
            ids.extend(self._parse_citation_ids(group))
        return ids

    def _extract_doi_from_text(self, text: Optional[str]) -> Optional[str]:
        if not text:
            return None
        import re
        match = re.search(r"(10\\.[0-9]{4,9}/[^\\s\"<>]+)", text, flags=re.IGNORECASE)
        if not match:
            return None
        return _normalize_doi(match.group(1).rstrip(").,;"))

    def _map_foundational_relationships(self, data: Dict[str, Any], paper_id: Optional[str], dry_run: bool):
        if not isinstance(data, dict):
            return
        background = data.get("research_narrative", {}).get("background", {})
        foundational = background.get("foundational_works", []) if isinstance(background, dict) else []
        references = data.get("multimedia_content", {}).get("references", {})
        ref_list = references.get("reference_list", []) if isinstance(references, dict) else []
        ref_by_id: Dict[str, Dict[str, Any]] = {}
        ref_by_doi: Dict[str, Dict[str, Any]] = {}
        ref_by_citation: Dict[str, Dict[str, Any]] = {}
        ref_index: Dict[str, int] = {}
        if isinstance(ref_list, list):
            for idx, ref in enumerate(ref_list):
                if not isinstance(ref, dict):
                    continue
                ref_id = ref.get("id")
                if ref_id is not None:
                    ref_id = str(ref_id).strip()
                    ref_by_id[ref_id] = ref
                    ref_index[ref_id] = idx
                doi = _normalize_doi(ref.get("doi"))
                if doi:
                    ref_by_doi[doi] = ref
                citation = ref.get("citation")
                if isinstance(citation, str) and citation.strip():
                    ref_by_citation[_normalize_text(citation)] = ref

        if not isinstance(foundational, list):
            return
        for item in foundational:
            if not isinstance(item, dict):
                continue
            citation = item.get("citation", {})
            citation_value = ""
            citation_text = ""
            if isinstance(citation, dict):
                citation_value = str(citation.get("value") or "").strip()
                citation_text = str(citation.get("citation_text") or "").strip()
            ids = self._parse_citation_ids(citation_value)
            if not ids and citation_text:
                ids = self._extract_ids_from_text(citation_text)

            targets: List[Tuple[Optional[str], Optional[Dict[str, Any]]]] = []
            if ids:
                for cid in ids:
                    targets.append((cid, ref_by_id.get(cid)))
            else:
                ref_item = None
                doi = self._extract_doi_from_text(citation_text)
                if doi:
                    ref_item = ref_by_doi.get(doi)
                if ref_item is None and citation_text:
                    ref_item = ref_by_citation.get(_normalize_text(citation_text))
                targets.append((None, ref_item))

            for cid, ref_item in targets:
                if ref_item is None:
                    seed = citation_text or cid or json.dumps(item, ensure_ascii=False)
                    ref_key_type, ref_key_value = ("citation_hash", _citation_hash(seed))
                    ref_props = self._build_reference_properties({}, ref_key_type, ref_key_value)
                else:
                    ref_key_type, ref_key_value = self._paper_key_from_reference(ref_item)
                    ref_props = self._build_reference_properties(ref_item, ref_key_type, ref_key_value)

                ref_node = None
                if not dry_run:
                    ref_node = self._merge_paper_node(ref_key_type, ref_key_value, ref_props)

                rel_props = {}
                if cid:
                    rel_props["citation_id"] = cid
                if citation_value:
                    rel_props["citation_value"] = citation_value
                if citation_text:
                    rel_props["citation_text"] = citation_text
                if isinstance(citation, dict):
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

                cite_props = {}
                if cid:
                    cite_props["citation_id"] = cid
                if ref_item is not None:
                    if cid and cid in ref_index:
                        cite_props["reference_index"] = ref_index.get(cid)
                    if isinstance(ref_item.get("citation"), str):
                        cite_props["citation_text"] = ref_item.get("citation")
                    if isinstance(ref_item.get("purpose"), str) and ref_item.get("purpose").strip():
                        cite_props["purpose"] = ref_item.get("purpose").strip()
                elif citation_text:
                    cite_props["citation_text"] = citation_text
                cite_props = {k: v for k, v in cite_props.items() if v not in (None, "", [])}
                self._ensure_relationship(paper_id, ref_node, "CITES", dry_run, cite_props)

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

        formulas = multimedia.get("formulas")
        if isinstance(formulas, dict):
            formula_list = formulas.get("formula_list")
            if isinstance(formula_list, list):
                formulas_uid = _hash_id(f"{multimedia_uid}:formulas")
                formulas_props = {"_path": "multimedia_content.formulas"}
                formulas_id = self._ensure_node("Formulas", formulas_uid, formulas_props, dry_run)
                self._ensure_relationship(multimedia_id, formulas_id, "HAS_FORMULAS", dry_run)
                for idx, item in enumerate(formula_list):
                    item_path = f"multimedia_content.formulas.formula_list[{idx}]"
                    formula_uid = _hash_id(f"{formulas_uid}:formula:{idx}")
                    if isinstance(item, dict):
                        formula_props = _collect_scalar_props(item)
                    else:
                        formula_props = {"value": item}
                    formula_props["index"] = idx
                    formula_props["_path"] = item_path
                    formula_id = self._ensure_node("Formula", formula_uid, formula_props, dry_run)
                    self._ensure_relationship(formulas_id, formula_id, "HAS_FORMULA", dry_run)

        # Map remaining multimedia fields, skipping references list handled above
        self._map_object(
            multimedia,
            parent_id=multimedia_id,
            parent_uid=multimedia_uid,
            path_prefix="multimedia_content",
            dry_run=dry_run,
            skip_keys={"references", "images", "formulas"},
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
            if key == "foundational_works" and path_prefix.startswith("research_narrative.background"):
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
                # Flatten scalar lists onto parent node properties
                scalar_list: Optional[List[Any]] = None
                if all(_is_scalar(item) for item in value):
                    scalar_list = [item for item in value if _is_scalar(item) and item not in (None, "", [], {})]
                elif all(isinstance(item, dict) for item in value):
                    values = []
                    all_value = True
                    for item in value:
                        if not isinstance(item, dict):
                            all_value = False
                            break
                        v = item.get("value")
                        if _is_scalar(v) and v not in (None, "", [], {}):
                            values.append(v)
                        else:
                            all_value = False
                            break
                    if all_value:
                        scalar_list = values

                if scalar_list is not None:
                    self._update_node_props_by_uid(parent_uid, {key: scalar_list}, dry_run)
                    continue

                # Non-scalar lists: create child nodes directly (no container / item nodes)
                label_override = {
                    "formula_list": "Formula",
                }
                item_label = label_override.get(key, _to_label(key))
                for idx, item in enumerate(value):
                    item_path = f"{path}[{idx}]"
                    item_uid = _hash_id(f"{parent_uid}:{item_path}")
                    if _is_scalar(item):
                        item_props = {"value": item, "index": idx, "_path": item_path}
                        item_id = self._ensure_node(item_label, item_uid, item_props, dry_run)
                        self._ensure_relationship(parent_id, item_id, f"HAS_{item_label.upper()}", dry_run)
                    elif isinstance(item, dict):
                        item_props = _collect_scalar_props(item)
                        item_props["index"] = idx
                        item_props["_path"] = item_path
                        item_id = self._ensure_node(item_label, item_uid, item_props, dry_run)
                        self._ensure_relationship(parent_id, item_id, f"HAS_{item_label.upper()}", dry_run)
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
