from __future__ import annotations

import sys
from typing import List, Optional, Sequence, Tuple

if sys.version_info < (3, 8):
    from typing_extensions import Final
else:
    from typing import Final

import numpy as np

from bisheng_unstructured.cleaners.core import clean_bullets, replace_unicode_quotes
from bisheng_unstructured.documents.base import Page
from bisheng_unstructured.documents.elements import (
    Address,
    Element,
    ElementMetadata,
    EmailAddress,
    Link,
    ListItem,
    NarrativeText,
    Table,
    Text,
    Title,
)
from bisheng_unstructured.documents.xml import VALID_PARSERS, XMLDocument
from bisheng_unstructured.partition.text_type import (
    is_bulleted_text,
    is_email_address,
    is_possible_narrative_text,
    is_possible_title,
    is_us_city_state_zip,
)

TEXT_TAGS: Final[List[str]] = ["p", "a", "td", "span", "font"]
LIST_ITEM_TAGS: Final[List[str]] = ["li", "dd"]
HEADING_TAGS: Final[List[str]] = ["h1", "h2", "h3", "h4", "h5", "h6"]
TABLE_TAGS: Final[List[str]] = ["table", "tbody", "td", "tr"]
PAGEBREAK_TAGS: Final[List[str]] = ["hr"]
HEADER_OR_FOOTER_TAGS: Final[List[str]] = ["header", "footer"]
EMPTY_TAGS: Final[List[str]] = ["br", "hr"]

import re

RE_MULTISPACE_INCLUDING_NEWLINES = re.compile(pattern=r"\s+", flags=re.DOTALL)

import lxml
from loguru import logger
from lxml import etree
from lxml.html.clean import Cleaner


def norm_text(e):
    return re.sub(RE_MULTISPACE_INCLUDING_NEWLINES, " ", str(e) or "").strip()


def markdown_table(rows):
    def _format_row(r):
        content = " | ".join(r)
        content = "| " + content + " |"
        return content

    def _format_header(n):
        r = ["---"] * n
        content = " | ".join(r)
        content = "| " + content + " |"
        return content

    if not rows:
        return ""
    r0 = rows[0]
    max_cols = max(map(len, rows))
    first_cols = len(r0)

    content = [_format_row(r0)]
    content.append(_format_header(first_cols))
    for r in rows[1:]:
        content.append(_format_row(r))

    return "\n".join(content)


def parse_table_element(table_node, field_sep=" "):
    rows = []
    for tr in table_node.xpath(".//tr"):
        row = []
        for e in tr.getchildren():
            texts = tuple(e.xpath(".//text()"))
            texts = map(norm_text, texts)
            texts = [t for t in texts if t]
            field_text = field_sep.join(texts)
            if field_text:
                row.append(field_text)

        if row:
            rows.append(row)

    table_html = etree.tostring(table_node)

    cleaner = Cleaner(
        remove_unknown_tags=False,
        allow_tags=[
            "table",
            "tbody",
            "td",
            "tr",
            "th",
        ],
        style=True,
        page_structure=False,
    )
    clean_table_html = cleaner.clean_html(table_html).decode()
    text = markdown_table(rows)

    metadata = ElementMetadata(text_as_html=clean_table_html)
    return Table(text=text, metadata=metadata)


class TagsMixin:
    """Mixin that allows a class to retain tag information."""

    def __init__(
        self,
        *args,
        tag: Optional[str] = None,
        ancestortags: Sequence[str] = (),
        links: Sequence[Link] = [],
        emphasized_texts: Sequence[dict] = [],
        **kwargs,
    ):
        if tag is None:
            raise TypeError("tag argument must be passed and not None")
        else:
            self.tag = tag
        self.ancestortags = ancestortags
        self.links = links
        self.emphasized_texts = emphasized_texts
        super().__init__(*args, **kwargs)


class HTMLText(TagsMixin, Text):
    """Text with tag information."""

    pass


class HTMLAddress(TagsMixin, Address):
    """Address with tag information."""

    pass


class HTMLEmailAddress(TagsMixin, EmailAddress):
    """EmailAddress with tag information"""

    pass


class HTMLTitle(TagsMixin, Title):
    """Title with tag information."""

    pass


class HTMLNarrativeText(TagsMixin, NarrativeText):
    """NarrativeText with tag information."""

    pass


class HTMLListItem(TagsMixin, ListItem):
    """NarrativeText with tag information."""

    pass


class HTMLDocument(XMLDocument):
    """Class for handling HTML documents. Uses rules based parsing to identify sections
    of interest within the document."""

    def __init__(
        self,
        stylesheet: Optional[str] = None,
        parser: VALID_PARSERS = None,
        assemble_articles: bool = True,
    ):
        self.assembled_articles = assemble_articles
        super().__init__(stylesheet=stylesheet, parser=parser)

        self.cleaner = Cleaner(
            remove_unknown_tags=False,
            allow_tags=[
                "div",
                "main",
                "article",
                "figcaption",
                "header",
                "footer",
                "p",
                "br",
                "b",
                "span",
                "font",
                "li",
                "dd",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "table",
                "tbody",
                "td",
                "tr",
                "th",
            ],
            style=True,
            page_structure=False,
        )

    def _read(self) -> List[Page]:
        """Reads and structures and HTML document. If present, looks for article tags.
        if there are multiple article sections present, a page break is inserted between them.
        """
        if self._pages:
            return self._pages
        logger.info("Reading document ...")
        pages: List[Page] = []
        etree.strip_elements(self.document_tree, ["script"])
        root = _find_main(self.document_tree)

        content = etree.tostring(root, pretty_print=True)
        clean_html = self.cleaner.clean_html(content).decode()
        doc_tree = lxml.html.fromstring(clean_html)
        root = _find_main(doc_tree)

        language = "zh"
        articles = _find_articles(root, assemble_articles=True)
        page_number = 0
        page = Page(number=page_number)
        for article in articles:
            descendanttag_elems: Tuple[etree.Element, ...] = ()
            for tag_elem in article.iter():
                if tag_elem in descendanttag_elems:
                    # Prevent repeating something that's been flagged as text as we chase it
                    # down a chain
                    continue

                if tag_elem.tag == "table":
                    tables = []
                    emb_table = tag_elem.xpath(".//table")

                    filtered_table = []
                    groups = [tuple(e.iterdescendants()) for e in emb_table]
                    n = len(groups)
                    inds = list(range(n))
                    inds = sorted(inds, key=lambda i: len(groups[i]))
                    overlap = np.zeros((n, n))
                    for i in range(n):
                        for j in range(i + 1, n):
                            ind_j, ind_i = inds[j], inds[i]
                            g_j, g_i = groups[ind_j], groups[ind_i]
                            b = np.all([n in g_j for n in g_i])
                            if b:
                                overlap[i, j] = b
                                break

                    for i in range(n):
                        if np.sum(overlap[i, :]) == 0:
                            filtered_table.append(emb_table[inds[i]])

                    emb_table = filtered_table if filtered_table else [tag_elem]
                    for node in emb_table:
                        element = parse_table_element(node)
                        page.elements.append(element)

                    descendanttag_elems = tuple(tag_elem.iterdescendants())

                elif _is_text_tag(tag_elem):
                    element = _parse_tag(tag_elem, language)
                    if element is not None:
                        element.text = norm_text(element.text)
                        page.elements.append(element)
                        descendanttag_elems = tuple(tag_elem.iterdescendants())

                        # print('text tag', element.tag, str(element), element.to_dict())

                elif _is_container_with_text(tag_elem):
                    links = _get_links_from_tag(tag_elem)
                    emphasized_texts = _get_emphasized_texts_from_tag(tag_elem)
                    element = _text_to_element(
                        tag_elem.text, "div", (), links, emphasized_texts, language=language
                    )
                    if element is not None:
                        element.text = norm_text(element.text)
                        page.elements.append(element)

                        # print('container text tag', element.tag, str(element).strip())

                elif _is_bulleted_table(tag_elem):
                    bulleted_text = _bulleted_text_from_table(tag_elem)
                    page.elements.extend(bulleted_text)
                    descendanttag_elems = tuple(tag_elem.iterdescendants())

                elif is_list_item_tag(tag_elem):
                    element, next_element = _process_list_item(tag_elem)
                    if element is not None:
                        page.elements.append(element)
                        descendanttag_elems = _get_bullet_descendants(tag_elem, next_element)

                        # print('list item tag', element.tag, str(element).strip())

                elif tag_elem.tag in PAGEBREAK_TAGS and len(page.elements) > 0:
                    pages.append(page)
                    page_number += 1
                    page = Page(number=page_number)

            if len(page.elements) > 0:
                pages.append(page)
                page_number += 1
                page = Page(number=page_number)

        return pages

    def _read_ori(self) -> List[Page]:
        """Reads and structures and HTML document. If present, looks for article tags.
        if there are multiple article sections present, a page break is inserted between them.
        """
        if self._pages:
            return self._pages
        logger.info("Reading document ...")
        pages: List[Page] = []
        etree.strip_elements(self.document_tree, ["script"])
        root = _find_main(self.document_tree)
        articles = _find_articles(root, assemble_articles=self.assembled_articles)
        page_number = 0
        page = Page(number=page_number)

        for article in articles:
            descendanttag_elems: Tuple[etree.Element, ...] = ()
            for tag_elem in article.iter():
                if tag_elem in descendanttag_elems:
                    # Prevent repeating something that's been flagged as text as we chase it
                    # down a chain
                    continue

                if _is_text_tag(tag_elem):
                    element = _parse_tag(tag_elem)
                    if element is not None:
                        page.elements.append(element)
                        descendanttag_elems = tuple(tag_elem.iterdescendants())

                elif _is_container_with_text(tag_elem):
                    links = _get_links_from_tag(tag_elem)
                    emphasized_texts = _get_emphasized_texts_from_tag(tag_elem)
                    element = _text_to_element(tag_elem.text, "div", (), links, emphasized_texts)
                    if element is not None:
                        page.elements.append(element)

                elif _is_bulleted_table(tag_elem):
                    bulleted_text = _bulleted_text_from_table(tag_elem)
                    page.elements.extend(bulleted_text)
                    descendanttag_elems = tuple(tag_elem.iterdescendants())

                elif is_list_item_tag(tag_elem):
                    element, next_element = _process_list_item(tag_elem)
                    if element is not None:
                        page.elements.append(element)
                        descendanttag_elems = _get_bullet_descendants(tag_elem, next_element)

                elif tag_elem.tag in PAGEBREAK_TAGS and len(page.elements) > 0:
                    pages.append(page)
                    page_number += 1
                    page = Page(number=page_number)

            if len(page.elements) > 0:
                pages.append(page)
                page_number += 1
                page = Page(number=page_number)

        return pages

    def doc_after_cleaners(
        self,
        skip_headers_and_footers=False,
        skip_table_text=False,
        inplace=False,
    ) -> HTMLDocument:
        """Filters the elements and returns a new instance of the class based on the criteria
        specified. Note that the number of pages can change in the case that all elements on a
        page are filtered out.
        Parameters
        ----------
        skip_table_text:
            If True, skips text that is contained within a table element
        skip_headers_and_footers:
            If True, ignores any content that is within <header> or <footer> tags
        inplace:
            If True, document is modified in place and returned.
        """

        excluders = []
        if skip_headers_and_footers:
            excluders.append(in_header_or_footer)
        if skip_table_text:
            excluders.append(has_table_ancestor)

        pages = []
        page_number = 0
        new_page = Page(number=page_number)
        for page in self.pages:
            elements: List[Element] = []
            for el in page.elements:
                if not isinstance(el, TagsMixin):
                    raise ValueError(
                        f"elements of class {self.__class__} should be of type HTMLTitle "
                        f"HTMLNarrativeText, or HTMLListItem but "
                        f"object has an element of type {type(el)}",
                    )
                if not any(excluder(el) for excluder in excluders):
                    elements.append(el)
                if skip_headers_and_footers and "footer" in tuple(el.ancestortags) + (el.tag,):
                    break
            if elements:
                new_page.elements = elements
                pages.append(new_page)
                page_number += 1
                new_page = Page(number=page_number)
        if inplace:
            self._pages = pages
            self._elements = None
            return self
        else:
            out = self.__class__.from_pages(pages)
            if not isinstance(out, HTMLDocument):
                # NOTE(robinson) - Skipping for test coverage because this condition is impossible.
                # Added type check because from_pages is a method on Document. Without the type
                # check, mypy complains about returning Document instead of HTMLDocument
                raise ValueError(f"Unexpected class: {self.__class__.__name__}")  # pragma: no cover
            return out


def _get_links_from_tag(tag_elem: etree.Element) -> List[Link]:
    links: List[Link] = []
    href = tag_elem.get("href")
    if href:
        links.append({"text": tag_elem.text, "url": href})

    for tag in tag_elem.iterdescendants():
        href = tag.get("href")
        if href:
            links.append({"text": tag.text, "url": href})
    return links


def _get_emphasized_texts_from_tag(tag_elem: etree.Element) -> List[dict]:
    """Get emphasized texts enclosed in <strong>, <em>, <span>, <b>, <i> tags
    from a tag element in HTML"""
    emphasized_texts = []
    tags_to_track = ["strong", "em", "span", "b", "i"]
    if tag_elem is None:
        return []

    if tag_elem.tag in tags_to_track:
        text = _construct_text(tag_elem, False)
        if text:
            emphasized_texts.append({"text": text, "tag": tag_elem.tag})

    for descendant_tag_elem in tag_elem.iterdescendants(*tags_to_track):
        text = _construct_text(descendant_tag_elem, False)
        if text:
            emphasized_texts.append({"text": text, "tag": descendant_tag_elem.tag})

    return emphasized_texts


def _parse_tag(
    tag_elem: etree.Element,
    language="en",
) -> Optional[Element]:
    """Converts an etree element to a Text element if there is applicable text in the element.
    Ancestor tags are kept so they can be used for filtering or classification without
    processing the document tree again. In the future we might want to keep descendants too,
    but we don't have a use for them at the moment."""
    ancestortags: Tuple[str, ...] = tuple(el.tag for el in tag_elem.iterancestors())[::-1]
    links = _get_links_from_tag(tag_elem)
    emphasized_texts = _get_emphasized_texts_from_tag(tag_elem)

    if tag_elem.tag == "script":
        return None
    text = _construct_text(tag_elem)
    if not text:
        return None
    return _text_to_element(
        text,
        tag_elem.tag,
        ancestortags,
        links=links,
        emphasized_texts=emphasized_texts,
        language=language,
    )


def _text_to_element(
    text: str,
    tag: str,
    ancestortags: Tuple[str, ...],
    links: List[Link] = [],
    emphasized_texts: List[dict] = [],
    language="en",
) -> Optional[Element]:
    """Given the text of an element, the tag type and the ancestor tags, produces the appropriate
    HTML element."""
    if is_bulleted_text(text):
        if not clean_bullets(text):
            return None
        return HTMLListItem(
            text=clean_bullets(text),
            tag=tag,
            ancestortags=ancestortags,
            links=links,
            emphasized_texts=emphasized_texts,
        )
    elif is_us_city_state_zip(text):
        return HTMLAddress(
            text=text,
            tag=tag,
            ancestortags=ancestortags,
            links=links,
            emphasized_texts=emphasized_texts,
        )
    elif is_email_address(text):
        return HTMLEmailAddress(text=text, tag=tag, links=links, emphasized_texts=emphasized_texts)

    if len(text) < 2:
        return None
    elif is_narrative_tag(text, tag):
        return HTMLNarrativeText(
            text,
            tag=tag,
            ancestortags=ancestortags,
            links=links,
            emphasized_texts=emphasized_texts,
        )
    elif is_possible_title(text, language=language):
        return HTMLTitle(
            text,
            tag=tag,
            ancestortags=ancestortags,
            links=links,
            emphasized_texts=emphasized_texts,
        )
    else:
        return HTMLText(
            text,
            tag=tag,
            ancestortags=ancestortags,
            links=links,
            emphasized_texts=emphasized_texts,
        )


def _is_container_with_text(tag_elem: etree.Element) -> bool:
    """Checks if a tag is a container that also happens to contain text.
    Example
    -------
    <div>Hi there,
        <div>This is my message.</div>
        <div>Please read my message!</div>
    </div>
    """
    if tag_elem.tag != "div" or len(tag_elem) == 0:
        return False

    if tag_elem.text is None or tag_elem.text.strip() == "":
        return False

    return True


def is_narrative_tag(text: str, tag: str) -> bool:
    """Uses tag information to infer whether text is narrative."""
    return tag not in HEADING_TAGS and is_possible_narrative_text(text)


def _construct_text(tag_elem: etree.Element, include_tail_text: bool = True) -> str:
    """Extracts text from a text tag element."""
    text = ""
    for item in tag_elem.itertext():
        if item:
            text += item

    if include_tail_text and tag_elem.tail:
        text = text + tag_elem.tail

    text = replace_unicode_quotes(text)
    return text.strip()


def _is_text_tag(tag_elem: etree.Element, max_predecessor_len: int = 5) -> bool:
    """Deteremines if a tag potentially contains narrative text."""
    # NOTE(robinson) - Only consider elements with limited depth. Otherwise,
    # it could be the text representation of a giant div
    # Exclude empty tags from tag_elem
    empty_elems_len = len([el for el in tag_elem.getchildren() if el.tag in EMPTY_TAGS])
    if len(tag_elem) > max_predecessor_len + empty_elems_len:
        return False

    if tag_elem.tag in TEXT_TAGS + HEADING_TAGS:
        return True

    # NOTE(robinson) - This indicates that a div tag has no children. If that's the
    # case and the tag has text, its potential a text tag
    children = tag_elem.getchildren()
    if tag_elem.tag == "div" and len(children) == 0:
        return True

    if _has_adjacent_bulleted_spans(tag_elem, children):
        return True

    return False


def _process_list_item(
    tag_elem: etree.Element,
    max_predecessor_len: int = 5,
) -> Tuple[Optional[Element], etree.Element]:
    """If an etree element contains bulleted text, extracts the relevant bulleted text
    and converts it to ListItem objects. Also returns the next html elements so that
    we can skip processing if bullets are found in a div element."""
    if tag_elem.tag in LIST_ITEM_TAGS:
        text = _construct_text(tag_elem)
        links = _get_links_from_tag(tag_elem)
        emphasized_texts = _get_emphasized_texts_from_tag(tag_elem)
        return (
            HTMLListItem(
                text=text,
                tag=tag_elem.tag,
                links=links,
                emphasized_texts=emphasized_texts,
            ),
            tag_elem,
        )

    elif tag_elem.tag == "div":
        text = _construct_text(tag_elem)
        next_element = tag_elem.getnext()
        if next_element is None:
            return None, None
        next_text = _construct_text(next_element)
        # NOTE(robinson) - Only consider elements with limited depth. Otherwise,
        # it could be the text representation of a giant div
        empty_elems_len = len([el for el in tag_elem.getchildren() if el.tag in EMPTY_TAGS])
        if len(tag_elem) > max_predecessor_len + empty_elems_len:
            return None, None
        if next_text:
            return HTMLListItem(text=next_text, tag=next_element.tag), next_element

    return None, None


def _get_bullet_descendants(element, next_element) -> Tuple[etree.Element, ...]:
    descendants = []
    if element is not None and next_element is not None:
        descendants += list(next_element.iterdescendants())
    descendanttag_elems = tuple(descendants)
    return descendanttag_elems


def is_list_item_tag(tag_elem: etree.Element) -> bool:
    """Checks to see if a tag contains bulleted text."""
    if tag_elem.tag in LIST_ITEM_TAGS or (
        tag_elem.tag == "div" and is_bulleted_text(_construct_text(tag_elem))
    ):
        return True
    return False


def _bulleted_text_from_table(table) -> List[Element]:
    """Extracts bulletized narrative text from a table.
    NOTE: if a table has mixed bullets and non-bullets, only bullets are extracted.
    I.e., _read() will drop non-bullet narrative text in the table.
    """
    bulleted_text: List[Element] = []
    rows = table.findall(".//tr")
    for row in rows:
        text = _construct_text(row)
        if is_bulleted_text(text):
            bulleted_text.append(HTMLListItem(text=clean_bullets(text), tag=row.tag))
    return bulleted_text


def _is_bulleted_table(tag_elem) -> bool:
    """Checks to see if a table element contains bulleted text."""
    if tag_elem.tag != "table":
        return False

    rows = tag_elem.findall(".//tr")
    for row in rows:
        text = _construct_text(row)
        if text and not is_bulleted_text(text):
            return False

    return True


def _has_adjacent_bulleted_spans(tag_elem: etree.Element, children: List[etree.Element]) -> bool:
    """Checks to see if a div contains two or more adjacent spans beginning with a bullet. If
    this is the case, it is treated as a single bulleted text element."""
    if tag_elem.tag == "div":
        all_spans = all(child.tag == "span" for child in children)
        _is_bulleted = children[0].text is not None and is_bulleted_text(children[0].text)
        if all_spans and _is_bulleted:
            return True
    return False


def has_table_ancestor(element: TagsMixin) -> bool:
    """Checks to see if an element has ancestors that are table elements. If so, we consider
    it to be a table element rather than a section of narrative text."""
    return any(ancestor in TABLE_TAGS for ancestor in element.ancestortags)


def in_header_or_footer(element: TagsMixin) -> bool:
    """Checks to see if an element is contained within a header or a footer tag."""
    if any(ancestor in HEADER_OR_FOOTER_TAGS for ancestor in element.ancestortags):
        return True
    return False


def _find_main(root: etree.Element) -> etree.Element:
    """Finds the main tag of the HTML document if it exists. Otherwise, returns the
    whole document."""
    main_tag_elem = root.find(".//main")
    return main_tag_elem if main_tag_elem is not None else root


def _find_articles(root: etree.Element, assemble_articles: bool = True) -> List[etree.Element]:
    """Tries to break the HTML document into distinct articles. If there are no article
    tags, the entire document is returned as a single item list."""
    if assemble_articles is False:
        return root

    articles = root.findall(".//article")
    if len(articles) == 0:
        # NOTE(robinson) - ref: https://schema.org/Article
        articles = root.findall(".//div[@itemprop='articleBody']")
    return [root] if len(articles) == 0 else articles
