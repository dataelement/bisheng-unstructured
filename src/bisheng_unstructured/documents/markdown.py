import re

from loguru import logger
import lxml
from lxml import etree
from lxml.html.clean import Cleaner

RE_MULTISPACE_INCLUDING_NEWLINES = re.compile(pattern=r"\s+", flags=re.DOTALL)


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
    head_cols_threhold = 3
    if max_cols - first_cols <= head_cols_threhold:
        first_cols = max_cols

    content = [_format_row(r0)]
    content.append(_format_header(first_cols))
    for r in rows[1:]:
        content.append(_format_row(r))

    return "\n".join(content)


def norm_texts(texts):
    return [t for t in map(norm_text, texts) if t]


def transform_html_table_to_md(html_table_str, field_sep=" "):
    try:
        table_node = lxml.html.fromstring(html_table_str)
    except Exception as e:
        logger.error("html table parse error: %s", e)
        return dict(text="", html="", error=str(e))
    rows = []
    for thead_node in table_node.xpath(".//thead"):
        row = norm_texts(thead_node.xpath(".//th//text()"))
        if row:
            rows.append(row)

    rowspan_data = {}  # 用于存储跨行单元格的数据
    for index, tr in enumerate(table_node.xpath(".//tr")):
        row = []
        col_offset = 0  # 用于处理行合并，遍历的偏移
        for col_index, e in enumerate(tr.getchildren()):
            col_real_index = col_index + col_offset
            if index in rowspan_data and col_real_index in rowspan_data[index]:
                # 表示这个单元格是合并的，需要补充合并的，再往下走
                row.extend(rowspan_data[index][col_real_index])
                col_offset += len(rowspan_data[index][col_real_index])

            texts = norm_texts(e.xpath(".//text()"))
            field_text = field_sep.join(texts)

            colspan = int(e.get('colspan', 1))
            rowspan = int(e.get('rowspan', 1))

            inner_col = []
            inner_col.extend([field_text] * colspan)
            col_offset += colspan - 1  # 当有单元合并，补充offset

            if rowspan > 1:
                for i in range(1, rowspan):
                    if (index + i) not in rowspan_data:
                        rowspan_data[index + i] = {}
                    # 行合并 列合并
                    for j in range(colspan):
                        rowspan_data[index + i][col_real_index] = inner_col

            row.extend(inner_col)

        if row:
            rows.append(row)

    table_html = etree.tostring(table_node)

    cleaner = Cleaner(
        remove_unknown_tags=False,
        allow_tags=[
            "table",
            "thead",
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

    return dict(text=text, html=clean_table_html)


def merge_md_tables(tables, has_header=False) -> str:
    if not tables:
        return ""
    content = tables[0]
    for t in tables[1:]:
        rows = t.split("\n")
        rows = rows[2:] if has_header else [rows[0]] + rows[2:]
        content += "\n" + "\n".join(rows)

    return content


def merge_html_tables(tables, has_header=False) -> str:
    if not tables:
        return ""

    # print('---table0/1---', has_header)
    # print(tables[0])
    # print(tables[1])

    contents = ["<table>"]
    table_node = lxml.html.fromstring(tables[0])

    for thead_node in table_node.xpath(".//thead"):
        contents.append(etree.tostring(thead_node))

    for tr in table_node.xpath("./tbody//tr"):
        contents.append(etree.tostring(tr))

    for t in tables[1:]:
        table_node = lxml.html.fromstring(t)
        if has_header:
            for tr in table_node.xpath("./tbody//tr"):
                contents.append(etree.tostring(tr))
        else:
            tds = []
            trs = []
            for thead_node in table_node.xpath(".//thead"):
                row = []
                texts = tuple(thead_node.xpath(".//th//text()"))
                for text in texts:
                    tds.append("<td>{}</td>".format(text))

                for tr in thead_node.xpath(".//tr"):
                    trs.append(etree.tostring(tr))

            if tds:
                tr = "<tr>{}</tr>".format("".join(tds))
                contents.append(tr)

            if trs:
                tr = b"\n".join(trs)
                contents.append(tr)

            for tr in table_node.xpath("./tbody//tr"):
                contents.append(etree.tostring(tr))

    contents.append("</table>")

    tables = []
    for e in contents:
        tables.append(e.decode().strip() if isinstance(e, bytes) else e)
    return "\n".join(tables)


def transform_list_to_table(cols):
    contents = ["<table><thead>"]
    for col in cols:
        contents.append("<th>{}</th>".format(col))

    contents.append("</thead></table>")
    return "\n".join(contents)


def clean_html_table(table_html):
    cleaner = Cleaner(
        remove_unknown_tags=False,
        allow_tags=[
            "table",
            "td",
            "tr",
            "th",
        ],
        style=True,
        page_structure=False,
    )
    return cleaner.clean_html(table_html)
