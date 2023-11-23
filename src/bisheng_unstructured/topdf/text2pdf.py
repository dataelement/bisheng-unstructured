import os
import shutil
import tempfile
from html import parser

import lxml.html
import numpy as np
from lxml import etree
from lxml.html.clean import Cleaner


def clean_html(ori_file, new_file):
    cleaner = Cleaner(
        remove_unknown_tags=False,
        allow_tags=[
            "html",
            "head",
            "title",
            "p",
            "br",
            "b",
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
        safe_attrs=[],
        safe_attrs_only=True,
    )
    ori_text = open(ori_file).read()
    # print(ori_text)
    new_text = cleaner.clean_html(ori_text)
    new_text = parser.unescape(new_text)
    root = lxml.html.fromstring(new_text)
    descendanttag_elems: Tuple[etree.Element, ...] = ()

    def _repr_elem(e):
        return etree.tostring(e, pretty_print=True, encoding="UTF-8").decode()

    # latex not supported table embedding in table, remove embedding tables
    for tag_elem in root.iter():
        if tag_elem.tag == "table":
            elems = tag_elem.xpath(".//table")
            filtered_elems = []
            groups = [tuple(e.iterdescendants()) for e in elems]
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
                    filtered_elems.append(elems[inds[i]])

            # todo: improve it, not clean up fully
            for e in filtered_elems:
                for emb_e in e.findall(".//table"):
                    emb_e.drop_tag()

            descendanttag_elems = tuple(tag_elem.iterdescendants())

    new_text = _repr_elem(root)
    with open(new_file, "w") as fout:
        fout.write(new_text)


class Text2PDF(object):
    def __init__(self, kwargs={}):
        cmd_template = """
          pandoc -o {1} --pdf-engine=xelatex
              --lua-filter=/opt/pandoc/unnested-table.lua
              {0}
              -V mainfont="Alibaba PuHuiTi"
              -V sansfont="Alibaba PuHuiTi"
              -V monofont="Cascadia Mono"
              -V CJKmainfont="Alibaba PuHuiTi"
              -V CJKsansfont="Alibaba PuHuiTi"
              -V CJKmonofont="Adobe Heiti Std"
        """

        cmd_template2 = """
            soffice --headless --convert-to pdf --outdir \"{1}\" \"{0}\"
        """

        def _norm_cmd(cmd):
            return " ".join([p.strip() for p in cmd.strip().split()])

        self.cmd_template = _norm_cmd(cmd_template)
        self.cmd_template2 = _norm_cmd(cmd_template2)

    @staticmethod
    def run(cmd):
        try:
            exit_code = os.system(cmd)
            if exit_code != 0:
                raise Exception("error in transforming text to pdf")
        except Exception as e:
            raise e

    def render(self, input_file, output_file=None, to_bytes=False):
        type_ext = input_file.rsplit(".", 1)[-1]
        filename = os.path.basename(input_file)
        temp_dir = os.path.dirname(input_file)
        output_filename = filename.rsplit(".", 1)[0] + ".pdf"

        if output_file is None:
            output_file = os.path.join(temp_dir, output_filename)

        assert type_ext in ["txt", "md", "html"]

        if type_ext == "txt":
            cmd = self.cmd_template2.format(input_file, temp_dir)
            Text2PDF.run(cmd)
            if output_file is not None:
                tmp_file = os.path.join(temp_dir, output_filename)
                shutil.move(tmp_file, output_file)

        elif type_ext == "md":
            cmd = self.cmd_template.format(input_file, output_file)
            Text2PDF.run(cmd)

        elif type_ext == "html":
            with tempfile.TemporaryDirectory() as internal_tmp_dir:
                tmp_input = os.path.join(internal_tmp_dir, filename)
                clean_html(input_file, tmp_input)
                cmd = self.cmd_template.format(tmp_input, output_file)
                Text2PDF.run(cmd)

        if to_bytes:
            return open(output_file, "rb").read()
