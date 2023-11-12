import os
import tempfile
from tempfile import SpooledTemporaryFile
from typing import IO, BinaryIO, List, Optional, Union, cast

import pandas as pd
from lxml.html.soupparser import fromstring as soupparser_fromstring

from bisheng_unstructured.documents.elements import (
    Element,
    ElementMetadata,
    Table,
    process_metadata,
)
from bisheng_unstructured.documents.markdown import transform_html_table_to_md
from bisheng_unstructured.file_utils.filetype import FileType, add_metadata_with_filetype
from bisheng_unstructured.partition.common import (
    exactly_one,
    get_last_modified_date,
    get_last_modified_date_from_file,
    spooled_to_bytes_io_if_needed,
)


def run_cmd(cmd):
    try:
        exit_code = os.system(cmd)
        if exit_code != 0:
            raise Exception("error in transforming xlx to xlsx")
    except Exception as e:
        raise e


@process_metadata()
@add_metadata_with_filetype(FileType.XLS)
def partition_xls(
    filename: Optional[str] = None,
    file: Optional[Union[IO[bytes], SpooledTemporaryFile]] = None,
    metadata_filename: Optional[str] = None,
    include_metadata: bool = True,
    metadata_last_modified: Optional[str] = None,
    include_header: bool = True,
    **kwargs,
) -> List[Element]:
    """Partitions Microsoft Excel Documents in .xlsx format into its document elements.

    Parameters
    ----------
    filename
        A string defining the target filename path.
    file
        A file-like object using "rb" mode --> open(filename, "rb").
    include_metadata
        Determines whether or not metadata is included in the output.
    metadata_last_modified
        The day of the last modification
    include_header
        Determines whether or not header info info is included in text and medatada.text_as_html
    """
    cmd_template = "soffice --headless --convert-to xlsx --outdir {1} {0}"
    sheets = None

    exactly_one(filename=filename, file=file)
    last_modification_date = None
    if filename:
        with tempfile.TemporaryDirectory() as temp_dir:
            new_file = os.path.basename(filename).rsplit(".", 1)[0] + ".xlsx"
            tmp_file = os.path.join(temp_dir, new_file)
            run_cmd(cmd_template.format(filename, temp_dir))
            sheets = pd.read_excel(tmp_file, sheet_name=None)
        last_modification_date = get_last_modified_date(filename)
    elif file:
        f = spooled_to_bytes_io_if_needed(
            cast(Union[BinaryIO, SpooledTemporaryFile], file),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            ori_src = os.path.join(temp_dir, "ori.xls")
            ori_tgt = os.path.join(temp_dir, "ori.xlsx")
            with open(ori_src, "wb") as fout:
                if isinstance(f, bytes):
                    fout.write(f)
                else:
                    fout.write(f.read())

            run_cmd(cmd_template.format(ori_src, temp_dir))
            sheets = pd.read_excel(ori_tgt, sheet_name=None)

        last_modification_date = get_last_modified_date_from_file(file)

    elements: List[Element] = []
    page_number = 0
    for sheet_name, table in sheets.items():
        page_number += 1
        html_text = table.to_html(index=False, header=include_header, na_rep="")
        text = transform_html_table_to_md(html_text)["text"]

        # text = soupparser_fromstring(html_text).text_content()

        if include_metadata:
            metadata = ElementMetadata(
                text_as_html=html_text,
                page_name=sheet_name,
                page_number=page_number,
                filename=metadata_filename or filename,
                last_modified=metadata_last_modified or last_modification_date,
            )
        else:
            metadata = ElementMetadata()

        table = Table(text=text, metadata=metadata)
        elements.append(table)

    return elements
