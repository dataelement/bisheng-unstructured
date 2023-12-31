from typing import IO, List, Optional

from bisheng_unstructured.documents.elements import Element, process_metadata
from bisheng_unstructured.file_utils.filetype import FileType, add_metadata_with_filetype
from bisheng_unstructured.partition.common import (
    get_last_modified_date,
    get_last_modified_date_from_file,
)
from bisheng_unstructured.partition.docx import convert_and_partition_docx


@process_metadata()
@add_metadata_with_filetype(FileType.ODT)
def partition_odt(
    filename: Optional[str] = None,
    file: Optional[IO[bytes]] = None,
    include_metadata: bool = True,
    metadata_filename: Optional[str] = None,
    metadata_last_modified: Optional[str] = None,
    **kwargs,
) -> List[Element]:
    """Partitions Open Office Documents in .odt format into its document elements.

    Parameters
    ----------
    filename
        A string defining the target filename path.
    file
        A file-like object using "rb" mode --> open(filename, "rb").
    metadata_last_modified
        The last modified date for the document.
    """

    last_modification_date = None
    if filename:
        last_modification_date = get_last_modified_date(filename)
    elif file:
        last_modification_date = get_last_modified_date_from_file(file)

    return convert_and_partition_docx(
        source_format="odt",
        filename=filename,
        file=file,
        metadata_filename=metadata_filename,
        metadata_last_modified=metadata_last_modified or last_modification_date,
    )
