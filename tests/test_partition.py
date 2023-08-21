import re

from unstructured.documents.coordinates import PixelSpace
from unstructured.documents.elements import (
    CoordinatesMetadata,
    ElementMetadata,
    NarrativeText,
    Text,
    Title,
)
from unstructured.partition import pdf, strategies
from unstructured.partition.auto import partition


RE_MULTISPACE_INCLUDING_NEWLINES = re.compile(pattern=r"\s+", flags=re.DOTALL)


def test_partition_html():
    elements = partition(filename="./examples/docs/maoxuan_wikipedia.html")

    idx = 0
    for e in elements:
        idx  += 1
        text = re.sub(
            RE_MULTISPACE_INCLUDING_NEWLINES, ' ',
            str(e) or "").strip()

        # if idx >= 10: break
        print(e.category, text)

    # print('-' * 100)
    # for e in elements:
    #     print(str(e))


test_partition_html()