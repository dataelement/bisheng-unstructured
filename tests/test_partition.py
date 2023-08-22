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
    texts = []
    first_element = True
    for e in elements:

        idx  += 1
        if e.category != 'Table':
            text = e.text
        else:
            text = '\n' + str(e) + '\n'

        if e.category == 'Title':
            if first_element:
                first_element = False
            else:
                text = '\n' + text

        texts.append(text)

    print('\n'.join(texts))


test_partition_html()