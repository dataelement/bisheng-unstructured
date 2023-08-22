from unstructured.partition.html import partition_html


def test_partition_html():
    elements = partition_html(filename="./examples/docs/maoxuan_wikipedia.html")

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


def test_url():
    url="https://fans.sports.qq.com/post.htm?id=1774874199910776989&mid=145#1_allWithElite"
    elements = partition_html(url=url)

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


test_url()