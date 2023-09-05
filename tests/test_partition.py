from bisheng_unstructured.partition.html import partition_html
from bisheng_unstructured.documents.html_utils import visualize_html, save_to_txt


def test_html1():
    elements = partition_html(
        filename="./examples/docs/maoxuan_wikipedia.html")
    visualize_html(elements, 'data/maoxuan_wikipedia-v1_1.html')
    save_to_txt(elements, 'data/maoxuan_wikipedia-v1_1.txt')


def test_html2():
    elements = partition_html(filename="./examples/docs/静静的顿河-wiki.html")
    visualize_html(elements, 'data/river-v1_1.html')
    save_to_txt(elements, 'data/river-v1_1.txt')


def test_url():
    url = ("https://fans.sports.qq.com/post.htm"
           "?id=1774874199910776989&mid=145#1_allWithElite")
    elements = partition_html(url=url)
    visualize_html(elements, 'data/qq_news-v1_1.html')
    save_to_txt(elements, 'data/qq_news-v1_1.txt')


test_html1()
# test_html2()
# test_url()