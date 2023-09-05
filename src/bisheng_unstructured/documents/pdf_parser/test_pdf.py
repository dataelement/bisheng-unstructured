import base64
import io

import cv2
import fitz
import numpy as np
import pypdfium2
from shapely import Polygon
from shapely import box as Rect

from bisheng_unstructured.documents.pdf_parser.blob import Blob
from bisheng_unstructured.models import LayoutAgent, OCRAgent, TableAgent


def draw_polygon(image, bbox, text=None, color=(255, 0, 0), thickness=1):
    bbox = bbox.astype(np.int32)
    is_rect = bbox.shape[0] == 4
    if is_rect:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    else:
        polys = [bbox.astype(np.int32).reshape((-1, 1, 2))]
        cv2.polylines(image, polys, True, color=color, thickness=thickness)
        start_point = (polys[0][0, 0, 0], polys[0][0, 0, 1])

    if text:
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 0, 255)
        image = cv2.putText(image, text, start_point, fontFace, fontScale, color, 1)

    return image


def get_image_blobs(pages, pdf_reader, n, start=0):
    blobs = []
    for pg in range(start, start + n):
        bytes_img = None
        page = pages.load_page(pg)
        mat = fitz.Matrix(1, 1)
        try:
            pm = page.get_pixmap(matrix=mat, alpha=False)
            bytes_img = pm.getPNGData()
        except Exception:
            # some pdf input cannot get render image from fitz
            page = pdf_reader.get_page(pg)
            pil_image = page.render().to_pil()
            img_byte_arr = io.BytesIO()
            pil_image.save(img_byte_arr, format="PNG")
            bytes_img = img_byte_arr.getvalue()

            blobs.append(Blob(data=bytes_img))
    return blobs


def merge_rects(bboxes):
    x0 = np.min(bboxes[:, 0])
    y0 = np.min(bboxes[:, 1])
    x1 = np.max(bboxes[:, 2])
    y1 = np.max(bboxes[:, 3])
    return [x0, y0, x1, y1]


def extract_blocks(textpage):
    blocks = []
    blocks_words_info = []
    page_dict = textpage.extractDICT()
    for block in page_dict["blocks"]:
        block_type = block["type"]
        block_no = block["number"]
        block_bbox = block["bbox"]
        if block_type != 0:
            block_text = ""
            block_info = (
                block_bbox[0],
                block_bbox[1],
                block_bbox[2],
                block_bbox[3],
                block_text,
                block_no,
                block_type,
            )
            blocks.append(block_info)
            blocks_words_info.append((None, None))

        lines = block["lines"]
        block_words = []
        block_words_bbox = []
        block_lines = []
        for line in lines:
            block_words.extend([span["text"] for span in line["spans"]])
            block_words_bbox.extend([span["bbox"] for span in line["spans"]])
            line_text = "".join([span["text"] for span in line["spans"]])
            block_lines.append(line_text)

        block_text = "".join(block_lines)
        block_info = (
            block_bbox[0],
            block_bbox[1],
            block_bbox[2],
            block_bbox[3],
            block_text,
            block_no,
            block_type,
        )
        blocks.append(block_info)
        blocks_words_info.append((block_words, block_words_bbox))

    return blocks, blocks_words_info


def extract_blocks2(textpage):
    blocks = []
    blocks_words_info = []
    page_dict = textpage.extractRAWDICT()
    for block in page_dict["blocks"]:
        block_type = block["type"]
        block_no = block["number"]
        block_bbox = block["bbox"]
        if block_type != 0:
            block_text = ""
            block_info = (
                block_bbox[0],
                block_bbox[1],
                block_bbox[2],
                block_bbox[3],
                block_text,
                block_no,
                block_type,
            )
            blocks.append(block_info)
            blocks_words_info.append((None, None))

        lines = block["lines"]
        block_words = []
        block_words_bbox = []
        block_lines = []
        for line in lines:
            line_text = []
            line_bboxes = []
            for span in line["spans"]:
                span_text = []
                span_bbox = []

                cont_bboxes = []
                cont_text = []
                for char in span["chars"]:
                    c = char["c"]
                    if c == " ":
                        if cont_bboxes:
                            word_bbox = merge_rects(np.asarray(cont_bboxes))
                            word = "".join(cont_text)
                            line_text.append(word)
                            line_bboxes.append(word_bbox)
                            cont_bboxes = []
                            cont_text = []
                    else:
                        cont_bboxes.append(char["bbox"])
                        cont_text.append(c)

                if cont_bboxes:
                    word_bbox = merge_rects(np.asarray(cont_bboxes))
                    word = "".join(cont_text)
                    line_text.append(word)
                    line_bboxes.append(word_bbox)

                block_words_bbox.extend(line_bboxes)
                block_words.extend(line_text)

            line_text = "".join([t for t in line_text])
            block_lines.append(line_text)

        block_text = "".join(block_lines)
        block_info = (
            block_bbox[0],
            block_bbox[1],
            block_bbox[2],
            block_bbox[3],
            block_text,
            block_no,
            block_type,
        )
        blocks.append(block_info)
        blocks_words_info.append((block_words, block_words_bbox))

    return blocks, blocks_words_info


def extract_lines(textpage):
    line_blocks = []
    line_words_info = []
    page_dict = textpage.extractRAWDICT()
    for block in page_dict["blocks"]:
        block_type = block["type"]
        block_no = block["number"]
        if block_type != 0:
            bbox = block["bbox"]
            block_text = ""
            block_info = (bbox[0], bbox[1], bbox[2], bbox[3], block_text, block_no, block_type)
            line_blocks.append(block_info)
            line_words_info.append((None, None))

        lines = block["lines"]

        for line in lines:
            bbox = line["bbox"]
            words = []
            words_bboxes = []
            for span in line["spans"]:
                cont_bboxes = []
                cont_text = []
                for char in span["chars"]:
                    c = char["c"]
                    if c == " ":
                        if cont_bboxes:
                            word_bbox = merge_rects(np.asarray(cont_bboxes))
                            word = "".join(cont_text)
                            words.append(word)
                            words_bboxes.append(word_bbox)
                            cont_bboxes = []
                            cont_text = []
                    else:
                        cont_bboxes.append(char["bbox"])
                        cont_text.append(c)

                if cont_bboxes:
                    word_bbox = merge_rects(np.asarray(cont_bboxes))
                    word = "".join(cont_text)
                    words.append(word)
                    words_bboxes.append(word_bbox)

            line_words_info.append((words, words_bboxes))

            line_text = "".join([char["c"] for span in line["spans"] for char in span["chars"]])
            bb0, bb1, bb2, bb3 = merge_rects(np.asarray(words_bboxes))

            block_info = (bb0, bb1, bb2, bb3, line_text, block_no, block_type)
            line_blocks.append(block_info)

    return line_blocks, line_words_info


def test_vis():
    # file_path = 'examples/docs/layout-parser-paper-fast.pdf'
    # output_prefix = 'layout-parser-paper-fast'
    # start, end, n = 0, 2, 2

    file_path = "examples/docs/达梦数据库招股说明书.pdf"
    output_prefix = "达梦数据库招股说明书"
    start, n = 212, 1

    layout_ep = "http://192.168.106.12:9001/v2.1/models/elem_layout_v1/infer"
    layout_agent = LayoutAgent(layout_ep=layout_ep)

    blob = Blob.from_path(file_path)
    pages = None
    image_blobs = []
    with blob.as_bytes_io() as file_path:
        pages = fitz.open(file_path)
        print("pages", pages)
        pdf_reader = pypdfium2.PdfDocument(file_path, autoclose=True)
        image_blobs = get_image_blobs(pages, pdf_reader, n, start)

    assert len(image_blobs) == n

    for i, blob in enumerate(image_blobs):
        idx = i + start

        page = pages.load_page(idx)
        textpage = page.get_textpage()
        # blocks, blocks_words = extract_blocks(textpage)
        # blocks, blocks_words = extract_blocks2(textpage)
        blocks, blocks_words = extract_lines(textpage)

        line_bbox = []
        for b in blocks:
            line_bbox.append([b[0], b[1], b[2], b[3]])
        line_bbox = np.asarray(line_bbox)

        words_bbox = []
        for texs, bbs in blocks_words:
            for bb in bbs:
                words_bbox.append(bb)
        words_bbox = np.asarray(words_bbox)

        b64_data = base64.b64encode(blob.as_bytes()).decode()
        layout_inp = {"b64_image": b64_data}
        layout = layout_agent.predict(layout_inp)["result"]

        bboxes = []
        labels = []
        for r in layout:
            bboxes.append(r["bbox"])
            labels.append(str(r["category_id"]))

        bboxes = np.asarray(bboxes)

        with open("data/p212.png", "wb") as fout:
            fout.write(blob.as_bytes())

        bytes_arr = np.frombuffer(blob.as_bytes(), dtype=np.uint8)
        image = cv2.imdecode(bytes_arr, flags=1)

        for bbox, text in zip(bboxes, labels):
            image = draw_polygon(image, bbox, text)

        # for bbox in words_bbox:
        #   image = draw_polygon(image, bbox, 'x', color=(0, 0, 255))

        for bbox in line_bbox:
            image = draw_polygon(image, bbox, "n", color=(0, 0, 255))

        outf = f"./data/{output_prefix}_layout_p{idx+1}_vis.png"
        cv2.imwrite(outf, image)


test_vis()
