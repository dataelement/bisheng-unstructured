"""Pdf Creator with multiple layers"""
import base64
import io
import math
import re

import fitz
import numpy as np
from PIL import Image, ImageOps

from bisheng_unstructured.models import OCRAgent

ZH_CHAR = re.compile("[\u4e00-\u9fa5]")


def get_hori_rect(rot_rect):
    arr = np.asarray(rot_rect, dtype=np.float32).reshape((4, 2))
    x0 = np.min(arr[:, 0])
    x1 = np.max(arr[:, 0])
    y0 = np.min(arr[:, 1])
    y1 = np.max(arr[:, 1])
    return [float(e) for e in (x0, y0, x1, y1)]


def rescale_rect(r, s):
    return [r[0] * s[0], r[1] * s[1], r[2] * s[0], r[3] * s[1]]


def banjiao_to_quanjiao(texts):
    b2q_map = {",": "，", "!": "！", "(": "（", ")": "）"}
    left_quota = "“"
    right_quota = "”"
    eng_quota = '"'

    new_texts = []
    has_left_quota = False
    for text in texts:
        chars = [b2q_map.get(t, t) for t in text.strip() if t != " "]
        for i, c in enumerate(chars):
            if c == eng_quota:
                if not has_left_quota:
                    has_left_quota = True
                    chars[i] = left_quota
                else:
                    chars[i] = right_quota
                    has_left_quota = False

        new_texts.append("".join(chars))

    return new_texts


def reorient_image(im):
    try:
        image_exif = im._getexif()
        image_orientation = image_exif[274]
        if image_orientation in (2, "2"):
            return im.transpose(Image.FLIP_LEFT_RIGHT)
        elif image_orientation in (3, "3"):
            return im.transpose(Image.ROTATE_180)
        elif image_orientation in (4, "4"):
            return im.transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (5, "5"):
            return im.transpose(Image.ROTATE_90).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (6, "6"):
            return im.transpose(Image.ROTATE_270)
        elif image_orientation in (7, "7"):
            return im.transpose(Image.ROTATE_270).transpose(Image.FLIP_TOP_BOTTOM)
        elif image_orientation in (8, "8"):
            return im.transpose(Image.ROTATE_90)
        else:
            return im
    except (KeyError, AttributeError, TypeError, IndexError):
        return im


class PdfCreator(object):
    """Double Layer pdf creator.

    Refer: omni-read-infrastructure/src/main/java/org/czhouyi/omniread/
      infrastructure/adaptors/PdfRepositoryImpl.java

    # Pymupdf has not character spacing api, replace PdfCreator with java pdfbox
    """

    def __init__(self, model_params: dict):
        self.ocr_agent = OCRAgent(**model_params)

    def text_angle(self, position):
        deltaX = position[0][0] - position[1][0]
        deltaY = position[0][1] - position[1][1]

        if deltaX == 0:
            if deltaY < 0:
                return -90
            elif deltaY > 0:
                return 90

        angle = math.atan(deltaY * 1.0 / deltaX) * (-180) / math.pi
        if deltaX > 0:
            return float(angle + 180)
        else:
            return float(angle)

    def compute_font_size(self, font, font_size, text, line_width):
        def _font_length(text, font_size, font):
            return sum([font.text_length(c, font_size) for c in text])

        # 1pt = 1/72th of 1Inch
        # 1px = 1/96th of 1Inch
        # unit = 72.0 / 96.0
        unit = 1
        # real_width = font.text_length(text, font_size) * unit
        real_width = _font_length(text, font_size, font) * unit
        while real_width > line_width:
            font_size -= 1
            # real_width = font.text_length(text, font_size) * unit
            real_width = _font_length(text, font_size, font) * unit

        return font_size, real_width

    def render_text(self, page, image_file, scale, xref):
        b64_data = base64.b64encode(open(image_file, "rb").read()).decode()
        inp = {"b64_image": b64_data}
        ocr_result = self.ocr_agent.predict(inp)
        texts = ocr_result["result"]["ocr_result"]["texts"]
        bboxes = ocr_result["result"]["ocr_result"]["bboxes"]

        fontfile = "./examples/docs/alibaba/Alibaba-PuHuiTi-Regular.ttf"
        # font = fitz.Font(fontfile=fontfile)

        # blue = (0, 0, 1)
        # red = (1, 0, 0)
        writer = fitz.TextWriter(page.rect)

        unit = 1
        # unit = 72.0 / 96.0

        texts_str = "".join(texts)
        char_cnt = len(texts_str)
        zh_n = len(re.findall(ZH_CHAR, texts_str))
        is_zh = False
        if zh_n > 50 or (char_cnt > 0 and zh_n / char_cnt > 0.5):
            is_zh = True
            font = fitz.Font(fontname="china-ss")
            texts = banjiao_to_quanjiao(texts)
        else:
            font = fitz.Font()

        gap_threhold = 10
        x_delta = 0
        x_right_delta = 5
        y_delta = 2 if is_zh else 0

        for text, bbox in zip(texts, bboxes):
            wx = bbox[0][0] - bbox[1][0]
            wy = bbox[0][1] - bbox[1][1]
            hx = bbox[0][0] - bbox[3][0]
            hy = bbox[0][1] - bbox[3][1]
            lineHeight = int(math.sqrt(hx * hx + hy * hy))
            lineWidth = int(math.sqrt(wx * wx + wy * wy))
            initFontSize = int(lineHeight)

            fontSize, realWidth = self.compute_font_size(font, initFontSize, text, lineWidth)

            # charSpace = 0
            # if len(text) > 1:
            #     charSpace = (lineWidth - realWidth) / (len(text) - 1)
            gap = lineWidth - realWidth
            rect = get_hori_rect(bbox)
            # https://github.com/pymupdf/PyMuPDF/issues/259
            # rect = rescale_rect(rect, scale)
            # print('---rect', page.rect, rect, gap, realWidth, text)
            if gap > gap_threhold:
                continue
                lineWidth += gap if is_zh else gap / 3
                if lineWidth + rect[0] >= page.rect.width:
                    lineWidth = page.rect.width - rect[0] - x_right_delta

                fontSize1, realWidth = self.compute_font_size(font, initFontSize, text, lineWidth)
                # print('---font', fontSize, fontSize1)
                if fontSize1 > gap:
                    realWidth = rect[2] - rect[0]
                    x_right_delta = fontSize1

                rect_delta = [
                    rect[0],
                    rect[1] + y_delta,
                    rect[0] + realWidth + x_right_delta,
                    rect[3] + y_delta,
                ]

                fill_rect = fitz.Rect(*rect_delta)
                writer.fill_textbox(
                    fill_rect,
                    text,
                    align=fitz.TEXT_ALIGN_LEFT,
                    warn=True,
                    fontsize=fontSize1,
                    font=font,
                )

            else:
                h = rect[3] - rect[1]
                if h >= fontSize:
                    fontSize = int(fontSize * 0.8)

                y_delta = 0
                x_right_delta = 10
                rect_delta = [
                    rect[0] + x_delta,
                    rect[1] + y_delta,
                    rect[2] + x_delta + x_right_delta,
                    rect[3] + y_delta,
                ]
                fill_rect = fitz.Rect(*rect_delta)
                # print('---fill textbox', fill_rect, text, fontSize)
                writer.fill_textbox(
                    fill_rect,
                    text,
                    align=fitz.TEXT_ALIGN_LEFT,
                    warn=True,
                    fontsize=fontSize,
                    font=font,
                )

            realWidth = int(font.text_length(text, fontSize))

            # shape = page.new_shape()
            # shape.draw_rect(fill_rect)  # the rect within which we had to stay
            # shape.finish(color=red, width=0.1)  # show in red color
            # shape.draw_rect(rect)  # the rect within which we had to stay
            # shape.finish(color=blue, width=0.1)  # show in red color
            # shape.commit()

        writer.write_text(page, oc=xref, render_mode=3)

    def render_image_v0(self, doc, image_file, xref):
        img = Image.open(image_file)
        img = reorient_image(img)

        max_longer_edge = 1600
        width = img.width
        height = img.height
        ratio = max_longer_edge / max(width, height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        img = img.resize((new_width, new_height), Image.BILINEAR)
        byte_arr = io.BytesIO()
        img.save(byte_arr, format="PNG")
        pil_bytes = byte_arr.getvalue()
        # pil_bytes = open(image_file, 'rb').read()

        scale = (1, 1)
        new_rect = fitz.Rect(0, 0, new_width, new_height)
        page = doc.new_page(width=new_width, height=new_height)
        page.insert_image(new_rect, stream=pil_bytes, oc=xref)
        # print('---dpi', width, height, scale, new_rect)
        return page, scale

    def render_image(self, doc, image_file, xref):
        img = fitz.open(image_file)
        rect = img[0].rect
        mat = fitz.Matrix(1, 1)
        pixmap = img[0].get_pixmap(matrix=mat)
        pdfbytes = img.convert_to_pdf()
        img.close()

        dpiX = pixmap.xres if pixmap.xres > 0 else 72
        dpiY = pixmap.yres if pixmap.yres > 0 else 72
        scaleX = dpiX / 72.0
        scaleY = dpiY / 72.0
        # scaleX = 1
        # scaleY = 1
        width = int(rect.width * scaleX)
        height = int(rect.height * scaleY)
        scale = (scaleX, scaleY)
        # print('---dpi', scale, rect, pixmap.xres, pixmap.yres, pixmap.width, pixmap.height)
        new_rect = fitz.Rect(0, 0, width, height)
        imgPDF = fitz.open("pdf", pdfbytes, dpi=dpiX)
        page = doc.new_page(width=width, height=height)
        page.show_pdf_page(new_rect, imgPDF, 0, oc=xref)
        return page, scale

    def render(self, image_file, output_file=None, to_bytes=False):
        doc = fitz.open()

        xref_image = doc.add_ocg("Graphics", on=True, intent=["View", "Design"], usage="Artwork")
        xref_text = doc.add_ocg("Texts", on=True, intent=["View", "Design"], usage="Artwork")

        page, scale = self.render_image_v0(doc, image_file, xref_image)
        # page, scale = self.render_image(doc, image_file, xref_image)
        # self.render_text(page, image_file, scale, xref_text)

        if to_bytes:
            return doc.tobytes()
        else:
            doc.save(output_file)
