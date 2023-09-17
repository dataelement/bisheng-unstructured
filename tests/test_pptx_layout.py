import pptx
from lxml import etree


def _order_shapes(shapes):
    """Orders the shapes from top to bottom and left to right."""
    return sorted(shapes, key=lambda x: (x.top or 0, x.left or 0))


def pptx_layout(filename):
    presentation = pptx.Presentation(filename)
    docs = []
    for i, slide in enumerate(presentation.slides):
        shape_infos = []
        for shape in _order_shapes(slide.shapes):
            # print('shape xml', etree.tostring(shape.element).decode())
            # print('text_frame', [shape.text_frame.text])

            shape_info = []
            for para in shape.text_frame.paragraphs:
                # align = para.alignment if hasattr(para, 'alignment') else None
                # print('para-align:', alig)
                # print('para', para.font, para.font.name, [para.text])
                currents = []
                # print('runs len', len(para.runs))
                if para.runs:
                    r0 = para.runs[0]
                    # align = para.alignment if hasattr(r0, 'alignment') else None
                    # print('r0:', align)

                    color = r0.font.color
                    color = str(color.rgb) if hasattr(color, "rgb") else None
                    currents = [r0.font.name, r0.font.size, color, r0.text]
                # print('currents0', currents, bbox)

                for run in para.runs[1:]:
                    font_name = run.font.name
                    font_size = run.font.size
                    color = run.font.color
                    color = str(color.rgb) if hasattr(color, "rgb") else None
                    if (
                        font_name == currents[0]
                        and font_size == currents[1]
                        and color == currents[2]
                    ):
                        currents[3] += run.text
                    else:
                        if currents[3]:
                            info = {
                                "font_name": currents[0],
                                "font_size": currents[1],
                                "font_color": currents[2],
                                "text": currents[3],
                            }
                            shape_info.append(info)
                        # reset
                        font_name = run.font.name
                        font_size = run.font.size
                        color = run.font.color
                        color = color.rgb if hasattr(color, "rgb") else None
                        currents = [font_name, font_size, color, run.text]

                    # print('---run---', font_name, run.text, currents)

                if currents and currents[3]:
                    info = {
                        "font_name": currents[0],
                        "font_size": currents[1],
                        "font_color": currents[2],
                        "text": currents[3],
                    }
                    shape_info.append(info)

            shape_infos.append({"bbox": bbox, "shapes": shape_info})
        docs.append((i, shape_infos))

    return docs
