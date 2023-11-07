import json


def format_colors(colors):
    temp = "\\definecolor{color_%s{rgb}{%s}"
    defs = []
    mapping = {}
    for i, color in enumerate(colors):
        r = round(int(color[:2], 16) / 255.0, 3)
        g = round(int(color[2:4], 16) / 255.0, 3)
        b = round(int(color[4:], 16) / 255.0, 3)
        rgb = "{},{},{}".format(r, g, b)
        ind = "{:03d}".format(i)
        color_def = temp % (ind, rgb)
        defs.append(color_def)
        mapping[color] = color_def

    result = "\n".join(defs)

    return result, mapping


def format_string(text):
    return text


def pptx_layout_to_tex(input_file):
    content = json.load(open(input_file))

    default_size = 28
    default_bold = False
    default_align = "default"
    default_spacing = 1.0
    default_font_color = "000000"

    colors = set()
    for page in content["pages"]:
        for shape in page[1]:
            bbox = shape["bbox"]
            for line in shape["lines"]:
                for run in line:
                    font_color = run.get("font_color", default_font_color)
                    if font_color is None:
                        font_color = default_font_color
                    colors.add(font_color)

    colors = sorted(list(colors))
    color_defs, mapping = format_colors(colors)
    # print(color_defs)

    textblock_begin_prefix = "\\begin{textblock}{%f}(%f,%f)"
    textblock_end_suffix = "\\end{textblock}"
    run_bold_temp = "{\\fontsize{font_size}{font_spacing}\\selectfont \\color{color} \textbf {text}"
    run_nobold_temp = "{\\fontsize{font_size}{font_spacing}\\selectfont \\color{color} {text}"

    for page in content["pages"]:
        page_content = []
        for shape in page[1]:
            b = shape["bbox"]
            page_content.append(textblock_begin_prefix % (b[0], b[1], b[2]))
            for line in shape["lines"]:
                for run in line:
                    font_size = run.get("font_size", default_size)
                    font_bold = run.get("font_bold", default_bold)
                    font_color = run.get("font_color", default_font_color)
                    line_spacing = run.get("line_spacing", default_spacing)
                    align = run.get("align", default_spacing)
                    text = run.get("text")


def test1():
    file = "./out.txt"
    out = pptx_layout_to_tex(file)


test1()
