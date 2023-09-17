from bisheng_unstructured.documents.markdown import transform_html_table_to_md


def visualize_html(elements, output_file=None):
    html_prefix = """
    <html>
    <head>
    <style>
        table {
          font-family: arial, sans-serif;
          border-collapse: collapse;
          width: 100%;
        }

        td, th {
          border: 1px solid #dddddd;
          text-align: left;
          padding: 8px;
        }

        tr:nth-child(even) {
          background-color: #dddddd;
        }
    </style>
    </head>
    <body>
    """

    html_suffix = "</body></html>"

    styles = ['style="background-color: #EBEBEB;"', 'style="background-color: #ABBAEA;"']
    idx = 0

    table_style = 'style="border:1px solid black;"'

    texts = []
    for el in elements:
        if el.category == "Title":
            text = f"<h1>{el.text}</h1>"
        elif el.category == "Table":
            text = el.metadata.text_as_html
            text = text.replace("\n", " ")
        else:
            text = el.text.replace("\n", "<br>")
            text = f"<p {styles[idx % 2]}>{text}</p>"
            idx += 1

        if text:
            texts.append(text)

    body_content = "\n".join(texts)
    html_str = html_prefix + body_content + html_suffix

    if output_file:
        with open(output_file, "w") as fout:
            fout.write(html_str)
    else:
        return html_str


def save_to_txt(elements, output_file=None):
    text_elem_sep = "\n"
    content_page = []
    is_first_elem = True
    last_label = ""
    for el in elements:
        label, text = el.category, el.text
        if is_first_elem:
            f_text = text + "\n" if label == "Title" else text
            content_page.append(f_text)
            is_first_elem = False
        else:
            if last_label == "Title" and label == "Title":
                content_page.append("\n" + text + "\n")
            elif label == "Title":
                content_page.append("\n\n" + text + "\n")
            elif label == "Table":
                content_page.append("\n\n" + text + "\n")
            else:
                content_page.append(text_elem_sep + text)

        last_label = label

    if output_file:
        with open(output_file, "w") as fout:
            fout.write("".join(content_page))
    else:
        return "".join(content_page)
