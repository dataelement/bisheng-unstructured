import base64
from bisheng_unstructured.models import (
    FormulaAgent,
    LayoutAgent,
    OCRAgent,
    RTLayoutAgent,
    RTOCRAgent,
    RTTableAgent,
    RTTableDetAgent,
    TableAgent,
    TableDetAgent,
)

url = f"http://10.60.38.67:3011/v2.1/models/"
layout_ep = url + "elem_layout_v1/infer"
cell_model_ep = url + "elem_table_cell_detect_v1/infer"
rowcol_model_ep = url + "elem_table_rowcol_detect_v1/infer"
table_model_ep = url + "elem_table_detect_v1/infer"

model_params = {
    "layout_ep": layout_ep,
    "cell_model_ep": cell_model_ep,
    "rowcol_model_ep": rowcol_model_ep,
    "table_model_ep": table_model_ep,
}
class PDFDocumentTest():
    def __init__():
        self.layout_agent = RTLayoutAgent(**model_params)
        self.table_agent = RTTableAgent(**model_params)
        self.ocr_agent = RTOCRAgent(**model_params)
        self.table_det_agent = RTTableDetAgent(**model_params)  
    def _task(textpage_info, bytes_img, img, is_scan, lang, rot_matirx, page_index: int):
                b64_data = base64.b64encode(bytes_img).decode()
                layout_inp = {"b64_image": b64_data}
                layout = self.layout_agent.predict(layout_inp)
                blocks = self._allocate_semantic(
                    textpage_info, layout, b64_data, img, is_scan, lang, rot_matrix
                )
                return blocks, page_index
            
