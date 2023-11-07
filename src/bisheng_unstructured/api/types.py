from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class UnstructuredInput(BaseModel):
    filename: str
    url: Optional[str] = None
    b64_data: Optional[List[str]] = None
    parameters: Optional[Dict] = {}
    mode: str = "text"  # text, partition, vis, topdf
    file_path: Optional[str] = None
    file_type: Optional[str] = None


class UnstructuredOutput(BaseModel):
    status_code: int = 200
    status_message: str = "success"
    text: Optional[str] = None
    html_text: Optional[str] = None
    partitions: List[Dict[str, Any]] = []
    b64_pdf: Optional[str] = None


class ConfigInput(BaseModel):
    pdf_model_params: Optional[Dict[str, Any]] = None
    rt_ep: Optional[str] = None
