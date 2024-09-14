import os
import re
from typing import Dict, List, Optional

import yaml
from loguru import logger
from pydantic import BaseModel, BaseSettings, Field, validator


class LoggerConf(BaseModel):
    level: str = "INFO"
    format: str = "[%(asctime)s][%(levelname)s] %(message)s"
    handlers: List[Dict] = []

    @classmethod
    def parse_logger_sink(cls, sink: str) -> str:
        match = re.search(r"\{(.+?)\}", sink)
        if not match:
            return sink
        env_keys = {}
        for one in match.groups():
            env_keys[one] = os.getenv(one, "")
        return sink.format(**env_keys)

    @validator("handlers", pre=True)
    @classmethod
    def set_handlers(cls, value):
        if value is None:
            value = []
        for one in value:
            one["sink"] = cls.parse_logger_sink(one["sink"])
            if one.get("filter"):
                one["filter"] = eval(one["filter"])
        return value


class PdfModelParams(BaseModel):
    layout_ep: Optional[str]
    cell_model_ep: Optional[str]
    rowcol_model_ep: Optional[str]
    table_model_ep: Optional[str]
    ocr_model_ep: Optional[str]


class OcrConf(BaseModel):
    params: Optional[Dict]
    scene_mapping: Optional[Dict]


class Settings(BaseSettings):
    logger_conf: LoggerConf = LoggerConf()
    pdf_model_params: PdfModelParams = PdfModelParams()
    ocr_conf: OcrConf = OcrConf()
    is_all_ocr: bool = Field(default=False)


def load_settings_from_yaml(file_path: str) -> Settings:
    # Get current path
    current_path = os.path.dirname(os.path.abspath(__file__))
    # Check if a string is a valid path or a file name
    if "/" not in file_path:
        file_path = os.path.join(current_path, file_path)

    with open(file_path, "r", encoding="utf-8") as f:
        settings_dict = yaml.safe_load(f)

    for key in settings_dict:
        if key not in Settings.__fields__.keys():
            raise KeyError(f"Key {key} not found in settings")
        logger.debug(f"Loading {key} from {file_path}")

    return Settings(**settings_dict)


config_file = os.getenv("config", "config.yaml")
settings = load_settings_from_yaml(config_file)
