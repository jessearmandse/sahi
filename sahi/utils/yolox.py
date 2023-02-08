import urllib.request
from os import path
from pathlib import Path
from typing import Optional

class YOLOXTestConstants:
    YOLOX_S_MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth"
    YOLOX_S_MODEL_PATH = "tests/data/models/yolox/yolox_s.pth"
    YOLOX_S_EXP_PATH = "tests/data/models/yolox/yolox_s.py"

def download_yolox_s_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = YOLOXTestConstants.YOLOX_S_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            YOLOXTestConstants.YOLOX_S_MODEL_URL,
            destination_path,
        )
