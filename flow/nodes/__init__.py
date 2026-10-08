# -*- coding: utf-8 -*-
"""内置算子节点。"""
from . import basic  # noqa: F401   (触发注册)
from . import classification  # noqa: F401
from . import detection  # noqa: F401
from . import segmentation  # noqa: F401
from . import seg_filter  # noqa: F401   (推理参数过滤：分割后处理)
from . import ocr  # noqa: F401
from . import ocv  # noqa: F401
from . import output  # noqa: F401
from . import visualize  # noqa: F401
from . import calculator  # noqa: F401
