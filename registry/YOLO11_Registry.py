from typing import Dict, Callable
from ..models import Yolo11

# Simple registry dictionary
_Model_Registry: Dict[str, Callable] = {}

def Register_Model(func: Callable) -> Callable:
    """Decorator to register a YOLO model variant"""
    _Model_Registry[func.__name__] = func
    return func

@Register_Model
def yolo_v11_n(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 16, 32, 64, 128, 256]
    return Yolo11(width, depth, csp, num_classes)

@Register_Model
def yolo_v11_t(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 24, 48, 96, 192, 384]
    return Yolo11(width, depth, csp, num_classes)

@Register_Model
def yolo_v11_s(num_classes: int = 80):
    csp = [False, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 32, 64, 128, 256, 512]
    return Yolo11(width, depth, csp, num_classes)

@Register_Model
def yolo_v11_m(num_classes: int = 80):
    csp = [True, True]
    depth = [1, 1, 1, 1, 1, 1]
    width = [3, 64, 128, 256, 512, 512]
    return Yolo11(width, depth, csp, num_classes)

@Register_Model
def yolo_v11_l(num_classes: int = 80):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 64, 128, 256, 512, 512]
    return Yolo11(width, depth, csp, num_classes)

@Register_Model
def yolo_v11_x(num_classes: int = 80):
    csp = [True, True]
    depth = [2, 2, 2, 2, 2, 2]
    width = [3, 96, 192, 384, 768, 768]
    return Yolo11(width, depth, csp, num_classes)

def Get_Yolo11_Model(name: str, *args, **kwargs):
    """Retrieve a YOLO model by name"""
    if name not in _Model_Registry:
        raise ValueError(f"Model '{name}' not found in registry. Available: {list(_Model_Registry.keys())}")
    return _Model_Registry[name](*args, **kwargs)
