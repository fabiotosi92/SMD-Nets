from .HSMNet import HSMNet
from .PSMNet import PSMNet
from .Stereodepth import SDFilter

BACKBONES = {
    "HSMNet": HSMNet,
    "PSMNet": PSMNet,
    "Stereodepth": SDFilter
}

def get_backbone(name: str):
    """Get backbone given the name"""
    if name not in BACKBONES.keys():
        raise ValueError(
            f"Backbone {name} not in backbone list. Valid backbonesa are {BACKBONES.keys()}"
        )
    return BACKBONES[name]