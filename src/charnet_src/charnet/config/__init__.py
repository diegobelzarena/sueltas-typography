import os
from pathlib import Path
from .defaults import _C as cfg

# Root directory of the charnet_src package (i.e. src/charnet_src/)
CHARNET_SRC_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_charnet_paths(config: "CfgNode") -> None:
    """Resolve WEIGHT, CHAR_DICT_FILE, and WORD_LEXICON_PATH.

    If a path is relative it is resolved against the charnet_src root
    directory, so the config works regardless of where the user launches
    the script from.
    """
    for key in ("WEIGHT", "CHAR_DICT_FILE", "WORD_LEXICON_PATH"):
        value = getattr(config, key, "")
        if not value:
            continue
        p = Path(value)
        if not p.is_absolute() and not p.exists():
            resolved = CHARNET_SRC_ROOT / p
            if resolved.exists():
                config.defrost()
                setattr(config, key, str(resolved))
                config.freeze()
