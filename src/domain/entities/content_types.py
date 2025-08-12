from __future__ import annotations
from enum import Enum

class ArtifactKind(str, Enum):
    passage = "passage"
    audio_script = "audio_script"
    image_caption = "image_caption"
