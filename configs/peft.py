from dataclasses import dataclass
from typing import List

@dataclass
class lora_config:
    r: int=8
    lora_alpha: int=32
    target_modules: List[str]= field(default_factory=lambda: ['q_proj', "v_proj"])