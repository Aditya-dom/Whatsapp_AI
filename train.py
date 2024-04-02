import os
from pkg_resources import packaging

import fire  
import random
import torch
import torch.optim as optim
from peft import get_peft_model, prepare_model_for_int8_training
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from configs.fsdp import fsdp_config as FSDP_CONFIG