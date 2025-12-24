#!/usr/bin/env python3
"""
Training script for Kontext model with DeepSpeed and Accelerate
"""

import torch
from torch.optim import AdamW
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig
import argparse
from my_datasets.replace5k import Replace5kDataset
from torch.utils.data import DataLoader
import os
import logging
from accelerate.logging import get_logger
import diffusers
import datasets
import transformers
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers import FluxPipeline
from pipeline_dreamomni2 import DreamOmni2Pipeline
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
    is_wandb_available
)
from safetensors.torch import save_file, load_file
import shutil
from utils.infer_utils import _encode_prompt_with_clip, _encode_prompt_with_t5
import os


def main():
    base_model = "/home/yanzhang/models/FLUX.1-Kontext-dev"
    pipe = DreamOmni2Pipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16).to("cuda")

    edit_lora_path = "./lora_ckpt/checkpoint-6500/lora.safetensors"
    # edit_lora_path = "/home/yanzhang/models/DreamOmni2/edit_lora/pytorch_lora_weights.safetensors"
    edit_lora_dict = load_file(edit_lora_path)
    edit_lora_dict_2 = {k[12:] : v for k,v in edit_lora_dict.items()}

    dit = pipe.transformer
    lora_rank = 32
    lora_alpha = 32
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha, target_modules=["to_q", "to_v", "to_k", "to_out.0"], lora_dropout=0.05
    )
    dit.add_adapter(lora_config, adapter_name="edit")
    missing, unexpected = dit.load_state_dict(edit_lora_dict_2, strict=False)
    dit.eval()

    # 加载数据 
    dataset = Replace5kDataset(json_file="/home/yanzhang/datasets/replace-5k/test.json", is_image_preprocess=False)
    data = dataset[222]
    src_image = data["src_image"]
    ref_image = data["ref_image"]
    data["src_image"].save("./src.jpg")
    data["ref_image"].save("./ref.jpg")
    with open("./instruction.txt", "w") as f:
        f.write(data["prompt"])

    image = pipe(
        images=[src_image, ref_image],
        height=src_image.height,
        width=src_image.width,
        prompt=data["prompt"],
        num_inference_steps=30,
        guidance_scale=3.5,
        max_area=1024**2
    ).images[0]
    image.save("tgt.jpg")
    

if __name__ == "__main__":
    main()