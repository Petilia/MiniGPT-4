import argparse
import os
import random
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import (
    Chat,
    CONV_VISION_LLama2,
    CONV_VISION_Vicuna0,
    StoppingCriteriaSub,
)

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--group-id", required=True, help="tracks group id")
    parser.add_argument(
        "--gpu-id", type=int, default=0, help="specify the gpu to load the model."
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


conv_dict = {
    "pretrain_vicuna0": CONV_VISION_Vicuna0,
    "pretrain_llama2": CONV_VISION_LLama2,
}

#### ARGS

args = parse_args()
cfg = Config(args)

#### MINIGPT4 MODEL

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to("cuda:{}".format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(
    vis_processor_cfg
)

stop_words_ids = [[835], [2277, 29937]]
stop_words_ids = [
    torch.tensor(ids).to(device="cuda:{}".format(args.gpu_id)) for ids in stop_words_ids
]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

chat = Chat(
    model,
    vis_processor,
    device="cuda:{}".format(args.gpu_id),
    stopping_criteria=stopping_criteria,
)

print("Initialization Finished")


def pred_description(
    path_img, chat, num_beams=1, temperature=1, user_message="describe this scene"
):
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(path_img, chat_state, img_list)
    chat.encode_img(img_list)

    chat.ask(user_message, chat_state)

    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        num_beams=num_beams,
        temperature=temperature,
        max_new_tokens=300,
        max_length=2000,
    )[0]

    return llm_message


#### GET DESCR


def get_groups(all_groups, N_GROUPS=9):
    n_tracks_per_group = len(all_groups) // N_GROUPS
    groups = []

    for cur_group_ind in range(N_GROUPS):
        if cur_group_ind == N_GROUPS - 1:
            group = all_groups[n_tracks_per_group * cur_group_ind :]
        else:
            group = all_groups[
                n_tracks_per_group
                * cur_group_ind : n_tracks_per_group
                * (cur_group_ind + 1)
            ]

        groups.append(group)

    return groups


def get_all_tracks(robotcar_root):
    all_tracks = [i for i in robotcar_root.iterdir() if i.is_dir()]
    all_tracks = sorted(all_tracks)
    return all_tracks


cur_group = int(args.group_id)

robotcar_root = Path("/home/docker_current/datasets/pnvlad_oxford_robotcar_full")
descriptions_root = Path("/home/docker_current/MiniGPT-4/robotcar_descr")

large_images_folder = "images_large"
cameras = ["mono_right", "mono_left", "stereo_centre", "mono_rear"]

if not descriptions_root.is_dir():
    descriptions_root.mkdir()

all_tracks = get_all_tracks(robotcar_root)
groups = get_groups(all_tracks)

cur_tracks = groups[cur_group]

for track in cur_tracks:
    trackname = str(track).split("/")[-1]

    if not (descriptions_root / trackname).is_dir():
        (descriptions_root / trackname).mkdir()

    if not (descriptions_root / trackname / "descriptions").is_dir():
        (descriptions_root / trackname / "descriptions").mkdir()

    for cam in cameras:
        image_paths = []
        descriptions = []

        imgages_path = track / large_images_folder / cam

        for img_path in imgages_path.iterdir():
            description = pred_description(img_path, chat)
            # description = "it's description"
            image_paths.append(img_path)
            descriptions.append(description)

        df_dict = {"path": image_paths, "description": descriptions}
        df = pd.DataFrame(df_dict)
        save_path = descriptions_root / trackname / "descriptions" / (cam + ".csv")
        df.to_csv(save_path, index=False)
