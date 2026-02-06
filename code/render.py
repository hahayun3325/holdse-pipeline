import torch
import os
from src.hold.hold import HOLD
from src.utils.parser import parser_args
import os.path as op
from common.torch_utils import reset_all_seeds
import numpy as np
from pprint import pprint
from loguru import logger
import sys

sys.path = [".."] + sys.path
from src.datasets.utils import create_dataset
import common.thing as thing


def main():
    device = "cuda:0"
    args, opt = parser_args()

    logger.info("Working dir:", os.getcwd())
    exp_key = args.load_ckpt.split("/")[1]
    args.log_dir = op.join("logs", exp_key, "test")

    logger.info(args)

    model = HOLD(opt, args)
    testset = create_dataset(opt.dataset.test, args)

    logger.info("img_paths: ")
    img_paths = np.array(testset.dataset.dataset.img_paths)
    logger.info(img_paths[:3])
    logger.info("...")
    logger.info(img_paths[-3:])
    reset_all_seeds(1)
    ckpt_path = None if args.ckpt_p == "" else args.ckpt_p
    sd = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    # disable barf masks
    nodes = model.model.nodes
    for node in nodes.values():
        node.implicit_network.embedder_obj.eval()
    model.model.background.bg_implicit_network.embedder_obj.eval()
    model.model.background.bg_rendering_network.embedder_obj.eval()
    for batch in testset:
        with torch.no_grad():
            batch = thing.thing2dev(batch, device)
            out = model.inference_step(batch)
            model.validation_epoch_end([out])


if __name__ == "__main__":
    main()
'''
export COMET_API_KEY="4hhuylWTxYQBirmxKwuwGv4Q5"
export COMET_WORKSPACE="cloudy"
python render.py --case hold_MC1_ho3d --load_ckpt logs/7dacf8bc6_000002000/checkpoints/last.ckpt --config confs/render_stage3_hold_MC1_ho3d_sds_from_official.yaml --mute --agent_id -1
'''