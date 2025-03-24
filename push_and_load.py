import torch
from argparse import ArgumentParser

from model.model import TwinLiteNetPlus


def detect(args):

    # load model
    model = TwinLiteNetPlus(args)
    model = model

    # load weights
    state_dict = torch.load(args.weight, map_location="cpu")
    model.load_state_dict(state_dict)

    # push model
    model.push_to_hub(f"nielsr/twinlitenetplus-{args.config}")

    # test load model
    model = TwinLiteNetPlus.from_pretrained(f"nielsr/twinlitenetplus-{args.config}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--weight', type=str, default='pretrained/large.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default='inference/videos', help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--config', type=str, choices=["nano", "small", "medium", "large"], help='Model configuration')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(opt)