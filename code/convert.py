from mapper import AffineMapper
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", "-p", required = True, type = str)
args = parser.parse_args()
mapper = AffineMapper(128, 768)

mapper.load_state_dict(torch.load(os.path.join(args.path, "ckpt.bin"))["model"])
mapper.save(os.path.join(args.path, "Affine.bin"))