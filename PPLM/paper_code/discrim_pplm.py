import os
import sys
import argparse
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np
from IPython import embed
from operator import add
from style_utils import to_var, top_k_logits
import pickle
import csv

from gpt2tunediscrim import ClassificationHead


from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
from pplm_refactoring import perturb_past, latent_perturb, sample_from_hidden

SmallConst = 1e-15

def get_from_pretrained(path, device="cpu"):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    enc = GPT2Tokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False
    pass

    return model, enc


def run_model_discrim(raw_text, model, enc, args):
    # TODO: check what this token is
    seq = enc.encode(raw_text)

    out1, out_perturb, discrim_loss_list, loss_in_time_list = latent_perturb(model=model, args=args, context=out,
                                                                device=device, enc=enc)

    return enc.decode(out1.tolist()[0]), enc.decode(out_perturb.tolist()[0])