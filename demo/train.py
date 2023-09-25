# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import inspect
import os
import pickle
import subprocess
from typing import Any, Dict
import datetime

import json
import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import poptorch
import torch
import tqdm
from torch import Tensor, nn

import matplotlib.axes
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch


from nanoGPT.config import train_shakespeare_char
from nanoGPT.model import GPTConfig, GPT

DEFAULT_CONFIGS = dict(
    beta1=0.9,
    weight_decay=1e-1,
    compute_batch_size=2,
    replication_factor=4,
    profile=False,
)


def config_dict_from_module(module) -> Dict[str, Any]:
    return {k: v for k, v in vars(module).items() if not k.startswith("__")}


def extract_model_params(config):
    return {
        k: v
        for k, v in config.items()
        if k in inspect.signature(GPTConfig).parameters.keys()
    }


_general_config_dict = config_dict_from_module(train_shakespeare_char)
_general_config_dict["compile"] = False  # We'll do this in the notebook when necessary

# Flash attention demo changes:
_general_config_dict["dropout"] = 0.0
_general_config_dict["block_size"] = 1024
_general_config_dict["max_iters"] = 2000
_general_config_dict["lr_decay_iters"] = 2000
_general_config_dict["learning_rate"] = 3e-4

_model_config_dict = extract_model_params(_general_config_dict)
_model_config_dict["vocab_size"] = 65  # Generated from data/shakespeare_char/prepare.py
config = GPTConfig(**_model_config_dict)
for key, value in _general_config_dict.items():
    setattr(config, key, value)

data_dir = "nanoGPT/data/shakespeare_char"


def download_train_data():
    cwd = os.getcwd()
    os.chdir(data_dir)
    print(f"Downloading training data/tokenizer to: {data_dir}")
    subprocess.run(["python", "prepare.py"])
    os.chdir(cwd)


class NanoGPTTokenizer:
    def __init__(self):
        meta_file = "nanoGPT/data/shakespeare_char/meta.pkl"
        if not os.path.exists(meta_file):
            download_train_data()

        with open(meta_file, "rb") as f:
            meta = pickle.load(f)
            stoi = meta["stoi"]
            self.encode_fn = lambda s: [stoi.get(c, stoi[" "]) for c in s]

    @property
    def pad_token(self):
        return self.encode_fn(" ")[0]

    def __call__(self, seqs, max_length, *args, **kwargs):
        batch = []
        for s in seqs:
            new_s = self.encode_fn(s)[:max_length]
            if len(new_s) < max_length:
                new_s += self.pad_token * (max_length - len(new_s))
            batch.append(new_s)
        batch = torch.tensor(batch)
        return {
            "input_ids": batch,
            "attention_mask": torch.ones_like(batch),  # nanoGPT ignores this anyway
        }


def plot(df: pd.DataFrame, name: str, save: bool = False) -> matplotlib.axes.Axes:
    sns.set_theme()
    ax = sns.lineplot(
        data=df,
        x="Steps",
        y="Loss",
        style="Train/Valid",
        label=name,
        solid_joinstyle="miter",
        solid_capstyle="butt",
        linewidth=1.5,
    )
    ax.set(xlim=(0, None), ylim=(0.4, 4.0))

    # remove duplicate legend entries
    entries = {}
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in entries:
            entries[l] = h
    entries[""] = plt.Line2D([0], [0], marker="none", linestyle="none", color="none")
    entries["validation"] = entries.pop("validation")  # move to bottom
    entries["training"] = entries.pop("training")  # move to bottom
    ax.legend(
        entries.values(), entries.keys()  # , bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    if save:
        plt.savefig(f"./local/{name}.jpeg")
    return ax


def run_training(
    model: nn.Module, config_dict: Dict[str, Any]
) -> Dict[str, Dict[str, List[float]]]:
    model_params_dict_overrides = extract_model_params(config_dict)
    model.config.__dict__.update(model_params_dict_overrides)

    cfg = Namespace(
        **{**DEFAULT_CONFIGS, **config_dict},
        model=model.config.__dict__,
    )

    if cfg.profile:
        profile = Path(f"profiles/{cfg.experiment_name}")
        profile.mkdir(exist_ok=True, parents=True)
        os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(
            {
                "autoReport.all": True,
                "autoReport.outputArchive": False,
                "autoReport.directory": str(profile),
            }
        )
        (profile / "app.json").write_text(json.dumps(cfg.__dict__))
        cfg.max_iters = 1

    if cfg.wandb_log:
        import wandb

        wandb.init(project=cfg.wandb_project, config=cfg.__dict__, reinit=True)

    if cfg.batch_size % (cfg.compute_batch_size * cfg.replication_factor) != 0:
        raise ValueError(
            f"Batch size {cfg.batch_size} not divisible by"
            " compute_batch_size * replication_factor"
            f" = {cfg.compute_batch_size} * {cfg.replication_factor}"
        )

    data_dir = Path("nanoGPT/data", cfg.dataset)
    data = {
        split: torch.frombuffer(
            (data_dir / f"{split}.bin").read_bytes(), dtype=torch.int16
        )
        for split in ["train", "val"]
    }

    def get_batch(split: str) -> Tuple[Tensor, Tensor]:
        idx = torch.randint(len(data[split]) - cfg.block_size - 1, (cfg.batch_size,))
        tokens = torch.stack([data[split][i : i + cfg.block_size + 1] for i in idx]).to(
            torch.long
        )
        return tokens[:, :-1].contiguous(), tokens[:, 1:].contiguous()

    def configure_optimizers(model, weight_decay, learning_rate, betas):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in model.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        return poptorch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

    def lr_schedule_fn(step: int) -> float:
        if step < cfg.warmup_iters:
            return step / cfg.warmup_iters
        min_ratio = cfg.min_lr / cfg.learning_rate
        progress = (step - cfg.warmup_iters) / (cfg.max_iters - cfg.warmup_iters)
        return min_ratio + (1 - min_ratio) * (0.5 + 0.5 * np.cos(np.pi * progress))

    options = poptorch.Options()
    options.replicationFactor(cfg.replication_factor)
    options.outputMode(poptorch.OutputMode.All)
    training_options, inference_options = options.clone(), options.clone()
    training_options.setAvailableMemoryProportion({"IPU0": 0.08})
    iterations = cfg.batch_size // (cfg.compute_batch_size * options.replication_factor)
    if iterations > 1:
        training_options.Training.gradientAccumulation(iterations)
        inference_options.deviceIterations(iterations)
    opt = configure_optimizers(
        model, cfg.weight_decay, cfg.learning_rate, betas=(cfg.beta1, cfg.beta2)
    )
    lr_schedule = torch.optim.lr_scheduler.LambdaLR(opt, lr_schedule_fn)
    opt._step_count = 1
    trainer = poptorch.trainingModel(model, options=training_options, optimizer=opt)
    evaluator = poptorch.inferenceModel(model, options=inference_options)

    results = {
        "train": {"iters": [], "losses": []},
        "valid": {"iters": [], "losses": []},
    }
    iter_num = 0

    def step() -> None:
        nonlocal iter_num
        if iter_num % cfg.eval_interval == 0 and cfg.eval_iters:
            if iter_num:
                trainer.detachFromDevice()
            losses = [evaluator(*get_batch("val"))[1] for _ in range(cfg.eval_iters)]
            val_loss = (
                float(torch.sum(torch.stack(losses)))
                * cfg.compute_batch_size
                / cfg.batch_size
                / cfg.eval_iters
            )
            results["valid"]["losses"].append(val_loss)
            results["valid"]["iters"].append(iter_num)
            if cfg.wandb_log:
                wandb.log(dict(val_loss=val_loss, step=iter_num), step=iter_num)
            evaluator.detachFromDevice()
        if iter_num < cfg.max_iters:
            loss = (
                float(torch.sum(trainer(*get_batch("train"))[1]))
                * cfg.compute_batch_size
                / cfg.batch_size
            )
            if iter_num % cfg.log_interval == 0:
                results["train"]["losses"].append(loss)
                results["train"]["iters"].append(iter_num)
                if cfg.wandb_log:
                    wandb.log(dict(loss=loss, step=iter_num), step=iter_num)
            lr_schedule.step()
            trainer.setOptimizer(opt)
            iter_num += 1

    try:
        step()  # trigger compilation before starting tqdm
        # +1 iteration for final validation only
        for _ in tqdm.tqdm(list(range(1, cfg.max_iters + 1))):
            step()
        if cfg.wandb_log:
            wandb.finish()
        return results
    except Exception as e:
        if cfg.wandb_log:
            wandb.run.summary["error"] = str(e)
            wandb.finish(1)
        raise
    finally:
        trainer.destroy()


def train(model: nn.Module, **config_overrides: Any) -> pd.DataFrame:
    device = "ipu"

    if not os.path.exists(f"{data_dir}/train.bin"):
        download_train_data()

    cfg = _general_config_dict.copy()
    cfg.update(
        device=device,
    )
    cfg.update(config_overrides)
    experiment_name = cfg["experiment_name"]
    print(f"Training {experiment_name} ...")
    try:
        results = run_training(model, cfg)
        do_plot = not DEFAULT_CONFIGS["profile"]
        if "profile" in cfg.keys():
            do_plot = not cfg["profile"]
        if do_plot:
            train_df = pd.DataFrame.from_dict(
                {
                    "Steps": results["train"]["iters"],
                    "Loss": results["train"]["losses"],
                }
            )
            valid_df = pd.DataFrame.from_dict(
                {
                    "Steps": results["valid"]["iters"],
                    "Loss": results["valid"]["losses"],
                }
            )
            train_df["Train/Valid"] = "training"
            valid_df["Train/Valid"] = "validation"
            df = pd.concat([train_df, valid_df])
            df["Model"] = experiment_name
            plot(df, experiment_name)
    except Exception as e:
        pass


if __name__ == "__main__":
    gpt = GPT(config)
    train(gpt, experiment_name="test", block_size=256)
