# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import pva
import pandas as pd
import matplotlib.pyplot as plt

M = 10**6
G = 10**9


def plot_liveness_trace(report_path: str, save: bool = False):
    report = pva.openReport(report_path)

    available_mem = report.compilation.target.bytesPerIPU / G

    const_mem = sum(var.size for var in report.compilation.alwaysLiveVariables) / G
    mem_per_step = []
    for step in report.compilation.livenessProgramSteps:
        mem_per_step.append(step.notAlwaysLiveMemory.bytes / G + const_mem)

    plt.plot(
        mem_per_step,
        color="C0",
        label="temporary memory (activations, gradients, etc.)",
    )
    plt.hlines(
        const_mem,
        0,
        len(mem_per_step),
        color="C1",
        label="constant memory (code, weights, etc.)",
    )
    plt.hlines(
        available_mem, 0, len(mem_per_step), "black", "--", label="max memory per IPU"
    )
    plt.fill_between(
        range(len(mem_per_step)),
        const_mem,
        mem_per_step,
        color="C0",
        alpha=0.2,
    )
    plt.fill_between(
        range(len(mem_per_step)),
        0,
        const_mem,
        color="C1",
        alpha=0.2,
    )

    plt.xlabel("Program Step")
    plt.ylabel("Memory required (GB)")
    plt.title("nanoGPT Memory usage over time per training step")
    plt.legend()

    # plt.xlim(0, 15000)
    plt.ylim(0, 1.2)

    if save:
        plt.savefig("./local/liveness.jpeg")


def get_step_variables(report_path: str, step_num=int):
    report = pva.openReport(report_path)
    temp_vars = {}
    step = report.compilation.livenessProgramSteps[step_num]
    for var in step.notAlwaysLiveMemory.variables:
        temp_vars[var.name] = {"MB": var.size / M}
        df = pd.DataFrame.from_dict(temp_vars, orient="index")
    return df.sort_values("MB", ascending=False)


def get_report_variables(report_path: str):
    report = pva.openReport(report_path)
    temp_vars = {}
    for step_num, step in enumerate(report.compilation.livenessProgramSteps):
        for var in step.notAlwaysLiveMemory.variables:
            temp_vars[var.name] = {"step": step_num, "MB": var.size / M}

    df = pd.DataFrame.from_dict(temp_vars, orient="index")
    return df.sort_values("MB", ascending=False)


if __name__ == "__main__":
    plot_liveness_trace(
        "/nethome/lukep/Projects/flash-attention-ipu-nanogpt-demo/profiles/230905-100000/training/profile.pop",
        save=True,
    )
