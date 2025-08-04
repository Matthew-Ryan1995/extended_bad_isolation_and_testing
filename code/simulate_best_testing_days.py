#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 11:40:25 2025

@author: Matt Ryan
"""
# %% Libraries
import pstats
import cProfile
from BaD import *
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor
from os import cpu_count


# %% Flags
FLAG_GEN_DATA = True

# %% Parameter set up

# Defined her to make sure they are kept constant throughout all simulations
K = 7
n = 3
params = load_param_defaults()
params["k"] = K

# This is only defined to make the event function work
M_tmp = bad(**params)

# These are the combinations of testing days to consider
combinations = list(itertools.combinations(range(K), n))
combinations = np.array(combinations)

# Start with small behaviour and infection in the community
B_0 = 1e-6
I_0 = 1e-6

target = "final_size"

# %% Functions


def event(t, Y, min_val=2e6, target="peak", *args):
    if target == "peak":
        model_val = 0
        for cc in M_tmp.CC:
            if "nc" in cc.name:
                continue
            if "I" in cc.name:
                model_val += Y[cc.value]
    elif target == "final_size":
        model_val = Y[M_tmp.CC["Inc"]]
    if model_val > min_val:
        ans = 0
    else:
        ans = 1
    return ans


def get_metric(M, target="peak"):
    if target == "peak":
        ans = M.get_I().max()
    elif target == "final_size":
        ans = M.get_incidence()[-1]
    return ans


def find_best_testing(pars, target,
                      t_start=0, t_end=300,  # Might not actually be finding Final size
                      P=1e6):

    # Initialise
    params = pars.copy()
    target_min = 2*P
    best_delta = []
    best_idx = 0

    # Turn off waning immunity for final size calculations
    if target == "final_size":
        params["immune_period"] = np.inf

    # Brute force - loop through all possible combinations, max 120
    for idx, x in enumerate(combinations):

        # Define and update the stopping event
        event_tmp = lambda t, Y, * \
            args: event(t, Y, min_val=target_min, target=target)
        event_tmp.terminal = True

        # Define testing days
        delta = np.zeros(K)
        delta[x] = 1.0

        # Update testing days
        params["delta"] = delta

        # Initialise model
        M = bad(**params)
        M.set_initial_conditions(pop_size=P,
                                 starting_B=B_0,
                                 starting_I=I_0)

        # Run model, stop early if exceed best statistic
        M.run(t_start=t_start, t_end=t_end, t_step=1,
              t_eval=True, flag_incidence_tracking=True,
              events=event_tmp)

        # If model completed running, update targets
        if M.terminated == 0:
            target_min = get_metric(M, target=target)
            best_delta = delta
            best_idx = idx

    return target_min, best_delta, best_idx


# def create_testing_region_data(input_params,
#                                target="final_size",
#                                disease_range=[0, 5], disease_step=0.01,
#                                behav_range=[0, 3], behav_step=0.01,
#                                generate_data_flag=False):

#     params = dict(input_params)

#     save_lbl = f"testing_regions_{target}"

#     # Find the line where R0 = 1
#     if generate_data_flag:
#         r0_b = np.arange(start=behav_range[0],
#                          stop=behav_range[1] + behav_step,
#                          step=behav_step)

#         R0_multiplier = params["infectious_period"] * \
#             (params["pA"]*params["qA"] + 1 - params["pA"])

#         r0_d = np.arange(start=disease_range[0],
#                          stop=disease_range[1] + disease_step,
#                          step=disease_step)

#         grid_vals = np.meshgrid(r0_b, r0_d)

#         iter_vals = (np.array(grid_vals)
#                      .reshape(2, len(r0_b)*len(r0_d))
#                      .T)

#         testing_idx_regions = list()

#         for idxx in range(len(iter_vals)):

#             tmp_params = dict(params)
#             tmp_r0d = iter_vals[idxx, 1]
#             tmp_r0b = iter_vals[idxx, 0]

#             tmp_params["w1"] = tmp_r0b * (tmp_params["a1"] + tmp_params["a2"])
#             if tmp_params["w1"] < 0:
#                 tmp_params["w1"] = 0

#             beta = tmp_r0d / R0_multiplier
#             tmp_params["transmission"] = beta

#             ans = find_best_testing(pars=tmp_params,
#                                     target=target)

#             testing_idx_regions.append(ans[2])

#         testing_idx_regions = np.array(
#             testing_idx_regions).reshape(grid_vals[0].shape)

#         dat = {
#             "grid_vals": grid_vals,
#             "testing_idx_regions": testing_idx_regions,
#             "r0_d": r0_d,
#             "r0_b": r0_b
#         }

#         with open(f"../outputs/{save_lbl}.pkl", "wb") as f:
#             pkl.dump(dat, f)
#     else:
#         with open(f"../outputs/{save_lbl}.pkl", "rb") as f:
#             dat = pkl.load(f)

#     return dat

def _evaluate_testing_idx(args):
    idx, iter_val, params, R0_multiplier, target = args

    tmp_params = dict(params)
    tmp_r0d = iter_val[1]
    tmp_r0b = iter_val[0]

    tmp_params["w1"] = tmp_r0b * (tmp_params["a1"] + tmp_params["a2"])
    if tmp_params["w1"] < 0:
        tmp_params["w1"] = 0

    beta = tmp_r0d / R0_multiplier
    tmp_params["transmission"] = beta

    try:
        ans = find_best_testing(pars=tmp_params, target=target)
        return idx, ans[2]
    except Exception:
        return idx, -1  # fallback if model fails


def create_testing_region_data(input_params,
                               target="final_size",
                               disease_range=[0, 5], disease_step=0.01,
                               behav_range=[0, 3], behav_step=0.01,
                               generate_data_flag=False):

    params = dict(input_params)
    save_lbl = f"testing_regions_{target}"

    if generate_data_flag:
        r0_b = np.arange(behav_range[0],
                         behav_range[1] + behav_step,
                         behav_step)
        r0_d = np.arange(disease_range[0],
                         disease_range[1] + disease_step,
                         disease_step)

        R0_multiplier = params["infectious_period"] * (
            params["pA"] * params["qA"] + 1 - params["pA"])

        grid_vals = np.meshgrid(r0_b, r0_d)
        iter_vals = np.array(grid_vals).reshape(2, -1).T

        args_list = [(i, iter_vals[i], params, R0_multiplier, target)
                     for i in range(len(iter_vals))]

        testing_idx_regions = [None] * len(iter_vals)

        with ProcessPoolExecutor(max_workers=cpu_count()-1) as executor:
            for idx, value in executor.map(_evaluate_testing_idx, args_list):
                testing_idx_regions[idx] = value

        testing_idx_regions = np.array(
            testing_idx_regions).reshape(grid_vals[0].shape)

        dat = {
            "grid_vals": grid_vals,
            "testing_idx_regions": testing_idx_regions,
            "r0_d": r0_d,
            "r0_b": r0_b
        }

        with open(f"../outputs/{save_lbl}.pkl", "wb") as f:
            pkl.dump(dat, f)
    else:
        with open(f"../outputs/{save_lbl}.pkl", "rb") as f:
            dat = pkl.load(f)

    return dat


# %% Run simulations
# ans = create_testing_region_data(input_params=params,
#                                  target=target,
#                                  disease_range=[0.0, 3.0],
#                                  behav_range=[0.0, 2.0],
#                                  disease_step=1.0,
#                                  behav_step=1.0,
#                                  generate_data_flag=FLAG_GEN_DATA)
if __name__ == "__main__":
    # with cProfile.Profile() as pr:
    #     create_testing_region_data(input_params=params,
    #                                target=target,
    #                                disease_range=[0.0, 3.0],
    #                                behav_range=[0.0, 2.0],
    #                                disease_step=1.0,
    #                                behav_step=1.0,
    #                                generate_data_flag=FLAG_GEN_DATA)

    # stats = pstats.Stats(pr)
    # # show top 10 functions by cumulative time
    # stats.sort_stats('cumtime').print_stats(10)

    for t in ["final_size", "peak"]:
        ans = create_testing_region_data(input_params=params,
                                         target=t,
                                         disease_range=[1.0, 5.0],
                                         behav_range=[0.0, 2.0],
                                         disease_step=0.01,
                                         behav_step=0.01,
                                         generate_data_flag=FLAG_GEN_DATA)

        plt.figure()
        plt.title(f"Target min: {t}")
        plt.tight_layout()
        im = plt.contourf(ans["grid_vals"][1], ans["grid_vals"][0],
                          ans["testing_idx_regions"],
                          # levels=lvls,
                          # cmap=cmap
                          )
        plt.contour(ans["grid_vals"][1], ans["grid_vals"][0],
                    ans["testing_idx_regions"],
                    # levels=[0, 1, 2, 3],
                    colors="black")
        plt.ylabel(
            "Social reproduction number ($\\mathcal{R}_0^{B}$)"
        )
        plt.xlabel(
            "Behaviour-free reproduction number ($\\mathcal{R}_0^{D}$)"
        )
        plt.colorbar(im)
        plt.show()
