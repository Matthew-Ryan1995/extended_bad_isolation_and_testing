#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 13:24:01 2025

@author: Matt Ryan
"""
# %%
from sympy import *
import numpy as np
from itertools import chain


# %% Functions

def get_update_paths(source="En", target="An"):
    # if source == "En":
    if target == "An":
        def update_paths(old_paths, i, **kwargs):

            kw_list = kwargs["kw_list"]
            ka_list = kwargs["ka_list"]
            w_list = kwargs["w_list"]
            a_list = kwargs["a_list"]

            if i == 0:
                if old_paths[0] == kwargs["w"]:
                    new_paths = [old_paths[0] * a_list[0]]
                elif old_paths[0] == kwargs["sw"]:
                    new_paths = [old_paths[0] * a_list[0]]
                else:
                    new_paths = [old_paths[0] * ka_list[0]]

                return new_paths

            new_paths = []
            # For every old path, we append a ka term
            for pp in old_paths:
                new_paths.append(pp*ka_list[i])

            # If the old path ends in ka, we append a loop a*w term
            for pp in old_paths:
                tmp = pp.subs((ka_list[i-1]), w_list[i]*a_list[i])
                if not tmp == pp:
                    new_paths.append(tmp)

            # If the old path ends in a, we push along a kw*a term
            for pp in old_paths:
                tmp = pp.subs((a_list[i-1]), kw_list[i]*a_list[i])
                if not tmp == pp:
                    new_paths.append(tmp)
            return new_paths
    elif target == "Ab":
        def update_paths(old_paths, i, **kwargs):

            kw_list = kwargs["kw_list"]
            ka_list = kwargs["ka_list"]
            w_list = kwargs["w_list"]
            a_list = kwargs["a_list"]

            if i == 0:
                if old_paths[0] == kwargs["w"]:
                    new_paths = [old_paths[0] * kw_list[0]]
                elif old_paths[0] == kwargs["sw"]:
                    new_paths = [old_paths[0] * kw_list[0]]
                else:
                    new_paths = [old_paths[0] * w_list[0]]

                return new_paths

            new_paths = []
            # For every old path, we append a kw term
            for pp in old_paths:
                new_paths.append(pp*kw_list[i])

            # If the old path ends in kw, we append a loop a*w term
            for pp in old_paths:
                tmp = pp.subs((kw_list[i-1]), w_list[i]*a_list[i])
                if not tmp == pp:
                    new_paths.append(tmp)

            # If the old path ends in w, we push along a ka*w term
            for pp in old_paths:
                tmp = pp.subs((w_list[i-1]), ka_list[i]*w_list[i])
                if not tmp == pp:
                    new_paths.append(tmp)
            return new_paths
    elif target == "In":
        def update_paths(old_paths, i, **kwargs):

            kw_list = kwargs["kw_list"]
            ka_list = kwargs["ka_list"]
            w_list = kwargs["w_list"]
            a_list = kwargs["a_list"]
            pT_list = kwargs["pT_list"]

            if i == 0:
                if old_paths[0] == kwargs["w"]:
                    new_paths = [old_paths[0] * (1-pT_list[0]) * a_list[0]]
                elif old_paths[0] == kwargs["sw"]:
                    new_paths = [old_paths[0] * (1-pT_list[0]) * a_list[0]]
                else:
                    new_paths = [old_paths[0] * ka_list[0]]

                return new_paths

            new_paths = []
            # For every old path, we append a ka term
            for pp in old_paths:
                new_paths.append(pp*ka_list[i])

            # If the old path ends in ka, we append a loop a*w*(1-pT) term
            for pp in old_paths:
                tmp = pp.subs(
                    (ka_list[i-1]), (1-pT_list[i]) * w_list[i] * a_list[i])
                if not tmp == pp:
                    new_paths.append(tmp)

            # If the old path ends in a, we push along a (1-pT) * kw*a term
            for pp in old_paths:
                tmp = pp.subs(
                    (a_list[i-1]), (1-pT_list[i]) * kw_list[i] * a_list[i])
                if not tmp == pp:
                    new_paths.append(tmp)
            return new_paths
    elif target == "Ib":
        def update_paths(old_paths, i, **kwargs):

            kw_list = kwargs["kw_list"]
            ka_list = kwargs["ka_list"]
            w_list = kwargs["w_list"]
            a_list = kwargs["a_list"]
            pT_list = kwargs["pT_list"]

            if i == 0:
                if old_paths[0] == kwargs["w"]:
                    new_paths = [old_paths[0] *
                                 (1-pT_list[0]) * kw_list[0]]
                elif old_paths[0] == kwargs["sw"]:
                    new_paths = [old_paths[0] *
                                 (1-pT_list[0]) * kw_list[0]]
                else:
                    new_paths = [old_paths[0] * w_list[0]]

                return new_paths

            new_paths = []
            # For every old path, we append a kw(1-pT) term
            for pp in old_paths:
                new_paths.append(pp*kw_list[i] * (1-pT_list[i]))

            # If the old path ends in kw, we append a loop a*w term
            for pp in old_paths:
                tmp = pp.subs((kw_list[i-1]),
                              w_list[i] * a_list[i])
                # tmp = pp.subs((kw_list[i-1]),
                #               w_list[i] * a_list[i] * (1-pT_list[i]))

                if not tmp == pp:
                    # tmp = tmp.subs((pT_list[i-1]), 0)
                    new_paths.append(tmp)

            # If the old path ends in w, we push along a ka*w term
            for pp in old_paths:
                tmp = pp.subs((w_list[i-1]),
                              ka_list[i] * w_list[i])
                # tmp = pp.subs((w_list[i-1]),
                #               ka_list[i] * w_list[i]*(1-pT_list[i]))

                if not tmp == pp:
                    # tmp = tmp.subs((pT_list[i-1]), 0)
                    new_paths.append(tmp)
            return new_paths
    elif target == "T":
        def update_paths(old_paths, i, **kwargs):

            kw_list = kwargs["kw_list"]
            ka_list = kwargs["ka_list"]
            w_list = kwargs["w_list"]
            a_list = kwargs["a_list"]
            pT_list = kwargs["pT_list"]

            if i == 0:
                if old_paths[0] == kwargs["w"]:
                    new_paths = [old_paths[0] * pT_list[0]]
                elif old_paths[0] == kwargs["sw"]:
                    new_paths = [old_paths[0] * pT_list[0]]
                else:
                    new_paths = old_paths

                return new_paths

            if i == 1:  # first path from En comes at i = 1
                if old_paths[0] == 0:
                    if kwargs["source"] == "En":
                        new_paths = [kwargs["sa"] * w_list[1]
                                     * pT_list[1] / kwargs["kaw"]]
                    elif kwargs["source"] == "Eb":
                        new_paths = [kwargs["a"] * w_list[1]
                                     * pT_list[1] / kwargs["kaw"]]
                    else:
                        Warning("Panic")
                    return new_paths

            new_paths = []

            # Attach kw * pT to the current paths
            for pp in old_paths:
                new_paths.append(
                    (pp*kw_list[i]*pT_list[i] / kwargs["kaw"]).subs((pT_list[i-1]), 1-pT_list[i-1]))

            # Replace kw*(1-pt) with a loop: a*w*pT
            for pp in old_paths:
                tmp = pp.subs((kw_list[i-1]),
                              a_list[i-1]*w_list[i]*pT_list[i] / kwargs["kaw"])
                if not tmp == pp:
                    tmp = tmp.subs((pT_list[i-1]), 1)
                    new_paths.append(tmp)

            # Replace w*pt with a push along the top: wk * w * pT
            for pp in old_paths:
                tmp = pp.subs(
                    (w_list[i-1]), ka_list[i-1]*w_list[i]*pT_list[i] / kwargs["kaw"])
                if not tmp == pp:
                    tmp = tmp.subs((pT_list[i-1]), 1)
                    new_paths.append(tmp)
            return new_paths
    else:
        assert False, "target must be either An, Ab, In, Ib, or T, the states of infectiousness."

    return update_paths


def simplify_paths(paths, l, simplify_pT=False, **kwargs):
    simplified_paths = []

    kw_list = kwargs["kw_list"]
    ka_list = kwargs["ka_list"]
    w_list = kwargs["w_list"]
    a_list = kwargs["a_list"]
    if simplify_pT:
        pT_list = kwargs["pT_list"]

    for pp in paths:
        for i in range(0, l):
            if pp == 0:
                continue
            pp = pp.subs((a_list[i]), kwargs["a"])
            pp = pp.subs((w_list[i]), kwargs["w"])
            pp = pp.subs((ka_list[i]), kwargs["ka"])
            pp = pp.subs((kw_list[i]), kwargs["kw"])

            if simplify_pT:
                pp = pp.subs((pT_list[i]), kwargs["pT"])
        simplified_paths.append(pp)

    return simplified_paths


def initialise_paths(source="En", target="An", **kwargs):
    sa = kwargs["sa"]
    sw = kwargs["sw"]
    a = kwargs["a"]
    w = kwargs["w"]

    if source == "En":

        if target == "T":
            initial_paths = [
                [0],  # Paths exiting En epidiemiolgically
                [w]  # Paths exiting Eb epidiemiolgically
            ]
        else:
            initial_paths = [
                [sa],  # Paths exiting En epidiemiolgically
                [w]  # Paths exiting En epidiemiolgically
            ]
    elif source == "Eb":
        if target == "T":
            initial_paths = [
                [0],  # Paths exiting En epidiemiolgically
                [sw]  # Paths exiting Eb epidiemiolgically
            ]
        else:
            initial_paths = [
                [a],  # Paths exiting En epidiemiolgically
                [sw]  # Paths exiting En epidiemiolgically
            ]
    else:
        assert False, "Source must be either En or Eb, the states at infection."

    return initial_paths

# TODO: evaluate outputs to derive fromulae


def t_source_target(l=7, source="En",
                    target="An",
                    simplify=True,
                    simplify_pT=True,
                    separate_symbols=False):

    if target == "An" or target == "Ab":
        simplify_pT = False

    sa = Symbol("(s + a)")
    sw = Symbol("(s + w)")

    a, w = symbols("a w")

    ka = Symbol("(kg + a)")
    kw = Symbol("(kg + w)")

    pA = Symbol("pA")
    pT = Symbol("pT")

    kg = Symbol("kg")
    saw = Symbol("(s+a+w)")
    kaw = Symbol("(kg+a+w)")

    initial_symbols = {
        "sa": sa,
        "sw": sw,
        "a": a,
        "w": w,
    }

    update_paths = get_update_paths(source=source,
                                    target=target)

    initial_paths = initialise_paths(source=source,
                                     target=target,
                                     **initial_symbols)
    paths = []

    for m, ppaths in enumerate(initial_paths):
        pp = [ppaths]
        a_list = [Symbol(f"a0")]
        w_list = [Symbol(f"w0")]
        ka_list = [Symbol(f"(kg + a0)")]
        kw_list = [Symbol(f"(kg + w0)")]
        pT_list = [Symbol(f"pT0")]

        terms_list = {
            "a_list": a_list.copy(),
            "w_list": w_list.copy(),
            "ka_list": ka_list.copy(),
            "kw_list": kw_list.copy(),
            "pT_list": pT_list.copy(),
            "sa": sa,  # Might just need to pass a and w, not sa and sw
            "sw": sw,
            "ka": ka,  # Might just need to pass a and w, not sa and sw
            "kw": kw,
            "a": a,
            "w": w,
            "pT": pT,
            "kaw": kaw,
            "source": source
        }
        for i in range(0, l):

            terms_list["ka_list"].append(Symbol(f"(kg + a{i+1})"))
            terms_list["kw_list"].append(Symbol(f"(kg + w{i+1})"))
            terms_list["a_list"].append(Symbol(f"a{i+1}"))
            terms_list["w_list"].append(Symbol(f"w{i+1}"))
            terms_list["pT_list"].append(Symbol(f"pT{i+1}"))

            old_paths = pp[i].copy()
            new_paths = update_paths(old_paths=old_paths,
                                     i=i,
                                     **terms_list)
            pp.append(new_paths)
        if target == "T":
            pp = list(chain.from_iterable(pp[1:]))
        else:
            pp = pp[-1]
        if simplify:
            simplified_pp = simplify_paths(paths=pp,
                                           l=l,
                                           simplify_pT=simplify_pT,
                                           **terms_list)
        else:
            simplified_pp = pp

        sum_pp = simplified_pp[0]
        for j in range(1, len(simplified_pp)):
            sum_pp += simplified_pp[j]

        paths.append(sum_pp)

    total_time = 0
    for component in paths:
        total_time += component

    if target == "An" or target == "Ab":
        total_time *= pA/(kg*saw*(kaw**l))
    elif target == "In" or target == "Ib":
        total_time *= (1-pA)/(kg*saw*(kaw**l))
    else:
        total_time *= (1-pA) / (kg * saw)

    if separate_symbols:
        k, g, s = symbols("k g s")
        total_time = total_time.subs((kg), k*g)
        total_time = total_time.subs((ka), k*g + a)
        total_time = total_time.subs((kw), k*g + w)
        total_time = total_time.subs((kaw), k*g + a + w)
        total_time = total_time.subs((sa), s+a)
        total_time = total_time.subs((sw), s+w)
        total_time = total_time.subs((saw), s+a+w)

    return total_time


def get_foi(k=1, source="En"):

    # Define symbols
    qA, qT = symbols("qA qT")
    beta_list = symbols(f"b1:{k+1}")

    # Define loops lists
    l_range = np.arange(start=1, stop=k+1, step=1)
    target_list = ["An", "Ab", "In", "Ib", "T"]

    foi = 0
    for i, l in enumerate(l_range):
        for t in target_list:
            tmp = t_source_target(l=l,
                                  source=source,
                                  target=t,
                                  simplify=True,
                                  simplify_pT=False,
                                  separate_symbols=True)
            if "A" in t:
                foi += qA * beta_list[i] * tmp
            elif "T" in t:
                foi += qT * beta_list[i] * tmp
            else:
                foi += beta_list[i] * tmp

    return foi


def get_R0(k=1):

    if k < 1:
        assert False, "There are no infection classes, try again"

    # Define symbols
    qB, B, N = symbols("qB B N")

    # Get FOIs
    Lambda_N = get_foi(k=k, source="En")
    Lambda_B = get_foi(k=k, source="Eb")

    # Leverage det = 0 in the NGM, take the trace
    R0 = Lambda_N * N + qB * Lambda_B * B

    return R0


def get_all_symbols(K=1):

    # Behavioural transistions
    a, w = symbols("a w")

    # Epi natural paratemeters
    k, g, s = symbols("k g s")

    # INfection free steady states
    B, N = symbols("B N")

    # FOI modifiers
    qB, qA, qT = symbols("qB qA qT")

    # Transmission rates
    beta_list = symbols(f"b1:{K+1}")

    # Probabilities
    pA = Symbol("pA")
    pT_list = []
    for i in range(K):
        pT_list.append(Symbol(f"pT{i}"))

    symb_dict = {
        # Behaviour transitions
        "a": a,
        "w": w,
        # Probabilities
        "pA": pA,
        "pT_list": pT_list,
        # Epi natural parameters
        "k": k,
        "g": g,
        "s": s,
        # Transmission rates
        "beta_list": beta_list,
        # FOI modifiers
        "qB": qB,
        "qA": qA,
        "qT": qT,
        # Infection free steady states
        "B": B,
        "N": N
    }

    return symb_dict


if __name__ == "__main__":

    # Check it simplifies when we expect it to.
    k = 3

    symbol_dict = get_all_symbols(K=k)

    R0 = get_R0(k=k)

    # Only no behaviour
    R0_simple = R0.subs((symbol_dict["N"]), 1).subs((symbol_dict["B"]), 0)
    R0_simple = R0_simple.subs(
        (symbol_dict["a"]), 0).subs((symbol_dict["w"]), 0)
    R0_simple = R0_simple.subs((symbol_dict["k"]), k)

    if k > 1:
        for i in range(1, k):
            R0_simple = R0_simple.subs(
                (symbol_dict["beta_list"][i]), symbol_dict["beta_list"][0])

    R0_simple = R0_simple.simplify()

    print(R0_simple)

# %% Testing

# def t_En_An(l=7):

#     # To make this ork with individual symbols, I need to think carefully
#     # about my update rules.
#     # pA, k, a, w, g, s = symbols("pA k a w g s")

#     # sa = s + a
#     # ka = k*g + a
#     # kw = k*g + w
#     # kg = k*g
#     # saw = s + a + w
#     # kaw = kg + a + w

#     sa = Symbol("(s + a)")
#     a, w = symbols("a w")
#     ka = Symbol("(kg + a)")
#     kw = Symbol("(kg + w)")
#     pA = Symbol("pA")
#     kg = Symbol("kg")
#     saw = Symbol("(s+a+w)")
#     kaw = Symbol("(k+a+w)")

#     # Paths exiting En
#     EN_paths = []

#     EN_paths.append(Mul(sa, ka, evaluate=False))

#     a_list = [a]
#     w_list = [w]
#     ka_list = [ka]
#     kw_list = [kw]

#     for i in range(1, l):

#         ka_list.append(Symbol(f"(kg + a{i})"))
#         kw_list.append(Symbol(f"(kg + w{i})"))
#         a_list.append(Symbol(f"a{i}"))
#         w_list.append(Symbol(f"w{i}"))

#         old_paths = EN_paths.copy()
#         new_paths = []

#         # For every old path, we append a ka term
#         for pp in old_paths:
#             new_paths.append(pp*ka_list[i])

#         # If the old path ends in ka, we append a loop a*w term
#         for pp in old_paths:
#             tmp = pp.subs((ka_list[i-1]), w_list[i]*a_list[i])
#             if not tmp == pp:
#                 new_paths.append(tmp)

#         # If the old path ends in a, we push along a kw*a term
#         for pp in old_paths:
#             tmp = pp.subs((a_list[i-1]), kw_list[i]*a_list[i])
#             if not tmp == pp:
#                 new_paths.append(tmp)
#         EN_paths = new_paths.copy()

#     simplified_EN_paths = []
#     for pp in EN_paths:
#         for i in range(1, l):
#             pp = pp.subs((a_list[i]), a_list[0]).subs((w_list[i]), w_list[0]).subs(
#                 (ka_list[i]), ka_list[0]).subs((kw_list[i]), kw_list[0])
#         simplified_EN_paths.append(pp)

#     cont_sum_EN = simplified_EN_paths[0]
#     for i in range(1, len(simplified_EN_paths)):
#         cont_sum_EN += simplified_EN_paths[i]

#     # Paths exiting Eb
#     EB_paths = []

#     EB_paths.append(Mul(w, a, evaluate=False))

#     a_list = [a]
#     w_list = [w]
#     ka_list = [ka]
#     kw_list = [kw]

#     for i in range(1, l):

#         ka_list.append(Symbol(f"(kg + a{i})"))
#         kw_list.append(Symbol(f"(kg + w{i})"))
#         a_list.append(Symbol(f"a{i}"))
#         w_list.append(Symbol(f"w{i}"))

#         old_paths = EB_paths.copy()
#         new_paths = []

#         # For every old path, we append a ka term
#         for pp in old_paths:
#             new_paths.append(pp*ka_list[i])

#         # If the old path ends in ka, we append a loop a*w term
#         for pp in old_paths:
#             tmp = pp.subs((ka_list[i-1]), w_list[i]*a_list[i])
#             if not tmp == pp:
#                 new_paths.append(tmp)

#         # If the old path ends in a, we push along a kw*a term
#         for pp in old_paths:
#             tmp = pp.subs((a_list[i-1]), kw_list[i]*a_list[i])
#             if not tmp == pp:
#                 new_paths.append(tmp)
#         EB_paths = new_paths.copy()

#     simplified_EB_paths = []
#     for pp in EB_paths:
#         for i in range(1, l):
#             pp = pp.subs((a_list[i]), a_list[0]).subs((w_list[i]), w_list[0]).subs(
#                 (ka_list[i]), ka_list[0]).subs((kw_list[i]), kw_list[0])
#         simplified_EB_paths.append(pp)

#     cont_sum_EB = simplified_EB_paths[0]
#     for i in range(1, len(simplified_EB_paths)):
#         cont_sum_EB += simplified_EB_paths[i]

#     total_time = cont_sum_EN + cont_sum_EB

#     total_time *= pA/(kg*saw*(kaw**l))

#     return total_time
