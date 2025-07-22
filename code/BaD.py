#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 13:34:08 2023

Class definitions and helper functions for a BaD SIRS model.

TODO:
    - Add incidence tracking
    - Add steady states
    - Add reproduction number

@author: Matt Ryan
"""


# %% Packages/libraries
import math
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib.ticker as tkr
import json
from enum import IntEnum
import sympyRo

# %% Plotting options
dpi = 300
flag_save_figs = False

# %% Class definitions


# class Compartments(IntEnum):
#     """
#     for speed ups whilst maintaining readability of code
#     """
#     Sn = 0
#     Sb = 1
#     En = 2
#     Eb = 3

#     An = 4
#     Ab = 5
#     In = 6
#     Ib = 7
#     T = 8

#     Rn = 9
#     Rb = 10

#     An_inc = 11
#     Ab_inc = 12
#     In_inc = 13
#     Ib_inc = 14
#     T_inc = 15
#     incidence = 16


def generate_compartments(k):
    # TODO: Add incidence trackers
    """Dynamically generate an IntEnum for BaD SEIAR with k stages for A, I and T."""
    names = ["Sn", "Sb", "En", "Eb"]
    names += [f"An{i+1}" for i in range(k)]
    names += [f"Ab{i+1}" for i in range(k)]
    names += [f"In{i+1}" for i in range(k)]
    names += [f"Ib{i+1}" for i in range(k)]
    names += [f"T{i+1}" for i in range(k)]
    names += ["Rn", "Rb"]
    names += ["A_inc", "I_inc", "T_inc", "Inc"]
    names += ["Tests"]
    return IntEnum("Compartment", {name: i for i, name in enumerate(names)})


class bad(object):
    """
    Implementation of the SIR model with behaviour (indicate will test if symptomatic) or not states for each
    compartment.  Explicitly, we assume proportion and not counts.
    Currently assuming no demography, no death due to pathogen, homogenous mixing, transitions between
    behaviour/no behaviour determined by social influence and fear of disease.  Currently assuming FD-like "infection"
    process for testing with fear of disease.
    """

    def __init__(self, **kwargs):
        """
        Written by: Roslyn Hickson
        Required parameters when initialising this class.
        :param transmission: double, the transmission rate from those infectious to those susceptible.
        :param infectious_period: scalar, the average infectious period.
        :param immune_period: scalar, average Immunity period (for SIRS)
        :param susc_B_efficacy: probability (0, 1), effectiveness in preventing disease contraction if S tests (c)
        :param inf_B_efficacy: probability (0, 1), effectiveness in preventing disease transmission if I tests (p)
        :param N_social: double, social influence of non-testers on testers (a1)
        :param N_fear: double, Fear of disease for testers to not test (a2)
        :param B_social: double, social influence of testers on non-testers (w1)
        :param B_fear: double, Fear of disease for non-testers to become testers (w2)
        :param av_lifespan: scalar, average life span in years
        """
        self.set_defaults()  # load default values from json file
        self.update_params(**kwargs)  # update with user specified values

        # set initial conditions
        self.CC = generate_compartments(k=self.k)
        self.set_initial_conditions()

    def set_defaults(self, filename="model_parameters.json"):
        """
        Written by: Roslyn Hickson
        Pull out default values from a file in json format.
        :param filename: json file containing default parameter values, which can be overridden by user specified values
        :return: loaded expected parameter values
        """
        with open(filename) as json_file:
            json_data = json.load(json_file)
        for key, value in json_data.items():
            json_data[key] = value["exp"]

        for key, value in json_data.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)
        # return json_data

    def update_params(self, **kwargs):
        args = kwargs
        for key, value in args.items():  # this is because I like the . notation. e.g. self.transmission
            self.__setattr__(key, value)

        # error handle period->rate conversions
        if self.immune_period == 0:
            self.nu = 1e6  # if immune period zero, means quickly move back to susceptible
        elif self.immune_period == np.inf:
            self.nu = 0  # if immune period is larege, means no move back to susceptible
        else:
            self.nu = 1/self.immune_period

        if self.infectious_period == 0:
            self.gamma = 1e6
        elif self.infectious_period == np.inf:
            self.gamma = 0
        else:
            self.gamma = 1/self.infectious_period
        if self.latent_period == 0:
            self.sigma = 1e6
        elif self.latent_period == np.inf:
            self.sigma = 0
        else:
            self.sigma = 1/self.latent_period

        self.k = int(self.k)

        # This feels hacky
        try:
            len(self.delta)
        except:
            self.delta = np.concatenate(([self.delta], np.zeros(self.k - 1)))
        assert len(
            self.delta) == self.k, "The length of delta much match the number of testing oppurtunities k"

        try:
            len(self.pT)
        except:
            self.pT = np.repeat(self.pT, repeats=self.k)
        assert len(
            self.pT) == self.k, "The length of delta much match the number of testing oppurtunities k"

        try:
            len(self.transmission)
        except:
            self.transmission = np.repeat(self.transmission, repeats=self.k)
        assert len(
            self.delta) == self.k, "The length of delta much match the number of testing oppurtunities k"

    def force_of_infection(self, prev_pop):
        """
        calculate force of infection, lambda

        Parameters
        ----------
        Ib : float
            current Ib(t)
        In : float
            current In(t)
        Ab : float
            current Ab(t)
        An : float
            current An(t)
        T : float
            current T(t)

        Returns
        -------
        lambda : float
            force of infection at time t.
        """

        foi = 0
        for l in range(self.k):
            foi += self.transmission[l] * (prev_pop[self.CC[f"In{l+1}"]] + prev_pop[self.CC[f"Ib{l+1}"]] + self.qA * (
                prev_pop[self.CC[f"An{l+1}"]] + prev_pop[self.CC[f"Ab{l+1}"]]) + self.qT * prev_pop[self.CC[f"T{l+1}"]])

        return foi

    # def ss_rate_to_infect(self, O, A, T):
    #     """
    #     calculate force of infection, lambda, at steady state

    #     Parameters
    #     ----------
    #     O : float
    #         Total in O=I_n+I_b+T at steady state
    #     A : float
    #         Total in A=A_n+A_b at steady state
    #     T : float
    #         Total in T at steady state

    #     Returns
    #     -------
    #     lambda : float
    #         force of infection at time steady state
    #     """
    #     return self.transmission * (O-T + self.qA * A + self.qT * T)

    def rate_to_test(self, B, T):  # i.e. omega
        return self.w1 * (B) + self.w2 * (T) + self.w3

    def rate_to_no_test(self, N):  # i.e. alpha
        return self.a1 * (N) + self.a2

    def odes(self, t, prev_pop, phi=False, psi=False, flag_track_incidence=False, flag_tests=False):
        """
        ODE set up to use spi.integrate.solve_ivp.  This defines the change in state at time t.

        Parameters
        ----------
        t : double
            time point.
        prev_pop : array
            State of the population at time t-1, in proportions.
            Assumes that it is of the form:
                [Sn, Sb, En, Eb, In, Ib, T, Rn, Rb]
        phi : a forcing function that multiplies the infection rate

        Returns
        -------
        Y : array
            rate of change of population compartments at time t.
        """

        Delta = self.delta
        if flag_tests:
            if prev_pop[self.CC["Tests"]] < 1:
                Delta = np.zeros(len(Delta))

        Y = np.zeros((len(prev_pop)))

        total_pop = prev_pop[0:(self.CC["Rb"] + 1)].sum()
        assert np.isclose(total_pop, 1.0), "total population deviating from 1"

        B_total = 0
        for cc in self.CC:
            if "b" in cc.name:
                B_total += prev_pop[cc.value]
        B_total /= total_pop

        T = 0
        for cc in self.CC:
            if "Test" in cc.name:
                continue
            if "nc" in cc.name:
                continue
            if "T" in cc.name:
                T += prev_pop[cc.value]
        T /= total_pop

        B_total += T

        # I_total = 0
        # for cc in CC:
        #     if ["I"] in cc.name:
        #         I_total += prev_pop[cc.value]
        # I_total /= total_pop

        # A_total = 0
        # for cc in CC:
        #     if ["A"] in cc.name:
        #         A_total += prev_pop[cc.value]
        # A_total /= total_pop

        lam = self.force_of_infection(prev_pop=prev_pop / total_pop)
        if phi:
            lam *= phi(t)

        omega = self.rate_to_test(B=B_total,
                                  T=T)
        alpha = self.rate_to_no_test(N=1.0-B_total)

        # Sn
        Y[self.CC["Sn"]] = -lam * prev_pop[self.CC["Sn"]] - omega * \
            prev_pop[self.CC["Sn"]] + alpha * \
            prev_pop[self.CC["Sb"]] + self.nu * prev_pop[self.CC["Rn"]]
        # En
        Y[self.CC["En"]] = lam * prev_pop[self.CC["Sn"]] - self.sigma * prev_pop[self.CC["En"]] - \
            omega * prev_pop[self.CC["En"]] + \
            alpha * prev_pop[self.CC["Eb"]]
        # An
        Y[self.CC["An1"]] = self.pA * self.sigma * prev_pop[self.CC["En"]] - self.k * self.gamma * \
            prev_pop[self.CC["An1"]] - omega * \
            prev_pop[self.CC["An1"]] + alpha * prev_pop[self.CC["Ab1"]]
        if self.k > 1:
            for j in range(2, self.k+1):
                Y[self.CC[f"An{j}"]] = self.k * self.gamma * prev_pop[self.CC[f"An{j-1}"]] - self.k * self.gamma * \
                    prev_pop[self.CC[f"An{j}"]] - omega * \
                    prev_pop[self.CC[f"An{j}"]] + \
                    alpha * prev_pop[self.CC[f"Ab{j}"]]
        # In
        Y[self.CC["In1"]] = (1.0-self.pA) * self.sigma * prev_pop[self.CC["En"]] - self.k * self.gamma * \
            prev_pop[self.CC["In1"]] - omega * \
            prev_pop[self.CC["In1"]] + alpha * prev_pop[self.CC["Ib1"]]
        if self.k > 1:
            for j in range(2, self.k+1):
                Y[self.CC[f"In{j}"]] = self.k * self.gamma * prev_pop[self.CC[f"In{j-1}"]] - self.k * self.gamma * \
                    prev_pop[self.CC[f"In{j}"]] - omega * \
                    prev_pop[self.CC[f"In{j}"]] + \
                    alpha * prev_pop[self.CC[f"Ib{j}"]]
        # Rn
        Y[self.CC["Rn"]] = self.k * self.gamma * (prev_pop[self.CC[f"An{self.k}"]] + prev_pop[self.CC[f"In{self.k}"]]) - omega * \
            prev_pop[self.CC["Rn"]] + alpha * \
            prev_pop[self.CC["Rb"]] - self.nu * prev_pop[self.CC["Rn"]]

        # Sb
        Y[self.CC["Sb"]] = -self.qB * lam * prev_pop[self.CC["Sb"]] + omega * \
            prev_pop[self.CC["Sn"]] - alpha * \
            prev_pop[self.CC["Sb"]] + self.nu * prev_pop[self.CC["Rb"]]
        # Eb
        Y[self.CC["Eb"]] = self.qB * lam * prev_pop[self.CC["Sb"]] - self.sigma * \
            prev_pop[self.CC["Eb"]] + omega * \
            prev_pop[self.CC["En"]] - alpha * prev_pop[self.CC["Eb"]]
        # Ab
        Y[self.CC["Ab1"]] = self.pA * self.sigma * prev_pop[self.CC["Eb"]] - self.k * self.gamma * \
            prev_pop[self.CC["Ab1"]] + omega * \
            prev_pop[self.CC["An1"]] - alpha * prev_pop[self.CC["Ab1"]]
        if self.k > 1:
            for j in range(2, self.k + 1):
                Y[self.CC[f"Ab{j}"]] = self.k * self.gamma * prev_pop[self.CC[f"Ab{j-1}"]] - self.k * self.gamma * \
                    prev_pop[self.CC[f"Ab{j}"]] + omega * \
                    prev_pop[self.CC[f"An{j}"]] - \
                    alpha * prev_pop[self.CC[f"Ab{j}"]]

        # Ib
        Y[self.CC["Ib1"]] = (1.0-self.pA) * (1.0-self.delta[0] * self.pT[0]) * self.sigma * prev_pop[self.CC["Eb"]] - self.k * self.gamma * \
            prev_pop[self.CC["Ib1"]] + omega * \
            prev_pop[self.CC["In1"]] - alpha * prev_pop[self.CC["Ib1"]]
        if self.k > 1:
            for j in range(2, self.k + 1):
                Y[self.CC[f"Ib{j}"]] = (1.0-self.delta[j-1] * self.pT[j-1]) * self.k * self.gamma * prev_pop[self.CC[f"Ib{j-1}"]] - self.k * self.gamma * \
                    prev_pop[self.CC[f"Ib{j}"]] + omega * \
                    prev_pop[self.CC[f"In{j}"]] - \
                    alpha * prev_pop[self.CC[f"Ib{j}"]]
        # T
        Y[self.CC["T1"]] = (1.0-self.pA) * self.delta[0] * self.pT[0] * self.sigma * \
            prev_pop[self.CC["Eb"]] - self.k * \
            self.gamma * prev_pop[self.CC["T1"]]
        if self.k > 1:
            for j in range(2, self.k + 1):
                Y[self.CC[f"T{j}"]] = self.k * self.gamma * prev_pop[self.CC[f"T{j-1}"]] - self.k * self.gamma * \
                    prev_pop[self.CC[f"T{j}"]] + \
                    self.delta[j-1] * self.pT[j-1] * self.k * \
                    self.gamma * prev_pop[self.CC[f"Ib{j-1}"]]

        # Rb
        Y[self.CC["Rb"]] = self.k * self.gamma * (prev_pop[self.CC[f"Ab{self.k}"]] + prev_pop[self.CC[f"Ib{self.k}"]] + prev_pop[self.CC[f"T{self.k}"]]) + \
            omega * prev_pop[self.CC["Rn"]] - alpha * \
            prev_pop[self.CC["Rb"]] - self.nu * prev_pop[self.CC["Rb"]]

        if not flag_track_incidence:
            assert np.isclose(Y.sum(), 0.0), "compartment RHSs not adding to 0"

        if flag_track_incidence:
            Y[self.CC["A_inc"]] = self.pA * self.sigma * \
                (prev_pop[self.CC["En"]] + prev_pop[self.CC["Eb"]])

            Y[self.CC["T_inc"]] = (1.0-self.pA) * self.delta[0] * \
                self.pT[0] * self.sigma * prev_pop[self.CC["Eb"]]
            if self.k > 1:
                for j in range(2, self.k + 1):
                    Y[self.CC["T_inc"]] += self.delta[j-1] * self.pT[j-1] * \
                        self.k * self.gamma * prev_pop[self.CC[f"Ib{j-1}"]]

            Y[self.CC["Inc"]] = lam * prev_pop[self.CC["Sn"]] + \
                self.qB * lam * prev_pop[self.CC["Sb"]]

            Y[self.CC["I_inc"]] = Y[self.CC["Inc"]] - \
                Y[self.CC["A_inc"]] - Y[self.CC["T_inc"]]

        return Y

    def set_initial_conditions(self, pop_size=1, starting_I=1e-6, starting_B=None):
        """

        :param pop_size: the total population size for behavioural status
        :param num_patches: the total number of patches
        :param starting_I: the total number of starting infectious
        :param starting_E: the total number of starting exposed
        :param starting_R: the total number of starting recovereds
        :return: the initial population by patch and disease status
        """

        # CC = generate_compartments(k=self.k)

        starting_population = np.zeros(max(self.CC) + 1)

        if starting_B is None:
            starting_B = pop_size*(1-self.get_ss_N(Tstar=0))
            # always have some behaviour unless pre specified
            starting_B = max(starting_B, 1e-6)

        # error handling
        assert pop_size > 0, "pop_size must be > 0"
        assert starting_B < pop_size, "starting_B must be smaller than pop_size"
        assert starting_B >= 0, "starting_B must be >= 0"
        assert starting_I >= 0, "starting_I must be >= 0"
        assert starting_I < pop_size - \
            starting_B, "starting_I must be smaller than pop_size - starting_B"

        starting_Sn = pop_size - starting_I - starting_B
        assert starting_Sn >= 0, "starting_Sn must be >= 0"

        # assign values
        starting_population[self.CC["In1"]] = starting_I
        starting_population[self.CC["Sb"]] = starting_B
        starting_population[self.CC["Sn"]] = starting_Sn

        self.init_cond = starting_population

    def run(self, t_start=0, t_end=200, t_step=1, t_eval=True, phi=False,
            events=[], flag_incidence_tracking=False):
        """
        Run the model and store data and time

        TO ADD: next gen matrix, equilibrium

        Parameters
        ----------
        IC : TYPE
            Initial condition vector
        t_start : TYPE
            starting time
        t_end : TYPE
            end time
        t_step : TYPE, optional
            time step. The default is 1.
        t_eval : TYPE, optional
            logical: do we evaluate for all time. The default is True.
        events:
            Can pass in a list of events to go to solve_ivp, i.e. stopping conditions
        flag_incidence_tracking:
            flag to indicate whether or not to track cumulative incidence
        phi : a forcing function that multiplies the infection rate

        Returns
        -------
        self with new data added

        """
        IC = self.init_cond

        # Set up positional arguments for odes
        args = []
        if phi:
            args.append(phi)
        else:
            args.append(False)

        # FIXME: Incidence not set up
        if flag_incidence_tracking:
            args.append(flag_incidence_tracking)
            IC = np.concatenate((IC, np.zeros(4)))
        else:
            args.append(False)

        if t_eval:
            t_range = np.arange(
                start=t_start, stop=t_end + t_step, step=t_step)
            self.t_range = t_range

            res = solve_ivp(fun=self.odes,
                            t_span=[t_start, t_end],
                            y0=IC,
                            t_eval=t_range,
                            events=events,
                            args=args
                            # rtol=1e-7, atol=1e-14
                            )
            self.results = res.y.T
        else:
            res = solve_ivp(fun=self.odes,
                            t_span=[t_start, t_end],
                            y0=IC,
                            events=events,
                            args=args)
            self.results = res.y.T

    def get_B(self):
        if hasattr(self, "results"):
            B_total = 0
            for cc in self.CC:
                if "Test" in cc.name:
                    continue
                if "nc" in cc.name:
                    continue
                if "b" in cc.name:
                    B_total += self.results[:, cc.value]
                if "T" in cc.name:
                    B_total += self.results[:, cc.value]
            return B_total
        else:
            print("Model has not been run")
            return np.nan

    def get_S(self):
        if hasattr(self, "results"):
            S_total = 0
            for cc in self.CC:
                if "S" in cc.name:
                    S_total += self.results[:, cc.value]
            return S_total
            # return np.sum(self.results[:, [self.CC["Sn"], self.CC["Sb"]]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_E(self):
        if hasattr(self, "results"):
            E_total = 0
            for cc in self.CC:
                if "E" in cc.name:
                    E_total += self.results[:, cc.value]
            return E_total
            # return np.sum(self.results[:, [self.CC["En"], self.CC["Eb"]]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_A(self):
        if hasattr(self, "results"):
            A_total = 0
            for cc in self.CC:
                if "nc" in cc.name:
                    continue
                if "A" in cc.name:
                    A_total += self.results[:, cc.value]
            return A_total
            # return np.sum(self.results[:, [self.CC["An"], self.CC["Ab"]]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_I(self):
        if hasattr(self, "results"):
            I_total = 0
            for cc in self.CC:
                if "nc" in cc.name:
                    continue
                if "I" in cc.name:
                    I_total += self.results[:, cc.value]
            return I_total
            # return np.sum(self.results[:, [self.CC["In"], self.CC["Ib"], self.CC["T"]]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_T(self):
        if hasattr(self, "results"):
            T_total = 0
            for cc in self.CC:
                if "nc" in cc.name:
                    continue
                if "Test" in cc.name:
                    continue
                if "T" in cc.name:
                    T_total += self.results[:, cc.value]
            return T_total
            # return np.sum(self.results[:, [self.CC["In"], self.CC["Ib"], self.CC["T"]]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_R(self):
        if hasattr(self, "results"):
            R_total = 0
            for cc in self.CC:
                if "R" in cc.name:
                    R_total += self.results[:, cc.value]
            return R_total
            # return np.sum(self.results[:, [self.CC["Rn"], self.CC["Rb"]]], 1)
            print("Model has not been run")
            return np.nan

    def get_all_infectious(self):
        if hasattr(self, "results"):
            Infectious_total = 0
            for cc in self.CC:
                if "nc" in cc.name:
                    continue
                if "Test" in cc.name:
                    continue
                if "A" in cc.name:
                    Infectious_total += self.results[:, cc.value]
                if "I" in cc.name:
                    Infectious_total += self.results[:, cc.value]
                if "T" in cc.name:
                    Infectious_total += self.results[:, cc.value]
            return Infectious_total
            # return np.sum(self.results[:, [self.CC["Ab"], self.CC["Ib"], self.CC["An"], self.CC["In"], self.CC["T"]]], 1)
        else:
            print("Model has not been run")
            return np.nan

    def get_incidence(self, comp=None):
        if hasattr(self, "results"):
            if comp is None:
                Incidence = self.results[:, self.CC["Inc"]]
            else:
                Incidence = self.results[:, self.CC[f"{comp}_inc"]]
            return Incidence
        else:
            print("Model has not been run")
            return np.nan

    def get_reproduction_number(self):

        R0 = sympyRo.get_R0(k=self.k)
        symbols = sympyRo.get_all_symbols(K=self.k)

        N = self.get_ss_N(Tstar=0)
        a = self.rate_to_no_test(N=N)
        w = self.rate_to_test(B=1-N, T=0)

        R0 = R0.subs((symbols["k"]), self.k)
        R0 = R0.subs((symbols["pA"]), self.pA)
        R0 = R0.subs((symbols["g"]), self.gamma)
        R0 = R0.subs((symbols["s"]), self.sigma)
        R0 = R0.subs((symbols["qB"]), self.qB)
        R0 = R0.subs((symbols["qA"]), self.qA)
        R0 = R0.subs((symbols["qT"]), self.qT)

        for i in range(self.k):
            R0 = R0.subs((symbols["beta_list"][i]), self.transmission[i])
            R0 = R0.subs((symbols["pT_list"][i]), self.delta[i] * self.pT[i])

        R0 = R0.subs((symbols["B"]), 1-N)
        R0 = R0.subs((symbols["N"]), N)
        R0 = R0.subs((symbols["a"]), a)
        R0 = R0.subs((symbols["w"]), w)

        return float(R0)

    # def get_ss_B_a_w(self, T):
    #     """
    #     Calculate the steady state of behaviour, and alpha and omega values at steady state when T is known.

    #     Parameters
    #     ----------
    #     T : float
    #         Chosen proportion who test positive at disease equilibrium

    #     Returns
    #     -------
    #     B : float
    #         Proportion of the population performing the behaviour at equilibrium
    #     a : float
    #         alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
    #     w : float
    #         omega transition value at equilibrium (w1 * B + w2 * I + w3)
    #     """

    #     # handle infeasible values of T
    #     if T < 1e-8:
    #         T= 0
    #     if T > 1:
    #         T= 1
    #     if np.isnan(T):
    #         T= 0

    #     # Writing equation in the form C N^2 - D N + E
    #     C= round(self.w1 - self.a1, 8)
    #     D= round(self.w2 * T + self.w1 + self.w3 +
    #               self.a2 - self.a1 * (1-T), 8)
    #     E= round(self.a2 * (1-T), 8)

    #     assert round(E,
    #                  8) >= 0, f"Condition of G(N=0)= alpha_2 * (1-T)>0 not held in get_ss_B_a_w, T={T}"
    #     assert round(C - D + E,
    #                  8) <= 0, f"Condition of G(N=1)=C-D+E<0 not held in get_ss_B_a_w, T={T}"

    #     if np.isclose(C, 0):
    #         if np.isclose(D, 0):
    #             N= 1
    #         else:
    #             N= E/D
    #     else:
    #         N_opt1= round((D - np.sqrt(D**2 - 4*C*E))/(2*C), 8)
    #         N_opt2= round((D + np.sqrt(D**2 - 4*C*E))/(2*C), 8)
    #         if N_opt1 >= 0 and N_opt1 <= 1:
    #             N= N_opt1
    #         elif N_opt2 >= 0 and N_opt2 <= 1:
    #             N= N_opt2
    #         else:
    #             print(
    #                 f"Error: no valid steady state solution found for N in get_B_a_w, N_opt1={N_opt1}, N_opt2={N_opt2}, T={T}")
    #             print(f"C: {C}")
    #             print(f"D: {D}")
    #             print(f"E: {E}")
    #             N= 0.5  # =N_opt1 would be equivalent to code before I added this to explore what was happening
    #     B= 1 - N
    #     a= self.rate_to_no_test(N)  # already defined elsewhere
    #     w= self.rate_to_test(B, T)

    #     return B, a, w

    # def get_ss_SEAR(self, O):
    #     """
    #     Calculate the steady state of S, E, A, and R when I is known.

    #     Parameters
    #     ----------
    #     I : float
    #         Desired prevalence of disease at equilibrium

    #     Returns
    #     -------
    #     R : float
    #         Proportion of the population recovered at equilibrium
    #     S : float
    #         Proportion of the population susceptible at equilibrium
    #     """
    #     E= (self.gamma * O)/((1.0-self.pA) * self.sigma)
    #     A= (self.pA * O)/(1.0-self.pA)
    #     R= (self.gamma * O)/((1.0-self.pA)*self.nu)
    #     S= 1.0 - (E + A + O + R)

    #     return S, E, A, R

    # def get_ss_Eb(self, S, E, a, w, lam):
    #     """
    #     Calculate the steady state of Eb when steady states for S and E are known.

    #     Parameters
    #     ----------
    #     S : float
    #         Steady state for total in S
    #     E : float
    #         Steady state for total in E
    #     a : float
    #         Steady state for alpha: movement out of behaviour
    #     w : float
    #         Steady state for omega: movement into behaviour
    #     lam : float
    #         Steady state force of infection

    #     Returns
    #     -------
    #     Eb : float
    #         Proportion of the population in Eb at equilibrium
    #     """
    #     numer= self.qB * lam * S - (self.qB * (w + self.sigma) - w) * E
    #     denom= (1.0-self.qB) * (a + w + self.sigma)

    #     return numer/denom

    # def get_ss_T(self, Eb):
    #     """
    #     Calculate the steady state of T, based on parameters + steady state Eb

    #     Parameters
    #     ----------
    #     Eb : float
    #         Steady state for total in Eb

    #     Returns
    #     -------
    #     T : float
    #         Proportion of the population in T at equilibrium
    #     """
    #     return ((1-self.pA) * self.pT * self.sigma * Eb)/self.gamma

    # def get_ss_Ab(self, Eb, A, a, w):
    #     """
    #     Calculate the steady state of Ab when steady states for A and Eb are known.

    #     Parameters
    #     ----------
    #     Eb : float
    #         Steady state for Eb
    #     A : float
    #         Steady state for total in A
    #     a : float
    #         Steady state for alpha: movement out of behaviour
    #     w : float
    #         Steady state for omega: movement into behaviour

    #     Returns
    #     -------
    #     Ab : float
    #         Proportion of the population in Ab at equilibrium
    #     """

    #     numer= self.pA * self.sigma * Eb + w * A
    #     denom= self.gamma + a + w

    #     return numer/denom

    # def get_ss_Ib(self, O, T, Eb, a, w):
    #     """
    #     Calculate the proportion of people infected and performing the behaviour

    #     Parameters
    #     ----------
    #     O : float
    #         Desired prevalence of disease at equilibrium
    #     S : float
    #         Prevalence of susceptible at equilibrium
    #     a : float
    #         alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
    #     w : float
    #         omega transition value at equilibrium (w1 * B + w2 * I + w3)

    #     Returns
    #     -------
    #     Ib : float
    #         Prevalence of infected performing the behaviour.

    #     """

    #     numer= (1-self.pA) * (1-self.pT) * self.sigma * Eb + w * \
    #         (O-T)
    #     denom= self.gamma + a + w

    #     return numer/denom

    # def get_ss_Rb(self, T, R, Ib, Ab, a, w):
    #     """
    #     Calculate steady state proportion of recovereds doing behaviour.

    #     Parameters
    #     ----------
    #     R : float
    #         Proportion of individuals who are recovered.
    #     Ib : float
    #         Proportion of infected individuals doing the behaviour.
    #     a : float
    #         alpha transition value at equilibrium (a1 * N + a2 * (1 - I) + a3)
    #     w : float
    #         omega transition value at equilibrium (w1 * B + w2 * I + w3)

    #     Returns
    #     -------
    #     Rb : float
    #         Proportion of recovered individuals doing the behaviour.

    #     """
    #     numer= self.gamma * (Ab + Ib + T) + w * R
    #     denom= a + w + self.nu

    #     return numer/denom

    # def get_steady_states(self, O, T):
    #     """
    #     Calculate the steady state vector for a given disease prevalence and set of parameters.
    #     Parameters
    #     ----------
    #     O : float
    #         Desired prevalence of disease at equilibrium; O = In +Ib + T
    #     T : float
    #         value of T at equilibrium

    #     Returns
    #     -------
    #     ss : numpy.array
    #         Vector of steady states of the form [Sn, Sb, In, Ib, Rn, Rb]
    #     """

    #     B, a, w= self.get_ss_B_a_w(T)
    #     N= 1.0-B

    #     if O <= 0.0:
    #         ans= np.zeros(11)
    #         ans[self.CC["Sn"]]= N
    #         ans[self.CC["Sb"]]= B
    #         return ans

    #     S, E, A, R= self.get_ss_SEAR(O)
    #     lam= self.ss_rate_to_infect(O=O, A=A, T=T)
    #     Eb= self.get_ss_Eb(S, E, a, w, lam)
    #     Ab= self.get_ss_Ab(Eb, A, a, w)
    #     Ib= self.get_ss_Ib(O, T, Eb, a, w)
    #     Rb= self.get_ss_Rb(T, R, Ib, Ab, a, w)

    #     Sb= B - (Eb + Ab + Ib + T + Rb)

    #     Sn= S-Sb
    #     En= E-Eb
    #     An= A-Ab
    #     Rn= R-Rb

    #     In= N - (Sn+En+An+Rn)

    #     ans= np.zeros(11)
    #     ans[self.CC["Sn"]]= Sn
    #     ans[self.CC["En"]]= En
    #     ans[self.CC["An"]]= An
    #     ans[self.CC["In"]]= In
    #     ans[self.CC["Rn"]]= Rn
    #     ans[self.CC["Sb"]]= Sb
    #     ans[self.CC["Eb"]]= Eb
    #     ans[self.CC["Ab"]]= Ab
    #     ans[self.CC["Ib"]]= Ib
    #     ans[self.CC["T"]]= T
    #     ans[self.CC["Rb"]]= Rb

    #     return ans

    # def solve_ss_T_O(self, x):
    #     """
    #     Function to numerically find the disease prevalence at equilibrium for a given set
    #     of model parameters.  Designed to be used in conjunction with fsolve.

    #     Parameters
    #     ----------
    #     x : [float, float]
    #         Estimated disease prevalence at equilibrium for [T, O]. T=Total observed testing, O=In+Ib+T total symptomatic prevalence.

    #     Returns
    #     -------
    #     res : float
    #         The difference between the (lambda + w)*S_n and a*S_b + nu*R_n.
    #     """
    #     T= x[0]
    #     O= x[1]

    #     _, a, w= self.get_ss_B_a_w(T=T)

    #     ss_n= self.get_steady_states(O=O, T=T)

    #     A= ss_n[self.CC["An"]] + ss_n[self.CC["Ab"]]

    #     # T_est = ((1-pA) * pT * s * ss_n[self.CC["Eb])/g

    #     lam= self.ss_rate_to_infect(O=O, A=A, T=T)

    #     res= [T-self.get_ss_T(Eb=ss_n[self.CC["Eb"]]), (lam + w) * ss_n[self.CC["Sn"]] -
    #            (a * ss_n[self.CC["Sb"]] + self.nu * ss_n[self.CC["Rn"]])]

    #     return np.array(res)

    # def find_ss(self, init_cond=np.nan):
    #     """
    #     Calculate the steady states of the system for a given set of model parameters.  We first numerically
    #     solve for the disease prevalence I, then use I to find all other steady states.

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #     ss : numpy.array
    #         Vector of steady states of the form [Sn, Sb, In, Ib, Rn, Rb]
    #     Istar : float
    #         Estimated disease prevalence at equilibrium
    #     """

    #     if np.isnan(init_cond).all():
    #         # self.nu/(self.gamma + self.nu) - 1e-3
    #         # init_o = np.random.uniform(high=0.2, size=1)
    #         try:
    #             E= (self.transmission*(1-(1-self.qA)*self.pA) - self.gamma) / (self.transmission *
    #                                                                             (1-(1-self.qA)*self.pA) * (self.sigma/self.nu + self.sigma/self.gamma + 1))
    #             assert E >= 0
    #             init_o= (1-self.pA)*self.sigma/self.gamma * E - 1e-3
    #         except:
    #             init_o= np.random.uniform(high=0.2, size=1)
    #         init_t= init_o/10
    #     else:
    #         init_t= init_cond[0]
    #         init_o= init_cond[1]

    #     stop_while_flag= True
    #     R0= self.get_reproduction_number()
    #     c= 0
    #     while stop_while_flag:
    #         res= fsolve(self.solve_ss_T_O,
    #                      x0=[init_t, init_o],
    #                      xtol=1e-8)

    #         # Is solutions are too small then set to 0
    #         if (res[0] < self.init_cond[self.CC["In"]]):
    #             res[0]=0
    #         if (res[1] < self.init_cond[self.CC["In"]]):
    #             res[1]=0

    #         if res[0] > res[1]:  # T < O
    #             init_o=np.random.uniform(high=0.2, size=1)
    #             init_t=init_o/10
    #         elif res.sum() < 1e-8 and R0 > 1:  # DFE unstale
    #             init_o=np.random.uniform(high=0.2, size=1)
    #             init_t=init_o/10
    #         elif res.sum() > 1 or res[0] > 1 or res[1] > 1:  # T, O < 1
    #             init_o=np.random.uniform(high=0.2, size=1)
    #             init_t=init_o/10
    #         else:
    #             ss=self.get_steady_states(O=res[1], T=res[0])

    #             ss=ss.round(10)

    #             J=self.get_J(ss=ss)

    #             eigenValues, _=np.linalg.eig(J)
    #             # Ensure found ss is a solution
    #             if not (np.isclose(self.odes(t=0, prev_pop=ss), 0)).all():
    #                 init_o=np.random.uniform(high=0.2, size=1)
    #                 init_t=init_o/10
    #             elif (ss < 0).any():  # Feasibility must be positive
    #                 init_o=np.random.uniform(high=0.2, size=1)
    #                 init_t=init_o/10
    #             elif not np.isclose(ss.sum(), 1):  # Feasibility, must sum to 1
    #                 init_o=np.random.uniform(high=0.2, size=1)
    #                 init_t=init_o/10
    #             elif (np.real(eigenValues) > 0).any():  # ss must be stable
    #                 init_o=np.random.uniform(high=0.2, size=1)
    #                 init_t=init_o/10
    #             else:
    #                 stop_while_flag=False
    #         c += 1
    #         if c > 100:
    #             # When all else fails, calculate numerically.
    #             self.run(t_end=1000, t_step=0.5)
    #             ss=self.results[-1, :]
    #             res="fsolve did not converge"
    #             stop_while_flag=False
    #     return ss, res

    def get_ss_N(self, Tstar=np.nan):

        if np.isnan(Tstar):
            Tstar = self.results[-1, self.CC["T"]]

        A = self.w1 - self.a1
        B = -(self.w2*Tstar + self.w1 + self.w3 + self.a2 - self.a1 * (1-Tstar))
        C = self.a2 * (1-Tstar)

        if np.isclose(A, 0):
            ans = -C/B
        else:
            ans = (-B - np.sqrt(B**2 - 4*A*C)) / (2*A)

        ans = min(1, ans)
        ans = max(0, ans)

        return ans

    # def get_reproduction_number(self):

    #     T= 0
    #     B0, alpha, omega= self.get_ss_B_a_w(T)
    #     N0= 1-B0

    #     if self.infectious_period == 0:
    #         gamma= 0
    #     else:
    #         gamma= 1/self.infectious_period
    #     if self.latent_period == 0:
    #         sigma= 0
    #     else:
    #         sigma= 1/self.latent_period

    #     saw= sigma + alpha + omega
    #     gaw= gamma + alpha + omega

    #     denom= gamma*saw*gaw

    #     tnAn= self.pA * (alpha * saw + gamma * (sigma + alpha)) / denom
    #     tnIn= (1-self.pA) * (alpha * saw + gamma *
    #                           (sigma + alpha) - self.pT * alpha * omega) / denom
    #     tnAb= self.pA * omega * (gamma + sigma + alpha + omega) / denom
    #     tnIb= (1-self.pA) * omega * (sigma + alpha +
    #                                   (1-self.pT)*(gamma + omega)) / denom
    #     tnT= (1-self.pA) * self.pT * omega / (gamma * saw)

    #     tbAn= self.pA * alpha * (gamma + sigma + alpha + omega) / denom
    #     tbIn= (1-self.pA) * alpha * (gamma + alpha +
    #                                   (1-self.pT)*(sigma + omega)) / denom
    #     tbAb= self.pA * (omega * saw + gamma * (sigma + omega))/denom
    #     tbIb= (1-self.pA)*(omega * saw + gamma * (sigma + omega) -
    #                         self.pT * (sigma + omega) * (gamma + omega)) / denom

    #     # tbAb = (self.pA * omega * (gamma + sigma + alpha + omega) +
    #     #         self.pA * sigma * gamma) / denom
    #     # tbIb = ((1-self.pA) * alpha * omega + (1-self.pA) *
    #     #         (1-self.pT) * (sigma*gamma + omega * (sigma + gamma + omega))) / denom
    #     tbT= (1-self.pA) * self.pT * (sigma + omega) / (gamma * saw)

    #     Lambda_N= self.transmission * \
    #         (tnIn + tnIb + self.qA * (tnAn + tnAb) + self.qT * tnT)
    #     Lambda_B= self.transmission * \
    #         (tbIn + tbIb + self.qA * (tbAn + tbAb) + self.qT * tbT)

    #     R0= Lambda_N * N0 + self.qB * Lambda_B * B0

    #     return R0

    # def get_J0(self, N, T, a, w, gamma):

    #     dwdB= self.w1
    #     dadN= self.a1
    #     dwdT= self.w2

    #     J0= np.array(
    #         [[dwdB * N + dadN * (1-N - T) - w - a, -dwdT * N - a],
    #          [0, -gamma]]
    #     )

    #     return J0

    # def get_Jn0(self, ss, nu):

    #     dwdB= self.w1
    #     dadN= self.a1
    #     dwdT= self.w2

    #     Sn= ss[self.CC["Sn"]]
    #     En= ss[self.CC["En"]]
    #     An= ss[self.CC["An"]]
    #     In= ss[self.CC["In"]]

    #     Sb= ss[self.CC["Sb"]]
    #     Eb= ss[self.CC["Eb"]]
    #     Ab= ss[self.CC["Ab"]]
    #     Ib= ss[self.CC["Ib"]]

    #     Jn0= np.array(
    #         [
    #             [dwdB * Sn + dadN * Sb + nu, -self.qT *
    #                 self.transmission * Sn - dwdT * Sn],
    #             [dwdB * En + dadN * Eb, self.qT *
    #                 self.transmission * Sn - dwdT * En],
    #             [dwdB * An + dadN * Ab, -dwdT * An],
    #             [dwdB * In + dadN * Ib, -dwdT * In]
    #         ]
    #     )

    #     return Jn0

    # def get_Jb0(self, ss, nu):

    #     dwdB= self.w1
    #     dadN= self.a1
    #     dwdT= self.w2

    #     Sn= ss[self.CC["Sn"]]
    #     En= ss[self.CC["En"]]
    #     An= ss[self.CC["An"]]
    #     In= ss[self.CC["In"]]

    #     Sb= ss[self.CC["Sb"]]
    #     Eb= ss[self.CC["Eb"]]
    #     Ab= ss[self.CC["Ab"]]
    #     Ib= ss[self.CC["Ib"]]

    #     Jb0= np.array(
    #         [
    #             [-dwdB * Sn - dadN * Sb - nu, -self.qB * self.qT *
    #                 self.transmission * Sb + dwdT * Sn - nu],
    #             [-dwdB * En - dadN * Eb, self.qB * self.qT *
    #                 self.transmission * Sb + dwdT * En],
    #             [-dwdB * An - dadN * Ab, dwdT * An],
    #             [-dwdB * In - dadN * Ib, dwdT * In]
    #         ]
    #     )

    #     return Jb0

    # def get_Jnn(self, lam, w, nu, sigma, gamma, Sn):

    #     Jnn= np.array(
    #         [
    #             [-lam - w - nu, -nu, -self.qA*self.transmission *
    #                 Sn - nu, -self.transmission*Sn - nu],
    #             [lam, -sigma - w, self.qA*self.transmission*Sn, self.transmission*Sn],
    #             [0, self.pA*sigma, -gamma - w, 0],
    #             [0, (1-self.pA)*sigma, 0, -gamma - w]
    #         ]
    #     )

    #     return Jnn

    # def get_Jnb(self, a, Sn):

    #     Jnb= np.identity(4) * a

    #     Jnb[0, 2]= -self.qA * self.transmission * Sn
    #     Jnb[1, 2]= self.qA * self.transmission * Sn
    #     Jnb[0, 3]= -self.transmission * Sn
    #     Jnb[1, 3]= self.transmission * Sn

    #     return Jnb

    # def get_Jbn(self, w, Sb):

    #     Jbn= np.identity(4) * w

    #     Jbn[0, 2]= -self.qB * self.qA * self.transmission * Sb
    #     Jbn[1, 2]= self.qB * self.qA * self.transmission * Sb
    #     Jbn[0, 3]= -self.qB * self.transmission * Sb
    #     Jbn[1, 3]= self.qB * self.transmission * Sb

    #     return Jbn

    # def get_Jbb(self, lam, a, nu, sigma, gamma, Sb):

    #     Jbb= np.array(
    #         [
    #             [-self.qB*lam - a - nu, -nu, -self.qB*self.qA*self.transmission *
    #                 Sb - nu, -self.qB*self.transmission*Sb - nu],
    #             [self.qB*lam, -sigma - a, self.qA*self.qB *
    #                 self.transmission*Sb, self.qB*self.transmission*Sb],
    #             [0, self.pA*sigma, -gamma - a, 0],
    #             [0, (1-self.pA)*(1-self.pT)*sigma, 0, -gamma - a]
    #         ]
    #     )

    #     return Jbb

    # def get_J(self, ss=np.nan, numeric=False):

    #     if np.isnan(ss).any():
    #         if numeric:
    #             ss= self.results[-1, :]
    #         else:
    #             ss, _= self.find_ss()

    #     if self.infectious_period == 0:
    #         gamma= 0
    #     else:
    #         gamma= 1/self.infectious_period
    #     if self.latent_period == 0:
    #         sigma= 0
    #     else:
    #         sigma= 1/self.latent_period
    #     if self.immune_period == 0:
    #         nu= 0
    #     else:
    #         nu= 1/self.immune_period

    #     N= ss[[self.CC["Sn"], self.CC["En"],
    #             self.CC["An"], self.CC["In"],
    #             self.CC["Rn"]]].sum()

    #     Sn= ss[self.CC["Sn"]]
    #     En= ss[self.CC["En"]]
    #     An= ss[self.CC["An"]]
    #     In= ss[self.CC["In"]]

    #     Sb= ss[self.CC["Sb"]]
    #     Eb= ss[self.CC["Eb"]]
    #     Ab= ss[self.CC["Ab"]]
    #     Ib= ss[self.CC["Ib"]]
    #     T= ss[self.CC["T"]]

    #     _, a, w= self.get_ss_B_a_w(T)
    #     lam= self.rate_to_infect(Ib, In, An, Ab, T)

    #     J0= self.get_J0(N, T, a, w, gamma)

    #     J0n= np.zeros((2, 4))

    #     J0b= np.zeros((2, 4))
    #     J0b[1, 1]= (1-self.pA) * self.pT * sigma

    #     Jn0= self.get_Jn0(ss, nu)
    #     Jb0= self.get_Jb0(ss, nu)

    #     Jnn= self.get_Jnn(lam, w, nu, sigma, gamma, Sn)
    #     Jbb= self.get_Jbb(lam, a, nu, sigma, gamma, Sb)

    #     Jbn= self.get_Jbn(w, Sb)
    #     Jnb= self.get_Jnb(a, Sn)

    #     J= np.block(
    #         [
    #             [J0, J0n, J0b],
    #             [Jn0, Jnn, Jnb],
    #             [Jb0, Jbn, Jbb]
    #         ]
    #     )

    #     return J

    # def get_effective_reproduction_number(self):

    #     if hasattr(self, "results"):
    #         T= self.results[:, self.CC["T"]]

    #         B= self.get_B()
    #         omega= self.rate_to_test(B=B, T=T)
    #         alpha= self.rate_to_no_test(N=1-B)
    #         N0= self.results[:, self.CC["Sn"]]
    #         B0= self.results[:, self.CC["Sb"]]

    #         if self.infectious_period == 0:
    #             gamma= 0
    #         else:
    #             gamma= 1/self.infectious_period
    #         if self.latent_period == 0:
    #             sigma= 0
    #         else:
    #             sigma= 1/self.latent_period

    #         saw= sigma + alpha + omega
    #         gaw= gamma + alpha + omega

    #         denom= gamma*saw*gaw

    #         tnAn= self.pA * (alpha * saw + gamma * (sigma + alpha)) / denom
    #         tnIn= (1-self.pA) * (alpha * saw + gamma *
    #                               (sigma + alpha) - self.pT * alpha * omega) / denom
    #         tnAb= self.pA * omega * (gamma + sigma + alpha + omega) / denom
    #         tnIb= (1-self.pA) * omega * (sigma + alpha +
    #                                       (1-self.pT)*(gamma + omega)) / denom
    #         tnT= (1-self.pA) * self.pT * omega / (gamma * saw)

    #         tbAn= self.pA * alpha * (gamma + sigma + alpha + omega) / denom
    #         tbIn= (1-self.pA) * alpha * (gamma + alpha +
    #                                       (1-self.pT)*(sigma + omega)) / denom
    #         tbAb= self.pA * (omega * saw + gamma * (sigma + omega))/denom
    #         tbIb= (1-self.pA)*(omega * saw + gamma * (sigma + omega) -
    #                             self.pT * (sigma + omega) * (gamma + omega)) / denom

    #         tbT= (1-self.pA) * self.pT * (sigma + omega) / (gamma * saw)

    #         Lambda_N= self.transmission * \
    #             (tnIn + tnIb + self.qA * (tnAn + tnAb) + self.qT * tnT)
    #         Lambda_B= self.transmission * \
    #             (tbIn + tbIb + self.qA * (tbAn + tbAb) + self.qT * tbT)

    #         R0= Lambda_N * N0 + self.qB * Lambda_B * B0

    #         return R0

    #     else:
    #         print("Model has not been run")
    #         return np.nan

    def plot(self):

        if not hasattr(self, 'results'):
            print("Model not run")
            return np.nan

        P = self.init_cond[0:(self.CC["Rb"] + 1)].sum()

        # plt.figure()
        # plt.title("BaD: Compartment dynamics per strata")
        # plt.plot(self.t_range,
        #          self.results[:, self.compartments.Sn] / P, label="$S_N$")
        # plt.plot(self.t_range,
        #          self.results[:, self.compartments.Sb] / P, label="$S_B$")
        # plt.plot(self.t_range,
        #          self.results[:, self.compartments.In] / P, label="$I_N$")
        # plt.plot(self.t_range,
        #          self.results[:, self.compartments.Ib] / P, label="$I_B$")
        # plt.plot(self.t_range,
        #          self.results[:, self.compartments.Rn] / P, label="$R_N$")
        # plt.plot(self.t_range,
        #          self.results[:, self.compartments.Rb] / P, label="$R_B$")
        # plt.legend(loc=(1.01, 0.5))
        # plt.xlabel("Time")
        # plt.ylabel("Proportion of population")
        # plt.show()

        plt.figure()
        plt.title("BaD: Compartment dynamics")
        plt.plot(self.t_range,
                 self.get_S() / P, label="$S$")
        plt.plot(self.t_range,
                 self.get_E() / P, label="$E$")
        plt.plot(self.t_range,
                 self.get_A() / P, label="$A$")
        plt.plot(self.t_range,
                 self.get_I() / P, label="$I$")
        plt.plot(self.t_range,
                 self.get_T() / P, label="$T$")
        plt.plot(self.t_range,
                 self.get_R() / P, label="$R$")
        plt.legend(loc=(1.01, 0.5))
        plt.xlabel("Time")
        plt.ylabel("Proportion of population")
        plt.show()

        # plt.figure()
        # plt.title("BaD: Infection dynamics")
        # plt.plot(self.t_range,
        #          self.get_I(), label="$I$")
        # plt.plot(self.t_range,
        #          self.results[:, self.compartments.In], label="$I_N$")
        # plt.plot(self.t_range,
        #          self.results[:, self.compartments.Ib], label="$I_B$")
        # plt.legend()
        # plt.xlabel("Time")
        # plt.ylabel("Proportion of population")
        # plt.show()

        plt.figure()
        plt.title("BaD: Behaviour dynamics")
        plt.plot(self.t_range, self.get_B())
        plt.xlabel("Time")
        plt.ylabel("Proportion of population doing behaviour")
        plt.show()

        plt.figure()
        plt.title("S - I Phase Plane")
        plt.plot(self.get_S(), self.get_all_infectious())
        plt.xlabel("Susceptibles (Sn + Sb)")
        plt.ylabel("Infectious (I + A + T)")
        plt.show()

        plt.figure()
        plt.title("B - T Phase Plane")
        plt.plot(self.get_B(), self.get_T())
        plt.xlabel("Behaviour (Sb + Eb + Ab + Ib + T + Rb)")
        plt.ylabel("Positive tests (T)")
        plt.show()

        # plt.figure()
        # plt.title("BaD: Effective reproduction number")
        # plt.plot(self.t_range,
        #          self.get_effective_reproduction_number(), color="green")
        # plt.xlabel("time")
        # plt.ylabel("$\\mathcal{R}_{eff}$")
        # plt.show()


# %% functions external to class


def load_param_defaults(filename="model_parameters.json"):
    """
    Written by: Roslyn Hickson
    Pull out default values from a file in json format.
    :param filename: json file containing default parameter values, which can be overridden by user specified values
    :return: loaded expected parameter values
    """
    with open(filename) as json_file:
        json_data = json.load(json_file)
    for key, value in json_data.items():
        json_data[key] = value["exp"]
    return json_data


# %%


if __name__ == "__main__":

    # set up parameter values for the simulations
    flag_use_defaults = True
    flag_simple_plots = True
    num_days_to_run = 100

    cust_params = load_param_defaults()
    if not flag_use_defaults:  # version manually overriding values in json file
        w1 = 8
        R0 = 5
        gamma = 1/7
        sigma = 1/3

        cust_params = load_param_defaults()
        cust_params["transmission"] = R0*gamma
        cust_params["infectious_period"] = 1/gamma
        cust_params["immune_period"] = 240
        cust_params["latent_period"] = 1/sigma  # Turning off demography

        cust_params["a1"] = cust_params["a1"]*gamma
        cust_params["w1"] = cust_params["w1"]*gamma
        cust_params["a2"] = cust_params["a1"]*gamma
        cust_params["w2"] = cust_params["w2"]*gamma
        cust_params["w3"] = cust_params["w3"]*gamma

    cust_params["delta"] = [1, 1, 1, 0]
    # cust_params["pT"] = [0.1, 0.2, 0.3, 0.4, 0.9, 0.9, 0.9]
    cust_params["k"] = 4
    M1 = bad(**cust_params)

    M1.run(t_end=num_days_to_run, t_step=0.1, flag_incidence_tracking=True)

# %% Plots

    if flag_simple_plots:
        M1.plot()
    else:
        # Phase plot examples
        cust_params = load_param_defaults()
        cust_params["k"] = 7

        # Test on day one
        cust_params["delta"] = [1, 0, 0, 0, 0, 0, 0]
        M1 = bad(**cust_params)
        M1.run(t_end=num_days_to_run, t_step=0.1)
        # Test on day four
        cust_params["delta"] = [0, 0, 0, 1, 0, 0, 0]
        M2 = bad(**cust_params)
        M2.run(t_end=num_days_to_run, t_step=0.1)
        # Test on day seven
        cust_params["delta"] = [0, 0, 0, 0, 0, 0, 1]
        M3 = bad(**cust_params)
        M3.run(t_end=num_days_to_run, t_step=0.1)

        plt.figure()
        plt.title("S - I phase plane")
        plt.plot(M1.get_S(), M1.get_all_infectious(),
                 linestyle="-",
                 label="Testing days: 1")
        plt.plot(M2.get_S(), M2.get_all_infectious(),
                 linestyle="--",
                 label="Testing days: 4")
        plt.plot(M3.get_S(), M3.get_all_infectious(),
                 linestyle=":",
                 label="Testing days: 7")
        plt.legend(loc=(0.6, 0.75))
        plt.xlabel("Susceptibles (Sn + Sb)")
        plt.ylabel("Infectious (I + A + T)")
        if flag_save_figs:
            plt.savefig("../img/phasePlane_S_I_oneTests.png",
                        dpi=dpi,
                        bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        plt.figure()
        plt.title("B - T phase plane")
        plt.plot(M1.get_B(), M1.get_T(),
                 linestyle="-",
                 label="Testing days: 1")
        plt.plot(M2.get_B(), M2.get_T(),
                 linestyle="--",
                 label="Testing days: 4")
        plt.plot(M3.get_B(), M3.get_T(),
                 linestyle=":",
                 label="Testing days: 7")
        plt.legend(loc=(0.01, 0.75))
        plt.xlabel("Behaviour (Sb + Eb + Ab + Ib + T + Rb)")
        plt.ylabel("Positive tests (T)")
        if flag_save_figs:
            plt.savefig("../img/phasePlane_B_T_oneTests.png",
                        dpi=dpi,
                        bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        # Test on day one, two, three
        cust_params["delta"] = [1, 1, 1, 0, 0, 0, 0]
        M1 = bad(**cust_params)
        M1.run(t_end=num_days_to_run, t_step=0.1)
        # Test on day one, four, seven
        cust_params["delta"] = [1, 0, 0, 1, 0, 0, 1]
        M2 = bad(**cust_params)
        M2.run(t_end=num_days_to_run, t_step=0.1)
        # Test on day five, six, seven
        cust_params["delta"] = [0, 0, 0, 0, 1, 1, 1]
        M3 = bad(**cust_params)
        M3.run(t_end=num_days_to_run, t_step=0.1)

        plt.figure()
        plt.title("S - I phase plane")
        plt.plot(M1.get_S(), M1.get_all_infectious(),
                 linestyle="-",
                 label="Testing days: 1, 2, 3")
        plt.plot(M2.get_S(), M2.get_all_infectious(),
                 linestyle="--",
                 label="Testing days: 1, 4, 7")
        plt.plot(M3.get_S(), M3.get_all_infectious(),
                 linestyle=":",
                 label="Testing days: 5, 6, 7")
        plt.legend(loc=(0.55, 0.75))
        plt.xlabel("Susceptibles (Sn + Sb)")
        plt.ylabel("Infectious (I + A + T)")
        if flag_save_figs:
            plt.savefig("../img/phasePlane_S_I_threeTests.png",
                        dpi=dpi,
                        bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        plt.figure()
        plt.title("B - T phase plane")
        plt.plot(M1.get_B(), M1.get_T(),
                 linestyle="-",
                 label="Testing days: 1, 2, 3")
        plt.plot(M2.get_B(), M2.get_T(),
                 linestyle="--",
                 label="Testing days: 1, 4, 7")
        plt.plot(M3.get_B(), M3.get_T(),
                 linestyle=":",
                 label="Testing days: 5, 6, 7")
        plt.legend(loc=(0.01, 0.75))
        plt.xlabel("Behaviour (Sb + Eb + Ab + Ib + T + Rb)")
        plt.ylabel("Positive tests (T)")
        if flag_save_figs:
            plt.savefig("../img/phasePlane_B_T_threeTests.png",
                        dpi=dpi,
                        bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        # Test on day one,
        cust_params["delta"] = [1, 0, 0, 0, 0, 0, 0]
        M1 = bad(**cust_params)
        M1.run(t_end=num_days_to_run, t_step=0.1)
        # Test on day one, two
        cust_params["delta"] = [1, 1, 0, 0, 0, 0, 0]
        M2 = bad(**cust_params)
        M2.run(t_end=num_days_to_run, t_step=0.1)
        # Test on day one, two, thee
        cust_params["delta"] = [1, 1, 1, 0, 0, 0, 0]
        M3 = bad(**cust_params)
        M3.run(t_end=num_days_to_run, t_step=0.1)

        plt.figure()
        plt.title("S - I phase plane")
        plt.plot(M1.get_S(), M1.get_all_infectious(),
                 linestyle="-",
                 label="Testing days: 1")
        plt.plot(M2.get_S(), M2.get_all_infectious(),
                 linestyle="--",
                 label="Testing days: 1, 2")
        plt.plot(M3.get_S(), M3.get_all_infectious(),
                 linestyle=":",
                 label="Testing days: 1, 2, 3")
        plt.legend(loc=(0.55, 0.75))
        plt.xlabel("Susceptibles (Sn + Sb)")
        plt.ylabel("Infectious (I + A + T)")
        if flag_save_figs:
            plt.savefig("../img/phasePlane_S_I_differentNumbers.png",
                        dpi=dpi,
                        bbox_inches="tight")
            plt.close()
        else:
            plt.show()

        plt.figure()
        plt.title("B - T phase plane")
        plt.plot(M1.get_B(), M1.get_T(),
                 linestyle="-",
                 label="Testing days: 1")
        plt.plot(M2.get_B(), M2.get_T(),
                 linestyle="--",
                 label="Testing days: 1, 2")
        plt.plot(M3.get_B(), M3.get_T(),
                 linestyle=":",
                 label="Testing days: 1, 2, 3")
        plt.legend(loc=(0.01, 0.75))
        plt.xlabel("Behaviour (Sb + Eb + Ab + Ib + T + Rb)")
        plt.ylabel("Positive tests (T)")
        if flag_save_figs:
            plt.savefig("../img/phasePlane_B_T_differentNumbers.png",
                        dpi=dpi,
                        bbox_inches="tight")
            plt.close()
        else:
            plt.show()

# %% Testing R0

    R0 = 2.0

    cust_params = load_param_defaults()
    cust_params["transmission"] = 1
    cust_params["delta"] = [1, 0, 0, 1, 0, 0, 1]

    M = bad(**cust_params)

    R0_multiplier = M.get_reproduction_number()

    cust_params["transmission"] = R0/R0_multiplier

    M = bad(**cust_params)

    M.run()
    M.plot()

# %% Static vs Dynamic behaviour

    num_days_to_run = 2*365

    gamma = 1/7
    sigma = 1/3
    R0 = 3.24

    cust_params = load_param_defaults()
    cust_params["transmission"] = 1
    cust_params["infectious_period"] = 1/gamma
    cust_params["immune_period"] = 240
    cust_params["latent_period"] = 1/sigma  # Turning off demography

    cust_params["a1"] = cust_params["a1"]*gamma
    cust_params["w1"] = cust_params["w1"]*gamma
    cust_params["a2"] = cust_params["a1"]*gamma
    cust_params["w2"] = cust_params["w2"]*gamma
    cust_params["w3"] = cust_params["w3"]*gamma

    cust_params["delta"] = [1, 0, 0, 1, 0, 1, 0]
    # cust_params["pT"] = [0.1, 0.2, 0.3, 0.4, 0.9, 0.9, 0.9]
    cust_params["k"] = 7

    M_tmp = bad(**cust_params)
    R0_multiplier = M_tmp.get_reproduction_number()

    cust_params["transmission"] = R0 / R0_multiplier

    M_dynamic = bad(**cust_params)
    M_static = bad(**cust_params)

    M_static.w1 = 0
    M_static.w2 = 0
    M_static.w3 = 0
    M_static.a1 = 0
    M_static.a2 = 0

    M_dynamic.run(t_end=num_days_to_run, t_step=0.1)
    M_static.run(t_end=num_days_to_run, t_step=0.1)

    plt.figure()
    plt.title("S - I phase plane")
    plt.plot(M_dynamic.get_S(), M_dynamic.get_all_infectious(),
             linestyle=":", color="grey",
             label="Dynamic behaviour")
    plt.plot(M_static.get_S(), M_static.get_all_infectious(),
             linestyle="-", color="green",
             label="Static behaviour")
    plt.legend(loc=(0.6, 0.75))
    plt.xlabel("Susceptibles (Sn + Sb)")
    plt.ylabel("Infectious (I + A + T)")
    plt.xlim(0, 1)
    plt.ylim(0, 0.5)
    if flag_save_figs:
        plt.savefig("../img/phasePlane_S_O_dyanmic_static.png",
                    dpi=dpi,
                    bbox_inches="tight")
        plt.close()
    else:
        plt.show()
# %%
    params_static = cust_params.copy()

    B_range = np.arange(start=0.2, stop=1, step=0.2)

    M = bad(**params_static)
    M.w1 = 0
    M.w2 = 0
    M.w3 = 0
    M.a1 = 0
    M.a2 = 0

    IC = M.init_cond

    plt.figure()
    for b in B_range:
        IC[M.CC.Sn] = 1 - b - IC[M.CC.In1]
        IC[M.CC.Sb] = b

        M.init_cond = IC
        M.run(t_end=num_days_to_run, t_step=0.1)

        plt.title("S - I phase plane")
        plt.plot(M.get_S(), M.get_all_infectious(),
                 label=f"$B(0)$: {round(b, 1)}")
    plt.plot(M_dynamic.get_S(), M_dynamic.get_all_infectious(),
             linestyle=":", color="grey",
             label="Dynamic behaviour")
    plt.legend(loc=(0.55, 0.5))
    plt.xlabel("Susceptibles (Sn + Sb)")
    plt.ylabel("Infectious (I + A + T)")
    plt.xlim(0, 1)
    plt.ylim(0, 0.5)
    if flag_save_figs:
        plt.savefig("../img/phasePlane_S_O_vary_static.png",
                    dpi=dpi,
                    bbox_inches="tight")
        plt.close()
    else:
        plt.show()
