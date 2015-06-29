import glob
import itertools
import os
import pdb
import random
import sys

from lingpy import *
from lingpy.evaluate.acd import *
import numpy as np

from fileio import read_data, get_cogids, extend_csv, read_gold_partitions
from crpclusterer import Clusterer as Clusterer
from crpsimulator import CrpSimulator as Simulator

def main():
    if len(sys.argv) == 1 or sys.argv[1].lower() == "mcmc":
        mode = "mcmc"
    elif sys.argv[1].lower() == "map":
        mode = "map"
    else:
        print("what?")
        sys.exit

    if mode == "mcmc":
        filenames = ("mcmc_theta.csv",
                "mcmc_w_mu.csv",
                "mcmc_w_sigma.csv",
                "mcmc_b_mu.csv",
                "mcmc_b_sigma.csv",
                "mcmc_full_splits.csv")
    else:
        filenames = ("map_theta.csv",
                "map_w_mu.csv",
                "map_w_sigma.csv",
                "map_b_mu.csv",
                "map_b_sigma.csv",
                "map_full_splits.csv")

    fps = [open(fname,"w") for fname in filenames]
    for i in range(0,30):
        # Simulate data from random parameters
        truth, parts, mats = do_sim()

        # Initialise a CrpClusterer with that simulated data
        clusterer = Clusterer(mats)
        clusterer.init_partitions(method="thresh")
        clusterer.change_params = True
        clusterer.change_params = True
        clusterer.verbose = False

        # Estimate the parameter values,  either...
        # ...as the medians of MCMC distributions
        if mode == "mcmc":
            estimates = do_mcmc(clusterer)
        # or as MAP parameter values
        elif mode == "map":
            estimates = do_map(clusterer)

        # Save the true and estimated parameters side-by-side
        for fp, (t,m) in zip(fps, zip(truth, estimates)):
            fp.write("%f, %f\n" % (t, m))
    [fp.close() for fp in fps]

def do_sim(meanings=110, words=7):

    # Initialise simulator with MAP partitions
    truth = []
    truth.append(random.random()*3)
    truth.append(random.random()*0.4)
    truth.append(random.random()*0.2)
    truth.append(0.6 + random.random()*0.4)
    truth.append(random.random()*0.2)
    simulator = Simulator(*truth)
    mats = []
    parts = []
    for i in range(0,meanings):
        part, matrix = simulator.simulate_datapoint(words)
        parts.append(part)
        mats.append(matrix)
    return truth, parts, mats

def do_map(clusterer):
    clusterer.find_MAP(5000)
    return (clusterer.theta,
            clusterer.within_mu,
            clusterer.within_sigma,
            clusterer.between_mu,
            clusterer.between_sigma,
            )

def do_mcmc(clusterer):
    samples = list(clusterer.sample_posterior(50000,5000,500,filename="mcmc.log"))
    medians = []
    for i in range(1, 6):
        values = [s[i] for s in samples]
        values.sort()
        medians.append(values[int(0.5*len(values))])
    return medians

if __name__ == "__main__":
    main()
