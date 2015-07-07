import glob
import itertools
import math
import os
import pdb
import random
import sys

import scipy.stats

from lingpy import *
from lingpy.evaluate.acd import *
import numpy as np

try:
    import piiprogressbar as pb
    progbar = True
except ImportError:
    progbar = False

from fileio import read_data, get_cogids, extend_csv, read_gold_partitions
from crpclusterer import Clusterer as Clusterer
from crpsimulator import CrpSimulator as Simulator
#
def main():
    if len(sys.argv) == 1 or sys.argv[1].lower() == "mcmc":
        mode = "mcmc"
    elif sys.argv[1].lower() == "map":
        mode = "map"
    else:
        print("what?")
        sys.exit
    do_test(mode, 10, 2500)

def do_test(mode, sims, iters):

    if progbar:
        pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar()], maxval=sims*iters).start()
        def pbar_hook():
            pbar.update(pbar.currval+1)
    else:
        def pbar_hook():
            return

    if mode == "mcmc":
        filenames = ("mcmc_theta.csv",
                "mcmc_w_mu.csv",
                "mcmc_w_sigma.csv",
                "mcmc_b_mu.csv",
                "mcmc_b_sigma.csv",
                "mcmc_full_splits.csv")

        hpd_hits = 0
        hpd_tries = 0
    else:
        filenames = ("map_theta.csv",
                "map_w_mu.csv",
                "map_w_sigma.csv",
                "map_b_mu.csv",
                "map_b_sigma.csv",
                "map_full_splits.csv")

    fps = [open(fname,"w") for fname in filenames]
    for i in range(0,sims):
        # Simulate data from random parameters
        truth, parts, mats = do_sim()

        # Initialise a CrpClusterer with that simulated data
        clusterer = Clusterer(mats)
        clusterer.init_partitions(method="thresh")
        clusterer.verbose = False

        # Estimate the parameter values,  either...
        # ...as the medians of MCMC distributions
        if mode == "mcmc":
#            clusterer.theta = truth[0]
            estimates, hpds = do_mcmc(clusterer, "mcmc_%02d.log" % i, iters)
            for t, (lower, upper) in zip(truth, hpds):
                hpd_tries += 1
                print(lower, t, upper, ": OKAY" if lower <= t <= upper else ": FAILED!")
                if lower <= t <= upper:
                    hpd_hits += 1
        # or as MAP parameter values
        elif mode == "map":
            estimates = do_map(clusterer, iters, pbar_hook)

        # Save the true and estimated parameters side-by-side
        for fp, (t,m) in zip(fps, zip(truth, estimates)):
            fp.write("%f, %f\n" % (t, m))
    [fp.close() for fp in fps]
    if progbar:
        print("")

    if mode == "mcmc":
        # Print HDP statistics
        print("Estimated a total of %d parameter HPDs." % hpd_tries)
        print("Of these %.2f%% contained the true value." % (100.0 * hpd_hits / hpd_tries))

def do_sim(meanings=110, words=7):

    truth = []
#    truth.append(scipy.stats.gamma(1.2128, loc=0.0, scale=1.0315).rvs(1)[0])
    truth.append(0.5 + random.random()*0.5)
#    truth.append(scipy.stats.beta(2,5).rvs(1)[0])
    truth.append(random.random()*0.4)
    truth.append(0.05+random.random()*0.2)
#    truth.append(scipy.stats.beta(5,2).rvs(1)[0])
    truth.append(0.6 + random.random()*0.4)
    truth.append(0.05+random.random()*0.2)
    print("Truth:", truth)
    simulator = Simulator(*truth)
    mats = []
    parts = []
    for i in range(0,meanings):
        part, matrix = simulator.simulate_datapoint(words)
        parts.append(part)
        mats.append(matrix)
    return truth, parts, mats

def do_map(clusterer, iters, pbar_hook):
#    set_good_starting_point(clusterer)
    clusterer.change_parameters = False
    clusterer.change_partitions = True
    clusterer.find_MAP(1000, pbar_hook)
    clusterer.change_parameters = True
    clusterer.change_partitions = False
    clusterer.find_MAP(1000, pbar_hook)
    clusterer.change_parameters = True
    clusterer.change_partitions = True
    clusterer.find_MAP(3000, pbar_hook)
    return (clusterer.max_theta,
            clusterer.max_within_mu,
            clusterer.max_within_sigma,
            clusterer.max_between_mu,
            clusterer.max_between_sigma,
            )

def do_mcmc(clusterer, filename, iters):
#    set_good_starting_point(clusterer)
    samples = list(clusterer.sample_posterior(iters,int(0.1*iters),100,filename=filename))
    estimates = []
    hpds = []
    for i in range(1, 6):
        values = [s[i] for s in samples]
        values.sort()
        estimates.append(values[int(0.5*len(values))])
        hpds.append((values[int(0.05*len(values))], values[int(0.95*len(values))]))
    print("Estimates: ", estimates)
    for op in sorted(clusterer.op_norms):
        print("%s: %f" % (op, clusterer.op_hits.get(op,0)/clusterer.op_norms[op]))
    return estimates, hpds

def set_good_starting_point(clusterer):
    # Initialise theta
    max_theta = 0
    max_lh = -999999999999
    for t in [x/10.0 for x in range(1,51)]:
        clusterer.theta = t
        clusterer.dirty_theta = True
        lh = clusterer.get_partition_lh()
        if lh > max_lh:
            max_lh = lh
            max_theta = t
    clusterer.theta = max_theta
    # Initialise mu
    w_sum = 0
    w_min = 100
    w_max = 0
    w_norm = 0
    b_sum = 0
    b_min = 100
    b_max = 0
    b_norm = 0
    for part, mat in zip(clusterer.partitions, clusterer.matrices):
        for x,y in itertools.combinations(range(0,len(mat)),2):
            if x == y:
                continue
            if any([x in bit and y in bit for bit in part]):
                w_sum += mat[x][y]
                w_norm += 1
                if mat[x][y] > w_max:
                    w_max = mat[x][y]
                elif mat[x][y] < w_min:
                    w_min = mat[x][y]
            else:
                b_sum += mat[x][y]
                b_norm += 1
                if mat[x][y] > b_max:
                    b_max = mat[x][y]
                elif mat[x][y] < b_min:
                    b_min = mat[x][y]
    clusterer.within_mu = w_sum / w_norm
    clusterer.within_sigma = abs(w_max - clusterer.within_mu)/2.0
    clusterer.between_mu = b_sum / b_norm
    clusterer.between_sigma = abs(b_max - clusterer.between_mu)/2.0

if __name__ == "__main__":
    main()
