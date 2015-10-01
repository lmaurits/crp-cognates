import glob
import os
import sys

from lingpy import *
from lingpy.evaluate.acd import *

from fileio import read_data, get_cogids, extend_csv
from crpclusterer import Clusterer as Clusterer

def get_mcmc_consensus(ids, matrices, filename):
    # Sample!
    clusterer = Clusterer(matrices)
    clusterer.verbose = False
    clusterer.init_partitions()
    partitions = []
    for n, sample in enumerate(clusterer.sample_posterior(200,100,10, filename)):
        print("\r%d" %n,end="")
        partitions.append(sample[-1])
    consensus = []
    for i in range(len(ids)):
        cc = CladeCounter()
        for part in [p[i] for p in partitions]:
            cc.update_counts(part)
        consensus.append(cc.get_consensus())
    assert len(ids) == len(consensus)
    return consensus

if len(sys.argv) > 1:
    data_files = (sys.argv[1],)
else:
    data_files = glob.glob("./Test set/*.csv")

for method in ("sca", "lex"):
    for filename in data_files:
        print(method, filename)
        # Derive matrix filename from .csv file and method
        dirr = os.path.dirname(filename)
        base = os.path.basename(filename)
        root = base.rsplit(".",1)[0]
        matrix_filename = os.path.join(dirr, "%s_%s.matrices" % (root, method))

        # Read matrix data
        ids, matrices = read_data(matrix_filename)

        log_filename = "mcmc_%s_%s.log" % (root, method)
        fp = open("mcmc_%s_%s.part" % (root, method), "w")
        # Sample!
        clusterer = Clusterer(matrices)
        clusterer.verbose = False
        clusterer.init_partitions()
        for sample in clusterer.sample_posterior(200,100,10, log_filename):
            partitions = sample[-1]
            fp.write(str(partitions)+"\n")
        fp.close()
