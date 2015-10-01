import os
import pdb
import glob

from lingpy import *
from lingpy.evaluate.acd import *

from fileio import read_data, get_cogids, extend_csv
from mcmc_post import CladeCounter
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

def get_map(ids, matrices):
    clusterer = Clusterer(matrices)
    clusterer.verbose = False
    clusterer.init_partitions()
    clusterer.find_MAP()
    return clusterer.partitions

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

        # Open MCMC output
        fp = open("mcmc_%s_%s.part" % (root, method), "r")
        consensus = []
        for i in range(len(ids)):
            cc = CladeCounter()
            for line in fp:
                partitions = eval(line.strip())
                partition = partitions[i]
                cc.update_counts(partition)
            consensus.append(cc.get_consensus())
            fp.seek(0)

        # Generate CogIDs from consensus partitions
        results = get_cogids(ids, consensus)

        # Extend the corresponding .csv file to include the new
        # CogIDs
        in_filename = os.path.join(dirr, "%s.csv" % root)
        if not os.path.exists("CRP results"):
            os.makedirs("CRP results")
        out_filename = os.path.join("CRP results", "%s_crp_%s.csv" % (root,method))
        extend_csv(in_filename, out_filename, results, "crp_%s" % method)
