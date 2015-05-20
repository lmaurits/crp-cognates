import os
import glob

from lingpy import *
from lingpy.evaluate.acd import *

from fileio import read_data, get_cogids, extend_csv
from crpclusterer import Clusterer

data_files = glob.glob("./Test set/*.csv")

for method in ("sca", "lex"):
    for filename in data_files:
        # Derive matrix filename from .csv file and method
        dirr = os.path.dirname(filename)
        base = os.path.basename(filename)
        root = base.rsplit(".",1)[0]
        matrix_filename = os.path.join(dirr, "%s_%s.matrices" % (root, method))

        # Read matrix data
        ids, matrices = read_data(matrix_filename)

        # Find MAP partitions
        clusterer = Clusterer(matrices)
        clusterer.init_partitions()
        clusterer.find_MAP()

        # Generate CogIDs from MAP partitions
        results = get_cogids(ids, clusterer.partitions)

        # Extend the corresponding .csv file to include the new
        # CogIDs
        in_filename = os.path.join(dirr, "%s.csv" % root)
        if not os.path.exists("CRP results"):
            os.makedirs("CRP results")
        out_filename = os.path.join("CRP results", "%s_crp_%s.csv" % (root,method))
        extend_csv(in_filename, out_filename, results, "crp_%s" % method)
