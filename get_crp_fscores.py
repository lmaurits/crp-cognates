import glob
import os.path
from lingpy import *
from lingpy.evaluate.acd import *
import pandas as pd

# Our Pandas Dataframe will be built from these two lists
results = []
names = []

# Now compute F-scores for the already run CRP methods
for glob_pattern in ("./CRP results/*_crp_sca.csv", "./CRP results/*_crp_lex.csv"):
    r = {}
    for filename in glob.glob(glob_pattern):
        dataset, method = os.path.basename(filename).rsplit(".",1)[0].split("_",1)
        lex = LexStat(filename)
        res = bcubes(lex, 'cogid', method)[2]
        r[dataset] = res 
    names.append(method)
    results.append(r)

df = pd.DataFrame(results,index=names)
df.to_csv("crp_fscores.csv")
