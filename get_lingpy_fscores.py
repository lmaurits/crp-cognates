import glob
import os.path
from lingpy import *
from lingpy.evaluate.acd import *
import pandas as pd

# Our Pandas Dataframe will be built from these two lists
results = []
names = []

data_files = glob.glob("./Test set/*.csv")
for method in ("edit-dist", "lexstat", "sca", "turchin"):
    if method in ("edit-dist","lexstat","sca"):
        # Run parameterised methods with ten different values
        for i in range(1,11):
            t = i/10.0
            name = method + ("-%d" % i)
            names.append(name)
            r = {}
            for filename in data_files:
                dataset = os.path.basename(filename).rsplit(".",1)[0]
                lex = LexStat(filename)
                if method == "lexstat":
                    lex.get_scorer()
                lex.cluster(method=method, threshold=t, ref=name+"_id")
                res = bcubes(lex, 'cogid', name+"_id")[2]
                r[dataset] = res
            results.append(r)
    else:
        # Run un-parameterised methods once only
        names.append(method)
        r = {}
        for filename in data_files:
            dataset = os.path.basename(filename).rsplit(".",1)[0]
            lex = LexStat(filename)
            lex.cluster(method=method, ref=method+"_id")
            res = bcubes(lex, 'cogid', method+"_id")[2]
            r[dataset] = res
        results.append(r)

df = pd.DataFrame(results,index=names)
df.to_csv("lingpy_fscores.csv")
