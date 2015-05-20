from glob import glob
from sys import argv

from lingpy import *
from lingpy.convert.strings import *

for filename in glob("Test set/*.csv"):
    # SCA
    lex = LexStat(filename)
    with open(filename.split('.')[0]+'_sca.matrices', 'w') as f:
        m = lex._get_matrices(method="sca")
        for a,b,c in m: # a is the concept, b is the index array, c is the matrix
            print ("Analyzing concept {0}".format(a))
            # now write stuff to file
            f.write(matrix2dst(c, taxa=[str(x) for x in b], taxlen=0))
            f.write('\n#\n')

    # LexStat
    lex = LexStat(filename)
    lex.get_scorer()
    with open(filename.split('.')[0]+'_lex.matrices', 'w') as f:
        m = lex._get_matrices(method="lexstat")
        for a,b,c in m: # a is the concept, b is the index array, c is the matrix
            print ("Analyzing concept {0}".format(a))
            # now write stuff to file
            f.write(matrix2dst(c, taxa=[str(x) for x in b], taxlen=0))
            f.write('\n#\n')




