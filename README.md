# crp-cognates

This repository contains rough and ready proof-of-concept code for making
cognacy judgements from pairwise phonetic distance matrices using a Bayesian
mixture model based on the Chinese Restaurant Process.  It compares the
CRP method as well as the deterministic methods implmented in LingPy against
some Gold Standard judgements (which must be provided separately).

# Dependencies

* Python 3 (http://www.python.org)
* LingPy (http://lingpy.org/)
* SciPy (http://www.scipy.org/scipylib/index.html)
* Pandas (http://pandas.pydata.org/)

# Prerequisites

The scripts in this repository expect a directory named "Test set" to exist
in the top level of the working copy, and this directory should contain a
list of LingPy .csv files which include a Gold Standard cognacy judgement
column named "CogID".

# Scripts

## get_matrices.py

This will parse all .csv files in "Test set" and create two .matrices files
for each, one using the SCA method and one using the LexStat method.

## run_crp.py

This will run the CRP cognate judgement method on all .matrices files in the
"Test set" directory.  For each matrix file, a LingPy .csv file will be
created in the directory "CRP results" which contains the estimated cognate
IDs.

## get_crp_fscores.py

This will create the "crp_fscores.csv" file, which will contain the F-score
for the CRP SCA and CRP LexStat methods for all data files in "Test set".  It
uses the contents of the "CRP results" directory and as such MUST ONLY BE RUN
AFTER run_crp.py has run.

## get_lingpy_fscores.py

This will create the "lingpy_fscores.csv" file, which will contain the
F-score for all LingPy methods for all data files in "Test set".

## comparison.py

This will:
(i)  combine crp_fscores.csv and lingpy_fscores.csv into a single
    "combined_fscores.csv" file.

(ii) print a report comparing the performance of the CRP methods to the LingPy
     methods.

## do_everything.sh

This shell script will execute all of the above scripts in an appropriate
order.  After some time you'll see the performance report.

## update_crp.sh

This shell script will run run_crp.py, get_crp_fscores.py and comparison.py.
It should be run after a change is made to crpclusterer.py to get a newly
valid performance report, without re-running the LingPy methods (which must
have already been run once).

# Code

## crpclusterer.py

This defines the Cluster class which does all of the work in run_crp.py.

## fileio.py

This defines a small number of functions which are used in the above scripts
to transform data between formats which work nicely with the Cluster class and
formats which work nicely with LingPy.
