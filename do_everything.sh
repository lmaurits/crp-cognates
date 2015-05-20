#!/bin/sh
python3 ./get_matrices.py
python3 ./run_crp.py
python3 ./get_crp_fscores.py
python3 ./get_lingpy_fscores.py
python3 ./comparison.py
