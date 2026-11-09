"""Fit Equation 4 and its uncertainty from the counts and ratings."""
from paths import DATA, DATA_ROOT, OUTPUT, FIGURES
import argparse, sys
from pathlib import Path
import formula_lib as L

def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['cv', 'fit'])
    p.add_argument('--outcome', choices=['G_rmask', 'G_tests'], default='G_rmask')
    p.add_argument('--labels', choices=['corrected', 'original'], default='corrected')
    p.add_argument('--design', choices=['new_repositories', 'new_conditions'])
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--output', type=Path, default=OUTPUT / 'formula')
    a = p.parse_args()
    L.DATA = a.data
    L.OUT = str(a.output)
    L.LABELS = a.labels
    L.known_coding()
    if a.mode == 'cv':
        import formula_cv
        sys.argv = ['formula_cv', a.outcome] + ([a.design] if a.design else [])
        formula_cv.main()
    else:
        import formula_final
        sys.argv = ['formula_final', a.outcome, str(a.bootstrap), 'category']
        formula_final.main()
if __name__ == '__main__':
    main()
