import os
import numpy as np
from .utils.isotope import isotope, SYMBOLS
from pathlib import Path
CURRENT_DIR = Path(os.path.abspath(__file__)).parent

class SolRef:
    '''Load observational data with optional NLTE correction'''
    def __init__(
        self, solref='As09', silent=False
    ):
        self.silent = silent
        self.load_solref(solref)
    
    def __call__(self, species):
        if hasattr(species, '__iter__') and not isinstance(species, str):
            return [self.__call__(spec_) for spec_ in species]
        else:
            if not isinstance(species, isotope):
                species = isotope(species)
            return self.logeps[species.Z-1]

    def load_solref(self, solref):
        path = CURRENT_DIR / (f'data/sol/{solref.upper()}.dat')
        sol_logeps = np.full(len(SYMBOLS), np.nan)
        with open(path, 'r') as rfile:
            lines = rfile.read().splitlines()
        for line in lines:
            linesplt = line.split()
            if len(linesplt) == 2:
                iso, logeps = linesplt
                iso = isotope(iso)
                sol_logeps[iso.Z-1] = logeps
        sol_logeps += (-sol_logeps[0]+12)
        self.logeps = sol_logeps
        self.reference = solref
        if not self.silent:
            print(f'Loaded solar reference data from {path}')