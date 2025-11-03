import re
import numpy as np
log10 = lambda X: np.log10(X, 
    out=np.full(X.shape, np.nan), 
    where=X>0)

SYMBOLS = np.array([
    'h',  'he', 'li', 'be', 'b',  'c',  'n',   'o',  'f', 'ne', 
    'na', 'mg', 'al', 'si', 'p',  's',  'cl', 'ar',  'k', 'ca',
    'sc', 'ti', 'v',  'cr', 'mn', 'fe', 'co', 'ni', 'cu', 'zn',
    'ga', 'ge', 'as', 'se', 'br', 'kr', 'rb', 'sr',  'y', 'zr',
    'nb', 'mo', 'tc', 'ru', 'rh', 'pd', 'ag', 'cd', 'in', 'sn', 
    'sb', 'te', 'i',  'xe', 'cs', 'ba', 'la', 'ce', 'pr', 'nd', 
    'pm', 'sm', 'eu', 'gd', 'tb', 'dy', 'ho', 'er', 'tm', 'yb', 
    'lu', 'hf', 'ta', 'w',  're', 'os', 'ir', 'pt', 'au', 'hg', 
    'tl', 'pb', 'bi', 'po', 'at', 'rn', 'fr', 'ra', 'ac', 'th', 
    'pa', 'u'
])
SYMBOLS = np.char.capitalize(SYMBOLS)
ATOMIC_MASS = np.array([
      1.0079,   4.0026,    6.941,   9.0122,   10.811,  12.0107,  14.0067,  15.9994,  18.9984,  20.1797,
     22.9898,  24.305 ,  26.9815,  28.0855,  30.9738,  32.065 ,  35.453 ,  39.948 ,  39.0983,  40.078 ,
     44.9559,  47.867 ,  50.9415,  51.9961,  54.938 ,  55.845 ,  58.9332,  58.6934,  63.546 ,  65.409 ,
     69.723 ,  72.64  ,  74.9216,  78.96  ,  79.904 ,  83.796 ,  85.4678,  87.62  ,  88.9059,  91.224 ,
     92.9064,  95.94  ,   np.nan, 101.07  , 102.9055, 106.42  , 107.8682, 112.411 , 114.818 , 118.71  , 
    121.76  , 127.6   , 126.9045, 131.293 , 132.9055, 137.327 , 138.9055, 140.116 , 140.9077, 144.24  , 
      np.nan, 150.36  , 151.964 , 157.25  , 158.9253, 162.5   , 164.9303, 167.259 , 168.9342, 173.04  , 
    174.967 , 178.49  , 180.9479, 183.84  , 186.207 , 190.23  , 192.217 , 195.078 , 196.9666, 200.59  ,
    204.3833, 207.2   , 208.9804,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan,   np.nan, 232.0381, 
      np.nan, 238.0298
])

def Symbol2Z(symb):
    if isinstance(symb, str):
        Z = SYMBOLS.tolist().index(symb)+1
    elif hasattr(symb, '__iter__'):
        Z = np.array([SYMBOLS.tolist().index(_) for _ in symb])+1
    return Z
def Z2Symbol(Z):
    if isinstance(Z, int):
        symb = SYMBOLS[Z-1]
    elif isinstance(Z, float):
        symb = SYMBOLS[int(np.floor(Z))-1]
    elif hasattr(Z, '__iter__'):
        Z = np.array(np.floor(Z), dtype=int)
        symb = np.array([SYMBOLS[_-1] for _ in Z])
    return symb

class isotope():
    def __init__(self, identifier, type=None) -> None:
        if isinstance(identifier, str):
            iden_rstripI = identifier.rstrip('I')
            if iden_rstripI:
                identifier = iden_rstripI
            else:
                identifier = 'I'
            string_pattern = r'([A-Za-z]{1,2})([0-9]*)'
            re_result = re.match(string_pattern, identifier)
            symb, A = re_result.groups()
            symb = symb.lower().capitalize()
            if symb in SYMBOLS:
                Z = Symbol2Z(symb)
                if A == '':
                    A = None
                else:
                    A = int(A)
                    if A < Z:
                        raise ValueError('Mass Number should be larger than Atomic Number.')
                mass = ATOMIC_MASS[Z-1]
            else:
                raise ValueError('symb cannot be identified. Note that element beyond U is not allowed!')
        elif isinstance(identifier, int):
            Z = identifier
            if Z <= 92:
                symb = Z2Symbol(Z)
                self.symb = symb
                A = None
                mass = ATOMIC_MASS[Z-1]
            else:
                raise ValueError('Element beyond U is not allowed!')
        self.symb = symb
        self.Z = Z
        self.A = A
        self.mass =mass
        if type is not None:
            self.type = type
        else:
            if self.A is None:
                self.type = 'Element'
            else:
                self.type = 'Isotope'
        if self.type == 'Element':
            self.identifier = r'Element %s'%(self.symb)
        elif self.type == 'Isotope':
            self.identifier = r'Isotope %s%d'%(self.symb, self.A)
    
    def __repr__(self):
        return self.identifier
    
    def __del__(self):
        pass