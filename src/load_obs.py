import os
import numpy as np
import pandas as pd
from .sol_ref import SolRef
from .abu_derive import CurveOfGrowth
from pathlib import Path
OBS_DIR = Path(os.path.abspath(__file__)).parent.parent/'obs'

class Obs:
    """
    A class to handle observational data, including loading, corrections, and filtering.

    Attributes:
    - obsref: str
        Reference name for the observational data.
    - logeps: pd.DataFrame
        Logarithmic abundances for elements.
    - elogeps: pd.DataFrame
        Uncertainties in logarithmic abundances.
    """

    def __init__(
        self, obsref, kind=['raw', 'lte', 'nlte'], overwrite=False, 
        sol_abu=None, cogref=None, silent=False
    ):
        """
        Initialize the Obs class.

        Parameters:
        - obsref: str
            Reference name for the observational data.
        - overwrite: bool, optional
            Whether to overwrite existing NLTE/LTE corrections. Default is False.
        - default_select: bool, optional
            Whether to apply default selection to the observational data. Default is True.
        - user_exclusions: list, optional
            List of user-specified exclusions for observational data.
        - sol_abu: SolRef, optional
            Solar abundance reference data.
        - cogref: str, optional
            Reference name for NLTE grids. Default is 'lind2022'.
        - silent: bool, optional
            If True, suppresses output messages. Default is False.
        """
        self.obsref = obsref  # Store obsref as an attribute
        self.silent = silent
        self.sol_abu = sol_abu or SolRef(silent=silent)  # Use provided SolRef or create a default one
        if cogref is None:
            cogref = 'lind2022'
        self.cogref = cogref
        if self.cogref is not None:
            self.cog = CurveOfGrowth(self.cogref)
        if isinstance(kind, str):
            kind = [kind]
        for k_ in kind:
            self.load_observation(k_, overwrite)

    def load_observation(self, kind, overwrite):
        """
        Load observational data and apply corrections if necessary.

        Parameters:
        - kind: str
            Type of observational data to load (e.g., 'nlte', 'lte').
        - overwrite: bool
            Whether to overwrite existing NLTE/LTE corrections.
        """
        # Check if obsref is a list
        if isinstance(self.obsref, list):
            combined_data = []
            for ref in self.obsref:
                ref_data = self._load_single_observation(ref, kind, overwrite)
                combined_data.append(ref_data)
            df_kdat = pd.concat(combined_data, ignore_index=True)
            if not self.silent:
                print('Loaded and combined observations from multiple references')
        else:
            # Single obsref case
            df_kdat = self._load_single_observation(self.obsref, kind, overwrite)
        
        # Quick Access to Data
        logeps = df_kdat.loc[:, 'A(X)'].copy()
        logeps.dropna(axis='columns', how='all', inplace=True)
        elogeps = df_kdat.loc[:, 'eA(X)'].copy()
        elogeps = elogeps.loc[:, logeps.columns]
        exec(f'self.{kind} = AbuPreview(logeps=logeps, elogeps=elogeps)')

    def _load_single_observation(self, obsref, kind, overwrite):
        # Calculate abundances from a new CoG
        # Load data based on KIND
        obs_dir = OBS_DIR / obsref
        data_files = list(obs_dir.glob(f'{kind}.csv'))
        if kind == 'raw':
            if data_files == []:
                raise FileNotFoundError(f'No {kind.upper()} data file found for {obs_dir}')
            else:
                data = pd.read_csv(data_files[0], header=[0, 1])
                if data.columns[0] in ['Unnamed: 0', ('Unnamed: 0_level_0', 'Unnamed: 0_level_1')]:
                    data = pd.read_csv(data_files[0], header=[0, 1], index_col=0)
        else:
            if (data_files == []) or overwrite:
                raw_files = list(obs_dir.glob('raw.csv'))
                if raw_files == []:
                    raise FileNotFoundError(f'No data file found for raw data. Please put "raw.csv" in {obs_dir}.')
                # corr_files = [] if overwrite else list(obs_dir.glob(f'{kind}.csv'))
                # if corr_files == []:
                #     # Load LTE data and apply NLTE correction if NLTE is enabled
                raw_file = raw_files[0]  # Use the first matching file
                raw_data = pd.read_csv(raw_file, header=[0, 1])
                if raw_data.columns[0] in ['Unnamed: 0', ('Unnamed: 0_level_0', 'Unnamed: 0_level_1')]:
                    raw_data = pd.read_csv(raw_file, header=[0, 1], index_col=0)
                cog = self.cog
                cog.get_abundance_from_table(raw_data, atm=kind, save_dir=obs_dir)
                if not self.silent:
                    print(f'No {self.cogref.upper()} {kind.upper()} file is found or "OVERWRITE" is True.')
                    print(f'{self.cogref.upper()} {kind.upper()} correction applied to raw data for {obsref}')
                data_files = list(obs_dir.glob(f'{kind}.csv'))
            data = pd.read_csv(data_files[0], header=[0, 1])
            if data.columns[0] in ['Unnamed: 0', ('Unnamed: 0_level_0', 'Unnamed: 0_level_1')]:
                data = pd.read_csv(data_files[0], header=[0, 1], index_col=0)
        return data


    def _apply_default_selection(self):
        """
        Apply default selection filters to the observational data.
        Filters include EMP, VMP, CEMP, and binary star criteria.
        """
        try:
            # Initialize filters
            emp_filter = vmp_filter = None
            cemp = cemp_s = cemp_rs = binary = None
            
            # FeH_sol filter
            if 'Fe' in self.logeps.columns:
                Fe_logeps = self.logeps['Fe'].values
                FeH_sol = Fe_logeps - self.sol_abu('Fe')
                emp_filter = FeH_sol <= -3.
                vmp_filter = FeH_sol <= -2.
            else:
                if not self.silent:
                    print('Warning! Skipping filters relative to [Fe/H]: Missing column Fe')

            # CFe_sol filter
            if emp_filter is not None:
                if 'C' in self.logeps.columns:
                    C_logeps = self.logeps['C'].values
                    CFe_sol = (C_logeps - self.sol_abu('C')) - FeH_sol
                    cemp = vmp_filter & (CFe_sol >= 1)
                else:
                    if not self.silent:
                        print('Warning! Skipping filters relative to CEMP: Missing column C')

            # BaFe_sol filter
            if cemp is not None:
                if 'Ba' in self.logeps.columns:
                    Ba_logeps = self.logeps['Ba'].values
                    BaFe_sol = (Ba_logeps - self.sol_abu('Ba')) - FeH_sol
                    cemp_rs = (BaFe_sol >= 0.) & (BaFe_sol <= .5) & cemp
                else:
                    if not self.silent:
                        print('Warning! Skipping CEMP-s and CEMP-r/s filters: Missing column Ba')

            # EuFe_sol filter
            if cemp_rs is not None:
                if 'Eu' in self.logeps.columns:
                    Eu_logeps = self.logeps['Eu'].values
                    EuFe_sol = (Eu_logeps - self.sol_abu('Eu')) - FeH_sol
                    cemp_s = (BaFe_sol > 1.) & ((BaFe_sol - EuFe_sol) > .5) & cemp
                else:
                    if not self.silent:
                        print('Warning! Skipping CEMP-r/s filter: Missing column Eu')

            # Binary filter
            if 'RUWE' in self.logeps.columns:
                ruwe = self.logeps['RUWE'].values
                binary = ruwe >= 1.4
            else:
                if not self.silent:
                    print('Warning! Skipping binary filter: Missing column RUWE')

            # Combine filters
            exclusion_filter = np.logical_or.reduce([f for f in [cemp_s, cemp_rs, binary] if f is not None])
            if exclusion_filter is not np.False_:
                combined_filter = np.logical_and(emp_filter, ~exclusion_filter)
            elif emp_filter is not None:
                combined_filter = emp_filter
            else:
                combined_filter = None
            # Apply combined filter
            if combined_filter is not None:
                self.ew_tab = self.ew_tab.loc[combined_filter, :]
                self.logeps = self.logeps.loc[combined_filter, :]
                self.elogeps = self.elogeps.loc[combined_filter, :]
            if not self.silent:
                print('Applied default selection to observations.')
        except KeyError as e:
            print(f'ERROR during default selection: Missing column {str(e)}')

    def _apply_user_exclusions(self):
        """
        Apply user-specified exclusions to the observational data.
        """
        if self.user_exclusions is not None:
            try:
                object_names = self.ew_tab.loc[:, ('info', 'name')].values
                user_exclusion_filter = ~np.isin(object_names, self.user_exclusions)
                self.ew_tab = self.ew_tab.loc[user_exclusion_filter, :]
                if not self.silent:
                    print('Applied user-specified exclusions to observations.')
            except KeyError as e:
                print(f'ERROR during user exclusions: Missing column {str(e)}')
            except Exception as e:
                print(f'Unexpected error during user exclusions: {str(e)}')

class AbuPreview:
    def __init__(self, logeps, elogeps):
        self.logeps = logeps
        self.elogeps = elogeps