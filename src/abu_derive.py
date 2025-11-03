import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from pathlib import Path
CURRENT_DIR = Path(os.path.abspath(__file__)).parent
elem_corr_arr = np.array(['Na', 'Mg', 'Al'])

class CurveOfGrowth:
    """
    A class to handle curve-of-growth models for abundance calculations.

    Attributes:
    - element_cog_dict: dict
        Dictionary containing curve-of-growth data for different elements.
    """

    def __init__(self, cogref='lind2022', silent=False):
        """
        Initialize the CurveOfGrowth class.

        Parameters:
        - cogref: str, optional
            Reference name for the curve-of-growth model. Default is 'lind2022'.
        - silent: bool, optional
            If True, suppresses output messages. Default is False.
        """
        self.silent = silent
        self.load_cogmodel(cogref)

    def load_cogmodel(self, cogref):
        """
        Load the curve-of-growth model data.

        Parameters:
        - cogref: str
            Reference name for the curve-of-growth model.
        """
        try:
            # Assuming the model data is stored as a CSV file for each element
            cog_dir = CURRENT_DIR / f'data/cog/{cogref.capitalize()}'
            element_cog_dict = {}
            for elem in elem_corr_arr:  # Replace with actual element list
                file_path = cog_dir / f'{elem.lower()}.h5'
                element_cog_dict[elem] = pd.read_hdf(file_path)
            self.element_cog_dict = element_cog_dict
        except Exception as e:
            raise ValueError(f'Failed to load model data from {cog_dir}: {e}')

    def get_abundance_from_line(
            self, elem, line, 
            teff, logg, feh, vt, 
            ew, atm,
            eteff=100, elogg=0.2, efeh=0.0, evt=0.2
        ):
        """
        Compute abundance using the curve-of-growth model for a single line.

        Parameters:
        - elem: str
            Element symbol (e.g., 'Fe').
        - line: int, str, or float
            Line identifier.
        - teff: float
            Effective temperature.
        - logg: float
            Surface gravity.
        - feh: float
            Metallicity [Fe/H].
        - vt: float
            Microturbulence velocity.
        - ew: float
            Equivalent width.
        - atm: str
            Atmosphere type ('lte' or 'nlte').

        Returns:
        - list: [abundance, uncertainty]
        """
        if isinstance(line, str):
            line = int(line)  # Convert string to integer
        elif isinstance(line, float):
            line = int(round(line))  # Round float to nearest integer
        line = int(line)  # Ensure line is an integer
        indep_vars = ['Teff', 'logg', '[Fe/H]', 'Vturb', 'EquivWidth']
        atm_vars = ['Teff', 'logg', '[Fe/H]', 'Vturb']
        sigma_atm_vars = np.array([eteff, elogg, efeh, evt])

        obs = pd.Series(
            [teff, logg, feh, vt, ew],
            index=indep_vars
        )
        if np.any(pd.isna(obs)):
            return [np.nan, np.nan]
        curvgrow_elem = self.element_cog_dict[elem]
        if line not in curvgrow_elem['obs']:
            return [np.nan, np.nan]
        match atm:
            case 'nlte':
                cog_column = 'logWN'
            case 'lte':
                cog_column = 'logWL'
        curvgrow = curvgrow_elem[curvgrow_elem['obs'] == line]
        obs['logEW'] = np.log10(obs['EquivWidth'])
        if len(curvgrow.loc[:, atm_vars]) == 0:
            print(line)
            print(atm_vars)
            print(obs)
        distance = np.sum(np.abs(curvgrow.loc[:, atm_vars] - obs[atm_vars]), axis=1)
        closest_model = curvgrow.loc[distance.idxmin(), atm_vars]

        logeps = pd.DataFrame(
            columns=atm_vars,
            index=[
                'Param_cnt', 'logeps_cnt',
                'Param_var', 'interp_y'
            ]
        )
        for var_interp in atm_vars:
            fill_value = 'extrapolate'
            vars_fixed = atm_vars.copy()
            vars_fixed.remove(var_interp)
            mask_fixed_vars = np.full(len(curvgrow), True)
            for f_var in vars_fixed:
                mask_fixed_vars &= (curvgrow.loc[:, f_var] == closest_model[f_var])
            curvgrow_fixed_vars = curvgrow.loc[mask_fixed_vars]
            interp_x = np.unique(curvgrow_fixed_vars[var_interp])
            interp_y = np.full_like(interp_x, np.nan, dtype=float)
            for idx, interp_xi in enumerate(interp_x):
                curvgrow_interp = curvgrow_fixed_vars[curvgrow_fixed_vars[var_interp] == interp_xi]
                if curvgrow_interp[cog_column].shape[0] < 2:
                    continue
                else:
                    interp_yi = interp1d(
                        curvgrow_interp[cog_column],
                        curvgrow_interp['logeps'],
                        fill_value=fill_value
                    )(obs['logEW'])
                interp_y[idx] = interp_yi
            mask_na = np.isnan(interp_y)
            interp_x = interp_x[~mask_na]
            interp_y = interp_y[~mask_na]
            if len(interp_y) > 1:
                logeps_atmcnt, logeps_atmvar = interp1d(
                    interp_x, interp_y,
                    fill_value=fill_value
                )([closest_model[var_interp], obs[var_interp]])
            else:
                logeps_atmcnt, logeps_atmvar = np.nan, np.nan
            logeps.loc[:, var_interp] = [
                closest_model[var_interp], logeps_atmcnt,
                obs[var_interp], logeps_atmvar
            ]
        d_logeps = logeps.loc['interp_y'] - logeps.loc['logeps_cnt']
        d_param = (logeps.loc['Param_var'] - logeps.loc['Param_cnt']).values
        logeps_sigma = np.divide(
            d_logeps, d_param,
            out=np.zeros_like(d_param), where=(d_param != 0)
        ) * sigma_atm_vars
        logeps_sigma = np.linalg.norm(logeps_sigma)
        logeps_mean = np.nansum(d_logeps) + np.nanmean(logeps.loc['logeps_cnt', :])
        return [np.round(logeps_mean, 2), np.round(logeps_sigma, 2)]

    def get_abundance_from_table(self, table, atm, save_dir=None):
        """
        Compute abundances for elements based on equivalent widths (EWs) using the curve-of-growth model.

        Parameters:
        - table: pd.DataFrame
            Input table containing equivalent widths (EWs) for different elements and lines.
            Each row corresponds to a star, and columns should include 'EW_{element}' for each element.
        - atm: str
            Atmosphere type ('lte' or 'nlte').
        - save_dir: Path or None, optional
            Directory to save the output tables ('nlte.csv' and 'detail.csv'). If None, no files are saved.
        """
        elem_obs  = list(table['A(X)'].columns)
        elem_corr = elem_corr_arr[np.isin(elem_corr_arr, elem_obs)]
        table_new = table.copy()
        colname_stats = pd.MultiIndex.from_product([elem_corr, ['ave', 'std', 'err']]).to_list()
        colname_logeps = []
        colname_elogeps = []
        for elem in elem_corr:
            colname_logeps += pd.MultiIndex.from_product(
                [[f'A({elem})'], table[f'EW_{elem}'].columns.to_list()]).to_list()
            colname_elogeps += pd.MultiIndex.from_product(
                [[f'eA({elem})'], table[f'EW_{elem}'].columns.to_list()]).to_list()
        table_detail = pd.DataFrame(
            index=table.index, 
            columns=pd.MultiIndex.from_tuples(colname_stats+colname_logeps+colname_elogeps), 
            dtype=float
        )
        for idx, row in table.iterrows():
            teff, logg, vt, feh = row.atmosphere[['Teff', 'logg', 'vt', '[Fe/H]']]
            if 'eTeff' in row.atmosphere:
                eteff = row.atmosphere['eTeff']
            else:
                eteff = 100
            if 'elogg' in row.atmosphere:
                elogg = row.atmosphere['elogg']
            else:
                elogg = 0.2
            if 'evt' in row.atmosphere:
                evt = row.atmosphere['evt']
            else:
                evt = 0.2
            if 'efeh' in row.atmosphere:
                efeh = row.atmosphere['efeh']
            else:
                efeh = 0.0
            for elem in elem_corr:
                row_elem = row[f'EW_{elem}']
                lines = row_elem.index.to_list()
                ews = row_elem.dropna()
                logeps_lines = pd.Series(index=lines)
                elogeps_lines = pd.Series(index=lines)
                for line, ew in ews.items():
                    logeps_line, elogeps_line = self.get_abundance_from_line(
                        elem, line, teff, logg, feh, vt, ew, atm, 
                        eteff=eteff, elogg=elogg, efeh=efeh, evt=evt
                    )
                    logeps_lines[line] = logeps_line
                    elogeps_lines[line] = elogeps_line
                    # Store individual line results
                    table_detail.loc[idx, (f'A({elem})', line)] = logeps_line
                    table_detail.loc[idx, (f'eA({elem})', line)] = elogeps_line
                
                # Compute and store statistics for the element
                nonan_logeps = logeps_lines[~np.isnan(logeps_lines)].values
                nonan_elogeps = elogeps_lines[~np.isnan(logeps_lines)].values
                ### Remove outliers 
                ### or Abnormal Maximum/Minimum if len(nonan_logeps) == 3
                nonan_logeps, mask = self.remove_outliers(nonan_logeps, return_mask=True)
                if mask is not None:
                    nonan_elogeps = nonan_elogeps[mask]
                if len(nonan_logeps) > 1:
                    nonan_ave = np.round(np.nanmean(nonan_logeps), 2)
                    nonan_std = np.round(np.nanstd(nonan_logeps, ddof=1), 2)
                    nonan_err = np.round(np.nanmean(nonan_elogeps), 2)
                    ### Remove divergent but ONLY 2 lines, default threshold = .3
                    if (len(nonan_logeps) == 2) & (nonan_std > .3):
                        nonan_ave = nonan_std = nonan_err = np.nan
                elif len(nonan_logeps) == 1:
                    nonan_ave = nonan_logeps[0]
                    nonan_std = nonan_err =np.nan
                else:
                    nonan_ave = nonan_std = nonan_err = np.nan
                table_detail.loc[idx, (elem, 'ave')] = nonan_ave
                table_detail.loc[idx, (elem, 'std')] = nonan_std
                table_detail.loc[idx, (elem, 'err')] = nonan_err

        for elem in elem_corr:
            table_new.loc[:, ('A(X)', elem)] = table_detail.loc[:, (elem, 'ave')].values
            table_new.loc[:, ('eA(X)', elem)] = np.round(np.linalg.norm(
                table_detail.loc[:, (elem, ['std', 'err'])].values, 
                axis=1), 2)
        if save_dir:
            if not (save_dir / 'detail').exists():
                (save_dir / 'detail').mkdir()
            table_new.to_csv(save_dir / f'{atm}.csv', index=True)
            table_detail.to_csv(save_dir / 'detail' / f'{atm}.csv', index=True)
    
    @staticmethod
    def remove_outliers(data, threshold=1.5, return_mask=False):
        """
        Remove outliers from a NumPy array using the IQR method.

        Parameters:
        - data: np.ndarray
            Input array.
        - threshold: float, optional
            The threshold for defining outliers. Default is 1.5.

        Returns:
        - np.ndarray
            Array with outliers removed.
        """
        if len(data) == 0:
            mask = None
        elif len(data) == 3:
            # Sort the array
            sorted_data = np.sort(data)
            median = sorted_data[1]
            diff_min = np.abs(sorted_data[0] - median)
            diff_max = np.abs(sorted_data[2] - median)

            # Define outliers based on a percentage of the range
            mask = np.array([
                (threshold-1) * diff_min <= diff_max,
                True,  # Median is never an outlier
                (threshold-1) * diff_max <= diff_min
            ])
            data = data[mask]
        else:
            # Calculate Q1 (25th percentile) and Q3 (75th percentile)
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1  # Interquartile range

            # Define the lower and upper bounds for outliers
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            # Filter the data to remove outliers
            mask = (data >= lower_bound) & (data <= upper_bound)
            data = data[mask]
        if return_mask:
            return data, mask
        else:
            return data
        