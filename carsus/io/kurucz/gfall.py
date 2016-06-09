import time
import re

import numpy as np
import pandas as pd

from astropy import units as u
from sqlalchemy import and_
from sqlalchemy.orm.exc import NoResultFound
from carsus.model import Atom, DataSource, Ion, Level, LevelEnergy

class GFALLReader(object):
    def __init__(self, fname):
        self.fname = fname
        self._gfall_raw = None
        self._gfall_df = None
        self._levels_df = None
        self._lines_df = None

    @property
    def gfall_raw(self):
        if self._gfall_raw is None:
            self._gfall_raw = self.read_gfall_raw()
        return self._gfall_raw

    @property
    def gfall_df(self):
        if self._gfall_df is None:
            self._gfall_df = self.parse_gfall()
        return self._gfall_df

    @property
    def levels_df(self):
        if self._levels_df is None:
            self._levels_df = self.extract_levels()
        return self._levels_df

    @property
    def lines_df(self):
        if self._lines_df is None:
            self._lines_df = self.extract_lines()
        return self._lines_df

    def read_gfall_raw(self, fname=None):
        """
        Reading in a normal gfall.dat (please remove any empty lines)

        Parameters
        ----------

        fname: ~str
            path to gfall.dat

        Returns
        -------
            : pandas.DataFrame
                pandas Dataframe represenation of gfall
        """

        if fname is None:
            fname = self.fname

        # FORMAT(F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,
        # 3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3,2I5,I6)

        kurucz_fortran_format = ('F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,'
                                 'A10,F6.2,F6.2,F6.2,A4,I2,I2,I3,F6.3,I3,F6.3,I5,I5,'
                                 '1X,I1,A1,1X,I1,A1,I1,A3,I5,I5,I6')

        number_match = re.compile(r'\d+(\.\d+)?')
        type_match = re.compile(r'[FIXA]')
        type_dict = {'F': np.float64, 'I': np.int64, 'X': 'S1', 'A': 'S10'}
        field_types = tuple([type_dict[item] for item in number_match.sub(
            '', kurucz_fortran_format).split(',')])

        field_widths = type_match.sub('', kurucz_fortran_format)
        field_widths = map(int, re.sub(r'\.\d+', '', field_widths).split(','))

        gfall = np.genfromtxt(fname, dtype=field_types, delimiter=field_widths,
                              skip_header=2)

        columns = ['wavelength', 'loggf', 'element_code', 'e_first', 'j_first',
                   'blank1', 'label_first', 'e_second', 'j_second', 'blank2',
                   'label_second', 'log_gamma_rad', 'log_gamma_stark',
                   'log_gamma_vderwaals', 'ref', 'nlte_level_no_first',
                   'nlte_level_no_second', 'isotope', 'log_f_hyperfine',
                   'isotope2', 'log_iso_abundance', 'hyper_shift_first',
                   'hyper_shift_second', 'blank3', 'hyperfine_f_first',
                   'hyperfine_note_first', 'blank4', 'hyperfine_f_second',
                   'hyperfine_note_second', 'line_strength_class', 'line_code',
                   'lande_g_first', 'lande_g_second', 'isotopic_shift']

        gfall = pd.DataFrame(gfall)
        gfall.columns = columns

        return gfall

    def parse_gfall(self, gfall_df=None):
        """
        GFall pandas dataframe from read_gfall
        :param gfall_df:
        :return:
        """

        if gfall_df is None:
            gfall_df = self.gfall_raw.copy()

        double_columns = [item.replace('_first', '') for item in gfall_df.columns if
                          item.endswith('first')]

        # due to the fact that energy is stored in 1/cm
        order_lower_upper = (gfall_df["e_first"].abs() <
                             gfall_df["e_second"].abs())

        for column in double_columns:
            data = pd.concat([gfall_df['{0}_first'.format(
                column)][order_lower_upper], gfall_df['{0}_second'.format(
                column)][~order_lower_upper]])

            gfall_df['{0}_lower'.format(column)] = data

            data = pd.concat([gfall_df['{0}_first'.format(
                column)][~order_lower_upper], gfall_df['{0}_second'.format(
                column)][order_lower_upper]])

            gfall_df['{0}_upper'.format(column)] = data

            del gfall_df['{0}_first'.format(column)]
            del gfall_df['{0}_second'.format(column)]

        gfall_df["label_lower"] = gfall_df["label_lower"].apply(
            lambda x: x.strip())
        gfall_df["label_upper"] = gfall_df["label_upper"].apply(
            lambda x: x.strip())

        gfall_df['e_lower_predicted'] = gfall_df["e_lower"] < 0
        gfall_df["e_lower"] = gfall_df["e_lower"].abs()
        gfall_df['e_upper_predicted'] = gfall_df["e_upper"] < 0
        gfall_df["e_upper"] = gfall_df["e_upper"].abs()

        gfall_df['atomic_number'] = gfall_df.element_code.astype(int)
        gfall_df['ion_charge'] = (
            (gfall_df.element_code.values -
             gfall_df.atomic_number.values) * 100).round().astype(int)

        del gfall_df['element_code']

        return gfall_df

    def extract_levels(self, gfall_df=None, selected_columns=None):
        """
        Extract the levels from the gfall dataframe

        Parameters
        ----------

        gfall_df: ~pandas.DataFrame
        selected_columns: list
            list of which columns to select (optional - default=None which selects
            a default set of columns)

        Returns
        -------
            : ~pandas.DataFrame
                a level DataFrame
        """

        if gfall_df is None:
            gfall_df = self.gfall_df

        if selected_columns is None:
            selected_columns = ['atomic_number', 'ion_charge', 'energy', 'j',
                                'label', 'theoretical']

        column_renames = {'e_{0}': 'energy', 'j_{0}': 'j', 'label_{0}': 'label',
                          'e_{0}_predicted': 'theoretical'}

        e_lower_levels = gfall_df.rename(
            columns=dict([(key.format('lower'), value)
                          for key, value in column_renames.items()]))

        e_upper_levels = gfall_df.rename(
            columns=dict([(key.format('upper'), value)
                          for key, value in column_renames.items()]))

        levels = pd.concat([e_lower_levels[selected_columns],
                            e_upper_levels[selected_columns]])

        levels = levels.drop_duplicates(['atomic_number', 'ion_charge', 'energy', 'j', 'label']). \
            sort(['atomic_number', 'ion_charge', 'energy', 'j', 'label'])

        levels["level_index"] = levels.groupby(['atomic_number', 'ion_charge'])['j'].\
            transform(lambda x: np.arange(len(x))).values

        levels.set_index(["atomic_number", "ion_charge", "level_index"], inplace=True)

        return levels

    def extract_lines(self, gfall_df=None, levels_df=None, selected_columns=None):

        if gfall_df is None:
            gfall_df = self.gfall_df

        if levels_df is None:
            levels_df = self.levels_df

        if selected_columns is None:
            selected_columns = ['wavelength', 'loggf', 'atomic_number', 'ion_charge']

        levels_df_idx = levels_df.reset_index()
        levels_df_idx = levels_df_idx.set_index(['atomic_number', 'ion_charge', 'energy', 'j', 'label'])

        lines = gfall_df[selected_columns].copy()
        lines["gf"] = np.power(10, lines["loggf"])
        lines = lines.drop(["loggf"], 1)

        level_lower_idx = gfall_df[['atomic_number', 'ion_charge', 'e_lower',
                                    'j_lower', 'label_lower']].values.tolist()
        level_lower_idx = [tuple(item) for item in level_lower_idx]

        level_upper_idx = gfall_df[['atomic_number', 'ion_charge', 'e_upper',
                                    'j_upper', 'label_upper']].values.tolist()
        level_upper_idx = [tuple(item) for item in level_upper_idx]

        lines['level_index_lower'] = levels_df_idx["level_index"].loc[level_lower_idx].values
        lines['level_index_upper'] = levels_df_idx["level_index"].loc[level_upper_idx].values

        return lines



class GFALLIngester(object):

    def __init__(self, session, fname, ds_short_name="ku_latest"):
        self.session = session
        self.gfall_reader = GFALLReader(fname)
        self.data_source = DataSource.as_unique(self.session, short_name=ds_short_name)
        self._atomic_number2atom_id = None

    @property
    def atomic_number2atom_id(self):
        if self._atomic_number2atom_id is None:

            # Select all existing atoms from this datasource
            q_atomic_number_atom_id = self.session.query(Atom).filter(Atom.data_source==self.data_source)

            # Create a DataFrame that maps atomic_number to atom_id
            atomic_number2atom_id_df = pd.read_sql_query(q_atomic_number_atom_id.selectable, self.session.bind,
                                              index_col="atomic_number")

            new_atoms = dict()

            for atomic_number, _ in self.gfall_reader.levels_df.groupby(level=["atomic_number"]):

                try:
                    assert atomic_number in atomic_number2atom_id_df.index
                except AssertionError:
                    atom = Atom(atomic_number=atomic_number, data_source=self.data_source)
                    self.session.add(atom)



    def ingest_levels(self, levels_df=None):

        if levels_df is None:
            levels_df = self.gfall_reader.levels_df


        for ion_index, ion_df in levels_df.groupby(level=["atomic_number", "ion_charge"]):

            atomic_number, ion_charge = ion_index
            atom = Atom(atomic_number=atomic_number, data_source=self.data_source)

