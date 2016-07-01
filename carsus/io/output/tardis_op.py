import numpy as np
import pandas as pd
import hashlib
import uuid

from pandas import HDFStore
from carsus.util import carsus_data_dir
from carsus.model import Atom, Ion, Line, Level, DataSource, ECollision
from carsus.model.meta import yield_limit
from sqlalchemy import and_, union_all, literal
from sqlalchemy.orm import joinedload
from sqlalchemy.orm.exc import NoResultFound
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity
from scipy import interpolate
from tardis.util import species_string_to_tuple

P_EMISSION_DOWN = -1
P_INTERNAL_DOWN = 0
P_INTERNAL_UP = 1

LINES_MAXRQ = 10000  # for yield_limit

ZETA_DATAFILE = carsus_data_dir("knox_long_recombination_zeta.dat")



class AtomData(object):
    """
    Class for creating the atomic dataframes for TARDIS

    Parameters:
    ------------
    session: SQLAlchemy session
    basic_atom_max_atomic_number: int
        The maximum atomic number to be stored in basic_atom_df
            (default: 30)
    levels_create_metastable_flags: bool
        Create the `metastable` column containing flags for metastable levels (levels that take a long time to de-excite)
        (default: True)
    levels_metastable_loggf_threshold: int
        log(gf) threshold for flagging metastable levels
        (default: -3)
    chianti_species: list of str in format <element_symbol> <ion_number>, eg. Fe 2
            The levels data for these ions will be taken from the CHIANTI database
            (default: None)
    chianti_short_name: str
        The short name of the CHIANTI database, if set to None the latest version will be used
        (default: None)
    kurucz_short_name: str
        The short name of the Kurucz database, if set to None the latest version will be used
        (default: None)
    temperatures: np.array
        The temperatures for calculating collision strengths

    Attributes:
    ------------
    session: SQLAlchemy session
    basic_atom_param: dict
        The parameters for creating basic_atom_df
    levels_param: dict
        The parameters for creating levels_df
    collisions_df: dict
        The parameters for creating collisins_df
    basic_atom_df
    ionization_df
    levels_df
    lines_df
    collisions_df
    macro_atom_df
    macro_atom_ref_df
    basic_atom_df_prepared
    ionization_df_prepared
    levels_df_prepared
    lines_df_prepared
    collisions_df_prepared
    macro_atom_df_prepared
    macro_atom_ref_df_prepared

    Methods:
    ---------
    create_basic_atom_df
    create_ionization_df
    create_levels_df
    create_lines_df
    create_collisions_df
    create_macro_atom_df
    create_macro_atom_ref_df
    prepare_basic_atom_df
    prepare_ionization_df
    prepare_levels_df
    prepare_lines_df
    prepare_collisions_df
    prepare_macro_atom_df
    prepare_macro_atom_ref_df

    """

    def __init__(self, session,
                 basic_atom_max_atomic_number=30, levels_create_metastable_flags=True,
                 levels_metastable_loggf_threshold=-3, chianti_species=None,
                 chianti_short_name=None, kurucz_short_name=None,
                 collisions_temperatures=None, zeta_datafile=None):

        self.session = session

        # Set the parameters for the dataframes
        self.basic_atom_param = {
            "max_atomic_number": basic_atom_max_atomic_number
        }

        self.levels_param = {
            "create_metastable_flags": levels_create_metastable_flags,
            "metastable_loggf_threshold": levels_metastable_loggf_threshold
        }

        if collisions_temperatures is None:
            collisions_temperatures = np.linspace(2000, 50000, 20)
        else:
            collisions_temperatures = np.array(collisions_temperatures)

        self.collisions_param = {
            "temperatures": collisions_temperatures
        }

        # Query the data sources
        if kurucz_short_name is None:
            kurucz_short_name = "ku_latest"
        try:
            self.ku_ds = session.query(DataSource).filter(DataSource.short_name == kurucz_short_name).one()
        except NoResultFound:
            print "Kurucz data source does not exist!"
            raise

        if chianti_species is not None:

            # Get a list of tuples (atomic_number, ion_charge) for the chianti species
            chianti_species = [tuple(species_string_to_tuple(species_str)) for species_str in chianti_species]

            if chianti_short_name is None:
                chianti_short_name = "chianti_v8.0.2"
            try:
                self.ch_ds = session.query(DataSource).filter(DataSource.short_name == chianti_short_name).one()
            except NoResultFound:
                print "Chianti data source does not exist!"
                raise
        self.chianti_species = chianti_species

        if zeta_datafile is None:
            self.zeta_datafile = ZETA_DATAFILE
        else:
            self.zeta_datafile = zeta_datafile

        self._basic_atom_df = None
        self._ionization_df = None
        self._levels_df = None
        self._lines_df = None
        self._collisions_df = None
        self._macro_atom_df = None
        self._macro_atom_ref_df = None
        self._zeta_data = None

    @property
    def basic_atom_df(self):
        if self._basic_atom_df is None:
            self._basic_atom_df = self.create_basic_atom_df(**self.basic_atom_param)
        return self._basic_atom_df

    def create_basic_atom_df(self, max_atomic_number):
        """
        Create a DataFrame with basic atomic data.

        Parameters
        ----------
        max_atomic_number: int
            The maximum atomic number to be stored in basic_atom_df

        Returns
        -------
        basic_atom_df : pandas.DataFrame
            DataFrame containing the *basic atomic data* with:
            index: none;
            columns: atomic_number, symbol, name, weight[u].
        """
        basic_atom_q = self.session.query(Atom). \
            filter(Atom.atomic_number <= max_atomic_number).\
            order_by(Atom.atomic_number)

        basic_atom_data = list()
        for atom in basic_atom_q.options(joinedload(Atom.weights)):
            weight = atom.weights[0].quantity.value if atom.weights else None  # Get the first weight from the collection
            basic_atom_data.append((atom.atomic_number, atom.symbol, atom.name, weight))

        basic_atom_dtype = [("atomic_number", np.int), ("symbol", "|S5"), ("name", "|S150"),
                            ("weight", np.float)]
        basic_atom_data = np.array(basic_atom_data, dtype=basic_atom_dtype)
        basic_atom_df = pd.DataFrame.from_records(basic_atom_data)

        return basic_atom_df

    @property
    def basic_atom_df_prepared(self):
        return self.prepare_basic_atom_df()

    def prepare_basic_atom_df(self):
        """
        Prepare the basic_atom_df for TARDIS

        Returns
        -------
        basic_atom_df : pandas.DataFrame
            DataFrame containing the *basic atomic data* with:
                index: atomic_number;
                columns: symbol, name, mass[cgs].
        """
        # Set index
        basic_atom_df = self.basic_atom_df.set_index("atomic_number")

        # Rename the `weight` column to `mass`
        basic_atom_df.rename(columns={"weight": "mass"}, inplace=True)

        # We have to use constants.u because astropy uses different values for the unit u and the constant.
        # This is changed in later versions of astropy (the value of constants.u is used in all cases)
        if u.u.cgs == const.u.cgs:
            basic_atom_df["mass"] = Quantity(basic_atom_df["mass"].values, "u").cgs
        else:
            basic_atom_df["mass"] = basic_atom_df["mass"].values * const.u.cgs

        return basic_atom_df

    @property
    def ionization_df(self):
        if self._ionization_df is None:
            self._ionization_df = self.create_ionization_df()
        return self._ionization_df

    def create_ionization_df(self):
        """
        Create a DataFrame with ionization data.

        Returns
        -------
        ionization_df : pandas.DataFrame
            DataFrame containing the *ionization data* with:
                index: none;
                columns: atomic_number, ion_number, ionization_energy[eV]
        """
        ionization_q = self.session.query(Ion).\
            order_by(Ion.atomic_number, Ion.ion_charge)

        ionization_data = list()
        for ion in ionization_q.options(joinedload(Ion.ionization_energies)):
            ionization_energy = ion.ionization_energies[0].quantity.value if ion.ionization_energies else None
            ionization_data.append((ion.atomic_number, ion.ion_charge, ionization_energy))

        ionization_dtype = [("atomic_number", np.int), ("ion_number", np.int), ("ionization_energy", np.float)]
        ionization_data = np.array(ionization_data, dtype=ionization_dtype)

        ionization_df = pd.DataFrame.from_records(ionization_data)

        return ionization_df

    @property
    def ionization_df_prepared(self):
        return self.prepare_ionization_df()

    def prepare_ionization_df(self):
        """
        Prepare ionization_df for TARDIS

        Returns
        -------
        ionization_df : pandas.DataFrame
            DataFrame containing the *ionization data* with:
                index: atomic_number, ion_number;
                columns: ionization_energy[cgs].

        Notes
        ------
        In TARDIS `ion_number` describes the final ion state,
        e.g. H I - H II is described with ion_number = 1
        On the other hand, in carsus `ion_number` describes the lower ion state,
        e.g. H I - H II is described with ion_number = 0
        For this reason we add 1 to `ion_number` in this prepare method.
        """
        ionization_df = self.ionization_df.copy()

        # See the Notes section
        ionization_df["ion_number"] += 1

        # Convert ionization energies to CGS
        ionization_df["ionization_energy"] = Quantity(ionization_df["ionization_energy"].values, "eV").cgs

        # Set index
        ionization_df.set_index(["atomic_number", "ion_number"], inplace=True)

        return ionization_df

    @property
    def levels_df(self):
        if self._levels_df is None:
            self._levels_df = self.create_levels_df(**self.levels_param)
        return self._levels_df

    def create_levels_df(self, create_metastable_flags, metastable_loggf_threshold):
        """
            Create a DataFrame with levels data.

            Parameters
            ----------
            create_metastable_flags: bool
                Create the `metastable` column containing flags for metastable levels (levels that take a long time to de-excite)
            metastable_loggf_threshold: int
                log(gf) threshold for flagging metastable levels

            Returns
            -------
            levels_df : pandas.DataFrame
                DataFrame containing the *levels data* with:
                    index: level_id
                    columns: atomic_number, ion_number, level_number, energy[eV], g[1]
        """

        if self.chianti_species is None:
            kurucz_levels_q = self.session.query(Level).\
                filter(Level.data_source == self.ku_ds)

            levels_q = kurucz_levels_q

        else:

            # To select ions we create a CTE (Common Table Expression), because sqlite doesn't support composite IN statements
            chianti_species_cte = union_all(
                *[self.session.query(
                    literal(atomic_number).label("atomic_number"),
                    literal(ion_charge).label("ion_charge"))
                  for atomic_number, ion_charge in self.chianti_species]
            ).cte("chianti_species_cte")

            # To select chianti ions we join on the CTE
            chianti_levels_q = self.session.query(Level).\
                join(chianti_species_cte, and_(Level.atomic_number == chianti_species_cte.c.atomic_number,
                                               Level.ion_charge == chianti_species_cte.c.ion_charge)).\
                filter(Level.data_source == self.ch_ds)

            # To select kurucz ions we do an outerjoin on the CTE and select rows that don't have a match from the CTE
            kurucz_levels_q = self.session.query(Level).\
                outerjoin(chianti_species_cte, and_(Level.atomic_number == chianti_species_cte.c.atomic_number,
                                               Level.ion_charge == chianti_species_cte.c.ion_charge)).\
                filter(chianti_species_cte.c.atomic_number.is_(None)).\
                filter(Level.data_source == self.ku_ds)

            levels_q = kurucz_levels_q.union(chianti_levels_q)

        # Get the levels data
        levels_data = list()
        for lvl in levels_q.options(joinedload(Level.energies)):
            try:
                energy = None
                # Try to find the measured energy for this level
                for nrg in lvl.energies:
                    if nrg.method == "meas":
                        energy = nrg.quantity
                        break
                # If the measured energy is not available, try to get the first one
                if energy is None:
                    energy = lvl.energies[0].quantity
            except IndexError:
                print "No energy is available for level {0}".format(lvl.level_id)
                continue
            levels_data.append((lvl.level_id, lvl.atomic_number, lvl.ion_charge, energy.value, lvl.g, lvl.data_source_id))

        # Create a dataframe with the levels data
        levels_dtype = [("level_id", np.int), ("atomic_number", np.int),
                        ("ion_number", np.int), ("energy", np.float), ("g", np.int), ("ds_id", np.int)]
        levels_data = np.array(levels_data, dtype=levels_dtype)
        levels_df = pd.DataFrame.from_records(levels_data, index="level_id")

        # Create level numbers
        levels_df.sort_values(["atomic_number", "ion_number", "energy", "g"], inplace=True)
        levels_df["level_number"] = levels_df.groupby(['atomic_number', 'ion_number'])['energy']. \
            transform(lambda x: np.arange(len(x))).values
        levels_df["level_number"] = levels_df["level_number"].astype(np.int)

        if create_metastable_flags:
            # Create metastable flags
            # ToDO: It is assumed that all lines are ingested. That may not always be the case

            levels_subq = self.session.query(Level). \
                filter(Level.level_id.in_(levels_df.index.values)).subquery()
            metastable_q = self.session.query(Line). \
                join(levels_subq, Line.upper_level)

            metastable_data = list()
            for line in yield_limit(metastable_q.options(joinedload(Line.gf_values)),
                                    Line.line_id, maxrq=LINES_MAXRQ):
                try:
                    # Currently it is assumed that each line has only one gf value
                    gf = line.gf_values[0].quantity  # Get the first gf value
                except IndexError:
                    print "No gf value is available for line {0}".format(line.line_id)
                    continue
                metastable_data.append((line.line_id, line.upper_level_id, gf.value))

            metastable_dtype = [("line_id", np.int), ("upper_level_id", np.int), ("gf", np.float)]
            metastable_data = np.array(metastable_data, dtype=metastable_dtype)
            metastable_df = pd.DataFrame.from_records(metastable_data, index="line_id")

            # Filter loggf on the threshold value
            metastable_df["loggf"] = np.log10(metastable_df["gf"])
            metastable_df = metastable_df.loc[metastable_df["loggf"] > metastable_loggf_threshold]

            # Count the remaining strong transitions
            metastable_df_grouped = metastable_df.groupby("upper_level_id")
            metastable_flags = metastable_df_grouped["upper_level_id"].count()
            metastable_flags.name = "metastable"

            # If there are no strong transitions for a level (the count is NaN) then the metastable flag is True
            # else (the count is a natural number) the metastable flag is False
            levels_df = levels_df.join(metastable_flags)
            levels_df["metastable"] = levels_df["metastable"].isnull()

        return levels_df

    @property
    def levels_df_prepared(self):
        return self.prepare_levels_df()

    def prepare_levels_df(self):
        """
        Prepare levels_df for TARDIS

        Returns
        -------
        levels_df : pandas.DataFrame
            DataFrame containing the *levels data* with:
                index: none;
                columns: atomic_number, ion_number, level_number, energy[cgs], g[1], metastable.
        """

        levels_df = self.levels_df.copy()

        # Set index
        levels_df.reset_index(inplace=True)
        # levels_df.set_index(["atomic_number", "ion_number", "level_number"], inplace=True)

        # Covert energy to CGS
        levels_df["energy"] = Quantity(levels_df["energy"].values, 'eV').cgs

        # Drop the unwanted columns
        levels_df.drop(["level_id", "ds_id"], axis=1, inplace=True)

        return levels_df

    @property
    def lines_df(self):
        if self._lines_df is None:
            self._lines_df = self.create_lines_df()
        return self._lines_df

    def create_lines_df(self):
        """
            Create a DataFrame with lines data.

            Returns
            -------
            lines_df : pandas.DataFrame
                DataFrame containing the *levels data* with:
                    index: line_id;
                    columns: atomic_number, ion_number, level_number_lower, level_number_upper,
                             wavelength[angstrom], nu[Hz], f_lu[1], f_ul[1], B_ul[?], B_ul[?], A_ul[1/s].
        """
        levels_df = self.levels_df.copy()

        levels_subq = self.session.query(Level.level_id.label("level_id")). \
            filter(Level.level_id.in_(levels_df.index.values)).subquery()

        lines_q = self.session.query(Line).\
            join(levels_subq, Line.lower_level_id == levels_subq.c.level_id)

        lines_data = list()
        for line in yield_limit(lines_q.options(joinedload(Line.wavelengths), joinedload(Line.gf_values)),
                                Line.line_id, maxrq=LINES_MAXRQ):
            try:
                # Try to get the first gf value
                gf = line.gf_values[0].quantity
            except IndexError:
                print "No gf value is available for line {0}".format(line.line_id)
                continue
            try:
                # Try to get the first wavelength
                wavelength = line.wavelengths[0].quantity
            except IndexError:
                print "No wavelength is available for line {0}".format(line.line_id)
                continue
            lines_data.append((line.line_id, line.lower_level_id, line.upper_level_id,
                               line.data_source_id,  wavelength.value, gf.value))

        lines_dtype = [("line_id", np.int), ("lower_level_id", np.int), ("upper_level_id", np.int),
                       ("ds_id", np.int), ("wavelength", np.float), ("gf", np.float)]
        lines_data = np.array(lines_data, dtype=lines_dtype)
        lines_df = pd.DataFrame.from_records(lines_data, index="line_id")

        # Join atomic_number, ion_number, level_number_lower, level_number_upper and set multiindex
        ions_df = levels_df[["atomic_number", "ion_number"]]

        lower_levels_df = levels_df.rename(columns={"level_number": "level_number_lower", "g": "g_l"}).\
            loc[:,["level_number_lower", "g_l"]]
        upper_levels_df = levels_df.rename(columns={"level_number": "level_number_upper", "g": "g_u"}).\
            loc[:,["level_number_upper", "g_u"]]

        lines_df = lines_df.join(ions_df, on="lower_level_id")
        lines_df = lines_df.join(lower_levels_df, on="lower_level_id")
        lines_df = lines_df.join(upper_levels_df, on="upper_level_id")

        # Calculate absorption oscillator strength f_lu and emission oscillator strength f_ul
        lines_df["f_lu"] = lines_df["gf"]/lines_df["g_l"]
        lines_df["f_ul"] = lines_df["gf"]/lines_df["g_u"]

        # Calculate frequency
        lines_df['nu'] = u.Unit('angstrom').to('Hz', lines_df['wavelength'], u.spectral())

        # Calculate Einstein coefficients
        einstein_coeff = (4 * np.pi**2 * const.e.gauss.value**2) / (const.m_e.cgs.value * const.c.cgs.value)
        lines_df['B_lu'] = einstein_coeff * lines_df['f_lu'] / (const.h.cgs.value * lines_df['nu'])
        lines_df['B_ul'] = einstein_coeff * lines_df['f_ul'] / (const.h.cgs.value * lines_df['nu'])
        lines_df['A_ul'] = 2 * einstein_coeff * lines_df['nu']**2 / const.c.cgs.value**2 * lines_df['f_ul']

        return lines_df

    @property
    def lines_df_prepared(self):
        return self.prepare_lines_df()

    def prepare_lines_df(self):
        """
            Prepare lines_df for TARDIS
            Parameters
            ----------
            session : SQLAlchemy session
            chianti_species: list of str in format <element_symbol> <ion_number>, eg. Fe 2
                The lines data for these ions will be taken from the CHIANTI database
                (default: None)
            chianti_short_name: str
                The short name of the CHIANTI database, if set to None the latest version will be used
                (default: None)
            kurucz_short_name: str
                The short name of the Kurucz database, if set to None the latest version will be used
                (default: None)
            Returns
            -------
            lines_df : pandas.DataFrame
                DataFrame containing the *levels data* with:
                    index: none;
                    columns: lind_id, atomic_number, ion_number, level_number_lower, level_number_upper,
                             wavelength[angstrom], wavelength_cm[CGS], nu[Hz], f_lu[1], f_ul[1], B_ul[?], B_ul[?], A_ul[1/s].
        """

        #Set the index
        lines_df = self.lines_df.reset_index()
        # lines_df.set_index(["atomic_number", "ion_number", "level_number_lower", "level_number_upper"], inplace=True)

        # Create a new columns with wavelengths in the CGS units
        lines_df['wavelength_cm'] = Quantity(lines_df['wavelength'], 'angstrom').cgs

        # Drop the unwanted columns
        lines_df.drop(["g_l", "g_u", "gf", "lower_level_id", "upper_level_id", "ds_id"], axis=1, inplace=True)

        return lines_df

    @property
    def collisions_df(self):
        if self._collisions_df is None:
            self._collistions_df = self.create_collisions_df(**self.collisions_param)
        return self._collistions_df

    def create_collisions_df(self, temperatures):
        """
            Create a DataFrame with collisions data.

            Parameters
            -----------
            temperatures: np.array
                The temperatures for calculating collision strengths

            Returns
            -------
            collisions_df : pandas.DataFrame
                DataFrame with the *electron collisions data* with:
        """

        levels_df = self.levels_df.copy()

        levels_subq = self.session.query(Level.level_id.label("level_id")). \
            filter(Level.level_id.in_(levels_df.index.values)).\
            filter(Level.data_source == self.ch_ds).subquery()

        collisions_q = self.session.query(ECollision). \
            join(levels_subq, ECollision.lower_level_id == levels_subq.c.level_id)

        collisions_data = list()
        for e_col in collisions_q.options(joinedload(ECollision.gf_values),
                                          joinedload(ECollision.temp_strengths)):

            # Try to get the first gf value
            try:
                gf = e_col.gf_values[0].quantity
            except IndexError:
                print "No gf is available for electron collision {0}".format(e_col.e_col_id)
                continue

            btemp, bscups = (list(ts) for ts in zip(*e_col.temp_strengths_tuple))

            collisions_data.append((e_col.e_col_id, e_col.lower_level_id, e_col.upper_level_id,
                e_col.data_source_id, btemp, bscups, e_col.bt92_ttype, e_col.bt92_cups, gf.value))

        collisions_dtype = [("e_col_id", np.int), ("lower_level_id", np.int), ("upper_level_id", np.int),
                            ("ds_id", np.int),  ("btemp", 'O'), ("bscups", 'O'), ("ttype", np.int),
                            ("cups", np.float), ("gf", np.float)]

        collisions_data = np.array(collisions_data, dtype=collisions_dtype)
        collisions_df = pd.DataFrame.from_records(collisions_data, index="e_col_id")

        # Join atomic_number, ion_number, level_number_lower, level_number_upper and set multiindex
        ions_df = levels_df[["atomic_number", "ion_number"]]

        lower_levels_df = levels_df.rename(columns={"level_number": "level_number_lower", "g": "g_l", "energy": "energy_lower"}). \
                              loc[:, ["level_number_lower", "g_l", "energy_lower"]]
        upper_levels_df = levels_df.rename(columns={"level_number": "level_number_upper", "g": "g_u", "energy": "energy_upper"}). \
                              loc[:, ["level_number_upper", "g_u", "energy_upper"]]

        collisions_df = collisions_df.join(ions_df, on="lower_level_id")
        collisions_df = collisions_df.join(lower_levels_df, on="lower_level_id")
        collisions_df = collisions_df.join(upper_levels_df, on="upper_level_id")

        # Calculate delta_e
        kb_ev = const.k_B.cgs.to('eV / K').value
        collisions_df["delta_e"] = (collisions_df["energy_upper"] - collisions_df["energy_lower"])/kb_ev

        def calculate_collisional_strength(row, temperatures):
            """
                Function to calculation upsilon from Burgess & Tully 1992 (TType 1 - 4; Eq. 23 - 38)
            """
            c = row["cups"]
            x_knots = np.linspace(0, 1, len(row["btemp"]))
            y_knots = row["bscups"]
            delta_e = row["delta_e"]
            g_u = row["g_u"]

            ttype = row["ttype"]
            if ttype > 5: ttype -= 5

            kt = kb_ev * temperatures

            spline_tck = interpolate.splrep(x_knots, y_knots)

            if ttype == 1:
                x = 1 - np.log(c) / np.log(kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func * np.log(kt / delta_e + np.exp(1))

            elif ttype == 2:
                x = (kt / delta_e) / (kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func

            elif ttype == 3:
                x = (kt / delta_e) / (kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func / (kt / delta_e + 1)

            elif ttype == 4:
                x = 1 - np.log(c) / np.log(kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func * np.log(kt / delta_e + c)

            elif ttype == 5:
                raise ValueError('Not sure what to do with ttype=5')

            #### 1992A&A...254..436B Equation 20 & 22 #####

            c_ul = 8.63e-6 * upsilon / (g_u * temperatures**.5)
            return tuple(c_ul)

        collisions_df["c_ul"] = collisions_df.apply(calculate_collisional_strength, axis=1, args=(temperatures,))

        # Calculate g_ratio
        collisions_df["g_ratio"] = collisions_df["g_l"] / collisions_df["g_u"]

        return collisions_df

    @property
    def collisions_df_prepared(self):
        return self.prepare_collisions_df()

    def prepare_collisions_df(self):
        """
            Prepare collisions_df for TARDIS

            Returns
            -------
            collisions_df : pandas.DataFrame
                DataFrame with the *electron collisions data* with:
                    index: atomic_number, ion_number, level_number_lower, level_number_upper;
                    columns: e_col_id, delta_e, g_ratio, c_ul.
        """

        collisions_df = self.collisions_df.copy()

        # Drop the unwanted columns
        collisions_df.drop(["lower_level_id", "upper_level_id", "ds_id", "btemp", "bscups",
                            "ttype", "energy_lower", "energy_upper", "gf", "g_l", "g_u", "cups"],  axis=1, inplace=True)

        # Set multiindex
        collisions_df.reset_index(inplace=True)
        collisions_df.set_index(["atomic_number", "ion_number", "level_number_lower", "level_number_upper"], inplace=True)

        return collisions_df

    @property
    def macro_atom_df(self):
        if self._macro_atom_df is None:
            self._macro_atom_df = self.create_macro_atom_df()
        return self._macro_atom_df

    def create_macro_atom_df(self):
        """
            Create a DataFrame with macro atom data.

            Returns
            -------
            macro_atom_df : pandas.DataFrame
                DataFrame with the *macro atom data* with:
                    index: none;
                    columns: atomic_number, ion_number, source_level_number, target_level_number,
                        transition_line_id, transition_type, transition_probability.

            Notes:
                Refer to the docs: http://tardis.readthedocs.io/en/latest/physics/plasma/macroatom.html

        """

        levels_df = self.levels_df.copy()
        lines_df = self.lines_df.copy()

        lvl_energy_lower_df = levels_df.rename(columns={"energy": "energy_lower"}).loc[:, ["energy_lower"]]
        lvl_energy_upper_df = levels_df.rename(columns={"energy": "energy_upper"}).loc[:, ["energy_upper"]]

        lines_df = lines_df.join(lvl_energy_lower_df, on="lower_level_id")
        lines_df = lines_df.join(lvl_energy_upper_df, on="upper_level_id")

        macro_atom_data = list()
        macro_atom_dtype = [("atomic_number", np.int), ("ion_number", np.int),
                            ("source_level_number", np.int), ("target_level_number", np.int),
                            ("transition_line_id", np.int), ("transition_type", np.int), ("transition_probability", np.float)]

        for line_id, row in lines_df.iterrows():
            atomic_number, ion_number = row["atomic_number"], row["ion_number"]
            level_number_lower, level_number_upper = row["level_number_lower"], row["level_number_upper"]
            nu = row["nu"]
            f_ul, f_lu = row["f_ul"], row["f_lu"]
            e_lower, e_upper = row["energy_lower"], row["energy_upper"]

            transition_probabilities_dict = dict()  # type : probability
            transition_probabilities_dict[P_EMISSION_DOWN] = 2 * nu**2 * f_ul / const.c.cgs.value**2 * (e_upper - e_lower)
            transition_probabilities_dict[P_INTERNAL_DOWN] = 2 * nu**2 * f_ul / const.c.cgs.value**2 * e_lower
            transition_probabilities_dict[P_INTERNAL_UP] = f_lu * e_lower / (const.h.cgs.value * nu)

            macro_atom_data.append((atomic_number, ion_number, level_number_upper, level_number_lower,
                                    line_id, P_EMISSION_DOWN, transition_probabilities_dict[P_EMISSION_DOWN]))
            macro_atom_data.append((atomic_number, ion_number, level_number_upper, level_number_lower,
                                    line_id, P_INTERNAL_DOWN, transition_probabilities_dict[P_INTERNAL_DOWN]))
            macro_atom_data.append((atomic_number, ion_number, level_number_lower, level_number_upper,
                                    line_id, P_INTERNAL_UP, transition_probabilities_dict[P_INTERNAL_UP]))

        macro_atom_data = np.array(macro_atom_data, dtype=macro_atom_dtype)
        macro_atom_df = pd.DataFrame(macro_atom_data)

        return macro_atom_df

    @property
    def macro_atom_df_prepared(self):
        return self.prepare_macro_atom_df()

    def prepare_macro_atom_df(self):
        """
            Prepare macro_atom_df for TARDIS

            Returns
            -------
            macro_atom_df : pandas.DataFrame
                DataFrame with the *macro atom data* with:
                    index: none;
                    columns: atomic_number, ion_number, source_level_number, destination_level_number,
                        transition_line_id, transition_type, transition_probability.

            Notes:
                Refer to the docs: http://tardis.readthedocs.io/en/latest/physics/plasma/macroatom.html

        """
        macro_atom_df = self.macro_atom_df.copy()

        # ToDo: choose between `target_level_number` and `destination_level_number`
        # Rename `target_level_number` to `destination_level_number` used in TARDIS
        # Personally, I think `target_level_number` is better so I use it in Carsus.
        macro_atom_df.rename(columns={"target_level_number": "destination_level_number"}, inplace=True)

        # macro_atom_df.set_index(["atomic_number", "ion_number",
        #                          "source_level_number", "destination_level_number"], inplace=True)
        # macro_atom_df.sort_index(level=["atomic_number", "ion_number", "source_level_number"], inplace=True)
        macro_atom_df.sort_values(["atomic_number", "ion_number", "source_level_number"], inplace=True)
        return macro_atom_df

    @property
    def macro_atom_ref_df(self):
        if self._macro_atom_ref_df is None:
            self._macro_atom_ref_df = self.create_macro_atom_ref_df()
        return self._macro_atom_ref_df

    def create_macro_atom_ref_df(self):
        """
            Create a DataFrame with macro atom reference data.

            Returns
            -------
            macro_atom_ref_df : pandas.DataFrame
                DataFrame with the *macro atom references* with:
                    index: level_id;
                    columns: atomic_number, ion_number, source_level_number, count_down, count_up, count_total.
        """

        levels_df = self.levels_df.copy()
        lines_df = self.lines_df.copy()

        macro_atom_ref_df = levels_df.rename(columns={"level_number": "source_level_number"}).\
                                       loc[:, ["atomic_number", "ion_number", "source_level_number"]]

        count_down = lines_df.groupby("upper_level_id").size()
        count_down.name = "count_down"

        count_up = lines_df.groupby("lower_level_id").size()
        count_up.name = "count_up"

        macro_atom_ref_df = macro_atom_ref_df.join(count_down).join(count_up)
        macro_atom_ref_df.fillna(0, inplace=True)
        macro_atom_ref_df["count_total"] = 2*macro_atom_ref_df["count_down"] + macro_atom_ref_df["count_up"]

        # Convert to int
        macro_atom_ref_df["count_down"] = macro_atom_ref_df["count_down"].astype(np.int)
        macro_atom_ref_df["count_up"] = macro_atom_ref_df["count_up"].astype(np.int)
        macro_atom_ref_df["count_total"] = macro_atom_ref_df["count_total"].astype(np.int)

        return macro_atom_ref_df

    @property
    def macro_atom_ref_df_prepared(self):
        return self.prepare_macro_atom_ref_df()

    def prepare_macro_atom_ref_df(self):
        """
            Prepare macro_atom_ref_df for TARDIS

            Returns
            -------
            macro_atom_ref_df : pandas.DataFrame
                DataFrame with the *macro atom references* with:
                    index: no_index;
                    columns: atomic_number, ion_number, source_level_number, count_down, count_up, count_total.
        """
        macro_atom_ref_df = self.macro_atom_ref_df.copy()

        macro_atom_ref_df.reset_index(drop=True, inplace=True)
        # macro_atom_ref_df.set_index(["atomic_number", "ion_number", "source_level_number"], inplace=True)

        return macro_atom_ref_df

    @property
    def zeta_data(self):
        if self._zeta_data is None:
            self._zeta_data = self.create_zeta_data(self.zeta_datafile)
        return self._zeta_data

    def create_zeta_data(self, zeta_datafile):
        zeta_data = np.loadtxt(zeta_datafile, usecols=xrange(1, 23), dtype=np.float64)
        t_rads = np.arange(2000, 42000, 2000)
        return pd.DataFrame(zeta_data[:,2:],
                            index=pd.MultiIndex.from_arrays(zeta_data[:,:2].transpose().astype(int)),
                            columns=t_rads)

    def to_hdf(self, hdf5_path, store_basic_atom=True, store_ionization=True,
               store_levels=True, store_lines=True, store_collisions=True, store_macro_atom=True,
               store_macro_atom_ref=True, store_zeta_data=True):
        """
            Store the dataframes in an HDF5 file

            Parameters
            ------------
            hdf5_path: str
                The path of the HDF5 file
            store_basic_atom: bool
                Store the basic atom DataFrame
                (default: True)
            store_ionization: bool
                Store the ionzation DataFrame
                (default: True)
            store_levels: bool
                Store the levels DataFrame
                (default: True)
            store_lines: bool
                Store the lines DataFrame
                (default: True)
            store_collisions: bool
                Store the electron collisions DataFrame
                (default: True)
            store_macro_atom: bool
                Store the macro_atom DataFrame
                (default: True)
            store_macro_atom_ref: bool
                Store the macro_atom_references DataFrame
                (default: True)
            store_zeta_data: bool
                Store `zeta_data`
        """

        with HDFStore(hdf5_path) as store:

            if store_basic_atom:
                store.put("basic_atom_df", self.basic_atom_df_prepared)

            if store_ionization:
                store.put("ionization_df", self.ionization_df_prepared)

            if store_levels:
                store.put("levels_df", self.levels_df_prepared)

            if store_lines:
                store.put("lines_df", self.lines_df_prepared)

            if store_collisions:
                store.put("collisions_df", self.collisions_df_prepared)
                store.get_storer("collisions_df").attrs["temperatures"] = self.collisions_param["temperatures"]

            if store_macro_atom:
                store.put("macro_atom_df", self.macro_atom_df_prepared)

            if store_macro_atom_ref:
                store.put("macro_atom_ref_df", self.macro_atom_ref_df_prepared)

            if store_zeta_data:
                store.put("zeta_data", self.zeta_data)

            # Set the root attributes
            # It seems that the only way to set the root attributes is to use `_v_attrs`
            store.root._v_attrs["database_version"] = "v0.9"

            print "Signing AtomData with MD5 and UUID1"

            md5_hash = hashlib.md5()
            for key in store.keys():
                md5_hash.update(store[key].values.data)

            uuid1 = uuid.uuid1().hex

            store.root._v_attrs['md5'] = md5_hash.hexdigest()
            store.root._v_attrs['uuid1'] = uuid1