import pytest
import os
import pandas as pd
import numpy as np

from carsus.io.output.tardis_op import AtomData
from carsus.model import DataSource
from numpy.testing import assert_almost_equal, assert_allclose
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose


with_test_db = pytest.mark.skipif(
    not pytest.config.getoption("--test-db"),
    reason="--testing database was not specified"
)


@pytest.fixture
def atom_data(test_session):
    atom_data = AtomData(test_session, chianti_species=["He 2", "N 6"])
    return atom_data


@pytest.fixture
def basic_atom_df_prepared(atom_data):
    return atom_data.basic_atom_df_prepared


@pytest.fixture
def ionization_df_prepared(atom_data):
    return atom_data.ionization_df_prepared


@pytest.fixture
def levels_df_prepared(atom_data):
    return atom_data.levels_df_prepared


@pytest.fixture
def lines_df_prepared(atom_data):
    return atom_data.lines_df_prepared


@pytest.fixture
def collisions_df_prepared(atom_data):
    return atom_data.collisions_df_prepared


@pytest.fixture
def macro_atom_df_prepared(atom_data):
    return atom_data.macro_atom_df_prepared


@pytest.fixture
def macro_atom_ref_df_prepared(atom_data):
    return atom_data.macro_atom_ref_df_prepared


@pytest.fixture
def hdf_store(request, data_dir, atom_data):
    hdf_path = os.path.join(data_dir, "test_hdf.hdf5")
    atom_data.to_hdf(hdf_path)
    hdf_store = pd.HDFStore(hdf_path)

    def fin():
        hdf_store.close()
        os.remove(hdf_path)
    request.addfinalizer(fin)

    return hdf_store


@with_test_db
@pytest.mark.parametrize("atomic_number, exp_mass", [
    (2, 4.002602),
    (11, 22.98976928)
])
def test_prepare_basic_atom_df(basic_atom_df_prepared, atomic_number, exp_mass):
    assert_almost_equal(basic_atom_df_prepared.loc[atomic_number]["mass"], exp_mass)


@with_test_db
def test_prepare_basic_atom_df_max_atomic_number(test_session):
    atom_data = AtomData(test_session, basic_atom_max_atomic_number=15)
    basic_atom_df = atom_data.basic_atom_df_prepared
    basic_atom_df.reset_index(inplace=True)
    assert basic_atom_df["atomic_number"].max() == 15


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, exp_ioniz_energy", [
    (8, 6, 138.1189),
    (11, 1,  5.1390767)
])
def test_prepare_ionizatinon_df(ionization_df_prepared, atomic_number, ion_number, exp_ioniz_energy):
    assert_almost_equal(ionization_df_prepared.loc[(atomic_number, ion_number)]["ionization_energy"],
                        exp_ioniz_energy)


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number, exp_energy",[
    (7, 5, 7, 3991860.0 * u.Unit("cm-1")),
    (4, 2, 2, 981177.5 * u.Unit("cm-1"))
])
def test_prepare_levels_df(levels_df_prepared, atomic_number, ion_number, level_number, exp_energy):
    energy = levels_df_prepared.loc[(atomic_number, ion_number, level_number)]["energy"] * u.eV
    energy = energy.to(u.Unit("cm-1"), equivalencies=u.spectral())
    assert_quantity_allclose(energy, exp_energy)


@with_test_db
def test_create_levels_df_wo_chianti_species(test_session):
    atom_data = AtomData(test_session)
    levels_df = atom_data.levels_df
    chianti_ds_id = test_session.query(DataSource.data_source_id).\
        filter(DataSource.short_name=="chianti_v8.0.2").scalar()
    assert all(levels_df["ds_id"]!=chianti_ds_id)


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number_lower, level_number_upper, exp_wavelength",[
    (7, 5, 0, 1, 29.5343 * u.Unit("angstrom")),
    (4, 2, 0, 3, 10.1693 * u.Unit("angstrom"))
])
def test_prepare_lines_df(lines_df_prepared, atomic_number, ion_number,
                          level_number_lower, level_number_upper, exp_wavelength):
    wavelength = lines_df_prepared.loc[(atomic_number, ion_number,
                                        level_number_lower, level_number_upper)]["wavelength"] * u.Unit("angstrom")
    assert_quantity_allclose(wavelength, exp_wavelength)


# ToDo: Implement real tests
@with_test_db
def test_prepare_collisions_df(collisions_df_prepared):
    assert True


@with_test_db
def test_prepare_macro_atom_df(macro_atom_df_prepared):
    assert True


@with_test_db
def test_prepare_macro_atom_ref_df(macro_atom_ref_df_prepared):
    assert True


@with_test_db
def test_atom_data_to_hdf_collisions_df_attrs(hdf_store):
    collisions_temperatures = hdf_store.get_storer("collisions_df").attrs["temperatures"]
    assert_allclose(collisions_temperatures, np.linspace(2000, 50000, 20))