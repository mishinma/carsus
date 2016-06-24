import pytest

from carsus.io.output.tardis_op import AtomData
from carsus.model import DataSource
from numpy.testing import assert_almost_equal
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
def basic_atom_df(atom_data):
    return atom_data.prepare_basic_atom_df()


@pytest.fixture
def ionization_df(atom_data):
    return atom_data.prepare_ionization_df()


@pytest.fixture
def levels_df(atom_data):
    return atom_data.prepare_levels_df()


@pytest.fixture
def lines_df(atom_data):
    return atom_data.prepare_lines_df()

@pytest.fixture
def collisions_df(atom_data):
    return atom_data.prepare_collisions_df()

@pytest.fixture
def macro_atom_df(atom_data):
    return atom_data.prepare_macro_atom_df()


@pytest.fixture
def macro_atom_ref_df(atom_data):
    return atom_data.prepare_macro_atom_ref_df()


@with_test_db
@pytest.mark.parametrize("atomic_number, exp_weight", [
    (2, 4.002602),
    (11, 22.98976928)
])
def test_create_basic_atom_df(basic_atom_df, atomic_number, exp_weight):
    assert_almost_equal(basic_atom_df.loc[atomic_number]["weight"],
                        exp_weight)

@with_test_db
def test_create_basic_atom_df_max_atomic_number(atom_data):
    basic_atom_df = atom_data.prepare_basic_atom_df(max_atomic_number=15)
    basic_atom_df.reset_index(inplace=True)
    assert basic_atom_df["atomic_number"].max() == 15


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, exp_ioniz_energy", [
    (8, 6, 138.1189),
    (11, 1,  5.1390767)
])
def test_create_ionizatinon_df(ionization_df, atomic_number, ion_number, exp_ioniz_energy):
    assert_almost_equal(ionization_df.loc[(atomic_number, ion_number)]["ionization_energy"],
                        exp_ioniz_energy)


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number, exp_energy",[
    (7, 6, 7, 3991860.0 * u.Unit("cm-1")),
    (4, 3, 2, 981177.5 * u.Unit("cm-1"))
])
def test_create_levels_df(levels_df, atomic_number, ion_number, level_number, exp_energy):
    energy = levels_df.loc[(atomic_number, ion_number, level_number)]["energy"]*u.eV
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
    (7, 6, 0, 1, 29.5343 * u.Unit("angstrom")),
    (4, 3, 0, 3, 10.1693 * u.Unit("angstrom"))
])
def test_create_lines_df(lines_df, atomic_number, ion_number, level_number_lower, level_number_upper, exp_wavelength):
    wavelength = lines_df.loc[(atomic_number, ion_number,
                               level_number_lower, level_number_upper)]["wavelength"]*u.Unit("angstrom")
    assert_quantity_allclose(wavelength, exp_wavelength)

# ToDo: Implement real tests
@with_test_db
def test_create_collisions_df(collisions_df):
    assert True


@with_test_db
def test_create_macro_atom_df(macro_atom_df):
    assert True

@with_test_db
def test_create_macro_atom_ref_df(macro_atom_ref_df):
    assert True
