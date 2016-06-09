import pytest
import os

from carsus.io.kurucz import GFALLReader, GFALLIngester
from numpy.testing import assert_almost_equal

@pytest.fixture()
def gfall_fname():
    return os.path.join(os.path.dirname(__file__), 'data', 'gf0402.all')


@pytest.fixture()
def gfall_rdr(gfall_fname):
    return GFALLReader(gfall_fname)

@pytest.fixture()
def gfall_ingester(test_session, gfall_fname):
    return GFALLIngester(test_session, gfall_fname)

def test_grall_reader_read_gfall_raw(gfall_rdr):
    gfall_rdr.gfall_raw


@pytest.mark.parametrize("row, wavelength",
                         [(5 , 58.2079),
                          (24, 3242.7095)
                          ])
def test_grall_reader_read_gfall_raw(gfall_rdr, row, wavelength):
    assert_almost_equal(gfall_rdr.gfall_raw.loc[row, "wavelength"], wavelength)

# ToDo: write a test for lower-upper level


def test_gfall_parse_gfall(gfall_rdr):
    gfall_rdr.gfall_df


def test_gfall_extract_levels(gfall_rdr):
    gfall_rdr.levels_df


def test_gfall_extract_lines(gfall_rdr):
    gfall_rdr.lines_df


def test_gfall_ingester_ingest_levels(gfall_ingester):
    gfall_ingester.ingest_levels()