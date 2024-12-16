import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os
import tempfile
from clean_data import PopulationDataPreprocessor


@pytest.fixture
def sample_population_data():
    """Create sample population data for testing with all years from 1990-2020."""
    years = list(range(1990, 2021))
    base_population = 100000
    base_women = 50000
    birth_rate = 20
    death_rate = 1000

    data = {
        "Unnamed: 0": [
            "US Population",
            "Number of women in US",
            "Crude birth rate per 1,000 women",
            "Population level Death Rates per 100,000 people",
        ]
    }

    # add data for each year with small increments
    for year in years:
        year_str = str(year)
        increment = (year - 1990) * 0.02  # 2% increase per year
        data[year_str] = [
            int(base_population * (1 + increment)),  # Population increases
            int(base_women * (1 + increment)),  # Women population increases
            birth_rate * (1 - increment * 0.5),  # Birth rate slowly decreases
            death_rate * (1 - increment * 0.2),  # Death rate slowly decreases
        ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        pd.DataFrame(data).to_csv(f.name, index=False)
        return f.name


@pytest.fixture
def sample_viral_data():
    """Create sample viral suppression data for testing with all years from 1990-2020."""
    years = [1994] + list(range(1997, 2021))
    data = {"Unnamed: 0": ["Viral Suppression Proportion"]}

    # generate realistic viral suppression proportions
    for year in years:
        year_str = str(year)
        if year < 1995:
            # very low suppression pre-1995
            value = 0.01 + (year - 1990) * 0.005
        elif year < 2000:
            # increasing suppression 1995-2000
            value = 0.05 + (year - 1995) * 0.03
        elif year < 2010:
            # steady increase 2000-2010
            value = 0.20 + (year - 2000) * 0.02
        else:
            # matching some CDC data points and interpolating between them
            if year == 2011:
                value = 0.30
            elif year == 2016:
                value = 0.53
            elif year == 2019:
                value = 0.66
            else:
                value = 0.30 + (year - 2011) * 0.05

        data[year_str] = [round(value, 3)]

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        pd.DataFrame(data).to_csv(f.name, index=False)
        return f.name


@pytest.fixture
def preprocessor(sample_population_data, sample_viral_data):
    """Create a preprocessor instance with sample data."""
    return PopulationDataPreprocessor(sample_population_data, sample_viral_data)


def test_load_population_data(preprocessor):
    """Test population data loading and preprocessing."""
    df = preprocessor.load_population_data()

    assert isinstance(df, pd.DataFrame)
    assert "Year" in df.columns
    assert "Number of Births" in df.columns
    assert "Number of Deaths" in df.columns
    assert len(df) == 31  # 1990-2020 (31 years)

    # test calculations for first year (1990)
    first_row = df.iloc[0]
    expected_births = 50000 * (20 / 1000)
    expected_deaths = 100000 * (1000 / 100000)
    assert np.isclose(first_row["Number of Births"], expected_births)
    assert np.isclose(first_row["Number of Deaths"], expected_deaths)


def test_load_viral_data(preprocessor):
    """Test viral data loading and preprocessing."""
    df = preprocessor.load_viral_data()

    assert isinstance(df, pd.DataFrame)
    assert "Year" in df.columns
    assert "Viral Suppression Proportion" in df.columns
    assert df["Year"].dtype == np.int64
    assert len(df) == 25  # 1997-2020 + 1994 (25 years)

    # test specific CDC data points
    assert np.isclose(
        df[df["Year"] == 2011]["Viral Suppression Proportion"].iloc[0], 0.30
    )
    assert np.isclose(
        df[df["Year"] == 2016]["Viral Suppression Proportion"].iloc[0], 0.53
    )
    assert np.isclose(
        df[df["Year"] == 2019]["Viral Suppression Proportion"].iloc[0], 0.66
    )


def test_logistic_function():
    """Test the static logistic function calculation."""
    result = PopulationDataPreprocessor._logistic_function(2011, L=1, x0=0, k=1)
    assert np.isclose(result, 0.5)


def test_fit_logistic_curve(preprocessor):
    """Test logistic curve fitting."""
    func, params = preprocessor.fit_logistic_curve()

    assert callable(func)
    assert len(params) == 3
    assert all(isinstance(p, float) for p in params)


def test_predict_early_viral_suppression(preprocessor):
    """Test prediction of early viral suppression values."""
    preprocessor.load_viral_data()
    df = preprocessor.predict_early_viral_suppression([1995, 1996])

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 27  # 1994-2020 (26 years)
    assert all(year in df["Year"].values for year in [1995, 1996])
    assert all(df["Viral Suppression Proportion"].notna())


def test_adjust_viral_suppression(preprocessor):
    """Test viral suppression adjustment."""
    preprocessor.load_viral_data()
    preprocessor.fit_logistic_curve()
    preprocessor.predict_early_viral_suppression([1995, 1996])
    df = preprocessor.adjust_viral_suppression()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 32  # 1990-2021 (32 years)
    assert all(df["Year"] == list(range(1990, 2022)))
    assert all(df["Viral Suppression Proportion"].notna())

    # test specific CDC data points are preserved
    assert np.isclose(
        df[df["Year"] == 2011]["Viral Suppression Proportion"].iloc[0], 0.30
    )
    assert np.isclose(
        df[df["Year"] == 2016]["Viral Suppression Proportion"].iloc[0], 0.53
    )
    assert np.isclose(
        df[df["Year"] == 2019]["Viral Suppression Proportion"].iloc[0], 0.66
    )


def test_process_data(preprocessor):
    """Test the complete data processing workflow."""
    final_df = preprocessor.process_data()

    assert isinstance(final_df, pd.DataFrame)
    assert "Viral Suppression Proportion" in final_df.columns
    assert all(final_df["Viral Suppression Proportion"].notna())
    assert len(final_df) == 31  # 1990-2020 (31 years)


def test_main_execution(sample_population_data, sample_viral_data, tmp_path):
    """Test the main execution function."""
    output_file = tmp_path / "updated_population_data.csv"

    preprocessor = PopulationDataPreprocessor(sample_population_data, sample_viral_data)
    final_df = preprocessor.process_data()
    final_df.to_csv(output_file, index=False)

    assert output_file.exists()
    result_df = pd.read_csv(output_file)
    assert "Viral Suppression Proportion" in result_df.columns
    assert len(result_df) == 31  # 1990-2020 (31 years)


def test_error_handling_invalid_files():
    """Test error handling for invalid file paths."""
    with pytest.raises(FileNotFoundError):
        preprocessor = PopulationDataPreprocessor("invalid.csv", "also_invalid.csv")
        preprocessor.process_data()


# clean up temporary files after tests
@pytest.fixture(autouse=True)
def cleanup(sample_population_data, sample_viral_data):
    yield
    os.unlink(sample_population_data)
    os.unlink(sample_viral_data)
