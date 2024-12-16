import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Matplotlib is currently using agg.*')

import matplotlib
matplotlib.use('Agg')  # Must be before importing plt
import pytest
import pandas as pd
import numpy as np
from HIVModel import HIVOpen
import matplotlib.pyplot as plt

# Keep existing warnings filter for runtime warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Mocks the savefig function to prevent file generation during tests
@pytest.fixture(autouse=True)
def no_file_generation(monkeypatch):
    """Prevent any file generation during tests."""
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return pd.DataFrame({
        'Year': [2010, 2011, 2012],
        'Population': [300000000, 301000000, 302000000],
        'New_HIV_Infections': [50000, 48000, 46000],
        'Total_HIV_Cases': [1000000, 1020000, 1040000],
        'HIV_Deaths': [15000, 14500, 14000],
        'Births': [4000000, 4010000, 4020000],
        'Natural_Death_Rate': [800, 790, 780],
        'Viral_Suppression': [0.50, 0.52, 0.54]
    })

@pytest.fixture
def column_mapping():
    """Define column mapping for the model."""
    return {
        'year': 'Year',
        'population': 'Population',
        'new_infections': 'New_HIV_Infections',
        'total_hiv': 'Total_HIV_Cases',
        'deaths_hiv': 'HIV_Deaths',
        'number_of_births': 'Births',
        'natural_death_rate': 'Natural_Death_Rate',
        'viral_suppression': 'Viral_Suppression'
    }

@pytest.fixture
def hiv_model():
    """Create a basic HIV model instance."""
    return HIVOpen(beta=0.0001, sigma=0.01, nu=0.1, mu=0.02, delta=0.03, gamma=0.01)

def test_model_initialization(hiv_model):
    """Test that the model initializes with correct parameters."""
    assert hiv_model.beta == 0.0001
    assert hiv_model.sigma == 0.01
    assert hiv_model.nu == 0.1
    assert hiv_model.mu == 0.02
    assert hiv_model.delta == 0.03
    assert hiv_model.gamma == 0.01
    assert hiv_model.results is None
    assert hiv_model.columns is None
    assert len(hiv_model.param_samples) == 0

def test_load_data_with_dataframe(hiv_model, sample_data, column_mapping):
    """Test loading data from a DataFrame."""
    hiv_model.load_data(sample_data, column_mapping)
    assert isinstance(hiv_model.data, pd.DataFrame)
    assert hiv_model.N == 300000000
    assert hiv_model.I0 == 50000

def test_load_data_with_csv(hiv_model, sample_data, column_mapping, tmp_path):
    """Test loading data from a CSV file."""
    csv_path = tmp_path / "test_data.csv"
    sample_data.to_csv(csv_path, index=False)
    hiv_model.load_data(str(csv_path), column_mapping)
    assert isinstance(hiv_model.data, pd.DataFrame)

def test_load_data_invalid_input(hiv_model, column_mapping):
    """Test loading data with invalid input."""
    with pytest.raises(ValueError, match="Data must be a file path or pandas DataFrame"):
        hiv_model.load_data([1, 2, 3], column_mapping)

def test_load_data_missing_columns(hiv_model, sample_data):
    """Test loading data with missing columns."""
    invalid_mapping = {'invalid_column': 'NonExistentColumn'}
    with pytest.raises(ValueError, match="Missing columns in the dataset"):
        hiv_model.load_data(sample_data, invalid_mapping)

def test_simulate_without_data(hiv_model):
    """Test simulation without loading data first."""
    with pytest.raises(ValueError, match="Data must be loaded before simulation"):
        hiv_model.simulate(5)

def test_simulate_with_data(hiv_model, sample_data, column_mapping):
    """Test basic simulation functionality."""
    hiv_model.load_data(sample_data, column_mapping)
    results = hiv_model.simulate(5)
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 5
    expected_columns = {'Year', 'Exposed', 'Infectious', 'Recovered (ART)', 'Deaths (HIV)'}
    assert all(col in results.columns for col in expected_columns)

def test_simulate_with_custom_parameters(hiv_model, sample_data, column_mapping):
    """Test simulation with custom parameters."""
    hiv_model.load_data(sample_data, column_mapping)
    custom_params = [0.0002, 0.02, 0.2, 0.03, 0.04, 0.02]
    results = hiv_model.simulate(5, params=custom_params)
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 5

def test_simulate_with_custom_initial_conditions(hiv_model, sample_data, column_mapping):
    """Test simulation with custom initial conditions."""
    hiv_model.load_data(sample_data, column_mapping)
    initial_conditions = [290000000, 1000, 40000, 500000, 10000]
    results = hiv_model.simulate(5, initial_conditions=initial_conditions)
    assert isinstance(results, pd.DataFrame)
    assert len(results) == 5

def test_calibrate_parameters(hiv_model, sample_data, column_mapping):
    """Test parameter calibration."""
    hiv_model.load_data(sample_data, column_mapping)
    hiv_model.calibrate_parameters(num_bootstrap=2)  # using small number for testing
    assert len(hiv_model.param_samples) > 0
    assert all(isinstance(param, np.ndarray) for param in hiv_model.param_samples)

def test_simulate_with_uncertainty_without_calibration(hiv_model, sample_data, column_mapping):
    """Test that simulate_with_uncertainty raises error without calibration."""
    hiv_model.load_data(sample_data, column_mapping)
    with pytest.raises(ValueError, match="No parameter samples found"):
        hiv_model.simulate_with_uncertainty(5)

def test_simulate_with_uncertainty_after_calibration(hiv_model, sample_data, column_mapping):
    """Test simulate_with_uncertainty after calibration."""
    hiv_model.load_data(sample_data, column_mapping)
    hiv_model.calibrate_parameters(num_bootstrap=2)  # using small number for testing
    try:
        hiv_model.simulate_with_uncertainty(5)
    except Exception as e:
        pytest.fail(f"simulate_with_uncertainty raised unexpected exception: {e}")
