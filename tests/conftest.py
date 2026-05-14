import warnings

from sklearn.exceptions import ConvergenceWarning, UndefinedMetricWarning


def pytest_configure(config):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
