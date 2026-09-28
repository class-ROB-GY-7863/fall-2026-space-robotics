from estimators.kalman import Kalman
from estimators.zero import Zero

# "perfect" feeds back the true state, no estimator
ESTIMATORS = {"perfect": None, "zero": Zero, "kalman": Kalman}
