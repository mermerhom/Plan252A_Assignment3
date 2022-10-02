import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

np.random.seed(252)

present_trips = np.array([[0, 300, 300],
                          [100, 0, 450],
                          [240, 200, 0]])
total_present_trips = np.transpose(np.vstack((present_trips.sum(axis=1), present_trips.sum(axis=0))))
future_trips = np.array([[900, 700],
                   [630, 600],
                   [570, 800]])
sum_total_future_trips = future_trips.sum(axis=0)[0]
growth_rate = future_trips/total_present_trips

"""Average Factor Method"""
def afm(present_trips, future_trips, growth_rate, afm_iteration):
    afm_iteration = afm_iteration
    present_trips = present_trips
    future_trips = future_trips
    growth_rate = growth_rate
    afm_base_growth_rate = np.zeros((growth_rate.shape[-2] * (afm_iteration + 2), growth_rate.shape[-1]))               ####
    afm_array = np.zeros(((afm_iteration + 2) * present_trips.shape[-2], present_trips.shape[-1]))                  ####
    afm_growth_rate = np.zeros_like(afm_array, dtype=float)
    afm_recharging_array = np.zeros((present_trips.shape[-2], present_trips.shape[-1]))
    afm_recharging_growth_rate = np.zeros_like(afm_recharging_array, dtype=float)
    for i in range(present_trips.shape[-2]):
        for j in range(present_trips.shape[-1]):
            for k in range(growth_rate.shape[-1]):
                afm_base_growth_rate[i, k] = growth_rate[i, k]
            afm_array[i, j] = present_trips[i, j]
    for t in range((afm_iteration) + 1):                ####
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                afm_recharging_growth_rate[i, j] = (afm_base_growth_rate[(present_trips.shape[-2] * t) + i, 0] +
                                                    afm_base_growth_rate[
                                                        (present_trips.shape[-2] * t) + j, 1]) * 0.5  # not advanced
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                afm_recharging_array[i, j] = afm_array[(present_trips.shape[-2] * t) + i, j]  # not advanced
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                afm_growth_rate[(present_trips.shape[-2] * t) + i, j] = afm_recharging_growth_rate[i, j]  # not advanced
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                afm_array[(present_trips.shape[-2] * (t + 1)) + i, j] = afm_recharging_array[i, j] * \
                                                                        afm_recharging_growth_rate[i, j]  # advanced
                new_recharging_array_sum = np.transpose(np.vstack(
                    (afm_array[(present_trips.shape[-2] * (t + 1)): (present_trips.shape[-2] * (t + 2)),  # advanced
                     0: (present_trips.shape[-1] + 1)].sum(axis=1),
                     afm_array[(present_trips.shape[-2] * (t + 1)): (present_trips.shape[-2] * (t + 2)),
                     0: (present_trips.shape[-1] + 1)].sum(axis=0))))
        new_recharging_growth_rate = future_trips / new_recharging_array_sum
        for i in range(present_trips.shape[-2]):
            for j in range(growth_rate.shape[-1]):
                afm_base_growth_rate[(present_trips.shape[-2] * (t + 1)) + i, j] = new_recharging_growth_rate[
                    i, j]  # advanced
    mock_convergence = np.ones_like(present_trips, dtype=float)
    afm_convergence = afm_growth_rate[
                      ((present_trips.shape[-2] * (afm_iteration + 1)) - present_trips.shape[-2]): present_trips.shape[
                                                                                                 -2] * (afm_iteration + 1),             ####
                      0: present_trips.shape[-1]]
    return np.amax(np.absolute(afm_convergence - mock_convergence)) * 100


"""Detroit Method"""
def dm(present_trips, future_trips, growth_rate, dm_iteration):
    dm_iteration = dm_iteration
    present_trips = present_trips
    future_trips = future_trips
    growth_rate = growth_rate
    dm_base_growth_rate = np.zeros((growth_rate.shape[-2] * (dm_iteration + 2), growth_rate.shape[-1]))             ####
    dm_array = np.zeros(((dm_iteration + 2) * present_trips.shape[-2], present_trips.shape[-1]))                ####
    dm_growth_rate = np.zeros_like(dm_array, dtype=float)
    dm_recharging_array = np.zeros((present_trips.shape[-2], present_trips.shape[-1]))
    dm_recharging_growth_rate = np.zeros_like(dm_recharging_array, dtype=float)
    for i in range(present_trips.shape[-2]):
        for j in range(present_trips.shape[-1]):
            for k in range(growth_rate.shape[-1]):
                dm_base_growth_rate[i, k] = growth_rate[i, k]
            dm_array[i, j] = present_trips[i, j]
    for t in range((dm_iteration) + 1):             ####
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                current_recharging_array_sum = dm_array[(present_trips.shape[-2] * t): (present_trips.shape[-2] * (t + 1)),  # not advanced
                                                                           0 : (present_trips.shape[-1] + 1)].sum(axis=1).sum(axis=0)
        current_total_growth_rate = (future_trips.sum(axis=0)[0])/current_recharging_array_sum
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                dm_recharging_growth_rate[i, j] = (dm_base_growth_rate[(present_trips.shape[-2] * t) + i, 0] *
                                                   dm_base_growth_rate[(present_trips.shape[-2] * t) + j, 1]) * (1/current_total_growth_rate)    # not advanced
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                dm_recharging_array[i, j] = dm_array[(present_trips.shape[-2] * t) + i, j]  # not advanced
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                dm_growth_rate[(present_trips.shape[-2] * t) + i, j] = dm_recharging_growth_rate[i, j]    # not advanced
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                dm_array[(present_trips.shape[-2] * (t + 1)) + i, j] = dm_recharging_array[i, j] * dm_recharging_growth_rate[i, j]    # advanced
                new_recharging_array_sum = np.transpose(np.vstack((dm_array[(present_trips.shape[-2] * (t + 1)): (present_trips.shape[-2] * (t + 2)),  # advanced
                                                                           0 : (present_trips.shape[-1] + 1)].sum(axis=1),
                                                                   dm_array[(present_trips.shape[-2] * (t + 1)): (present_trips.shape[-2] * (t + 2)),
                                                                           0 : (present_trips.shape[-1] + 1)].sum(axis=0))))
        new_recharging_growth_rate = future_trips/new_recharging_array_sum
        for i in range(present_trips.shape[-2]):
            for j in range(growth_rate.shape[-1]):
                dm_base_growth_rate[(present_trips.shape[-2] * (t + 1)) + i, j] = new_recharging_growth_rate[i, j]    # advanced
    mock_convergence = np.ones((present_trips.shape[-2], dm_base_growth_rate.shape[-1]), dtype=float)
    dm_convergence = dm_base_growth_rate[((present_trips.shape[-2] * (dm_iteration + 1)) - present_trips.shape[-2]):
                                         present_trips.shape[-2] * (dm_iteration + 1), 0: dm_base_growth_rate.shape[-1]]              ####
    return np.amax(np.absolute(dm_convergence - mock_convergence)) * 100


"""Fratar Method"""
def fm(present_trips, future_trips, growth_rate, fm_iteration):
    fm_iteration = fm_iteration
    present_trips = present_trips
    future_trips = future_trips
    growth_rate = growth_rate
    fm_base_growth_rate = np.zeros((growth_rate.shape[-2] * (fm_iteration + 2), growth_rate.shape[-1]))             ####
    fm_array = np.zeros(((fm_iteration + 2) * present_trips.shape[-2], present_trips.shape[-1]))                ####
    fm_growth_rate = np.zeros_like(fm_array, dtype=float)
    fm_recharging_array = np.zeros((present_trips.shape[-2], present_trips.shape[-1]))
    fm_recharging_growth_rate = np.zeros_like(fm_recharging_array, dtype=float)
    fm_current_base_growth_rate = np.zeros_like(present_trips, dtype=float)
    for i in range(present_trips.shape[-2]):
        for j in range(present_trips.shape[-1]):
            for k in range(growth_rate.shape[-1]):
                fm_base_growth_rate[i, k] = growth_rate[i, k]
            fm_array[i, j] = present_trips[i, j]
    for t in range((fm_iteration) + 1):             ####
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                fm_recharging_array[i, j] = fm_array[(present_trips.shape[-2] * t) + i, j]  # not advanced
        fm_initial_current_base_growth_rate = fm_base_growth_rate[(present_trips.shape[-2] * t): (present_trips.shape[-2] * (t + 1)), 0: growth_rate.shape[-1]]  # not advanced
        attraction_growth_rate_sum = np.matmul(fm_recharging_array, fm_initial_current_base_growth_rate)
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                fm_current_base_growth_rate[i, j] = (attraction_growth_rate_sum[:, 1])[i]
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                fm_recharging_growth_rate[i, j] = (fm_base_growth_rate[(present_trips.shape[-2] * t) + i, 0] *  # not advanced
                                                   fm_base_growth_rate[(present_trips.shape[-2] * t) + j, 1]) * \
                                                  (fm_recharging_array.sum(axis=1)[i]) / fm_current_base_growth_rate[i, 1]

        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                fm_growth_rate[(present_trips.shape[-2] * t) + i, j] = fm_recharging_growth_rate[i, j]    # not advanced
        for i in range(present_trips.shape[-2]):
            for j in range(present_trips.shape[-1]):
                fm_array[(present_trips.shape[-2] * (t + 1)) + i, j] = fm_recharging_array[i, j] * fm_recharging_growth_rate[i, j]    # advanced
                new_recharging_array_sum = np.transpose(np.vstack((fm_array[(present_trips.shape[-2] * (t + 1)): (present_trips.shape[-2] * (t + 2)),  # advanced
                                                                           0 : (present_trips.shape[-1] + 1)].sum(axis=1),
                                                                   fm_array[(present_trips.shape[-2] * (t + 1)): (present_trips.shape[-2] * (t + 2)),
                                                                           0 : (present_trips.shape[-1] + 1)].sum(axis=0))))
        new_recharging_growth_rate = future_trips/new_recharging_array_sum
        for i in range(present_trips.shape[-2]):
            for j in range(growth_rate.shape[-1]):
                fm_base_growth_rate[(present_trips.shape[-2] * (t + 1)) + i, j] = new_recharging_growth_rate[i, j]    # advanced
    mock_convergence = np.ones_like(present_trips, dtype=float)
    fm_convergence = fm_growth_rate[((present_trips.shape[-2] * (fm_iteration + 1)) - present_trips.shape[-2]):
                                    present_trips.shape[-2] * (fm_iteration + 1), 0: present_trips.shape[-1]]             ####
    return np.amax(np.absolute(fm_convergence - mock_convergence)) * 100


"""Comparison of Trip Distribution Methods (Random Normal Data)"""
"""First Comparison: Zones = 6, Initial Mean = 480, Initial Variance = 12.5% of Initial Mean, Mean Increase = 60%, Variance Increase = 15%"""
mock_1_present = np.random.normal(loc=480, scale=60, size=25).astype(int).reshape((5, 5))
np.fill_diagonal(mock_1_present, 0)
mock_1_future = np.random.normal(loc=768, scale=69, size=10).astype(int).reshape((5, 2))
total_present_trips_mock_1 = np.transpose(np.vstack((mock_1_present.sum(axis=1), mock_1_present.sum(axis=0))))
growth_rate_mock_1 = mock_1_future / total_present_trips_mock_1

"""Second Comparison: Zones = 10, Initial Mean = 480, Initial Variance = 12.5% of Initial Mean, Mean Increase = 60%, Variance Increase = 15%"""
mock_2_present = np.random.normal(loc=480, scale=60, size=100).astype(int).reshape((10, 10))
np.fill_diagonal(mock_2_present, 0)
mock_2_future = np.random.normal(loc=768, scale=69, size=20).astype(int).reshape((10, 2))
total_present_trips_mock_2 = np.transpose(np.vstack((mock_2_present.sum(axis=1), mock_2_present.sum(axis=0))))
growth_rate_mock_2 = mock_2_future / total_present_trips_mock_2

"""Third Comparison: Zones = 15, Initial Mean = 480, Initial Variance = 12.5% of Initial Mean, Mean Increase = 60%, Variance Increase = 15%"""
mock_3_present = np.random.normal(loc=480, scale=60, size=225).astype(int).reshape((15, 15))
np.fill_diagonal(mock_3_present, 0)
mock_3_future = np.random.normal(loc=768, scale=69, size=30).astype(int).reshape((15, 2))
total_present_trips_mock_3 = np.transpose(np.vstack((mock_3_present.sum(axis=1), mock_3_present.sum(axis=0))))
growth_rate_mock_3 = mock_3_future / total_present_trips_mock_3

comparison_df = pd.DataFrame({
    "Method" : np.hstack((np.array(["Average Factor Method"]*15),
                          np.array(["Detroit Method"]*15),
                          np.array(["Fratar Method"]*15),
           np.array(["Average Factor Method"]*15),
           np.array(["Detroit Method"]*15),
           np.array(["Fratar Method"]*15),
           np.array(["Average Factor Method"]*15),
           np.array(["Detroit Method"]*15),
           np.array(["Fratar Method"]*15))),
    "Number of Zones" : np.hstack((np.array(["5"]*45), np.array(["10"]*45), np.array(["15"]*45))),
    "Iteration" : np.array(["3", "4", "5", "6", "7", "8", "9", "10", "20", "50", "70", "75", "100", "250", "1000"]*9),
    "Convergence" : [afm(mock_1_present, mock_1_future, growth_rate_mock_1, 3),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 4),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 5),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 6),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 7),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 8),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 9),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 10),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 20),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 50),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 70),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 75),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 100),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 250),
                     afm(mock_1_present, mock_1_future, growth_rate_mock_1, 1000),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 3),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 4),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 5),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 6),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 7),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 8),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 9),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 10),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 20),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 50),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 70),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 75),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 100),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 250),
                     dm(mock_1_present, mock_1_future, growth_rate_mock_1, 1000),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 3),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 4),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 5),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 6),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 7),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 8),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 9),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 10),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 20),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 50),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 70),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 75),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 100),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 250),
                     fm(mock_1_present, mock_1_future, growth_rate_mock_1, 1000),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 3),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 4),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 5),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 6),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 7),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 8),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 9),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 10),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 20),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 50),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 70),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 75),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 100),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 250),
                     afm(mock_2_present, mock_2_future, growth_rate_mock_2, 1000),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 3),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 4),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 5),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 6),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 7),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 8),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 9),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 10),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 20),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 50),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 70),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 75),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 100),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 250),
                     dm(mock_2_present, mock_2_future, growth_rate_mock_2, 1000),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 3),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 4),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 5),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 6),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 7),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 8),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 9),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 10),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 20),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 50),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 70),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 75),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 100),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 250),
                     fm(mock_2_present, mock_2_future, growth_rate_mock_2, 1000),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 3),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 4),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 5),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 6),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 7),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 8),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 9),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 10),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 20),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 50),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 70),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 75),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 100),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 250),
                     afm(mock_3_present, mock_3_future, growth_rate_mock_3, 1000),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 3),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 4),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 5),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 6),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 7),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 8),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 9),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 10),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 20),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 50),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 70),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 75),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 100),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 250),
                     dm(mock_3_present, mock_3_future, growth_rate_mock_3, 1000),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 3),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 4),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 5),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 6),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 7),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 8),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 9),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 10),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 20),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 50),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 70),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 75),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 100),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 250),
                     fm(mock_3_present, mock_3_future, growth_rate_mock_3, 1000)
                     ]
}, columns=["Method", "Number of Zones", "Iteration", "Convergence"])
sns.set_style("dark", {"axes.facecolor" : "#F0F8FF"})
fig, comparison_df_plot = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
sns.lineplot(data=comparison_df, x="Iteration", y="Convergence", hue="Method", palette=["#1874CD", "#008B00", "#CD6889"],
             linewidth=3, markers=["^", "o", "s"], markersize=8, style="Number of Zones")
sns.despine(fig=fig, ax=comparison_df_plot, top=True, right=True, offset=10)
comparison_df_plot.set_title("Trip Distribution Methods Comparison (Forwarded)\nNumber of Zones = 5, 10 and 15, Initial Mean of Trips = 480,"
                               "\nInitial Variance = 12.5% of Initial Trip Mean,"
                               "\nTrip Mean Increase for Future Trips = 60%, Trip Variance Increase for Future Trips = 15%",
                               fontdict={"font": "Nirmala UI", "color": "black", "fontsize": 10, "weight": "bold"})
comparison_df_plot.set_xlabel("Number of Iterations", fontdict={"font": "Nirmala UI", "color": "black", "fontsize": 10, "weight": "bold"})
comparison_df_plot.set_ylabel("Convergence (%)", fontdict={"font": "Nirmala UI", "color": "black", "fontsize": 10, "weight": "bold"})
comparison_1_df_plot_legend = comparison_df_plot.legend(labelspacing=1, edgecolor = "#1874CD", fontsize=10, frameon=False, bbox_to_anchor=(0.95, 0.92),
                                                        prop={"family":"Nirmala UI", "weight": "bold", "style": "normal", "size": 10})#,
                                                        #title="Trip Distribution Methods")
title_comparison_1_df_plot = comparison_1_df_plot_legend.get_title()
title_comparison_1_df_plot.set_family("Nirmala UI")
title_comparison_1_df_plot.set_weight("bold")
title_comparison_1_df_plot.set_size(10)
plt.tight_layout()


"""Method Comparison for Data in  Assignment 3 """
present_trips = np.array([[0, 300, 300],
                          [100, 0, 450],
                          [240, 200, 0]])
total_present_trips = np.transpose(np.vstack((present_trips.sum(axis=1), present_trips.sum(axis=0))))
future_trips = np.array([[900, 700],
                   [630, 600],
                   [570, 800]])
sum_total_future_trips = future_trips.sum(axis=0)[0]
growth_rate = future_trips/total_present_trips

comparison_df_assignment = pd.DataFrame({
    "Method" : np.hstack((np.array(["Average Factor Method"]*15),
                          np.array(["Detroit Method"]*15),
                          np.array(["Fratar Method"]*15))),
    "Iteration" : np.array(["3", "4", "5", "10", "20", "50", "70", "75", "100", "250", "500", "1000", "2500", "5000", "10000"]*3),
    "Convergence" : [afm(present_trips, future_trips, growth_rate, 3),
                     afm(present_trips, future_trips, growth_rate, 4),
                     afm(present_trips, future_trips, growth_rate, 5),
                     afm(present_trips, future_trips, growth_rate, 10),
                     afm(present_trips, future_trips, growth_rate, 20),
                     afm(present_trips, future_trips, growth_rate, 50),
                     afm(present_trips, future_trips, growth_rate, 70),
                     afm(present_trips, future_trips, growth_rate, 75),
                     afm(present_trips, future_trips, growth_rate, 100),
                     afm(present_trips, future_trips, growth_rate, 250),
                     afm(present_trips, future_trips, growth_rate, 500),
                     afm(present_trips, future_trips, growth_rate, 1000),
                     afm(present_trips, future_trips, growth_rate, 2500),
                     afm(present_trips, future_trips, growth_rate, 5000),
                     afm(present_trips, future_trips, growth_rate, 10000),
                     dm(present_trips, future_trips, growth_rate, 3),
                     dm(present_trips, future_trips, growth_rate, 4),
                     dm(present_trips, future_trips, growth_rate, 5),
                     dm(present_trips, future_trips, growth_rate, 10),
                     dm(present_trips, future_trips, growth_rate, 20),
                     dm(present_trips, future_trips, growth_rate, 50),
                     dm(present_trips, future_trips, growth_rate, 70),
                     dm(present_trips, future_trips, growth_rate, 75),
                     dm(present_trips, future_trips, growth_rate, 100),
                     dm(present_trips, future_trips, growth_rate, 250),
                     dm(present_trips, future_trips, growth_rate, 500),
                     dm(present_trips, future_trips, growth_rate, 1000),
                     dm(present_trips, future_trips, growth_rate, 2500),
                     dm(present_trips, future_trips, growth_rate, 5000),
                     dm(present_trips, future_trips, growth_rate, 10000),
                     fm(present_trips, future_trips, growth_rate, 3),
                     fm(present_trips, future_trips, growth_rate, 4),
                     fm(present_trips, future_trips, growth_rate, 5),
                     fm(present_trips, future_trips, growth_rate, 10),
                     fm(present_trips, future_trips, growth_rate, 20),
                     fm(present_trips, future_trips, growth_rate, 50),
                     fm(present_trips, future_trips, growth_rate, 70),
                     fm(present_trips, future_trips, growth_rate, 75),
                     fm(present_trips, future_trips, growth_rate, 100),
                     fm(present_trips, future_trips, growth_rate, 250),
                     fm(present_trips, future_trips, growth_rate, 500),
                     fm(present_trips, future_trips, growth_rate, 1000),
                     fm(present_trips, future_trips, growth_rate, 2500),
                     fm(present_trips, future_trips, growth_rate, 5000),
                     fm(present_trips, future_trips, growth_rate, 10000),
                     ]
}, columns=["Method", "Iteration", "Convergence"])
sns.set_style("dark", {"axes.facecolor" : "#F0F8FF"})
fig_2, comparison_df_plot_assignment = plt.subplots(nrows=1, ncols=1, figsize=(15, 8))
sns.lineplot(data=comparison_df_assignment, x="Iteration", y="Convergence", hue="Method", palette=["#1874CD", "#008B00", "#CD6889"],
             linewidth=3, marker="o", markersize=8)
sns.despine(fig=fig_2, ax=comparison_df_plot_assignment, top=True, right=True, offset=10)
comparison_df_plot_assignment.set_title("Trip Distribution Methods Comparison (Forwarded)\nData: Assignment",
                                        fontdict={"font": "Nirmala UI", "color": "black", "fontsize": 10, "weight": "bold"})
comparison_df_plot_assignment.set_xlabel("Number of Iterations", fontdict={"font": "Nirmala UI", "color": "black", "fontsize": 10, "weight": "bold"})
comparison_df_plot_assignment.set_ylabel("Convergence (%)", fontdict={"font": "Nirmala UI", "color": "black", "fontsize": 10, "weight": "bold"})
comparison_df_plot_legend_assignment = comparison_df_plot_assignment.legend(labelspacing=1, edgecolor ="#1874CD", fontsize=10, frameon=False, bbox_to_anchor=(0.95, 0.9),
                                                                            prop={"family":"Nirmala UI", "weight": "bold", "style": "normal", "size": 10},
                                                                            title="Trip Distribution Methods")
title_comparison_df_plot_assignment = comparison_df_plot_legend_assignment.get_title()
title_comparison_df_plot_assignment.set_family("Nirmala UI")
title_comparison_df_plot_assignment.set_weight("bold")
title_comparison_df_plot_assignment.set_size(10)
plt.tight_layout()
plt.show()