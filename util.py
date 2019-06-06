"""Functions for use in the notebook"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor


def generate_dataset(sample_size, vehicle_types, add_interaction, random_seed=0):
    """Generate a toy dataset with two factors.

    Args:
        sample_size (int): Number of observations.
        vehicle_types (list of str): List of different vehicle
            types to include in the factor.
        add_interaction (bool): Whether or not an interaction
            term between the two factors should be added.
        random_seed (int, optional): Defaults to zero.

    Returns:
        pandas.DataFrame

    """

    np.random.seed(random_seed)

    age = np.random.uniform(16, 60, size=sample_size)
    vehicle = np.random.randint(0, len(vehicle_types), size=sample_size)
    noise = np.random.normal(0, 0.25, size=sample_size)

    linear_model = 0.8 - 0.1 * age - 0.5 * (vehicle % 3) + noise

    if add_interaction:
        linear_model += 0.000001 * (vehicle % 2) * age ** 3.75

    # The following creates the 'true' conditional means of the
    # distribution, np.exp implying a log link is appropriate
    true_mean = np.exp(linear_model).clip(0, 0.6)

    # Claims will be draws from Poisson dist with appropriate means
    claim_count = np.random.poisson(true_mean, size=sample_size)

    # Use string values for the vehicle factor
    vehicle_map = {i: name for i, name in enumerate(vehicle_types)}
    vehicle = [vehicle_map[x] for x in vehicle]

    return pd.DataFrame(
        {
            "age": age,
            "vehicle": pd.Categorical(vehicle, categories=vehicle_types, ordered=True),
            # "noise": noise,
            "true_mean": true_mean,
            "claim_count": claim_count,
        }
    )


def fit_and_plot_model(
    df,
    vehicles,
    colormap,
    allow_interactions=True,
    n_estimators=80,
    cut_age_into_bands=False,
    n_bands=20,
):

    tmp = df.copy()

    if cut_age_into_bands:
        tmp["age_bands"] = pd.cut(tmp["age"].values, n_bands).codes
        age_col = "age_bands"
    else:
        age_col = "age"

    est = LGBMRegressor(
        objective="poisson", max_depth=allow_interactions + 1, n_estimators=n_estimators
    )

    X = tmp[[age_col, "vehicle"]]
    y = tmp["claim_count"]

    est.fit(X, y)
    
    tmp["pred"] = est.predict(X)

    shap_vals = est.predict(X, pred_contrib=True)
    for i, col in enumerate(X.columns):
        tmp[col + "_relativity"] = np.exp(shap_vals[:, i])

    tmp["base_rate"] = np.exp(shap_vals[:, -1])

    samp = tmp.sample(1_000, random_state=0)
    f, ax = plt.subplots()

    for veh in vehicles:
        samp[samp.vehicle == veh].plot.scatter(
            "age", "true_mean", color=colormap[veh], ax=ax, alpha=0.3
        )
        su = tmp[tmp.vehicle == veh].groupby("age").pred.mean()
        su.plot(label=veh, color=colormap[veh], linewidth=2, ax=ax)

    plt.legend()
    plt.show()

    f, ax = plt.subplots()
    ax2 = ax.twinx()

    su = tmp.groupby(["age_bands", "vehicle"]).size().unstack()
    su.plot.bar(
        color=colormap.values(), alpha=0.4, edgecolor="white", hatch="///", ax=ax
    )
    ax.set_ylabel("number_of_observations")

    for veh in vehicles:
        su = (
            tmp[tmp.vehicle == veh].groupby("age_bands")[age_col + "_relativity"].mean()
        )
        ax2.plot(ax.get_xticks(), su, label=veh, color=colormap[veh], linewidth=2)
        ax2.plot(
            ax.get_xticks(),
            su,
            label=veh,
            color=colormap[veh],
            marker="o",
            markersize=8,
        )
        ax2.set_ylabel("relativity")
        ax2.grid(False)

    plt.show()

    f, ax = plt.subplots()
    ax2 = ax.twinx()

    su = tmp.groupby("vehicle").size()
    su.plot.bar(
        color=colormap.values(), alpha=0.4, edgecolor="white", hatch="///", ax=ax
    )
    ax.set_ylabel("number_of_observations")

    for veh in vehicles:
        su = tmp[tmp.vehicle == veh].groupby("vehicle")["vehicle_relativity"].mean()
        ax2.plot(ax.get_xticks(), su, label=veh, color=colormap[veh], linewidth=2)
        ax2.plot(
            ax.get_xticks(),
            su,
            label=veh,
            color=colormap[veh],
            marker="o",
            markersize=12,
        )
        ax2.set_ylabel("relativity")
        ax2.grid(False)

    plt.show()
