import os, sys
import pandas as pd
import numpy as np
import geopandas
import calendar
import matplotlib.pyplot as plt

import iwateruse.outliers_utils as outliers_utils


def extract_wu_from_swud(swud_df):
    monthly_cols = "JAN_MGD	FEB_MGD	MAR_MGD	APR_MGD	MAY_MGD	JUN_MGD	JUL_MGD	AUG_MGD	SEP_MGD	OCT_MGD	NOV_MGD	DEC_MGD"
    monthly_cols = monthly_cols.split()
    month_swud = swud_df[
        ["WSA_AGIDF", "YEAR", "F_PWSID"] + monthly_cols
    ].copy()

    for im, m in enumerate(monthly_cols):
        month_swud[im + 1] = month_swud[m].values
        del month_swud[m]
    month_swud.dropna(axis=0, inplace=True)

    month_swud = month_swud.melt(
        id_vars=["WSA_AGIDF", "YEAR", "F_PWSID"],
        var_name="Month",
        value_name="mWU",
    )
    month_swud["mWU"] = month_swud["mWU"].astype(float)
    month_swud = month_swud.groupby(by=["WSA_AGIDF", "YEAR", "Month"]).sum()
    month_swud.reset_index(inplace=True)

    month_swud["day"] = 1
    month_swud["date"] = pd.to_datetime(month_swud[["YEAR", "Month", "day"]])
    month_swud["days_in_month"] = month_swud["date"].dt.days_in_month
    month_swud["monthly_flow"] = (
        1e6 * month_swud["mWU"] * month_swud["days_in_month"]
    )  # monthly flow is G/month
    month_swud["monthly_flow"] = month_swud["monthly_flow"].abs()

    annual_df_from_months = month_swud.groupby(by=["WSA_AGIDF", "YEAR"]).sum()
    annual_df_from_months["annual_flow"] = annual_df_from_months[
        "monthly_flow"
    ]  # annual flow is G/year
    annual_df_from_months.reset_index(inplace=True)
    annual_df_from_months = annual_df_from_months[
        ["WSA_AGIDF", "YEAR", "annual_flow"]
    ]
    all_monthly_wu = month_swud.merge(
        annual_df_from_months, how="left", on=["WSA_AGIDF", "YEAR"]
    )
    all_monthly_wu = all_monthly_wu[
        [
            "WSA_AGIDF",
            "YEAR",
            "Month",
            "date",
            "days_in_month",
            "monthly_flow",
            "annual_flow",
        ]
    ]
    all_monthly_wu = all_monthly_wu.rename(
        columns={
            "monthly_flow": "monthly_wu_Gallons",
            "annual_flow": "annual_wu_Gallons",
        }
    )

    # mask = annual_df_from_months['YEAR'].mod(4) == 0
    # annual_df_from_months.loc[mask, 'annual_flow'] = (annual_df_from_months.loc[mask, 'annual_flow'] ).values
    # annual_df_from_months.loc[~mask, 'annual_flow'] = (annual_df_from_months.loc[~mask, 'annual_flow']).values

    # annual from annual
    ann_ann_df = swud_df[["WSA_AGIDF", "YEAR", "TOT_WD_MGD", "F_PWSID"]].copy()
    ann_ann_df = ann_ann_df.groupby(by=["WSA_AGIDF", "YEAR", "F_PWSID"]).sum()
    ann_ann_df.reset_index(inplace=True)
    del ann_ann_df["F_PWSID"]

    # make sure units are G
    mask = ann_ann_df["YEAR"].mod(4) == 0
    ann_ann_df.loc[mask, "annual_wu_G"] = (
        1e6 * (ann_ann_df.loc[mask, "TOT_WD_MGD"]).values * 366
    )
    ann_ann_df.loc[~mask, "annual_wu_G"] = (
        1e6 * (ann_ann_df.loc[~mask, "TOT_WD_MGD"]).values * 365
    )
    # ann_ann_df['annual_wu_G'] = ann_ann_df['TOT_WD_MGD'] * 1e6 # annual G/day
    ann_ann_df = ann_ann_df.groupby(by=["WSA_AGIDF", "YEAR"]).sum()
    ann_ann_df.reset_index(inplace=True)

    all_annual = ann_ann_df.merge(
        annual_df_from_months, how="outer", on=["WSA_AGIDF", "YEAR"]
    )
    all_annual.drop(["TOT_WD_MGD"], axis=1, inplace=True)
    all_annual.rename(
        columns={"annual_flow": "annual_wu_from_monthly_G"}, inplace=True
    )

    all_monthly_wu = all_monthly_wu[
        [
            "WSA_AGIDF",
            "YEAR",
            "Month",
            "monthly_wu_Gallons",
            "annual_wu_Gallons",
        ]
    ]
    all_monthly_wu.rename(
        columns={
            "annual_wu_Gallons": "annual_wu_G",
            "monthly_wu_Gallons": "monthly_wu_G",
        },
        inplace=True,
    )

    # in all_month_wu:
    #    - column annual_wu_G: is the summation of monthly water use in gallons
    #    - column monthly_wu_G: is monthly water use in gallons
    # in all_annual:
    #   - column annual_wu_G: is annual water use as reported by the swud file converted to gallons
    #   - Column annual_wu_from_monthly_G: is annual wu calculated from monthly data
    return all_annual, all_monthly_wu


def extract_wu_from_nonswud(non_swud_data):
    isleap = np.mod(non_swud_data["YEAR"].values, 4.0)
    ndays = np.zeros_like(isleap)
    ndays[isleap == 0] = 366
    ndays[isleap != 0] = 365

    cols = "JAN_MG	FEB_MG	MAR_MG	APR_MG	MAY_MG	JUN_MG	JUL_MG	AUG_MG	SEP_MG	OCT_MG	NOV_MG	DEC_MG"
    cols = cols.split()
    for im, m in enumerate(cols):
        non_swud_data[im + 1] = non_swud_data[m].values
        del non_swud_data[m]

    annual_df = non_swud_data[["WSA_AGIDF", "ANNUAL_MG", "YEAR"]].copy()
    monthly_df = non_swud_data.drop(
        ["STATE", "ANNUAL_MG", "DATA_FREQ", "POPULATION", "PA_FLAG"], axis=1
    )
    monthly_df.dropna(axis=0, inplace=True)
    monthly_df = monthly_df.melt(
        id_vars=["WSA_AGIDF", "YEAR"],
        var_name="Month",
        value_name="wu_rate_MG",
    )
    monthly_df["wu_rate_MG"] = 1e6 * monthly_df["wu_rate_MG"]

    annual_df.rename(columns={"ANNUAL_MG": "annual_wu_G"}, inplace=True)
    monthly_df.rename(columns={"wu_rate_MG": "monthly_wu_G"}, inplace=True)
    annual_df["annual_wu_G"] = annual_df["annual_wu_G"] * 1e6

    return annual_df, monthly_df


def join_swuds_nonswuds(swuds, nonswuds):
    """

    :param swuds:
    :param nonswuds:
    :return: return two files annual and month water use - units (Gallon/year or Gallon/month)
    """
    annual_nonswud, monthly_nonswud = extract_wu_from_nonswud(nonswuds)
    annual_swud, monthly_swud = extract_wu_from_swud(swuds)

    monthly_nonswud = monthly_nonswud.merge(
        annual_nonswud, how="left", on=["WSA_AGIDF", "YEAR"]
    )
    temp_ = monthly_nonswud[["WSA_AGIDF", "YEAR", "annual_wu_G"]].copy()
    temp_.drop_duplicates(subset=["WSA_AGIDF", "YEAR"], inplace=True)
    annual_nonswud = annual_nonswud.merge(
        temp_, how="left", on=["WSA_AGIDF", "YEAR"]
    )
    annual_nonswud.rename(
        columns={
            "annual_wu_G_x": "annual_wu_G",
            "annual_wu_G_y": "annual_wu_from_monthly_G",
        },
        inplace=True,
    )

    all_monthly = monthly_swud.merge(
        monthly_nonswud, how="outer", on=["WSA_AGIDF", "YEAR", "Month"]
    )
    all_annual = annual_swud.merge(
        annual_nonswud, how="outer", on=["WSA_AGIDF", "YEAR"]
    )

    all_annual.rename(
        columns={
            "annual_wu_G_x": "annual_wu_G_swuds",
            "annual_wu_G_y": "annual_wu_G_nonswuds",
            "annual_wu_from_monthly_G_x": "annual_wu_from_monthly_swuds_G",
            "annual_wu_from_monthly_G_y": "annual_wu_from_monthly_nonswuds_G",
        },
        inplace=True,
    )
    all_monthly.rename(
        columns={
            "monthly_wu_G_x": "monthly_wu_G_swuds",
            "monthly_wu_G_y": "monthly_wu_G_nonswuds",
            "annual_wu_G_x": "annual_wu_G_swuds",
            "annual_wu_G_y": "annual_wu_G_nonswuds",
        },
        inplace=True,
    )

    all_monthly.sort_values(by=["WSA_AGIDF", "YEAR", "Month"], inplace=True)
    all_monthly.reset_index(drop=True, inplace=True)

    all_annual.sort_values(by=["WSA_AGIDF", "YEAR"], inplace=True)
    all_annual.reset_index(drop=True, inplace=True)

    return all_annual, all_monthly


def update_population(dataset, pop_info):
    pop_info["pop"] = pop_info["pop_swud16"].copy()
    mask = pop_info["pop"].isna() | pop_info["pop"] == 0
    pop_info.loc[mask, "pop"] = pop_info[mask]["plc_pop_interpolated"]
    mask = pop_info["pop"].isna() | pop_info["pop"] == 0
    pop_info.loc[mask, "pop"] = pop_info[mask]["TPOPSRV"]
    mask = pop_info["pop"].isna() | pop_info["pop"] == 0
    pop_info.loc[mask, "pop"] = pop_info[mask]["tract_pop"]

    pop_df = pop_info[["sys_id", "pop", "Year"]]
    dataset = dataset.merge(
        pop_df,
        right_on=["sys_id", "Year"],
        left_on=["sys_id", "Year"],
        how="left",
    )
    return dataset


def code_wu_outliers(annual_wu, monthly_wu, master_pop):

    master_pop.rename(columns={"Year": "YEAR"}, inplace=True)
    annual_wu = annual_wu.merge(
        master_pop[["WSA_AGIDF", "YEAR", "pop_swud16", "pop", "TPOPSRV"]],
        how="left",
        on=["WSA_AGIDF", "YEAR"],
    )
    monthly_wu = monthly_wu.merge(
        master_pop[["WSA_AGIDF", "YEAR", "pop_swud16", "pop", "TPOPSRV"]],
        how="left",
        on=["WSA_AGIDF", "YEAR"],
    )

    # (1) find swuds systems where monthly data exist but not annual
    mask_1 = (annual_wu["annual_wu_from_monthly_swuds_G"] > 0) & (
        (annual_wu["annual_wu_G_swuds"] == 0)
        | (annual_wu["annual_wu_G_swuds"].isna())
    )
    mask_1n = (annual_wu["annual_wu_from_monthly_nonswuds_G"] > 0) & (
        (annual_wu["annual_wu_G_nonswuds"] == 0)
        | (annual_wu["annual_wu_G_nonswuds"].isna())
    )
    annual_wu["flg_no_annual"] = 0
    annual_wu.loc[(mask_1 | mask_1n), "flg_no_annual"] = 1

    # (2) in annual data flag systems where monthly and annual water use are not consistent.
    rrratio = (
        annual_wu["annual_wu_from_monthly_swuds_G"]
        / annual_wu["annual_wu_G_swuds"]
    )
    rrratio[rrratio.isin([np.inf, -np.inf])] = np.NAN
    ratio_mask = (rrratio > 1.1) | (rrratio < 0.90)
    annual_wu["flg_annual_isdiff_annualize_month_swud"] = 0
    annual_wu["ratio_annual_annualizeMonth_swud"] = rrratio
    annual_wu.loc[ratio_mask, "flg_annual_isdiff_annualize_month_swud"] = 2

    # same but non swud
    rrratio = (
        annual_wu["annual_wu_from_monthly_nonswuds_G"]
        / annual_wu["annual_wu_G_nonswuds"]
    )
    rrratio[rrratio.isin([np.inf, -np.inf])] = np.NAN
    ratio_mask = (rrratio > 1.1) | (rrratio < 0.90)
    annual_wu["flg_annual_isdiff_annualize_month_nonswud"] = 0
    annual_wu["ratio_annual_annualizeMonth_nonswud"] = rrratio
    annual_wu.loc[ratio_mask, "flg_annual_isdiff_annualize_month_nonswud"] = 3

    # (3) systems with large temporal change
    def compute_temporal_variability(df_, field):
        df = df_[df_[field] > 0]
        df.sort_values(by="YEAR", inplace=True)
        if len(df) > 1:
            pc = df[field] / df["pop"]
            wu_change = pc.diff() / df["YEAR"].diff()
            wu_change = 100 * wu_change.abs() / pc
            wu_change[wu_change.isin([np.inf, -np.inf])] = 99.99

            return wu_change.max()
        else:
            return 0

    temp_diff = annual_wu.groupby("WSA_AGIDF").apply(
        compute_temporal_variability, field="annual_wu_G_swuds"
    )
    temp_diff = (
        pd.DataFrame(temp_diff)
        .reset_index()
        .rename(columns={0: "prc_time_change_swud"})
    )
    annual_wu = annual_wu.merge(temp_diff, on="WSA_AGIDF", how="left")

    temp_diff = annual_wu.groupby("WSA_AGIDF").apply(
        compute_temporal_variability, field="annual_wu_G_nonswuds"
    )
    temp_diff = (
        pd.DataFrame(temp_diff)
        .reset_index()
        .rename(columns={0: "prc_time_change_nonswud"})
    )
    annual_wu = annual_wu.merge(temp_diff, on="WSA_AGIDF", how="left")

    # (4) compute per capita

    annual_wu["days_in_year"] = 365
    isleap = annual_wu["YEAR"].mod(4) == 0
    annual_wu.loc[isleap, "days_in_year"] = 366
    annual_wu["swuds_pc"] = annual_wu["annual_wu_G_swuds"] / (
        annual_wu["pop"] * annual_wu["days_in_year"]
    )
    annual_wu["swuds_pc"].replace([np.inf, -np.inf], np.nan, inplace=True)
    annual_wu["nonswuds_pc"] = annual_wu["annual_wu_G_nonswuds"] / (
        annual_wu["pop"] * annual_wu["days_in_year"]
    )
    annual_wu["nonswuds_pc"].replace([np.inf, -np.inf], np.nan, inplace=True)

    # (5) flag inconsistent annual swud vs noswud data
    annual_wu["swud_noswud_diff_prc"] = (
        100.0
        * np.abs(
            annual_wu["annual_wu_G_swuds"] - annual_wu["annual_wu_G_nonswuds"]
        )
        / annual_wu["annual_wu_G_swuds"]
    )

    # (6) flag inconsistent pop and TPOPSRV
    annual_wu["pswud16_tpop_ratio"] = (
        100.0
        * np.abs(annual_wu["TPOPSRV"] - annual_wu["pop_swud16"])
        / annual_wu["pop_swud16"]
    )
    annual_wu["ppop_tpop_ratio"] = (
        100.0
        * np.abs(annual_wu["TPOPSRV"] - annual_wu["pop"])
        / annual_wu["TPOPSRV"]
    )

    # (7) check normality of monthly fractions - swud
    monthly_wu["month_frac_swud"] = (
        monthly_wu["monthly_wu_G_swuds"] / monthly_wu["annual_wu_G_swuds"]
    )
    monthly_wu = outliers_utils.flag_monthly_wu_abnormal_fac(
        monthly_wu,
        sys_id="WSA_AGIDF",
        year="YEAR",
        month="Month",
        mon_wu="monthly_wu_G_swuds",
        ann_wu="annual_wu_G_swuds",
    )
    monthly_wu.rename(
        columns={"seasonality_simil": "seasonality_simil_swud"}, inplace=True
    )
    # (8) check normality of monthly fractions - nonswud
    monthly_wu["month_frac_nonswud"] = (
        monthly_wu["monthly_wu_G_nonswuds"]
        / monthly_wu["annual_wu_G_nonswuds"]
    )
    monthly_wu = outliers_utils.flag_monthly_wu_abnormal_fac(
        monthly_wu,
        sys_id="WSA_AGIDF",
        year="YEAR",
        month="Month",
        mon_wu="monthly_wu_G_nonswuds",
        ann_wu="annual_wu_G_nonswuds",
    )
    monthly_wu.rename(
        columns={"seasonality_simil": "seasonality_simil_nonswud"},
        inplace=True,
    )

    annual_wu["inWSA"] = 0
    annual_wu.loc[
        annual_wu["WSA_AGIDF"].isin(wsa_shp["WSA_AGIDF"]), "inWSA"
    ] = 1

    monthly_wu["inWSA"] = 0
    monthly_wu.loc[
        monthly_wu["WSA_AGIDF"].isin(wsa_shp["WSA_AGIDF"]), "inWSA"
    ] = 1

    monthly_wu.to_csv(r"monthly_wu.csv", index=False)
    annual_wu.to_csv(r"annual_wu.csv", index=False)
    x = 1


def resolve_conflicts(annual_wu, monthly_wu, regional_average):
    annual_wu = annual_wu[annual_wu["inWSA"] > 0]
    regional_average = regional_average[
        [
            "sys_id",
            "x",
            "y",
            "pop_tmean",
            "pop_median",
            "swud_tmean",
            "swud_median",
            "tpop_tmean",
            "tpop_median",
        ]
    ].copy()
    regional_average.rename(columns={"sys_id": "WSA_AGIDF"}, inplace=True)
    annual_wu = annual_wu.merge(regional_average, how="left", on="WSA_AGIDF")

    pass


if __name__ == "__main__":
    dataset = pd.read_csv(
        r"../../ml_experiments/annual_v_0_0/clean_train_db.csv"
    )
    # pop_info = pd.read_csv(r"pop_info.csv")

    agg_rules = pd.read_csv(
        r"../../ml_experiments/annual_v_0_0/aggregation_roles.csv"
    )

    swud = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\SWUDS_v17.csv",
        encoding="cp1252",
    )
    nonswud3 = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\nonswuds_wu_v5_clean.csv"
    )
    wsa_shp = geopandas.read_file(
        r"C:\work\water_use\mldataset\gis\wsa\WSA_v1.shp"
    )
    main_db = pd.read_csv(
        r"C:\work\water_use\mldataset\ml\training\train_datasets\Annual\wu_annual_training3.csv"
    )

    if 1:
        annual_wu, monthly_wu = join_swuds_nonswuds(swud, nonswud3)
        master_pop = pd.read_csv(
            r"C:\work\water_use\ml_experiments\annual_v_0_0\master_population.csv"
        )
        df = code_wu_outliers(annual_wu, monthly_wu, master_pop)

    if 0:
        annual_swud, monthly_swud = extract_wu_from_swud(swud16)
        annual_nonswud, monthly_nonswud = extract_wu_from_nonswud(nonswud3)

    if 1:
        monthly_wu = pd.read_csv(
            "../../ml_experiments/annual_v_0_0/monthly_wu.csv"
        )
        annual_wu = pd.read_csv(
            "../../ml_experiments/annual_v_0_0/annual_wu.csv"
        )
        regional_average = pd.read_csv(
            "../../ml_experiments/annual_v_0_0/spatial_pc_statistics_apply_max_min_pc.csv"
        )

        resolve_conflicts(annual_wu, monthly_wu, regional_average)
