import os, sys
import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt


def cluster_based_on_water_exchange():
    pass


#
# def extract_wu_from_swud(swud_df):
#     monthly_cols = 'JAN_MGD	FEB_MGD	MAR_MGD	APR_MGD	MAY_MGD	JUN_MGD	JUL_MGD	AUG_MGD	SEP_MGD	OCT_MGD	NOV_MGD	DEC_MGD'
#     monthly_cols = monthly_cols.split()
#     month_swud = swud_df[['WSA_AGIDF', 'YEAR', 'F_PWSID'] + monthly_cols].copy()
#
#     for im, m in enumerate(monthly_cols):
#         month_swud[im + 1] = month_swud[m].values
#         # month_swud[month_swud[im + 1] == 0] = np.NAN
#         del (month_swud[m])
#     month_swud.dropna(axis=0, inplace=True)
#
#     month_swud = month_swud.melt(id_vars=["WSA_AGIDF", "YEAR", "F_PWSID"], var_name="Month", value_name="mWU")
#     month_swud = month_swud.groupby(by=["WSA_AGIDF", 'YEAR', 'Month', "F_PWSID"]).sum()
#     month_swud.reset_index(inplace=True)
#     del (month_swud['F_PWSID'])
#
#     month_swud['day'] = 1
#     month_swud['date'] = pd.to_datetime(month_swud[['YEAR', "Month", "day"]])
#     month_swud['days_in_month'] = month_swud['date'].dt.days_in_month
#     month_swud['monthly_flow'] = 1e6 * month_swud['mWU'] * month_swud['days_in_month']
#     month_swud['monthly_flow'] = month_swud['monthly_flow'].abs()
#
#     annual_df_from_months = month_swud.groupby(by=['WSA_AGIDF', 'YEAR']).sum()
#     annual_df_from_months['annual_flow'] = annual_df_from_months['monthly_flow']
#     annual_df_from_months.reset_index(inplace=True)
#     annual_df_from_months = annual_df_from_months[['WSA_AGIDF', 'YEAR', 'annual_flow']]
#     all_monthly_wu = month_swud.merge(annual_df_from_months, how='left', on=['WSA_AGIDF', 'YEAR'])
#     all_monthly_wu = all_monthly_wu[
#         ['WSA_AGIDF', 'YEAR', 'Month', 'date', 'days_in_month', 'monthly_flow', 'annual_flow']]
#     all_monthly_wu = all_monthly_wu.rename(
#         columns={'monthly_flow': 'monthly_wu_Gallons', 'annual_flow': 'annual_wu_Gallons'})
#
#     mask = annual_df_from_months['YEAR'].mod(4) == 0
#     annual_df_from_months.loc[mask, 'annual_flow'] = (annual_df_from_months.loc[mask, 'annual_flow'] / 366).values
#     annual_df_from_months.loc[~mask, 'annual_flow'] = (annual_df_from_months.loc[~mask, 'annual_flow'] / 365).values
#
#     # annual from annual
#     ann_ann_df = swud_df[['WSA_AGIDF', 'YEAR', 'TOT_WD_MGD', "F_PWSID"]].copy()
#     ann_ann_df = ann_ann_df.groupby(by=['WSA_AGIDF', 'YEAR', 'F_PWSID']).sum()
#     ann_ann_df.reset_index(inplace=True)
#     del (ann_ann_df['F_PWSID'])
#     ann_ann_df['annual_wu_G'] = ann_ann_df['TOT_WD_MGD'] * 1e6
#     ann_ann_df = ann_ann_df.groupby(by=['WSA_AGIDF', 'YEAR']).sum()
#     ann_ann_df.reset_index(inplace=True)
#
#     all_annual = ann_ann_df.merge(annual_df_from_months, how='outer', on=['WSA_AGIDF', 'YEAR'])
#     all_annual.drop(['TOT_WD_MGD'], axis=1, inplace=True)
#     all_annual.rename(columns={"annual_flow": "annual_wu_from_monthly_G"}, inplace=True)
#
#     all_monthly_wu = all_monthly_wu[['WSA_AGIDF', 'YEAR', 'Month', 'monthly_wu_Gallons', 'annual_wu_Gallons']]
#     all_monthly_wu.rename(columns={"annual_wu_Gallons": "annual_wu_G",  'monthly_wu_Gallons':" monthly_wu_G" }, inplace=True)
#
#     # in all_month_wu:
#     #    - column annual_wu_G: is the summation of monthly water use in gallons
#     #    - column monthly_wu_G: is monthly water use in gallons
#     # in all_annual:
#     #   - column annual_wu_G: is annual water use as reported by the sud file converted to gallons
#     #   - Column annual_wu_from_monthly_G: is annual wu calculated from monthly data
#     return all_annual, all_monthly_wu
#
#
# def extract_wu_from_nonswud(non_swud_data):
#     isleap = np.mod(non_swud_data['YEAR'].values, 4.0)
#     ndays = np.zeros_like(isleap)
#     ndays[isleap == 0] = 366
#     ndays[isleap != 0] = 365
#
#     cols = "JAN_MG	FEB_MG	MAR_MG	APR_MG	MAY_MG	JUN_MG	JUL_MG	AUG_MG	SEP_MG	OCT_MG	NOV_MG	DEC_MG"
#     cols = cols.split()
#     for im, m in enumerate(cols):
#         non_swud_data[im + 1] = non_swud_data[m].values
#         del (non_swud_data[m])
#
#     annual_df = non_swud_data[['PWSID', 'ANNUAL_MG', 'YEAR']]
#     monthly_df = non_swud_data.drop(["STATE", 'SOURCE', 'ANNUAL_MG'], axis=1)
#     monthly_df.dropna(axis=0, inplace=True)
#     monthly_df = monthly_df.melt(id_vars=["PWSID", "YEAR"], var_name="Month", value_name="wu_rate_MG")
#
#     annual_df.rename(columns = {"PWSID":"WSA_AGIDF", "ANNUAL_MG": "annual_wu_G"}, inplace = True)
#     annual_df["annual_wu_G"] = 1e6*annual_df["annual_wu_G"]
#     monthly_df['monthly_wu_G'] = monthly_df['wu_rate_MG'] * 1e6
#     del(monthly_df['wu_rate_MG'])
#     return annual_df, monthly_df
#
#
# def update_water_use(old_df, new_data):
#     pass
#
#
# def update_population(dataset, pop_info):
#     pop_info['pop'] = pop_info['pop_swud16'].copy()
#     mask = pop_info['pop'].isna() | pop_info['pop'] == 0
#     pop_info.loc[mask, 'pop'] = pop_info[mask]['plc_pop_interpolated']
#     mask = pop_info['pop'].isna() | pop_info['pop'] == 0
#     pop_info.loc[mask, 'pop'] = pop_info[mask]['TPOPSRV']
#     mask = pop_info['pop'].isna() | pop_info['pop'] == 0
#     pop_info.loc[mask, 'pop'] = pop_info[mask]['tract_pop']
#
#     pop_df = pop_info[['sys_id', 'pop', 'Year']]
#     dataset = dataset.merge(pop_df, right_on=['sys_id', 'Year'], left_on=['sys_id', 'Year'], how='left')
#     return dataset
#
#
# def clean_swud(swud_df, pop_info):
#     annual_swud, monthly_swud = extract_wu_from_swud(swud_df)
#
#     # (1) find systems where monthly data exist but not annual
#     mask_1 = (annual_swud['annual_wu_from_monthly_G'] > 0) & (
#                 (annual_swud['annual_wu_G'] == 0) | (annual_swud['annual_wu_G'].isna()))
#     annual_swud['1'] = 0
#     annual_swud.loc[mask_1, '1'] = 1
#
#     # (2) in annual data flag systems where monthly and annual water use are not consistent.
#     #used_data_mask = ~(annual_swud['annual_wu_from_monthly_G'].isna() | annual_swud['annual_wu_G'].isna())
#     rrratio = annual_swud['annual_wu_from_monthly_G'] / annual_swud['annual_wu_G']
#     ratio_mask = (rrratio>1.1) | (rrratio<0.90)
#     annual_swud['2'] = 0
#     annual_swud.loc[ratio_mask, '2'] = 2
#
#     return 1


def up_scale_data(df, groupby_col, skip_cols, agg_rules):
    """
    take a dataset and aggregate it to a county level
    :return:
    """
    import copy

    col_means = copy.deepcopy(groupby_col)
    col_sums = copy.deepcopy(groupby_col)
    col_mode = copy.deepcopy(groupby_col)

    for (
        irow,
        feat,
    ) in agg_rules.iterrows():
        if feat["Feature"] in skip_cols:
            continue

        if not (feat["Feature"] in df.columns):
            continue

        if feat["aggregation method"] in ["mean"]:
            col_means.append(feat["Feature"])
        elif feat["aggregation method"] in ["sum"]:
            col_sums.append(feat["Feature"])
        elif feat["aggregation method"] in ["mode"]:
            col_mode.append(feat["Feature"])
        else:
            continue

    df1 = df[set(col_means)].groupby(by=groupby_col).mean()
    df2 = df[set(col_sums)].groupby(by=groupby_col).sum()
    df3 = (
        df[set(col_mode)]
        .groupby(by=groupby_col)
        .agg(lambda x: x.value_counts().index[0])
    )

    df_agg = pd.concat([df1, df2, df3], axis=1)
    df_agg.reset_index(inplace=True)

    return df_agg


if __name__ == "__main__":
    dataset = pd.read_csv(r"clean_train_db.csv")
    agg_rules = pd.read_csv(r"aggregation_roles.csv")

    annual_wu = pd.read_csv(r"annual_wu.csv")
    annual_wu["wu_rate_mean"] = annual_wu[
        ["annual_wu_G_swuds", "annual_wu_G_nonswuds"]
    ].mean(axis=1)
    annual_wu["wu_rate_mean"] = (
        annual_wu["wu_rate_mean"] / annual_wu["days_in_year"]
    )
    avg_wu = annual_wu[["WSA_AGIDF", "YEAR", "wu_rate_mean"]].copy()
    avg_wu.rename(
        columns={
            "WSA_AGIDF": "sys_id",
            "YEAR": "Year",
            "wu_rate_mean": "wu_rate",
        },
        inplace=True,
    )

    del dataset["wu_rate"]
    dataset = dataset.merge(avg_wu, on=["sys_id", "Year"], how="left")
    dataset = dataset[dataset["wu_rate"] > 0]

    df_agg = up_scale_data(
        df=dataset,
        groupby_col=["county_id", "Year"],
        skip_cols=[],
        agg_rules=agg_rules,
    )
    x = 1

    # pop_info = pd.read_csv(r"pop_info.csv")
    #
    #
    # swud16 = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\SWUDS_v16.csv")
    # nonswud3 = pd.read_csv(r"C:\work\water_use\mldataset\ml\training\targets\monthly_annually\nonswuds_wu_v3.csv")
    #
    # df = clean_swud(swud16, pop_info)
    # if 0:
    #     annual_swud, monthly_swud = extract_wu_from_swud(swud16)
    #     annual_nonswud, monthly_nonswud = extract_wu_from_nonswud(nonswud3)
    #
    # dataset = update_population(dataset, pop_info)
    #
    # up_scale_to_county(df=dataset, cols=["county_id", "Year"], agg_rules=agg_rules)
    # pass
