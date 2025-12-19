import numpy as np
import pandas as pd
import geopandas as gpd


# ---(DataFrame : Bevölkerung)---

# temp_pred_Bev_df - Step1
def temp_pred_Bev_base(BW_pred_Bev_df, Grp_1, Grp_2, Grp_3, EW_inc_rate, Grp_val_0):
    # Extract population predictions for Regionalverbände not belonging to Grp_1, Grp_2, or Grp_3
    temp_pred_Bev_base_df = BW_pred_Bev_df.loc[~((BW_pred_Bev_df['Regionalverband'].isin(Grp_1))|(BW_pred_Bev_df['Regionalverband'].isin(Grp_2))|(BW_pred_Bev_df['Regionalverband'].isin(Grp_3))), ['DN_DT', 'Regionalverband', 'Jahr', f'rate_{EW_inc_rate}']]
    # Add Group Column : Grp_val_0
    temp_pred_Bev_base_df['Group'] = Grp_val_0
    # Column Renaming for concat() : Bevölkerung insgesamt
    temp_pred_Bev_base_df = temp_pred_Bev_base_df.rename(columns={f'rate_{EW_inc_rate}':'Bevölkerung insgesamt'})
    return temp_pred_Bev_base_df


# temp_pred_Bev_df - Step2
def temp_pred_Bev_Grp(BW_pred_Bev_df, groups, group_vals, ew_inc_rates):
    # empty List for Dataframe
    temp_pred_Bev_grp_list = []

    for grp, grp_val, ew_rate in zip(groups, group_vals, ew_inc_rates):
        rate_col = f"rate_{ew_rate}"

        temp_df = BW_pred_Bev_df.loc[
            BW_pred_Bev_df['Regionalverband'].isin(grp),
            ['DN_DT', 'Regionalverband', 'Jahr', rate_col]
        ].copy()

        # Add Group Column : grp_val
        temp_df['Group'] = grp_val

        # Column Renaming for concat() : Bevölkerung insgesamt
        temp_df = temp_df.rename(columns={rate_col: 'Bevölkerung insgesamt'})

        temp_pred_Bev_grp_list.append(temp_df)

    # Concatenate group-specific dataframes
    temp_pred_Bev_Grp_df = pd.concat(temp_pred_Bev_grp_list, ignore_index=True)

    return temp_pred_Bev_Grp_df


# temp_pred_Bev_df - Step3(End)
def temp_pred_Bev(temp_pred_Bev_base_df, temp_pred_Bev_Grp_df):
    temp_pred_Bev_df = pd.concat([temp_pred_Bev_base_df, temp_pred_Bev_Grp_df], ignore_index=True).reset_index(drop=True)

    # Column used to distinguish line types in line plot visualizations
    temp_pred_Bev_df['Art'] = 'predict'
    return temp_pred_Bev_df


# temp_act_Bev_df - Step1(End)
def temp_act_Bev(BW_Bev_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):

    temp_act_Bev_df = BW_Bev_df[['DN_DT', 'Regionalverband', 'Jahr', 'Bevölkerung insgesamt']].copy()

    # Group1
    temp_act_Bev_df.loc[temp_act_Bev_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # Group2
    temp_act_Bev_df.loc[temp_act_Bev_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # Group3
    temp_act_Bev_df.loc[temp_act_Bev_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # Group0
    temp_act_Bev_df.loc[temp_act_Bev_df['Group'].isna(), 'Group'] = Grp_val_0

    # Column used to distinguish line types in line plot visualizations
    temp_act_Bev_df['Art'] = 'actual'
    return temp_act_Bev_df


# Duplicate 2023 data with Art = 'predict' to prevent line breaks between actual and predicted values in line plot visualizations.
# temp_middle_Bev_df - Step1(End)
def temp_middle_Bev(temp_act_Bev_df):

    # 2023 Data from 'actual' data
    temp_middle_Bev_df = temp_act_Bev_df[temp_act_Bev_df['Jahr']==2023].copy()

    # Column used to distinguish line types in line plot visualizations
    temp_middle_Bev_df['Art'] = 'predict'

    return temp_middle_Bev_df


# Bev_for_Vis_df - Step1(End)
def Bev_for_Vis(temp_pred_Bev_df, temp_middle_Bev_df, temp_act_Bev_df):
    # temp_act_Bev_df, temp_middle_Bev_df, temp_pred_Bev_df
    Bev_for_Vis_df = pd.concat([temp_act_Bev_df, temp_middle_Bev_df], axis=0)
    Bev_for_Vis_df = pd.concat([Bev_for_Vis_df, temp_pred_Bev_df], axis=0).reset_index(drop=True)
    return Bev_for_Vis_df



# ---HaushaltEV DataFrame for Visualizations---

# temp_pred_HausEV_df - Step1
def frame_pred_HausEV():

    frame_pred_HausEV_df = pd.DataFrame(columns=['DN_DT', 'Regionalverband', 'Jahr', 'Stromverbrauch(Haushalt)'])

    # Assign key future years (10-year intervals) to reduce dataframe size
    frame_pred_HausEV_df['Jahr'] = [2030, 2040, 2050, 2060, 2070]

    # Column used to distinguish line types in line plot visualizations
    frame_pred_HausEV_df['Art'] = 'predict'
    return frame_pred_HausEV_df

# temp_pred_HausEV_df - Step2
def temp_pred_HausEV_1(frame_pred_HausEV_df, TBD_df):
    # Empty Dict for Dataframes
    temp_dict = {}
    for reg in TBD_df['DN_DT'].drop_duplicates():
        # Select Regierungsverband
        rv = TBD_df.loc[TBD_df['DN_DT'] == reg, 'Regionalverband'].iloc[0]

        temp_df = frame_pred_HausEV_df.copy()
        temp_df['DN_DT'] = reg
        temp_df['Regionalverband'] = rv


        temp_dict[reg] = temp_df


    temp_pred_HausEV_df = pd.concat(temp_dict.values(), axis=0).reset_index(drop=True)
    return temp_pred_HausEV_df

# temp_pred_HausEV_df - Step3
def temp_pred_HausEV_2(temp_pred_HausEV_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    # Group1
    temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # Group2
    temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # Group3
    temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # Group0
    temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Group'].isna(), 'Group'] = Grp_val_0

    return temp_pred_HausEV_df


# temp_pred_HausEV_df - Step4(End)
def temp_pred_HausEV_3(temp_pred_HausEV_df, Bev_for_Vis_df, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0, EW_GJ_1, EW_GJ_2, EW_GJ_3, EW_GJ):
    # Estimate household electricity demand by multiplying predicted population with per-capita electricity consumption; convert results to TJ

    # Grp1
    for reg in temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Group']==Grp_val_1, 'DN_DT'].drop_duplicates():
        for year in temp_pred_HausEV_df['Jahr'].drop_duplicates():
            temp_pred_HausEV_df.loc[(temp_pred_HausEV_df['DN_DT']==reg)&(temp_pred_HausEV_df['Jahr']==year), 'Stromverbrauch(Haushalt)'] = (
                ((Bev_for_Vis_df.loc[(Bev_for_Vis_df['DN_DT']==reg)&(Bev_for_Vis_df['Jahr']==year), 'Bevölkerung insgesamt'] * EW_GJ_1) / 1000).iloc[0].round()
            )

    # Grp2
    for reg in temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Group']==Grp_val_2, 'DN_DT'].drop_duplicates():
        for year in temp_pred_HausEV_df['Jahr'].drop_duplicates():
            temp_pred_HausEV_df.loc[(temp_pred_HausEV_df['DN_DT']==reg)&(temp_pred_HausEV_df['Jahr']==year), 'Stromverbrauch(Haushalt)'] = (
                ((Bev_for_Vis_df.loc[(Bev_for_Vis_df['DN_DT']==reg)&(Bev_for_Vis_df['Jahr']==year), 'Bevölkerung insgesamt'] * EW_GJ_2) / 1000).iloc[0].round()
            )

    # Grp3
    for reg in temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Group']==Grp_val_3, 'DN_DT'].drop_duplicates():
        for year in temp_pred_HausEV_df['Jahr'].drop_duplicates():
            temp_pred_HausEV_df.loc[(temp_pred_HausEV_df['DN_DT']==reg)&(temp_pred_HausEV_df['Jahr']==year), 'Stromverbrauch(Haushalt)'] = (
                ((Bev_for_Vis_df.loc[(Bev_for_Vis_df['DN_DT']==reg)&(Bev_for_Vis_df['Jahr']==year), 'Bevölkerung insgesamt'] * EW_GJ_3) / 1000).iloc[0].round()
            )

    # Grp0
    for reg in temp_pred_HausEV_df.loc[temp_pred_HausEV_df['Group']==Grp_val_0, 'DN_DT'].drop_duplicates():
        for year in temp_pred_HausEV_df['Jahr'].drop_duplicates():
            temp_pred_HausEV_df.loc[(temp_pred_HausEV_df['DN_DT']==reg)&(temp_pred_HausEV_df['Jahr']==year), 'Stromverbrauch(Haushalt)'] = (
                ((Bev_for_Vis_df.loc[(Bev_for_Vis_df['DN_DT']==reg)&(Bev_for_Vis_df['Jahr']==year), 'Bevölkerung insgesamt'] * EW_GJ) / 1000).iloc[0].round()
            )

    return temp_pred_HausEV_df



# temp_act_HausEV_df - Step1(End)
def temp_act_HausEV(BW_HausEV_Kreis_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):

    temp_act_HausEV_df = BW_HausEV_Kreis_df[['DN_DT', 'Jahr', 'Regionalverband', 'Stromverbrauch(Haushalt)']].copy()

    # Column used to distinguish line types in line plot visualizations
    temp_act_HausEV_df['Art'] = 'actual'


    # Grp1
    temp_act_HausEV_df.loc[temp_act_HausEV_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # Grp2
    temp_act_HausEV_df.loc[temp_act_HausEV_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # Grp3
    temp_act_HausEV_df.loc[temp_act_HausEV_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # Grp0
    temp_act_HausEV_df.loc[temp_act_HausEV_df['Group'].isna(), 'Group'] = Grp_val_0

    return temp_act_HausEV_df

# Duplicate 2023 data with Art = 'predict' to prevent line breaks between actual and predicted values in line plot visualizations
# temp_middle_HausEV_df - Step1(End)
def temp_middle_HausEV(temp_act_HausEV_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):

    temp_middle_HausEV_df = temp_act_HausEV_df[temp_act_HausEV_df['Jahr']==2023].copy()
    temp_middle_HausEV_df['Art'] = 'predict'


    # Grp1
    temp_middle_HausEV_df.loc[temp_middle_HausEV_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # Grp2
    temp_middle_HausEV_df.loc[temp_middle_HausEV_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # Grp3
    temp_middle_HausEV_df.loc[temp_middle_HausEV_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # Grp0
    temp_middle_HausEV_df.loc[temp_middle_HausEV_df['Group'].isna(), 'Group'] = Grp_val_0

    return temp_middle_HausEV_df

# HausEV_for_Vis_df - Step1(End)
def HausEV_for_Vis(temp_act_HausEV_df, temp_middle_HausEV_df, temp_pred_HausEV_df):
    # temp_act_HausEV_df, temp_middle_HausEV_df, temp_pred_HausEV_df
    HausEV_for_Vis_df = pd.concat([temp_act_HausEV_df, temp_middle_HausEV_df], axis=0)
    HausEV_for_Vis_df = pd.concat([HausEV_for_Vis_df, temp_pred_HausEV_df], axis=0).reset_index(drop=True)
    return HausEV_for_Vis_df



# ---HaushaltEV Dataframe for Map-Visualizations---

# HausEV_for_Vismap_df - Step1(End)
def HausEV_for_Vismap(HausEV_for_Vis_df):
    # Use predicted values only (exclude actual data)
    HausEV_for_Vismap_df = HausEV_for_Vis_df.loc[HausEV_for_Vis_df['Art']=='predict'].copy()


    # Calculate absolute and relative changes using 2023 as the regional baseline

    # Extract regional baseline values for 2023
    # (index: DN_DT, value: household electricity consumption in 2023)
    baseline_2023 = (
        HausEV_for_Vismap_df
            .loc[HausEV_for_Vismap_df['Jahr'] == 2023]
            .set_index('DN_DT')['Stromverbrauch(Haushalt)']
    )

    # Compute absolute increase relative to 2023 baseline
    HausEV_for_Vismap_df['Zunahme(Haushalt)'] = (
        HausEV_for_Vismap_df['Stromverbrauch(Haushalt)']
        - HausEV_for_Vismap_df['DN_DT'].map(baseline_2023)
    )

    # Compute relative increase rate (%) compared to 2023 baseline
    HausEV_for_Vismap_df['Zunahmequote(Haushalt)'] = (
        HausEV_for_Vismap_df['Zunahme(Haushalt)'] /
        HausEV_for_Vismap_df['DN_DT'].map(baseline_2023) * 100
    )
    return HausEV_for_Vismap_df
