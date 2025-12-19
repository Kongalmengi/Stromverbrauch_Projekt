import numpy as np
import pandas as pd
import geopandas as gpd


# ---(Build empl_for_Vis_df : Beschäftigte Data)---

# pred_Bev_for_empl_df - step1(End)
def pred_Bev_for_empl(Bev_for_Vis_df, TBD_df, pred_empl_r_df, Grp_1, Grp_2, Grp_3, empl_r_inc_1, empl_r_inc_2, empl_r_inc_3, empl_r_inc):
    # Bev_for_Vis_df contains both historical actual values and future predictions.
    # For visualization purposes, the year 2023 is also marked as Art='predict', although it represents an actual observed value.
    # To extract only pure future predictions, the year 2023 is excluded here.
    pred_Bev_for_empl_df = Bev_for_Vis_df[(Bev_for_Vis_df['Art']=='predict')&~(Bev_for_Vis_df['Jahr']==2023)].copy()

    # Pre-create the 'Beschäftigte' column.
    # If no regions are selected for groups 1–3, the assignment loops won't run and the column would never be created. This would break the fallback (group 0) logic that checks for NA.
    if 'Beschäftigte' not in pred_Bev_for_empl_df.columns:
        pred_Bev_for_empl_df['Beschäftigte'] = pd.NA

    # Compute projected 'Beschäftigte' for each region and year:
    # Beschäftigte = employment_rate (scenario by group) * projected population, rounded.
    # Groups 1–3 use different employment-rate adjustments; remaining regions fall back to the default rate.

    # Group 1: use rate_{empl_r_inc_1}
    for reg in TBD_df.loc[TBD_df['Regionalverband'].isin(Grp_1), 'DN_DT']:
        for year in pred_empl_r_df['Jahr'].drop_duplicates():
            pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Beschäftigte'] = (pred_empl_r_df.loc[(pred_empl_r_df['DN_DT']==reg)&(pred_empl_r_df['Jahr']==year), f'rate_{empl_r_inc_1}'].iloc[0] * pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Bevölkerung insgesamt'].iloc[0]).round()

    # Group 2: use rate_{empl_r_inc_2}
    for reg in TBD_df.loc[TBD_df['Regionalverband'].isin(Grp_2), 'DN_DT']:
        for year in pred_empl_r_df['Jahr'].drop_duplicates():
            pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Beschäftigte'] = (pred_empl_r_df.loc[(pred_empl_r_df['DN_DT']==reg)&(pred_empl_r_df['Jahr']==year), f'rate_{empl_r_inc_2}'].iloc[0] * pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Bevölkerung insgesamt'].iloc[0]).round()

    # Group 3: use rate_{empl_r_inc_3}
    for reg in TBD_df.loc[TBD_df['Regionalverband'].isin(Grp_3), 'DN_DT']:
        for year in pred_empl_r_df['Jahr'].drop_duplicates():
            pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Beschäftigte'] = (pred_empl_r_df.loc[(pred_empl_r_df['DN_DT']==reg)&(pred_empl_r_df['Jahr']==year), f'rate_{empl_r_inc_3}'].iloc[0] * pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Bevölkerung insgesamt'].iloc[0]).round()

    # Group 0 (fallback): for regions not covered above, use rate_{empl_r_inc}
    for reg in pred_Bev_for_empl_df.loc[pred_Bev_for_empl_df['Beschäftigte'].isna(), 'DN_DT'].drop_duplicates():
        for year in pred_empl_r_df['Jahr'].drop_duplicates():
            pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Beschäftigte'] = (pred_empl_r_df.loc[(pred_empl_r_df['DN_DT']==reg)&(pred_empl_r_df['Jahr']==year), f'rate_{empl_r_inc}'].iloc[0] * pred_Bev_for_empl_df.loc[(pred_Bev_for_empl_df['DN_DT']==reg)&(pred_Bev_for_empl_df['Jahr']==year), 'Bevölkerung insgesamt'].iloc[0]).round()


    pred_Bev_for_empl_df = pred_Bev_for_empl_df.drop(columns=['Bevölkerung insgesamt'])
    pred_Bev_for_empl_df['Beschäftigte'] = pred_Bev_for_empl_df['Beschäftigte'].astype(int)

    return pred_Bev_for_empl_df


# act_Bev_for_empl_df - Step1(End)
# Prepare actual employee data in the same structure as the predicted data, so that actual and predicted values can be visualized together in a single line plot.
def act_Bev_for_empl(BW_Industrie_df, groups, group_vals, Grp_val_0):
    act_Bev_for_empl_df = BW_Industrie_df.drop(columns=['DName', 'DType', 'Betriebe', 'Gesamtumsatz']).copy()

    # Grp1, 2, 3
    for grp, val in zip(groups, group_vals):
        act_Bev_for_empl_df.loc[
            act_Bev_for_empl_df['Regionalverband'].isin(grp),
            'Group'
        ] = val

    # Grp0
    act_Bev_for_empl_df.loc[~((act_Bev_for_empl_df['Regionalverband'].isin(groups[0]))|(act_Bev_for_empl_df['Regionalverband'].isin(groups[1]))|(act_Bev_for_empl_df['Regionalverband'].isin(groups[2]))), 'Group'] = Grp_val_0

    # Column used to distinguish line types in line plot visualizations
    act_Bev_for_empl_df['Art'] = 'actual'

    return act_Bev_for_empl_df


# middle_Bev_for_empl_df
# Duplicate 2023 data with Art = 'predict' to prevent line breaks between actual and predicted values in line plot visualizations.
def middle_Bev_for_empl(act_Bev_for_empl_df):
    middle_Bev_for_empl_df = act_Bev_for_empl_df[act_Bev_for_empl_df['Jahr']==2023].copy()
    middle_Bev_for_empl_df['Art'] = 'predict'

    return middle_Bev_for_empl_df


# Combine actual, intermediate, and predicted employee data into a single DataFrame for visualization.
# empl_for_Vis_df
def empl_for_Vis(act_Bev_for_empl_df, middle_Bev_for_empl_df, pred_Bev_for_empl_df):
    empl_for_Vis_df = pd.concat([act_Bev_for_empl_df, middle_Bev_for_empl_df], axis=0)
    empl_for_Vis_df = pd.concat([empl_for_Vis_df, pred_Bev_for_empl_df], axis=0).reset_index(drop=True)

    return empl_for_Vis_df



# ---(Build Umsatz_for_Vis_df : Umsatz Data)---

# temp_pred_Umsatz_df - Step1
def temp_pred_Umsatz_base(BW_pred_Umsatz_df, Grp_1, Grp_2, Grp_3, Grp_val_0, umsatz_inc):
    # Build baseline revenue projections for regions not assigned to any group:
    # Select ungrouped regions, extract the revenue value for the given scenario (rate_{umsatz_inc}), and label them as Group0 to ensure a uniform structure for later concatenation.

    temp_pred_Umsatz_base_df = BW_pred_Umsatz_df.loc[~((BW_pred_Umsatz_df['Regionalverband'].isin(Grp_1))|(BW_pred_Umsatz_df['Regionalverband'].isin(Grp_2))|(BW_pred_Umsatz_df['Regionalverband'].isin(Grp_3))), ['DN_DT', 'Regionalverband', 'Jahr', f'rate_{umsatz_inc}']]

    temp_pred_Umsatz_base_df['Group'] = Grp_val_0

    temp_pred_Umsatz_base_df = temp_pred_Umsatz_base_df.rename(columns={f'rate_{umsatz_inc}':'Gesamtumsatz'})

    return temp_pred_Umsatz_base_df

# temp_pred_Umsatz_df - Step2
def temp_pred_Umsatz_Grp(BW_pred_Umsatz_df, groups, group_vals, umsatz_inc_rates):
    # Build group-specific revenue projections:
    # For each predefined group, select the corresponding regions,
    # apply the group-specific revenue growth scenario,
    # and standardize the output schema for later concatenation.

    # empty list for Dataframes
    temp_pred_Umsatz_grp_list = []

    for grp, grp_val, umsatz_rate in zip(groups, group_vals, umsatz_inc_rates):
        rate_col = f"rate_{umsatz_rate}"

        temp_df = BW_pred_Umsatz_df.loc[
            BW_pred_Umsatz_df['Regionalverband'].isin(grp),
            ['DN_DT', 'Regionalverband', 'Jahr', rate_col]
        ].copy()

        # Add Group Column
        temp_df['Group'] = grp_val

        temp_df = temp_df.rename(columns={rate_col: 'Gesamtumsatz'})

        temp_pred_Umsatz_grp_list.append(temp_df)

    temp_pred_Umsatz_Grp_df = pd.concat(temp_pred_Umsatz_grp_list, ignore_index=True)

    return temp_pred_Umsatz_Grp_df


# temp_pred_Umsatz_df - Step3(End)
def temp_pred_Umsatz(temp_pred_Umsatz_base_df, temp_pred_Umsatz_Grp_df):
    # Summary
    temp_pred_Umsatz_df = pd.concat([temp_pred_Umsatz_base_df, temp_pred_Umsatz_Grp_df], ignore_index=True).reset_index(drop=True)

    # Column used to distinguish line types in line plot visualizations
    temp_pred_Umsatz_df['Art'] = 'predict'

    return temp_pred_Umsatz_df


# temp_act_Umsatz_df - Step1(End)
# Prepare actual revenue (Umsatz) data in the same structure as the predicted data, so that actual and predicted values can be visualized together in a single line plot.
def temp_act_Umsatz(BW_Industrie_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):

    temp_act_Umsatz_df = BW_Industrie_df[['DN_DT', 'Regionalverband', 'Jahr', 'Gesamtumsatz']].copy()

    # Group1
    temp_act_Umsatz_df.loc[temp_act_Umsatz_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # Group2
    temp_act_Umsatz_df.loc[temp_act_Umsatz_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # Group3
    temp_act_Umsatz_df.loc[temp_act_Umsatz_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # Group0
    temp_act_Umsatz_df.loc[temp_act_Umsatz_df['Group'].isna(), 'Group'] = Grp_val_0

    # Column used to distinguish line types in line plot visualizations
    temp_act_Umsatz_df['Art'] = 'actual'

    return temp_act_Umsatz_df


# temp_middle_Umsatz_df - Step1(End)
# Duplicate 2023 data with Art = 'predict' to prevent line breaks between actual and predicted values in line plot visualizations.
def temp_middle_Umsatz(temp_act_Umsatz_df):

    temp_middle_Umsatz_df = temp_act_Umsatz_df[temp_act_Umsatz_df['Jahr']==2023].copy()

    temp_middle_Umsatz_df['Art'] = 'predict'

    return temp_middle_Umsatz_df


# Umsatz_for_Vis_df - Step1(End)
def Umsatz_for_Vis(temp_act_Umsatz_df, temp_middle_Umsatz_df, temp_pred_Umsatz_df):
    # Combine
    # temp_act_Umsatz_df, temp_middle_Umsatz_df, temp_pred_Umsatz_df
    Umsatz_for_Vis_df = pd.concat([temp_act_Umsatz_df, temp_middle_Umsatz_df], axis=0)
    Umsatz_for_Vis_df = pd.concat([Umsatz_for_Vis_df, temp_pred_Umsatz_df], axis=0).reset_index(drop=True)

    return Umsatz_for_Vis_df


# ---(Build fut_Invest_df : Investitionen Data)---
# temp_2023_Invest_df
# Extract 2023 industrial investment data as the baseline for future investment scenarios,
# and assign region groups to support group-level scenario adjustments.
def temp_2023_Invest(BW_Invest_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    
    temp_2023_Invest_df = BW_Invest_df[BW_Invest_df['Jahr']==2023].copy()
    temp_2023_Invest_df = temp_2023_Invest_df[['DN_DT', 'Regionalverband', 'Jahr', 'Investitionen']]

    # Add Group Column
    # Group1
    temp_2023_Invest_df.loc[temp_2023_Invest_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # Group2
    temp_2023_Invest_df.loc[temp_2023_Invest_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # Group3
    temp_2023_Invest_df.loc[temp_2023_Invest_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # Group0
    temp_2023_Invest_df.loc[temp_2023_Invest_df['Group'].isna(), 'Group'] = Grp_val_0

    return temp_2023_Invest_df


# temp_fut_Invest_df - Step1
# Build a future-year investment frame (2030–2070) by region and group,
# serving as a placeholder for scenario-driven investment allocations.
def temp_fut_Invest_1(TBD_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):

    frame_fut_Invest_df = pd.DataFrame(columns=['DN_DT', 'Regionalverband', 'Jahr', 'Investitionen'])

    frame_fut_Invest_df['Jahr'] = [2030, 2040, 2050, 2060, 2070]

    temp_dict = {}
    for reg in TBD_df['DN_DT'].drop_duplicates():
        # Regierungsverband
        rv = TBD_df.loc[TBD_df['DN_DT'] == reg, 'Regionalverband'].iloc[0]

        temp_df = frame_fut_Invest_df.copy()
        temp_df['DN_DT'] = reg
        temp_df['Regionalverband'] = rv

        # save temp_df in temp_dict
        temp_dict[reg] = temp_df

    # Combine
    temp_fut_Invest_df = pd.concat(temp_dict.values(), axis=0).reset_index(drop=True)

    # Create Group Column
    # Group1
    temp_fut_Invest_df.loc[temp_fut_Invest_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # Group2
    temp_fut_Invest_df.loc[temp_fut_Invest_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # Group3
    temp_fut_Invest_df.loc[temp_fut_Invest_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # Group0
    temp_fut_Invest_df.loc[temp_fut_Invest_df['Group'].isna(), 'Group'] = Grp_val_0

    return temp_fut_Invest_df


# temp_fut_Invest_df - Step2(End)
# Populate future investment values by applying group-specific percentage increases to the 2023 baseline investment for each region.
def temp_fut_Invest_2(temp_fut_Invest_df, temp_2023_Invest_df, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0, invest_inc_1, invest_inc_2, invest_inc_3, invest_inc):

    # Group-specific investment increase rates: invest_inc (Group0), invest_inc_1, invest_inc_2, invest_inc_3

    # Group1
    for reg in temp_fut_Invest_df.loc[temp_fut_Invest_df['Group']==Grp_val_1, 'DN_DT'].drop_duplicates():
        for year in temp_fut_Invest_df['Jahr'].drop_duplicates():
            temp_fut_Invest_df.loc[(temp_fut_Invest_df['DN_DT']==reg)&(temp_fut_Invest_df['Jahr']==year), 'Investitionen'] = (
                (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0] + (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0])*(invest_inc_1/100)).round()
            )

    # Group2
    for reg in temp_fut_Invest_df.loc[temp_fut_Invest_df['Group']==Grp_val_2, 'DN_DT'].drop_duplicates():
        for year in temp_fut_Invest_df['Jahr'].drop_duplicates():
            temp_fut_Invest_df.loc[(temp_fut_Invest_df['DN_DT']==reg)&(temp_fut_Invest_df['Jahr']==year), 'Investitionen'] = (
                (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0] + (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0])*(invest_inc_2/100)).round()
            )

    # Group3
    for reg in temp_fut_Invest_df.loc[temp_fut_Invest_df['Group']==Grp_val_3, 'DN_DT'].drop_duplicates():
        for year in temp_fut_Invest_df['Jahr'].drop_duplicates():
            temp_fut_Invest_df.loc[(temp_fut_Invest_df['DN_DT']==reg)&(temp_fut_Invest_df['Jahr']==year), 'Investitionen'] = (
                (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0] + (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0])*(invest_inc_3/100)).round()
            )

    # Group0
    for reg in temp_fut_Invest_df.loc[temp_fut_Invest_df['Group']==Grp_val_0, 'DN_DT'].drop_duplicates():
        for year in temp_fut_Invest_df['Jahr'].drop_duplicates():
            temp_fut_Invest_df.loc[(temp_fut_Invest_df['DN_DT']==reg)&(temp_fut_Invest_df['Jahr']==year), 'Investitionen'] = (
                (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0] + (temp_2023_Invest_df.loc[temp_2023_Invest_df['DN_DT']==reg, 'Investitionen'].iloc[0])*(invest_inc/100)).round()
            )

    return temp_fut_Invest_df


# fut_Invest_df
# Combine : temp_2023_Invest_df, temp_fut_Invest_df
def fut_Invest(temp_2023_Invest_df, temp_fut_Invest_df):
    fut_Invest_df = pd.concat([temp_2023_Invest_df, temp_fut_Invest_df], axis=0).reset_index(drop=True)

    return fut_Invest_df



# ---(Industrial Electricity Consumption Forecasting Section)---

# temp_pred_empl_for_Vis_df : predicted Beschäftigte
def temp_pred_empl_for_Vis(empl_for_Vis_df):
    temp_pred_empl_for_Vis_df = empl_for_Vis_df[empl_for_Vis_df['Art']=='predict'].copy()
    return temp_pred_empl_for_Vis_df

# temp_pred_Umsatz_for_Vis_df : predicted Gesamtumsatz
def temp_pred_Umsatz_for_Vis(Umsatz_for_Vis_df):
    temp_pred_Umsatz_for_Vis_df = Umsatz_for_Vis_df[Umsatz_for_Vis_df['Art']=='predict'].copy()
    return temp_pred_Umsatz_for_Vis_df


# IndustEV_for_Vismap_df : Combine temp_pred_empl_for_Vis_df, temp_pred_Umsatz_for_Vis_df, fut_Invest_df
def IndustEV_for_Vismap(temp_pred_empl_for_Vis_df, temp_pred_Umsatz_for_Vis_df, fut_Invest_df):
    IndustEV_for_Vismap_df = temp_pred_empl_for_Vis_df.merge(temp_pred_Umsatz_for_Vis_df, on=['DN_DT', 'Regionalverband', 'Jahr', 'Group', 'Art'])
    IndustEV_for_Vismap_df = IndustEV_for_Vismap_df.merge(fut_Invest_df, on=['DN_DT', 'Regionalverband', 'Jahr', 'Group'])

    IndustEV_for_Vismap_df['Gesamtumsatz'] = IndustEV_for_Vismap_df['Gesamtumsatz'].astype(int)
    IndustEV_for_Vismap_df['Investitionen'] = IndustEV_for_Vismap_df['Investitionen'].astype(int)

    return IndustEV_for_Vismap_df


# Scaling + Prediction : IndustEV_for_Vismap_df
def scal_pred_IndustEV_for_Vismap(IndustEV_for_Vismap_df, normal_reg_scaler, special_reg_scaler, normal_ridge_model, Alb_model, Boeblingen_model, Mannheim_model, Ortenaukreis_model, Rastatt_model, Stuttgart_model, Waldshut_model, Karlsruhe_model):
    """
    Scale regional industrial features and predict industrial electricity consumption
    using region-specific Ridge regression models.

    This function applies different StandardScaler instances to normal regions and
    special regions, then performs predictions using appropriate Ridge models for
    each Kreis. The resulting dataframe is intended to be merged with a GeoDataFrame
    for map-based visualization.
    """

    # 1. Region classification and filtering
    #  - Normal regions vs. special regions
    #  - Karlsruhe is handled separately due to its administrative structure


    mask = (IndustEV_for_Vismap_df['DN_DT'].str.contains('Ortenaukreis|Mannheim|Rastatt|Waldshut|Alb-Donau-Kreis|Stuttgart|Böblingen')|((IndustEV_for_Vismap_df['DN_DT'].str.contains('Karlsruhe'))&(IndustEV_for_Vismap_df['DN_DT'].str.contains('kreisfreie'))))

    # Normal regions  : ~mask
    # Special regions : mask

    # 2. Feature selection and scaling
    #  - X_nr : features for normal regions
    #  - X_sr : features for special regions

    X_nr = IndustEV_for_Vismap_df.loc[~mask, ['Beschäftigte', 'Gesamtumsatz', 'Investitionen']]
    X_sr = IndustEV_for_Vismap_df.loc[mask, ['Beschäftigte', 'Gesamtumsatz', 'Investitionen']]

    # Apply StandardScaler
    X_scaled_nr = normal_reg_scaler.transform(X_nr)
    X_scaled_sr = special_reg_scaler.transform(X_sr)

    # Store scaled features back into the dataframe
    IndustEV_for_Vismap_df.loc[~mask, ['scaled_Beschäftigte', 'scaled_Gesamtumsatz', 'scaled_Investitionen']] = X_scaled_nr
    IndustEV_for_Vismap_df.loc[mask, ['scaled_Beschäftigte', 'scaled_Gesamtumsatz', 'scaled_Investitionen']] = X_scaled_sr


    # 3. Electricity consumption prediction

    # 3-1. Prediction for normal regions using a common Ridge model
    y_nr_pred = normal_ridge_model.predict(X_scaled_nr)
    IndustEV_for_Vismap_df.loc[~mask, 'Stromverbrauch(Industrie)'] = y_nr_pred.round()

    # 3-2. Prediction for special regions (excluding Karlsruhe (kreisfreie Stadt))
    for reg, mod in {'Alb' : Alb_model, 'Böblingen' : Boeblingen_model, 'Mannheim' : Mannheim_model, 'Ortenaukreis' : Ortenaukreis_model, 'Rastatt' : Rastatt_model, 'Stuttgart' : Stuttgart_model, 'Waldshut' : Waldshut_model}.items():

        y_pred = mod.predict(IndustEV_for_Vismap_df.loc[IndustEV_for_Vismap_df['DN_DT'].str.contains(reg), ['scaled_Beschäftigte', 'scaled_Gesamtumsatz', 'scaled_Investitionen']].values)

        IndustEV_for_Vismap_df.loc[IndustEV_for_Vismap_df['DN_DT'].str.contains(reg), 'Stromverbrauch(Industrie)'] = y_pred.round()

    # 3-3. Prediction for Karlsruhe (kreisfreie Stadt)
    y_sr_Karl_pred = Karlsruhe_model.predict(IndustEV_for_Vismap_df.loc[(IndustEV_for_Vismap_df['DN_DT'].str.contains('Karlsruhe'))&(IndustEV_for_Vismap_df['DN_DT'].str.contains('kreisfreie')), ['scaled_Beschäftigte', 'scaled_Gesamtumsatz', 'scaled_Investitionen']].values)

    IndustEV_for_Vismap_df.loc[(IndustEV_for_Vismap_df['DN_DT'].str.contains('Karlsruhe'))&(IndustEV_for_Vismap_df['DN_DT'].str.contains('kreisfreie')), 'Stromverbrauch(Industrie)'] = y_sr_Karl_pred.round()

    return IndustEV_for_Vismap_df


# IndustEV_for_Graph_df - Step1 : DataFrame for Graph-Visualization
def temp_act_IndEV(BW_IndustEV_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0):
    temp_act_IndEV_df = BW_IndustEV_df[['DN_DT', 'Regionalverband', 'Jahr', 'Stromverbrauch(Industrie)']].copy()
    # Add Group Column
    # Group1
    temp_act_IndEV_df.loc[temp_act_IndEV_df['Regionalverband'].isin(Grp_1), 'Group'] = Grp_val_1

    # Group2
    temp_act_IndEV_df.loc[temp_act_IndEV_df['Regionalverband'].isin(Grp_2), 'Group'] = Grp_val_2

    # Group3
    temp_act_IndEV_df.loc[temp_act_IndEV_df['Regionalverband'].isin(Grp_3), 'Group'] = Grp_val_3

    # Group0
    temp_act_IndEV_df.loc[temp_act_IndEV_df['Group'].isna(), 'Group'] = Grp_val_0

    # Column used to distinguish line types in line plot visualizations
    temp_act_IndEV_df['Art'] = 'actual'

    return temp_act_IndEV_df


# IndustEV_for_Graph_df - Step2
# Duplicate 2022 data with Art = 'predict' to prevent line breaks between actual and predicted values in line plot visualizations.
def temp_middle_IndEV(temp_act_IndEV_df):

    temp_middle_IndEV_df = temp_act_IndEV_df[temp_act_IndEV_df['Jahr']==2022].copy()

    temp_middle_IndEV_df['Art'] = 'predict'

    return temp_middle_IndEV_df


# IndustEV_for_Graph_df - Step3
def temp_pred_IndEV(IndustEV_for_Vismap_df):
    temp_pred_IndEV_df = IndustEV_for_Vismap_df[['DN_DT', 'Regionalverband', 'Jahr', 'Stromverbrauch(Industrie)', 'Group', 'Art']].copy()
    return temp_pred_IndEV_df


# IndustEV_for_Graph_df - Step4(End)
def IndustEV_for_Graph(temp_act_IndEV_df, temp_middle_IndEV_df, temp_pred_IndEV_df):
    # Combine
    # temp_act_IndEV_df, temp_middle_IndEV_df, temp_pred_IndEV_df
    IndustEV_for_Graph_df = pd.concat([temp_act_IndEV_df, temp_middle_IndEV_df], axis=0)
    IndustEV_for_Graph_df = pd.concat([IndustEV_for_Graph_df, temp_pred_IndEV_df], axis=0).reset_index(drop=True)
    return IndustEV_for_Graph_df


# IndustEV_for_Map_df : DataFrame for Map-Visualization
def IndustEV_for_Map(IndustEV_for_Graph_df):
    # Use data from 2023 onward for map visualization
    IndustEV_for_Map_df = IndustEV_for_Graph_df[IndustEV_for_Graph_df['Jahr']>=2023].copy()

    # Extract 2023 baseline industrial electricity consumption by region (DN_DT)
    baseline_2023_2 = (
        IndustEV_for_Map_df
            .loc[IndustEV_for_Map_df['Jahr'] == 2023]
            .set_index('DN_DT')['Stromverbrauch(Industrie)']
    )

    # Calculate absolute increase relative to the 2023 baseline
    IndustEV_for_Map_df['Zunahme(Industrie)'] = (
        IndustEV_for_Map_df['Stromverbrauch(Industrie)']
        - IndustEV_for_Map_df['DN_DT'].map(baseline_2023_2)
    )

    # Calculate percentage increase relative to the 2023 baseline
    IndustEV_for_Map_df['Zunahmequote(Industrie)'] = (
        IndustEV_for_Map_df['Zunahme(Industrie)'] /
        IndustEV_for_Map_df['DN_DT'].map(baseline_2023_2) * 100
    )

    return IndustEV_for_Map_df



# BW_Summe_Vis_gdf - Step1
# Prepare a combined electricity consumption dataframe for map visualization
# (household + industrial consumption)
def Summe(HausEV_for_Vismap_df, IndustEV_for_Map_df):
    # Merge household and industrial electricity consumption data
    Summe_df = HausEV_for_Vismap_df.merge(IndustEV_for_Map_df, on=['DN_DT', 'Regionalverband', 'Jahr', 'Group', 'Art'])
    # Calculate total electricity consumption
    Summe_df['Stromverbrauch(Summe)'] = Summe_df['Stromverbrauch(Haushalt)'] + Summe_df['Stromverbrauch(Industrie)']
    # Extract 2023 baseline total consumption by region (DN_DT)
    baseline_2023_3 = (
        Summe_df
            .loc[Summe_df['Jahr'] == 2023]
            .set_index('DN_DT')['Stromverbrauch(Summe)']
    )

    # Calculate absolute increase relative to the 2023 baseline
    Summe_df['Zunahme(Summe)'] = (
        Summe_df['Stromverbrauch(Summe)']
        - Summe_df['DN_DT'].map(baseline_2023_3)
    )

    # Calculate percentage increase relative to the 2023 baseline
    Summe_df['Zunahmequote(Summe)'] = (
        Summe_df['Zunahme(Summe)'] /
        Summe_df['DN_DT'].map(baseline_2023_3) * 100
    )
    return Summe_df

# BW_Summe_Vis_gdf - Step2(End)
# Merge geometry information with total electricity consumption data for map visualization
def BW_Summe_Vis(BW_gdf, Summe_df):
    # Merge GeoDataFrame with summed electricity consumption data
    BW_Summe_Vis_gdf = BW_gdf.merge(Summe_df, on=['DN_DT', 'Regionalverband'], how='right')
    # Convert total electricity consumption columns to numeric
    BW_Summe_Vis_gdf['Stromverbrauch(Summe)'] = pd.to_numeric(BW_Summe_Vis_gdf['Stromverbrauch(Summe)'], errors='coerce')
    BW_Summe_Vis_gdf['Zunahme(Summe)'] = pd.to_numeric(BW_Summe_Vis_gdf['Zunahme(Summe)'], errors='coerce')
    BW_Summe_Vis_gdf['Zunahmequote(Summe)'] = pd.to_numeric(BW_Summe_Vis_gdf['Zunahmequote(Summe)'], errors='coerce')
    # Convert household electricity consumption columns to numeric
    BW_Summe_Vis_gdf['Stromverbrauch(Haushalt)'] = pd.to_numeric(BW_Summe_Vis_gdf['Stromverbrauch(Haushalt)'], errors='coerce')
    BW_Summe_Vis_gdf['Zunahme(Haushalt)'] = pd.to_numeric(BW_Summe_Vis_gdf['Zunahme(Haushalt)'], errors='coerce')
    BW_Summe_Vis_gdf['Zunahmequote(Haushalt)'] = pd.to_numeric(BW_Summe_Vis_gdf['Zunahmequote(Haushalt)'], errors='coerce')

    return BW_Summe_Vis_gdf
