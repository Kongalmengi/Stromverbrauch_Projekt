import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st

import joblib

# District_Data
@st.cache_data
def load_BW_District():
    BW_District_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_District.csv')
    return BW_District_df

@st.cache_data
def make_TBD_df(BW_District_df):
    TBD_df = BW_District_df.sort_values(by='DN_DT')[['DN_DT', 'Regionalverband']]
    return TBD_df


# Bevoelkerung_Data
@st.cache_data
def load_BW_Bev(TBD_df):
    BW_Bev_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_Bevoelkerung.csv')
    # prep
    BW_Bev_df = BW_Bev_df.merge(TBD_df, on='DN_DT', how='left')
    # columns reorder
    BW_Bev_df = BW_Bev_df[['DName', 'DType', 'DN_DT', 'Regionalverband', 'Jahr', 'Gemeindegebiet', 'Bevölkerung insgesamt', 'Bevölkerungsdichte', 'Landeswert(BD)']]
    return BW_Bev_df

# Bevoelkerung_pred_table
@st.cache_data
def load_BW_pred_Bev(TBD_df):
    BW_pred_Bev_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_pred_Bev.csv')
    BW_pred_Bev_df = BW_pred_Bev_df.merge(TBD_df, on='DN_DT', how='left')
    BW_pred_Bev_df = BW_pred_Bev_df[['DN_DT', 'Regionalverband', 'Jahr', 'rate_-70', 'rate_-60', 'rate_-50', 'rate_-40', 'rate_-30', 'rate_-20', 'rate_-10', 'rate_0', 'rate_10', 'rate_20', 'rate_30', 'rate_40', 'rate_50', 'rate_60', 'rate_70']]
    return BW_pred_Bev_df

# Industrie_data
@st.cache_data
def load_BW_Industrie(TBD_df):
    BW_Industrie_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_Industrie_Bet.csv')
    BW_Industrie_df = BW_Industrie_df.merge(TBD_df, on='DN_DT', how='left')
    BW_Industrie_df = BW_Industrie_df[['DName', 'DType', 'DN_DT', 'Regionalverband', 'Jahr', 'Betriebe', 'Beschäftigte', 'Gesamtumsatz']]
    return BW_Industrie_df

# Industrie Investitionen
@st.cache_data
def load_BW_Invest(TBD_df):
    BW_Invest_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_Industrie_Invest.csv')
    BW_Invest_df = BW_Invest_df.merge(TBD_df, on='DN_DT', how='left')
    BW_Invest_df = BW_Invest_df[['DName', 'DType', 'DN_DT', 'Regionalverband', 'Jahr', 'Investitionen', 'Investitionen je Beschäftigten']]
    return BW_Invest_df

# Stromverbrauch(Haushalt) pro EW
@st.cache_data
def load_BW_HausEV_EW():
    BW_HausEV_EW_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_HausEV_EW.csv')
    return BW_HausEV_EW_df

# HausEV_Kreis
@st.cache_data
def load_BW_HausEV_Kreis(TBD_df):
    BW_HausEV_Kreis_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_HausEV_Kreis.csv')
    BW_HausEV_Kreis_df = BW_HausEV_Kreis_df.merge(TBD_df, on='DN_DT', how='left')
    BW_HausEV_Kreis_df = BW_HausEV_Kreis_df[['DName', 'DType', 'DN_DT', 'Regionalverband', 'Jahr', 'Stromverbrauch(Haushalt)']]
    return BW_HausEV_Kreis_df

# IndustEV
@st.cache_data
def load_BW_IndustEV(TBD_df):
    BW_IndustEV_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_IndustEV.csv')
    # sort_values
    temp_list = []
    for reg in TBD_df['DN_DT'].drop_duplicates():
        temp_df = BW_IndustEV_df[BW_IndustEV_df['DN_DT']==reg].copy()
        temp_df = temp_df.sort_values(by='Jahr')
        temp_list.append(temp_df)

    BW_IndustEV_df = pd.concat(temp_list, ignore_index=True).reset_index(drop=True)

    BW_IndustEV_df = BW_IndustEV_df.merge(TBD_df, on='DN_DT', how='left')
    BW_IndustEV_df = BW_IndustEV_df[['DName', 'DType', 'DN_DT', 'Regionalverband', 'Jahr', 'Stromverbrauch(Industrie)']]
    return BW_IndustEV_df


# GeoJSON
@st.cache_data
def load_BW_gdf(TBD_df):
    BW_gdf = gpd.read_file('Data3/Baden-Wuerttemberg/BW_gdf.json')
    BW_gdf = BW_gdf.drop(columns=['District_Code', 'Size'])
    BW_gdf = BW_gdf.sort_values(by='DN_DT')
    BW_gdf = BW_gdf.merge(TBD_df, on='DN_DT', how='left')
    BW_gdf = BW_gdf[['Bundesland', 'Regierungsbezirk', 'DName', 'DType', 'DN_DT', 'Regionalverband', 'geometry']]
    return BW_gdf

# StandardScaler
@st.cache_data
def load_scaler(path):
    scaler = joblib.load(path)
    return scaler

# Ridge_Models
@st.cache_data
def load_Ridge_model(path):
    model = joblib.load(path)
    return model


# pred_empl_rate_table
@st.cache_data
def load_pred_empl_r(TBD_df):
    pred_empl_r_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_pred_empl_r.csv')
    pred_empl_r_df = pred_empl_r_df.merge(TBD_df, on='DN_DT', how='left')
    pred_empl_r_df = pred_empl_r_df[['DN_DT', 'Regionalverband', 'Jahr', 'rate_-50', 'rate_-40', 'rate_-30', 'rate_-20', 'rate_-10', 'rate_0', 'rate_10', 'rate_20', 'rate_30', 'rate_40', 'rate_50']]
    return pred_empl_r_df


# BW_pred_Umsatz_df_table
@st.cache_data
def load_BW_pred_Umsatz(TBD_df):
    BW_pred_Umsatz_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_pred_Umsatz.csv')
    BW_pred_Umsatz_df = BW_pred_Umsatz_df.merge(TBD_df, on='DN_DT', how='left')
    return BW_pred_Umsatz_df



# ------(files for Projekt_Overview.py)------

# combined
@st.cache_data
def load_BW_Combined_df(TBD_df):
    BW_Combined_df = pd.read_csv('Data3/Baden-Wuerttemberg/BW_Combined_Dataset.csv')
    BW_Combined_df = BW_Combined_df.merge(TBD_df, on='DN_DT', how='left')
    BW_Combined_df = BW_Combined_df[['DN_DT', 'Regionalverband', 'Jahr', 'Bevölkerung insgesamt', 'Betriebe', 'Beschäftigte', 'Gesamtumsatz', 'Investitionen', 'Stromverbrauch(Industrie)', 'Stromverbrauch(Haushalt)', 'Beschäftigungsquote']]
    return BW_Combined_df


# Bundesland_Indust
@st.cache_data
def load_BW_Indust_demmo():
    indu_total_df = pd.read_csv('Data3\DemoData\Bundesland_Indust.csv')
    return indu_total_df


# IndustrieData
@st.cache_data
def load_BW_raw_Indust():
    Indust_df = pd.read_csv('Data3\DemoData\IndustrieData.csv')
    return Indust_df


# Wrapper functions for joblib.load() to improve readability (separate loaders for scaler and model)

# LinearRegression
@st.cache_data
def load_LinearRegression(path):
    model = joblib.load(path)
    return model