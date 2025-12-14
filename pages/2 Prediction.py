import streamlit as st

import numpy as np
import pandas as pd
import geopandas as gpd


import plotly.express as px

# for the data loading
from utils.load_files import load_BW_District, make_TBD_df, load_BW_Bev, load_BW_pred_Bev, load_BW_Industrie, load_BW_Invest, load_BW_HausEV_EW, load_BW_HausEV_Kreis, load_BW_IndustEV, load_BW_Combined_df, load_BW_gdf, load_pred_empl_r, load_BW_pred_Umsatz

# for the scaler loading
from utils.load_files import load_scaler

# for the models loading
from utils.load_files import load_Ridge_model

# for HausEV prep.
from utils.HausEV_funcs import temp_pred_Bev_base, temp_pred_Bev_Grp, temp_pred_Bev, temp_act_Bev, temp_middle_Bev, Bev_for_Vis, frame_pred_HausEV, temp_pred_HausEV_1, temp_pred_HausEV_2, temp_pred_HausEV_3, temp_act_HausEV, temp_middle_HausEV, HausEV_for_Vis, HausEV_for_Vismap

# temp_pred_Bev_base_df
# temp_pred_Bev_Grp_df
# temp_pred_Bev_df
# temp_act_Bev_df
# temp_middle_Bev_df
# Bev_for_Vis_df
# frame_pred_HausEV_df
# temp_pred_HausEV_df_1
# temp_pred_HausEV_df_2
# temp_pred_HausEV_df_3
# temp_act_HausEV_df
# temp_middle_HausEV_df
# HausEV_for_Vis_df
# HausEV_for_Vismap_df
# BW_Bev_Vis_gdf : Deleted
# BW_gdf
# BW_Combined_df
# pred_empl_r_df
# BW_pred_Umsatz_df

# for IndustEV prep.
from utils.IndustEV_funcs import pred_Bev_for_empl, act_Bev_for_empl, middle_Bev_for_empl, empl_for_Vis, temp_pred_Umsatz_base, temp_pred_Umsatz_Grp, temp_pred_Umsatz, temp_act_Umsatz, temp_middle_Umsatz, Umsatz_for_Vis, temp_2023_Invest, temp_fut_Invest_1, temp_fut_Invest_2, fut_Invest

# pred_Bev_for_empl
# act_Bev_for_empl
# middle_Bev_for_empl
# empl_for_Vis
# temp_pred_Umsatz_base
# temp_pred_Umsatz_Grp
# temp_pred_Umsatz
# temp_act_Umsatz
# temp_middle_Umsatz
# Umsatz_for_Vis
# temp_2023_Invest
# temp_fut_Invest_1
# temp_fut_Invest_2
# fut_Invest

# for IndustEV_pred : scaling + pred.
from utils.IndustEV_funcs import temp_pred_empl_for_Vis, temp_pred_Umsatz_for_Vis, IndustEV_for_Vismap, scal_pred_IndustEV_for_Vismap, temp_act_IndEV, temp_middle_IndEV, temp_pred_IndEV, IndustEV_for_Graph
# temp_pred_empl_for_Vis
# temp_pred_Umsatz_for_Vis
# IndustEV_for_Vismap
# scal_pred_IndustEV_for_Vismap
# temp_act_IndEV
# temp_middle_IndEV
# temp_pred_IndEV
# IndustEV_for_Graph

# for IndustEV prep_2. : gpd
from utils.IndustEV_funcs import IndustEV_for_Map, Summe, BW_Summe_Vis
# IndustEV_for_Map
# BW_IndustEV_Vis : Deleted
# Summe
# BW_Summe_Vis


# for Graph
# about Haushalt Electricity Graph
from utils.graph_Vis import make_Bev_fig, make_HausEV_fig
# about Industrie Electricity Graph
from utils.graph_Vis import make_empl_fig, make_umsatz_fig, make_IndustEV_fig

# for Map_Vis
# about Haushalt Electricity Map
from utils.map_Vis import make_map_HausEV_fig, make_map_IndustEV_fig, make_map_SummeEV_fig


# page setting
st.set_page_config(
    page_title='Prognossen des Stromverbrauchs',
    page_icon=':zap:',
    layout='wide',
)


st.markdown(
    """
    # :house: Prognosen des Stromverbrauchs :factory:
    """
)

st.divider()


# Daten
BW_District_df = load_BW_District()
TBD_df = make_TBD_df(BW_District_df)
BW_Bev_df = load_BW_Bev(TBD_df)
BW_pred_Bev_df = load_BW_pred_Bev(TBD_df)
BW_Industrie_df = load_BW_Industrie(TBD_df)
BW_Invest_df = load_BW_Invest(TBD_df)
BW_HausEV_EW_df =  load_BW_HausEV_EW()
BW_HausEV_Kreis_df = load_BW_HausEV_Kreis(TBD_df)
BW_IndustEV_df = load_BW_IndustEV(TBD_df)
BW_Combined_df = load_BW_Combined_df(TBD_df)
BW_gdf = load_BW_gdf(TBD_df)
pred_empl_r_df = load_pred_empl_r(TBD_df)
BW_pred_Umsatz_df = load_BW_pred_Umsatz(TBD_df)

# Scalers
normal_reg_scaler = load_scaler('models/Scalers/normal_scaler.pkl')
special_reg_scaler = load_scaler('models/Scalers/special_scaler.pkl')

# models
normal_ridge_model = load_Ridge_model('models/Ridge_models/normal_ridge_model.pkl')

Alb_model = load_Ridge_model('models/Ridge_models/Alb-Donau-Kreis_ridge_model.pkl')
Boeblingen_model = load_Ridge_model('models/Ridge_models/Boeblingen_ridge_model.pkl')
Karlsruhe_model = load_Ridge_model('models/Ridge_models/Karlsruhe_ridge_model.pkl')
Mannheim_model = load_Ridge_model('models/Ridge_models/Mannheim_ridge_model.pkl')
Ortenaukreis_model = load_Ridge_model('models/Ridge_models/Ortenaukreis_ridge_model.pkl')
Rastatt_model = load_Ridge_model('models/Ridge_models/Rastatt_ridge_model.pkl')
Stuttgart_model = load_Ridge_model('models/Ridge_models/Stuttgart_ridge_model.pkl')
Waldshut_model = load_Ridge_model('models/Ridge_models/Waldshut_ridge_model.pkl')


def main():

    # List for Multiselect
    all_Verband = TBD_df['Regionalverband'].drop_duplicates().tolist()

    # session_state setting
    for key in ['group1', 'group2', 'group3']:
        if key not in st.session_state:
            st.session_state[key] = []


    # Columns for Widgets and Map

    main_cols = st.columns([1, 2])


    # Input setting Widgets
    with main_cols[0]:
        # Widget Container 1
        with st.container(border=True):
            st.markdown(
                """
                ### Gruppeneinstellungen
                """
            )
            st.divider()
            # Group1 Settings
            st.write('Regionseinstellungen - Gruppe 1')
            # Select Group1
            Grp_1 = st.multiselect('Regionen für Gruppe 1 auswählen', all_Verband, key='group1')
            # Exclude Group1 selections from Group2 and Group3
            st.session_state['group2'] = [
                k for k in st.session_state['group2']
                if k not in st.session_state['group1']
            ]
            st.session_state['group3'] = [
                k for k in st.session_state['group3']
                if k not in st.session_state['group1']
            ]

            # Group1 - Details
            with st.popover('Detaileinstellungen - Gruppe 1'):
                # Group1 values - Haushalt
                Grp_val_1 = 'group_1'
                EW_inc_rate_1 = st.slider('Wachstumsquote der Bevölkerung(%, bis 2070)', min_value=-70, max_value=70, value=10, step=10, key='EW_inc_rate_1')
                EW_GJ_1 = st.slider('Haushaltlicher Stromverbrauch pro Kopf(GJ)', min_value=4.0, max_value=7.5, value=4.96, step=0.01, key='EW_GJ_1')

                # Group1 values - Industrie
                empl_r_inc_1 = st.slider('Proportionale Veränderung der Beschäftigungsquote(%, bis 2070)', min_value=-50, max_value=50, value=10, step=10, key='empl_r_inc_1')
                umsatz_inc_1 = st.slider('Umsatzwachstumsrate (%, bis 2070)', min_value=-70, max_value=70, value=10, step=10, key='umsatz_inc_1')
                invest_inc_1 = st.slider('Investitionserhöhung (%, gegenüber 2023)', min_value=-70, max_value=70, value=0, step=1, key='invest_inc_1')

            st.divider()



            # Group2 Settings
            st.write('Regionseinstellungen - Gruppe 2')
            # Build the option list for Group 2 by excluding items already selected in Group 1 and Group 3
            remaining_for_group2 = [
                k for k in all_Verband
                if (k not in st.session_state['group1'])
                and (k not in st.session_state['group3'])
            ]
            # Select Group2
            Grp_2 = st.multiselect('Regionen für Gruppe 1 auswählen', remaining_for_group2, key='group2')

            # Remove items from Group 3 that are now selected in Group 2
            st.session_state['group3'] = [
                k for k in st.session_state['group3']
                if k not in st.session_state['group2']
            ]

            # Group2 - Details
            with st.popover('Detaileinstellungen - Gruppe 2'):
                # Group2 values - Haushalt
                Grp_val_2 = 'group_2'
                EW_inc_rate_2 = st.slider('Wachstumsquote der Bevölkerung(%, bis 2070)', min_value=-70, max_value=70, value=10, step=10, key='EW_inc_rate_2')
                EW_GJ_2 = st.slider('Haushaltlicher Stromverbrauch pro Kopf(GJ)', min_value=4.0, max_value=7.5, value=4.96, step=0.01, key='EW_GJ_2')

                # Group2 values - Industrie
                empl_r_inc_2 = st.slider('Proportionale Veränderung der Beschäftigungsquote(%, bis 2070)', min_value=-50, max_value=50, value=10, step=10, key='empl_r_inc_2')
                umsatz_inc_2 = st.slider('Umsatzwachstumsrate (%, bis 2070)', min_value=-70, max_value=70, value=10, step=10, key='umsatz_inc_2')
                invest_inc_2 = st.slider('Investitionserhöhung (%, gegenüber 2023)', min_value=-70, max_value=70, value=0, step=1, key='invest_inc_2')

            st.divider()




            # Group3 Settings
            st.write('Regionseinstellungen - Gruppe 3')
            # Build the option list for Group 3 by excluding items already selected in Group 1 and Group 2
            remaining_for_group3 = [
                k for k in all_Verband
                if (k not in st.session_state['group1'])
                and (k not in st.session_state['group2'])
            ]

            # Select Group3
            Grp_3 = st.multiselect('Regionen für Gruppe 1 auswählen', remaining_for_group3, key='group3')

            # Group3 - Details
            with st.popover('Detaileinstellungen - Gruppe 3'):
                # Group3 values - Haushalt
                Grp_val_3 = 'group_3'
                EW_inc_rate_3 = st.slider('Wachstumsquote der Bevölkerung(%, bis 2070)', min_value=-70, max_value=70, value=10, step=10, key='EW_inc_rate_3')
                EW_GJ_3 = st.slider('Haushaltlicher Stromverbrauch pro Kopf(GJ)', min_value=4.0, max_value=7.5, value=4.96, step=0.01, key='EW_GJ_3')

                # Group3 values - Industrie
                empl_r_inc_3 = st.slider('Proportionale Veränderung der Beschäftigungsquote(%, bis 2070)', min_value=-50, max_value=50, value=10, step=10, key='empl_r_inc_3')
                umsatz_inc_3 = st.slider('Umsatzwachstumsrate (%, bis 2070)', min_value=-70, max_value=70, value=10, step=10, key='umsatz_inc_3')
                invest_inc_3 = st.slider('Investitionserhöhung (%, gegenüber 2023)', min_value=-70, max_value=70, value=0, step=1, key='invest_inc_3')

            st.divider()



            # Other Region Settings
            st.write('Regionseinstellungen - Andere Regionen')
            # Other Region - Details
            with st.popover('Detaileinstellungen - Andere Regionen'):
                # Group0 values - Haushalt
                Grp_val_0 = 'group_0'
                EW_inc_rate = st.slider('Wachstumsquote der Bevölkerung(%, bis 2070)', min_value=-70, max_value=70, value=10, step=10, key='EW_inc_rate')
                # EV_1EW
                EW_GJ = st.slider('Haushaltlicher Stromverbrauch pro Kopf(GJ)', min_value=4.0, max_value=7.5, value=4.96, step=0.01, key='EW_GJ')

                # Group0 values - Industrie
                empl_r_inc = st.slider('Proportionale Veränderung der Beschäftigungsquote(%, bis 2070)', min_value=-50, max_value=50, value=10, step=10, key='empl_r_inc')
                umsatz_inc = st.slider('Umsatzwachstumsrate (%, bis 2070)', min_value=-70, max_value=70, value=10, step=10, key='umsatz_inc')
                invest_inc = st.slider('Investitionserhöhung (%, gegenüber 2023)', min_value=-70, max_value=70, value=0, step=1, key='invest_inc')

    # values_list
    groups = [Grp_1, Grp_2, Grp_3]
    group_vals = [Grp_val_1, Grp_val_2, Grp_val_3]
    ew_inc_rates = [EW_inc_rate_1, EW_inc_rate_2, EW_inc_rate_3]
    ew_gj_vals = [EW_GJ_1, EW_GJ_2, EW_GJ_3]
    umsatz_inc_rates = [umsatz_inc_1, umsatz_inc_2, umsatz_inc_3]

    # Add Widgets in main_cols[1]
    with main_cols[1]:
        with st.container(border=True):
            sub_cols = st.columns(2)

            with sub_cols[0]:
                st.markdown(
                    """
                    ### Karteneinstellungen
                    """
                )

                st.space(size='small')
                # year for Map Visual
                selected_year = st.select_slider('Jahr', options=[2023, 2030, 2040, 2050, 2060, 2070])

            with sub_cols[1]:
                # Map Option
                map_type = st.radio(
                    'Anzeigeoptionen der Karte',
                    ['Stromverbrauch', 'Zunahme', 'Zunahmequote'],
                    captions=['Jährlicher Stromverbrauch', 'Zunahme des Stromverbrauchs gegenüber 2023', 'Zunahmequote des Stromverbrauchs gegenüber 2023']
                    )
                if map_type=='Stromverbrauch':
                    colorbar_title = 'Stromverbrauch(TJ)'
                elif map_type=='Zunahme':
                    colorbar_title = 'Zunahme des Stromverbrauchs(TJ)'
                elif map_type=='Zunahmequote':
                    colorbar_title = 'Zunahmequote des Stromverbrauchs(%)'



    # ------(DataFrame preprocessing for Haushalt data)------

    # I. create Bev_for_Vis_df - (DataFrame : Bevölkerung)

    # 1. temp_pred_Bev_df - Step1
    temp_pred_Bev_base_df = temp_pred_Bev_base(BW_pred_Bev_df, Grp_1, Grp_2, Grp_3, EW_inc_rate, Grp_val_0)

    temp_pred_Bev_Grp_df = temp_pred_Bev_Grp(BW_pred_Bev_df, groups, group_vals, ew_inc_rates)

    temp_pred_Bev_df = temp_pred_Bev(temp_pred_Bev_base_df, temp_pred_Bev_Grp_df)

    # 2. temp_act_Bev_df - Step2
    temp_act_Bev_df = temp_act_Bev(BW_Bev_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0)

    # 3. temp_middle_Bev_df - Step3
    temp_middle_Bev_df = temp_middle_Bev(temp_act_Bev_df)

    # 4. create Bev_for_Vis_df
    Bev_for_Vis_df = Bev_for_Vis(temp_pred_Bev_df, temp_middle_Bev_df, temp_act_Bev_df)


    # II. create HausEV_for_Vis_df - (DataFrame : HaushaltEV (HausEV) for Visualization)

    # 1. create frame_pred_HausEV_df
    frame_pred_HausEV_df = frame_pred_HausEV()

    # 2. temp_pred_HausEV_df
    temp_pred_HausEV_df = temp_pred_HausEV_1(frame_pred_HausEV_df, TBD_df)

    temp_pred_HausEV_df = temp_pred_HausEV_2(temp_pred_HausEV_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0)

    temp_pred_HausEV_df = temp_pred_HausEV_3(temp_pred_HausEV_df, Bev_for_Vis_df, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0, EW_GJ_1, EW_GJ_2, EW_GJ_3, EW_GJ)

    # 3. create temp_act_HausEV_df
    temp_act_HausEV_df = temp_act_HausEV(BW_HausEV_Kreis_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0)

    # 4. create temp_middle_HausEV_df
    temp_middle_HausEV_df = temp_middle_HausEV(temp_act_HausEV_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0)

    # 5. create HausEV_for_Vis_df
    HausEV_for_Vis_df = HausEV_for_Vis(temp_act_HausEV_df, temp_middle_HausEV_df, temp_pred_HausEV_df)


    # III. create HausEV_for_Vismap_df

    HausEV_for_Vismap_df = HausEV_for_Vismap(HausEV_for_Vis_df)



    # ------(DataFrame preprocessing for Industrie data)------

    # I. create empl_for_Vis_df

    pred_Bev_for_empl_df = pred_Bev_for_empl(Bev_for_Vis_df, TBD_df, pred_empl_r_df, Grp_1, Grp_2, Grp_3, empl_r_inc_1, empl_r_inc_2, empl_r_inc_3, empl_r_inc)
    act_Bev_for_empl_df = act_Bev_for_empl(BW_Industrie_df, groups, group_vals, Grp_val_0)
    middle_Bev_for_empl_df = middle_Bev_for_empl(act_Bev_for_empl_df)
    empl_for_Vis_df = empl_for_Vis(act_Bev_for_empl_df, middle_Bev_for_empl_df, pred_Bev_for_empl_df)


    # II. create Umsatz_for_Vis_df
    temp_pred_Umsatz_base_df = temp_pred_Umsatz_base(BW_pred_Umsatz_df, Grp_1, Grp_2, Grp_3, Grp_val_0, umsatz_inc)
    temp_pred_Umsatz_Grp_df = temp_pred_Umsatz_Grp(BW_pred_Umsatz_df, groups, group_vals, umsatz_inc_rates)
    temp_pred_Umsatz_df = temp_pred_Umsatz(temp_pred_Umsatz_base_df, temp_pred_Umsatz_Grp_df)
    temp_act_Umsatz_df = temp_act_Umsatz(BW_Industrie_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0)
    temp_middle_Umsatz_df = temp_middle_Umsatz(temp_act_Umsatz_df)
    Umsatz_for_Vis_df = Umsatz_for_Vis(temp_act_Umsatz_df, temp_middle_Umsatz_df, temp_pred_Umsatz_df)


    # III. create fut_Invest_df
    temp_2023_Invest_df = temp_2023_Invest(BW_Invest_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0)
    temp_fut_Invest_df = temp_fut_Invest_1(TBD_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0)
    temp_fut_Invest_df = temp_fut_Invest_2(temp_fut_Invest_df, temp_2023_Invest_df, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0, invest_inc_1, invest_inc_2, invest_inc_3, invest_inc)
    fut_Invest_df = fut_Invest(temp_2023_Invest_df, temp_fut_Invest_df)
    

    # ------(for Visualization)------

    # I. create IndustEV_for_Vismap_df
    temp_pred_empl_for_Vis_df = temp_pred_empl_for_Vis(empl_for_Vis_df)
    temp_pred_Umsatz_for_Vis_df = temp_pred_Umsatz_for_Vis(Umsatz_for_Vis_df)
    IndustEV_for_Vismap_df = IndustEV_for_Vismap(temp_pred_empl_for_Vis_df, temp_pred_Umsatz_for_Vis_df, fut_Invest_df)
    IndustEV_for_Vismap_df = scal_pred_IndustEV_for_Vismap(IndustEV_for_Vismap_df, normal_reg_scaler, special_reg_scaler, normal_ridge_model, Alb_model, Boeblingen_model, Mannheim_model, Ortenaukreis_model, Rastatt_model, Stuttgart_model, Waldshut_model, Karlsruhe_model)

    # II. for IndustEV_for_Graph_df
    temp_act_IndEV_df = temp_act_IndEV(BW_IndustEV_df, Grp_1, Grp_2, Grp_3, Grp_val_1, Grp_val_2, Grp_val_3, Grp_val_0)
    temp_middle_IndEV_df = temp_middle_IndEV(temp_act_IndEV_df)
    temp_pred_IndEV_df = temp_pred_IndEV(IndustEV_for_Vismap_df)
    IndustEV_for_Graph_df = IndustEV_for_Graph(temp_act_IndEV_df, temp_middle_IndEV_df, temp_pred_IndEV_df)


    # IndustEV_for_Map_df : for IndustEV-Visualization
    IndustEV_for_Map_df = IndustEV_for_Map(IndustEV_for_Graph_df)


    # BW_Summe_Vis_gdf - sum HausEV and IndustEV for Visualization (gdf)
    Summe_df = Summe(HausEV_for_Vismap_df, IndustEV_for_Map_df)
    BW_Summe_Vis_gdf = BW_Summe_Vis(BW_gdf, Summe_df)



    # Add maps in main_cols[1]
    with main_cols[1]:
        # Map Visualization Section
        tab1, tab2, tab3 = st.tabs(['Stromverbrauch(Haushalt)', 'Stromverbrauch(Industrie)', 'Stromverbrauch(Gesamt)'])

        with tab1:
            st.header('Stromverbrauch(Haushalt)')
            # Map Visualization
            fig_map_HausEV = make_map_HausEV_fig(BW_Summe_Vis_gdf, selected_year, map_type, colorbar_title) # BW_Bev_Vis_gdf (함수 : BW_Bev_Vis) 를 원래 사용. BW_Summe_Vis_gdf로 바꿈. 이것이 종합 결과니까.

            st.plotly_chart(fig_map_HausEV, use_container_width=True, key='fig_map_HausEV')



        with tab2:
            st.header('Stromverbrauch(Industrie)')
            # Map Visualization
            fig_map_IndustEV = make_map_IndustEV_fig(BW_Summe_Vis_gdf, selected_year, map_type, colorbar_title) # BW_IndustEV_Vis_gdf (함수 : BW_IndustEV_Vis) 를 원래 사용. BW_Summe_Vis_gdf로 바꿈. 이것이 종합 결과니까.

            st.plotly_chart(fig_map_IndustEV, use_container_width=True, key='fig_map_IndustEV')


        with tab3:
            st.header('Stromverbrauch(Gesamt)')
            # Map Visualization
            fig_map__SummeEV = make_map_SummeEV_fig(BW_Summe_Vis_gdf, selected_year, map_type, colorbar_title)

            st.plotly_chart(fig_map__SummeEV, use_container_width=True, key='fig_map__SummeEV')


    st.divider()

    st.title('Regionaler Vergleich')
    # Regional Graph Section
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.header('Region 1')
            # select options 1 : Kreis
            var_reg_1 = st.selectbox('Wählen Sie bitte einen Kreis aus', TBD_df['DN_DT'].unique(), key='var_reg_1')

            # Visualization (Region 1) : Bevölkerung
            fig_Bev_1 = make_Bev_fig(Bev_for_Vis_df, var_reg_1)
            st.plotly_chart(fig_Bev_1, key='fig_Bev_1')

            st.divider()

            # Visualization (Region 1) : HausEV
            fig_HausEV_1 = make_HausEV_fig(HausEV_for_Vis_df, var_reg_1)
            st.plotly_chart(fig_HausEV_1, key='fig_HausEV_1')

            st.divider()

            # Visualization (Region 1) : Beschäftigte
            fig_empl_1 = make_empl_fig(empl_for_Vis_df, var_reg_1)
            st.plotly_chart(fig_empl_1, key='fig_empl_1')

            st.divider()

            # Visualization (Region 1) : Gesamtumsatz
            fig_umsatz_1 = make_umsatz_fig(Umsatz_for_Vis_df, var_reg_1)
            st.plotly_chart(fig_umsatz_1, key='fig_umsatz_1')

            st.divider()

            # Visualization (Region 1) : IndustEV
            fig_IndustEV_1 = make_IndustEV_fig(IndustEV_for_Graph_df, var_reg_1)
            st.plotly_chart(fig_IndustEV_1, key='fig_IndustEV_1')




    with col2:
        with st.container(border=True):
            st.header('Region 2')
            # select options 2 : Kreis
            var_reg_2 = st.selectbox('Wählen Sie bitte einen Kreis aus', TBD_df['DN_DT'].unique(), key='var_reg_2')

            # Visualization (Region 2) : Bevölkerung
            fig_Bev_2 = make_Bev_fig(Bev_for_Vis_df, var_reg_2)
            st.plotly_chart(fig_Bev_2, key='fig_Bev_2')

            st.divider()

            # Visualization (Region 2) : HausEV
            fig_HausEV_2 = make_HausEV_fig(HausEV_for_Vis_df, var_reg_2)
            st.plotly_chart(fig_HausEV_2, key='fig_HausEV_2')

            st.divider()

            # Visualization (Region 2) : Beschäftigte
            fig_empl_2 = make_empl_fig(empl_for_Vis_df, var_reg_2)
            st.plotly_chart(fig_empl_2, key='fig_empl_2')

            st.divider()

            # Visualization (Region 2) : Gesamtumsatz
            fig_umsatz_2 = make_umsatz_fig(Umsatz_for_Vis_df, var_reg_2)
            st.plotly_chart(fig_umsatz_2, key='fig_umsatz_2')

            st.divider()

            # Visualization (Region 2) : IndustEV
            fig_IndustEV_2 = make_IndustEV_fig(IndustEV_for_Graph_df, var_reg_2)
            st.plotly_chart(fig_IndustEV_2, key='fig_IndustEV_2')




if __name__ == "__main__":
    main()