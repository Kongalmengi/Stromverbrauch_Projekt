import numpy as np
import pandas as pd

# for Graph
import plotly.express as px
import plotly.graph_objects as go

# ------(for HaushaltEV Data)------

# Graph : Bevölkerung insgesamt
def make_Bev_fig(Bev_for_Vis_df, var_reg):
    fig = px.line(Bev_for_Vis_df[Bev_for_Vis_df['DN_DT']==var_reg], x='Jahr', y='Bevölkerung insgesamt', 
                 color='Art', line_dash='Art', markers=True, title=f'Entwicklung der Bevölkerung in {var_reg} (Ist + Prognose)')
    return fig


# Graph : Stromverbrauch(Haushalt)
def make_HausEV_fig(HausEV_for_Vis_df, var_reg):
    fig = px.line(HausEV_for_Vis_df[HausEV_for_Vis_df['DN_DT']==var_reg], x='Jahr', y='Stromverbrauch(Haushalt)', 
                 color='Art', line_dash='Art', markers=True, title=f'Stromverbrauch der Haushalte in {var_reg} (Ist + Prognose)')
    return fig



# ------(for Industrie Data)------

# Graph : Beschäftigte
def make_empl_fig(empl_for_Vis_df, var_reg):
    fig = px.line(empl_for_Vis_df[empl_for_Vis_df['DN_DT']==var_reg], x='Jahr', y='Beschäftigte', 
                 color='Art', line_dash='Art', markers=True, title=f'Entwicklung der Industriebeschäftigten in {var_reg} (Ist + Prognose)')
    return fig


# Graph : Gesamtumsatz
def make_umsatz_fig(Umsatz_for_Vis_df, var_reg):
    fig = px.line(Umsatz_for_Vis_df[Umsatz_for_Vis_df['DN_DT']==var_reg], x='Jahr', y='Gesamtumsatz', 
                 color='Art', line_dash='Art', markers=True, title=f'Entwicklung des Industrienumsatzes in {var_reg} (Ist + Prognose)')
    return fig

# Graph : Stromverbrauch(Industrie)
def make_IndustEV_fig(IndustEV_for_Graph_df, var_reg):
    fig = px.line(IndustEV_for_Graph_df[IndustEV_for_Graph_df['DN_DT']==var_reg], x='Jahr', y='Stromverbrauch(Industrie)', 
                 color='Art', line_dash='Art', markers=True, title=f'Stromverbrauch der Industrie in {var_reg} (Ist + Prognose)')
    return fig