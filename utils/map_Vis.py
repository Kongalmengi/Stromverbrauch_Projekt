import numpy as np
import pandas as pd

# for Map-Visualization
import plotly.express as px
import plotly.graph_objects as go


# ------(for HaushaltEV Data)------

def make_map_HausEV_fig(BW_Bev_Vis_gdf, selected_year, map_type, colorbar_title):
    fig = px.choropleth_map(
        BW_Bev_Vis_gdf[BW_Bev_Vis_gdf['Jahr']==selected_year],
        geojson=BW_Bev_Vis_gdf[BW_Bev_Vis_gdf['Jahr']==selected_year].__geo_interface__,
        locations='DN_DT',
        featureidkey='properties.DN_DT',
        color=f'{map_type}(Haushalt)',   # var1
        hover_name='DN_DT',
        hover_data={
            'Regionalverband': True,
            'DN_DT': False,
            'Stromverbrauch(Haushalt)': True,
            'Zunahme(Haushalt)' : True,
            'Zunahmequote(Haushalt)' : True
        },
        map_style='carto-positron',
        center={'lat': 48.5, 'lon': 9.4},
        zoom=6,
        opacity=0.6,
        color_continuous_scale='RdBu_r'
    )

    fig.update_layout(
        height=650,
        # coloraxis_colorbar_title=colorbar_title
        coloraxis_cmid=0,
        coloraxis_colorbar=dict(
            x=0.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            len=0.6,
            thickness=12,
            tickfont=dict(color='black'),
            title=dict(
                text=colorbar_title, # var2
                font=dict(color='black')
            )
        )
    )

    fig.update_traces(
        marker_line_width=0.5,
        marker_line_color='black'
    )

    return fig


# ------(for IndustrieEV Data)------
def make_map_IndustEV_fig(BW_IndustEV_Vis_gdf, selected_year, map_type, colorbar_title):
    fig = px.choropleth_map(
        BW_IndustEV_Vis_gdf[BW_IndustEV_Vis_gdf['Jahr']==selected_year],
        geojson=BW_IndustEV_Vis_gdf[BW_IndustEV_Vis_gdf['Jahr']==selected_year].__geo_interface__,
        locations='DN_DT',
        featureidkey='properties.DN_DT',
        color=f'{map_type}(Industrie)',   # var1
        hover_name='DN_DT',
        hover_data={
            'Regionalverband': True,
            'DN_DT': False,
            'Stromverbrauch(Industrie)': True,
            'Zunahme(Industrie)' : True,
            'Zunahmequote(Industrie)' : True
        },
        map_style='carto-positron',
        center={'lat': 48.5, 'lon': 9.4},
        zoom=6,
        opacity=0.6,
        color_continuous_scale='RdBu_r'
    )

    fig.update_layout(
        height=650,
        # coloraxis_colorbar_title=colorbar_title
        coloraxis_cmid=0,
        coloraxis_colorbar=dict(
            x=0.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            len=0.6,
            thickness=12,
            tickfont=dict(color='black'),
            title=dict(
                text=colorbar_title, # var2
                font=dict(color='black')
            )
        )
    )

    fig.update_traces(
        marker_line_width=0.5,
        marker_line_color='black'
    )

    return fig


# ------(HaushaltEV + IndustrieEV)------
def make_map_SummeEV_fig(BW_Summe_Vis_gdf, selected_year, map_type, colorbar_title):
    fig = px.choropleth_map(
        BW_Summe_Vis_gdf[BW_Summe_Vis_gdf['Jahr']==selected_year],
        geojson=BW_Summe_Vis_gdf[BW_Summe_Vis_gdf['Jahr']==selected_year].__geo_interface__,
        locations='DN_DT',
        featureidkey='properties.DN_DT',
        color=f'{map_type}(Summe)',   # var1
        hover_name='DN_DT',
        hover_data={
            'Regionalverband': True,
            'DN_DT': False,
            'Stromverbrauch(Summe)': True,
            'Zunahme(Summe)' : True,
            'Zunahmequote(Summe)' : True
        },
        map_style='carto-positron',
        center={'lat': 48.5, 'lon': 9.4},
        zoom=6,
        opacity=0.6,
        color_continuous_scale='RdBu_r'
    )

    fig.update_layout(
        height=650,
        # coloraxis_colorbar_title=colorbar_title
        coloraxis_cmid=0,
        coloraxis_colorbar=dict(
            x=0.02,
            y=0.5,
            xanchor='left',
            yanchor='middle',
            len=0.6,
            thickness=12,
            tickfont=dict(color='black'),
            title=dict(
                text=colorbar_title, # var2
                font=dict(color='black')
            )
        )
    )

    fig.update_traces(
        marker_line_width=0.5,
        marker_line_color='black'
    )

    return fig