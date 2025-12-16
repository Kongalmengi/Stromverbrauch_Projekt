import streamlit as st
import numpy as np
import pandas as pd

# page setting
st.set_page_config(
    page_title='Home',
    page_icon=':rocket:',
    layout='wide'
)

st.markdown(
    """
    <div style='text-align:center;'>
        <h1>Prognosen Projekt</h1>
        <h1>Stromverbrauch in Baden-Württemberg</h1>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align:right; font-size:0.9em; color:#777;'>
        Ver 1.0 · Veröffentlicht: Dez 2025<br>
        Entwickelt von Junho Song
    </div>
    """,
    unsafe_allow_html=True
)


st.space(size='large')

with st.expander('Projektziel'):
    st.markdown(
        """
        ### Projektziel

        Der weltweite Stromverbrauch nimmt derzeit kontinuierlich zu.

        Auch in Deutschland ist aufgrund des hohen Potenzials für industrielle Entwicklung davon auszugehen, dass der zukünftige Stromverbrauch unter dem Einfluss verschiedener Faktoren weiter ansteigen wird.

        Um auf diese Entwicklung angemessen reagieren zu können, sind der Ausbau von Kraftwerkskapazitäten sowie der Aufbau neuer Stromnetze erforderlich.

        Für eine effiziente und nachhaltige Planung ist eine zuverlässige Prognose des zukünftigen Stromverbrauchs daher unerlässlich.

        Die im Rahmen dieses Projekts entwickelte Anwendung hat zum Ziel, den zukünftigen Stromverbrauch zu prognostizieren und damit eine fundierte Grundlage für eine effiziente Planung und den Ausbau der Stromnetzinfrastruktur zu schaffen.
        """
    )


with st.expander('Projektstruktur und Aufgabenbereiche'):
    st.markdown(
        """
        #### 1. Projektkonzeption (Verantwortlich : Junho Song)
        - Entwicklung der Projektidee und Definition der Projektziele
        - Planung der Gesamtstruktur und des Vorgehens

        #### 2. Datenvorverarbeitung und Feature Engineering (Verantwortlich : Junho Song)
        - Datensammlung und Datenbereinigung
        - Strategie zur Behandlung fehlender Werte : Nutzung realer regionaler Modelle zur Ableitung plausibler Muster

        #### 3. Datenanalyse und Explorative Datenanalyse (Verantwortlich : Junho Song)
        - Analyse regionaler Datenmuster
        - Untersuchung von Multikollinearität zwischen Features

        #### 4. Modellierung und Evaluation (Verantwortlich : Junho Song)
        - Entwicklung einer Modellierungsstrategie auf Basis der EDA-Ergebnisse : Trennung in Gesamtmodell und regionale Modelle
        - Konzeption einer mehrstufigen Evaluationsstrategie

            a. Zentrale Bewertungsmetriken :

                i) R²

                ii) MAE

                iii) RMSE
            
            b. Ergänzende Bewertungsmetriken :

                i) MAE/Mittelwert

                ii) RMSE/Standardabweichung
        - Analyse der Modellgrenzen sowie Ableitung von Verbesserungspotenzialen auf Basis der Evaluationsergebnisse

        #### 5. Visualisierung von Ergebnissen und Geodaten (Verantwortlich : Junho Song)
        - Visualisierung zentraler Analyse- und Vergleichsergebnisse
        - Kartendarstellung auf Kreisebene in Baden-Württemberg mithilfe von Plotly

        #### 6. App-Entwicklung und Deployment (Verantwortlich : Junho Song)
        - Entwicklung und Bereitstellung einer webbasierten Anwendung auf Basis von Streamlit
        """
    )


with st.expander('Seitenübersicht und Navigation'):
    st.markdown(
        """
        #### Page1. Projekt Overview

        Auf dieser Seite können folgende Inhalte eingesehen werden:

        **- I. Datenaufbereitung & Analyse**

        **- II. Modellierung & Bewertung**

        Über den folgenden Link gelangen Sie direkt zur entsprechenden Projektseite:
        """
    )

    st.page_link('pages/1 Projekt_Overview.py', label='Link 1 : Projekt Overview', icon='➡️', help='Übersicht über Datenaufbereitung, Modellierung und Bewertung.')

    st.space(size='small')

    st.markdown(
        """
        #### Page2. Prediction

        Auf dieser Seite wird der zukünftige Stromverbrauch in Baden-Württemberg prognostiziert und die entsprechenden Ergebnisse werden visuell dargestellt.

        Über den folgenden Link gelangen Sie direkt zur entsprechenden Projektseite:
        """
    )
    st.page_link('pages/2 Prediction.py', label='Link 2 : Prediction', icon='➡️', help='Visualisierung der prognostizierten Stromverbrauchsentwicklung in Baden-Württemberg.')

    st.space(size='small')

    container_for_app_hinweis = st.container(border=True)
    with container_for_app_hinweis:
        st.markdown(
            """
            #### :rocket: Hinweis

            Die oben genannten Seiten sind ebenfalls über die Links in der Sidebar erreichbar.
            """
        )


with st.expander('Technologie-Stack'):
    stag_cols_1 = st.columns([1.1, 1.8, 1.1])
    with stag_cols_1[0]:
        stag_container_1 = st.container(border=True, height='stretch')
        with stag_container_1:
            st.markdown(
                """
                <div style='text-align:center;'>
                    <p style="font-size:20px; font-weight:600; margin:0;">
                        Programmiersprache
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            st.image('images/Python_logo.png', width=220)

    with stag_cols_1[1]:
        stag_container_2 = st.container(border=True)
        with stag_container_2:
            st.markdown(
                """
                <div style='text-align:center;'>
                    <p style="font-size:20px; font-weight:600; margin:0;">
                        Datenverarbeitung
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            stag_cols_1_sub = st.columns(2)
            with stag_cols_1_sub[0]:
                st.image('images/numpy.png', width=220)
            with stag_cols_1_sub[1]:
                st.image('images/pandas.png', width=220)

    with stag_cols_1[2]:
        stag_container_3 = st.container(border=True, height='stretch')
        with stag_container_3:
            st.markdown(
                """
                <div style='text-align:center;'>
                    <p style="font-size:20px; font-weight:600; margin:0;">
                        Geodatenverarbeitung
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.divider()

            st.image('images/geopandas.png', width=220)



    stag_cols_2 = st.columns(3)
    with stag_cols_2[0]:
        stag_container_4 = st.container(border=True, height='stretch')
        with stag_container_4:
            st.markdown(
                """
                <div style='text-align:center;'>
                    <p style="font-size:20px; font-weight:600; margin:0;">
                        Modellierung
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            st.image('images/Scikit_learn.png', width=300)

    with stag_cols_2[1]:
        stag_container_5 = st.container(border=True, height='stretch')
        with stag_container_5:
            st.markdown(
                """
                <div style='text-align:center;'>
                    <p style="font-size:20px; font-weight:600; margin:0;">
                        Datenvisualisierung
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            st.image('images/Plotly.png', width=300)


    with stag_cols_2[2]:
        stag_container_6 = st.container(border=True, height='stretch')
        with stag_container_6:
            st.markdown(
                """
                <div style='text-align:center;'>
                    <p style="font-size:20px; font-weight:600; margin:0;">
                        Web-App
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.divider()

            st.image('images/streamlit.png', width=300)
