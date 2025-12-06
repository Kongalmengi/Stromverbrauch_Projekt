import streamlit as st
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

# load_Daten
from utils.load_files import load_BW_District, make_TBD_df, load_BW_Combined_df, load_BW_Indust_demmo, load_BW_raw_Indust

# scaler and models
from utils.load_files import load_scaler, load_Ridge_model, load_LinearRegression

# Demo_funcs
from utils.Demo_funcs import divide_Jahr, Gopp_corr_umsatz_dfs, Gopp_corr_bes_dfs, umsatz_corr_dfs_for_Vis, bes_corr_dfs_for_Vis, colormaps_for_Vis, make_subset, make_pred_subset, make_subset_2, make_corr_mat, make_vif, test_gesamtmodell, kfold_result




# page setting
st.set_page_config(
    page_title='Projekt Overview',
    page_icon=':bar_chart:',
    layout='wide'
)


st.markdown(
    """
    # :bar_chart: Projekt Overview

    Hier können Sie die verwendeten Daten, Modelle und angewandten Methoden des Projekts einsehen.

    Bitte wählen Sie unten den gewünschten Tab aus.
    """
)

# Daten
BW_District_df = load_BW_District()
TBD_df = make_TBD_df(BW_District_df)
combined_df = load_BW_Combined_df(TBD_df)

Indust_df = load_BW_raw_Indust()
indu_total_df = load_BW_Indust_demmo()

# scalers
Gopp_Bes_scaler = load_scaler('models/Demo_models/scalers/Gopp_Bes_scaler.pkl')
Gopp_umsatz_scaler = load_scaler('models/Demo_models/scalers/Gopp_umsatz_scaler.pkl')
Ess_Bes_scaler = load_scaler('models/Demo_models/scalers/Ess_Bes_scaler.pkl')
Ess_umsatz_scaler = load_scaler('models/Demo_models/scalers/Ess_umsatz_scaler.pkl')

scaler_for_normal_reg = load_scaler('models/Scalers/normal_scaler.pkl')
scaler_for_special_reg = load_scaler('models/Scalers/special_scaler.pkl')

# models
lin_pre = load_LinearRegression('models/Demo_models/models/lin_pre.pkl')
lin_pan = load_LinearRegression('models/Demo_models/models/lin_pan.pkl')
lin_post = load_LinearRegression('models/Demo_models/models/lin_post.pkl')

normal_ridge_model = load_Ridge_model('models/Ridge_models/normal_ridge_model.pkl')

# Demo_funcs
# Year-based split (pre-2005 vs post-2005)
indu_under_2005, indu_upper_2006 = divide_Jahr(Indust_df)
# Umsatz corr
corr_under_2005, corr_upper_2006, dup_only_name = Gopp_corr_umsatz_dfs(indu_under_2005, indu_upper_2006)
Gopp_umsatz_corr_under_2005_df, Gopp_umsatz_corr_upper_2006_df = umsatz_corr_dfs_for_Vis(corr_under_2005, corr_upper_2006, dup_only_name)
# Bes corr
corr_under_2005_2, corr_upper_2006_2 = Gopp_corr_bes_dfs(indu_under_2005, indu_upper_2006)
Gopp_bes_corr_under_2005_2_df, Gopp_bes_corr_upper_2006_2_df = bes_corr_dfs_for_Vis(corr_under_2005_2, corr_upper_2006_2, dup_only_name)

# colormap for Vis
color_map_umsatz_2005, color_map_umsatz_2006, color_map_bes_2005, color_map_bes_2006 = colormaps_for_Vis(Gopp_umsatz_corr_under_2005_df, Gopp_umsatz_corr_upper_2006_df, Gopp_bes_corr_under_2005_2_df, Gopp_bes_corr_upper_2006_2_df)

# subset_df
subset_df = make_subset(Indust_df, Gopp_Bes_scaler, Gopp_umsatz_scaler, Ess_Bes_scaler, Ess_umsatz_scaler)


st.space(size='medium')

# main_tabs
main_tabs = st.tabs(['I. Datenaufbereitung & Analyse', 'II. Modellierung & Bewertung'])

with main_tabs[0]:
    st.markdown(
        """
        ## I. Datenaufbereitung & Analyse

        Hier finden Sie eine Übersicht über die Datenaufbereitung und die wichtigsten Analyseschritte.
    
        Bitte wählen Sie unten einen der Tabs aus.
        """
    )

    st.space(size='small')

    # sub_tabs_a
    sub_tabs_a = st.tabs(['1.Umgang mit fehlenden Werten I', '2.Umgang mit fehlenden Werten II', '3.Analyse der Multikollinearität', '4.Datenquellen'])

    with sub_tabs_a[0]:
        st.markdown(
            """
            ### 1. Umgang mit fehlenden Werten I

            Bei der Behandlung fehlender Werte werden häufig einfache Methoden wie Mittelwert-, Median- oder Modusersatz verwendet.
            Bei langen, zusammenhängenden Abschnitten mit fehlenden Werten können solche Verfahren jedoch den tatsächlichen Verlauf und die Verteilung der Daten deutlich verzerren.

            Auch eine Interpolation, bei der der bisherige Trend einfach fortgeschrieben wird, ist zwar möglich, kann jedoch zu unrealistischen Verläufen führen und unter Umständen sogar Werte erzeugen, die in der Realität nicht auftreten könnten.

            Um lange fehlende Abschnitte möglichst plausibel zu ergänzen, ist es daher notwendig, sich an den tatsächlich beobachtbaren strukturellen Mustern der Daten zu orientieren.
            In diesem Projekt wurde hierfür ein lineares Regressionsmodell eingesetzt.

            Der grundlegende Ansatz ist wie folgt:
            """
        )

        st.space(size='small')

        st.markdown(
            """
            i) Identifizieren Sie die Abschnitte, in denen im Merkmal Y von Region A fehlende Werte auftreten.

            ii) Um potenzielle Vergleichsregionen zu finden, filtern Sie zunächst jene Regionen heraus, deren Verlauf im selben Merkmal Y grundsätzlich mit dem von Region A vergleichbar ist.

            iii) Da allein das Merkmal Y nicht ausreicht, um die Ähnlichkeit der Regionen zuverlässig zu beurteilen, wird ein zusätzliches, vollständig verfügbares Hilfsmerkmal X ausgewählt.

            iv) Vergleichen Sie die zeitlichen Verläufe des Merkmals X der Kandidatenregionen mit dem von Region A und wählen Sie jene Region B aus, deren Muster am ähnlichsten ist.

            v) Unter der Annahme, dass die Ähnlichkeit im Hilfsmerkmal X auf vergleichbare strukturelle Bedingungen und Veränderungsmuster hinweist, wird der Verlauf des Merkmals Y von Region B genutzt, um die fehlenden Werte von Region A zu ergänzen.

            vi) Um Unterschiede in Skalen oder Größenordnungen zwischen den Regionen zu berücksichtigen, werden die Merkmale standardisiert (Scaling). Anschließend wird die Beziehung zwischen beiden Regionen mittels linearer Regression modelliert, um die fehlenden Werte im Merkmal Y zu schätzen.


            """
        )

        st.space(size='small')

        st.markdown(
            """
            In diesem Projekt gehört zu den mithilfe des oben beschriebenen Verfahrens bereinigten Datensätzen auch der fehlende Gesamtumsatz von Göppingen im Industriebereich.

            Der Prozess zur Behandlung dieser fehlenden Werte ist im Folgenden dargestellt.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            ### :rocket: Beispiel : Behandlung der fehlenden Gesamtumsatzdaten von Göppingen (Industrie)
            """
        )

        st.space(size='small')
        st.markdown(
            """
            #### i) Überprüfung der fehlenden Werte

            Zunächst werden die fehlenden Werte überprüft.

            Die fehlenden Gesamtumsatzdaten von Göppingen können im folgenden DataFrame eingesehen werden.
            """
        )

        st.dataframe(Indust_df[(Indust_df['DName']=='Göppingen')&(Indust_df['Gesamtumsatz'].isna())], height=200)

        st.space(size='small')

        st.markdown(
            """
            Anhand des oben gezeigten DataFrames lässt sich erkennen, dass die Gesamtumsatzdaten von Göppingen im Zeitraum von 2010 bis 2023 durchgehend fehlen.

            Um diese fehlenden Werte zu behandeln, werden Regionen gesucht, die in ihren Eigenschaften Göppingen ähneln.
            Zur eindeutigen Bestimmung der regionalen Ähnlichkeiten wird der Datensatz zudem in zeitliche Abschnitte unterteilt.

            Für die Festlegung dieser Zeitabschnitte wurden die aggregierten Umsatz- und Beschäftigtendaten des industriellen Sektors in Baden-Württemberg herangezogen.

            Um den industriellen Umsatz und die Beschäftigtenzahlen Baden-Württembergs auf einem vergleichbaren Niveau zu analysieren, wurde eine Standardisierung vorgenommen.
            Das Ergebnis dieser Standardisierung ist in der folgenden Grafik dargestellt.
            """
        )


        st.space(size='small')

        fig_indust_total = px.line(indu_total_df, x='Jahr', y=['Beschäftigte_norm', 'Gesamtumsatz_norm'], markers=True, title='fig1. Jährlicher Industrieumsatz und Beschäftigtenzahl in Baden-Württemberg (z-Score)')
        st.plotly_chart(fig_indust_total, key='fig_indust_total_1')

        st.space(size='small')

        st.markdown(
            """
            Aus dem obigen Diagramm lassen sich folgende Beobachtungen ableiten:
            """
        )

        st.space(size='small')

        st.markdown(
            """
            **(a) Ab dem Jahr 2006 besteht zwischen Industrieumsatz und Beschäftigtenzahl überwiegend ein proportionaler Zusammenhang.**

            **(b) Vor 2005 zeigt die Beziehung zwischen Industrieumsatz und Beschäftigtenzahl im Vergleich zu der Zeit nach 2006 deutlich stärkere Schwankungen.**
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Obwohl die Ursache für dieses Verhalten einer genaueren Untersuchung bedarf, lässt sich anhand der vorliegenden Daten vermuten, dass zwischen 2005 und 2006 eine Veränderung im Industriemarkt stattgefunden hat.

            Unter Berücksichtigung dieser Beobachtung wird die Korrelation des Industrieumsatzes der einzelnen Kreise in zwei Zeiträume aufgeteilt: vor 2005 und nach 2006.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            #### ii) Vergleich der Korrelationskoeffizienten der Umsatzdaten

            Im vorherigen Schritt wurden die Analysezeiträume für die industriellen Umsatzdaten festgelegt.

            Nun werden für jeden dieser Zeiträume die Korrelationskoeffizienten zwischen den Umsatzdaten Göppingens und denen der übrigen Regionen untersucht.

            Die zehn Regionen mit den höchsten Korrelationswerten werden ausgewählt und in der folgenden Grafik dargestellt.
            """
        )

        st.space(size='small')

        corr1_vis_cols1 = st.columns(2)

        with corr1_vis_cols1[0]:
            fig_umsatz_corr_2 = fig = px.bar(Gopp_umsatz_corr_under_2005_df, x='Korrelationskoeffizient', y=Gopp_umsatz_corr_under_2005_df.index, orientation='h', color='Kandidatenstatus', color_discrete_map=color_map_umsatz_2005, hover_data={'Kandidatenstatus':False}, category_orders={'DName':Gopp_umsatz_corr_under_2005_df.index.to_list()}, height=330, title='fig2. Korrelationskoeffizienten der Umsatzdaten (bis 2005)')
            st.plotly_chart(fig_umsatz_corr_2)

        with corr1_vis_cols1[1]:
            fig_umsatz_corr_1 = px.bar(Gopp_umsatz_corr_upper_2006_df, x='Korrelationskoeffizient', y=Gopp_umsatz_corr_upper_2006_df.index, orientation='h', color='Kandidatenstatus', color_discrete_map=color_map_umsatz_2006, hover_data={'Kandidatenstatus':False}, category_orders={'DName':Gopp_umsatz_corr_upper_2006_df.index.to_list()}, height=330, title='fig3. Korrelationskoeffizienten der Umsatzdaten (ab 2006)')
            st.plotly_chart(fig_umsatz_corr_1)

        st.space(size='small')

        st.markdown(
            """
            Das oben gezeigte Diagramm visualisiert die zehn Regionen mit den höchsten Korrelationswerten zu den Umsatzdaten Göppingens, getrennt nach den beiden Analysezeiträumen.

            In beiden Zeiträumen wurden zwei Regionen mit besonders hohen Korrelationskoeffizienten identifiziert: Esslingen und Karlsruhe. Diese Regionen sind im Diagramm rot markiert.

            Diese beiden Regionen wurden als Kandidaten ausgewählt, deren Umsatzentwicklung vermutlich eine ähnliche Trendstruktur wie die von Göppingen aufweist.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            #### iii) Einführung eines Zusatzindikators: Korrelationskoeffizienten der Beschäftigtendaten im Industriesektor

            Im vorherigen Schritt wurden die regionalen Korrelationskoeffizienten der Umsatzdaten untersucht und dabei Regionen identifiziert, deren Trendverlauf voraussichtlich dem von Göppingen ähnelt. Diese Regionen wurden als Kandidaten ausgewählt.

            Allerdings lässt sich allein aus diesen Ergebnissen noch nicht eindeutig bestimmen, welche Region tatsächlich eine ähnliche industrielle Marktstruktur wie Göppingen aufweist.

            Als ergänzenden Indikator zur Überprüfung dieser Ähnlichkeit wurde daher die Anzahl der Beschäftigten im Industriesektor ausgewählt.

            Analog zum Vorgehen bei den Umsatzdaten können auch hier die Korrelationskoeffizienten der Beschäftigtendaten für die einzelnen Kreise berechnet werden.

            In den folgenden beiden Diagrammen sind die Ergebnisse für die jeweiligen Zeiträume dargestellt, sortiert nach abnehmendem Korrelationswert.
            """
        )

        st.space(size='small')

        corr1_vis_cols2 = st.columns(2)

        with corr1_vis_cols2[0]:
            fig_bes_corr_2 = px.bar(Gopp_bes_corr_under_2005_2_df, x='Korrelationskoeffizient', y=Gopp_bes_corr_under_2005_2_df.index, orientation='h', color='Kandidatenstatus', color_discrete_map=color_map_bes_2005, hover_data={'Kandidatenstatus':False}, category_orders={'DName':Gopp_bes_corr_under_2005_2_df.index.to_list()}, height=330, title='fig4. Korrelationskoeffizienten der Beschäftigtendaten (bis 2005)')
            st.plotly_chart(fig_bes_corr_2)

        with corr1_vis_cols2[1]:
            fig_bes_corr_1 = px.bar(Gopp_bes_corr_upper_2006_2_df, x='Korrelationskoeffizient', y=Gopp_bes_corr_upper_2006_2_df.index, orientation='h', color='Kandidatenstatus', color_discrete_map=color_map_bes_2006, hover_data={'Kandidatenstatus':False}, category_orders={'DName':Gopp_bes_corr_upper_2006_2_df.index.to_list()}, height=330, title='fig5. Korrelationskoeffizienten der Beschäftigtendaten (ab 2006)')
            st.plotly_chart(fig_bes_corr_1)

        st.space(size='small')

        st.markdown(
            """
            In den oben dargestellten Diagrammen sind die zuvor ausgewählten Kandidatenregionen in Rot markiert.

            Diese Kandidatenregionen weisen auch bei den Beschäftigtendaten hohe Korrelationswerte auf. Besonders auffällig ist dabei Esslingen, das in beiden Zeitabschnitten zu den Regionen mit den höchsten Korrelationskoeffizienten gehört.

            Zudem liegt der Korrelationswert von Esslingen im Vergleich zu den übrigen Regionen — auch zu denen, die nicht als Kandidaten ausgewählt wurden — nahezu am höchsten.

            Zwar lässt sich aufgrund der begrenzten Datenbasis nicht mit absoluter Sicherheit behaupten, dass der industrielle Markt von Göppingen und Esslingen strukturell identisch ist.

            Jedoch deuten die Ähnlichkeiten in beiden betrachteten Indikatoren — Umsatz und Beschäftigtenzahl — darauf hin, dass zwischen den beiden Regionen eine gewisse strukturelle Nähe besteht.

            Daher kann Esslingen als geeignete Region zur Modellierung und zur Schätzung der fehlenden Umsatzdaten Göppingens ausgewählt werden.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            #### iv) Vergleich der Datenverläufe der finalen Kandidatenregion

            Auf Grundlage der vorherigen Analyseschritte wurde Esslingen als endgültiges Regionalmodell zur Schätzung der fehlenden Umsatzdaten Göppingens ausgewählt.

            Im nächsten Schritt wird anhand eines Diagramms überprüft, inwieweit der Datenverlauf der beiden Regionen tatsächlich Ähnlichkeiten aufweist.

            Allerdings bestehen Größenunterschiede in der industriellen Struktur der beiden Regionen.

            Um die Daten beider Regionen auf einem vergleichbaren Niveau analysieren zu können, wurde für jede Region eine Standardisierung der Daten durchgeführt.

            Die standardisierten Werte sind im folgenden Diagramm dargestellt.
            """
        )

        st.space(size='small')

        fig_verg_bes = px.line(subset_df, x='Jahr', y='Beschäftigte_norm', color='DName', markers=True, title='fig6. Vergleich der Beschäftigte (z-score)')
        st.plotly_chart(fig_verg_bes)

        fig_verg_umsatz = px.line(subset_df, x='Jahr', y='Gesamtumsatz_norm', color='DName', markers=True, title='fig7. Vergleich des Gesamtumsatzes (z-Score)')
        st.plotly_chart(fig_verg_umsatz)

        st.space(size='small')

        st.markdown(
            """
            Aus den beiden oben dargestellten Diagrammen lässt sich erkennen, dass die Datenverläufe von Göppingen und Esslingen eine deutliche Ähnlichkeit aufweisen.

            Damit wird Esslingen endgültig als Regionalmodell zur Schätzung der fehlenden industriellen Umsatzdaten Göppingens ausgewählt.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            #### v) Lösung der fehlenden Umsatzdaten - Teil 1

            Nun steht der nächste Schritt an: Die fehlenden Industrie-Umsatzdaten von Göppingen sollen mithilfe des Trends der Industrieumsatzdaten von Esslingen rekonstruiert werden.

            Dazu wird ein lineares Regressionsmodell verwendet. Die unabhängigen und abhängigen Variablen wurden wie folgt festgelegt:
            """
        )

        st.space(size='small')

        st.markdown(
            """
            - **Unabhängige Variablen : Jahr, Beschäftigte_norm**
            - **Abhängige Variable : Gesamtumsatz_norm**
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Im Folgenden wird kurz erläutert, warum die unabhängigen Variablen wie oben ausgewählt wurden.

            Zunächst wurde Beschäftigte_norm als unabhängige Variable gewählt, da der standardisierte Wert der Beschäftigtenzahl als Hilfsindikator für die strukturelle Ähnlichkeit der Industrie in beiden Regionen dient.

            Etwas ungewöhnlicher erscheint die Auswahl von Jahr als unabhängige Variable. Der Grund dafür wird jedoch deutlich, wenn man die in den Abbildungen fig1, 6 und 7 gezeigten Muster betrachtet.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            - **Beobachtung 1: In allen Grafiken ist ein deutlicher Rückgang in den Jahren 2009 und 2019 zu erkennen.**

            - **Beobachtung 2: Die Jahre mit diesen starken Einbrüchen - 2009 und 2019 - entsprechen den Zeitpunkten der globalen Finanzkrise (2008/2009) bzw. der COVID-19-Pandemie.**
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Aus diesen Beobachtungen lässt sich schließen, dass der Industriemarkt erheblich vom jeweiligen Jahr beeinflusst wird.

            Zwar könnte dieser Effekt auch durch die Einführung entsprechender Ereignis-Dummy-Variablen abgebildet werden, jedoch sollte hier eine möglichst umfassende Berücksichtigung verschiedener marktbeeinflussender Ereignisse erfolgen.

            Die Erstellung zahlreicher Ereignis-Dummys würde jedoch zu einer Erhöhung der Feature-Anzahl führen und damit das Risiko des sogenannten „Curse of Dimensionality“ erhöhen.

            Um dieses Problem zu vermeiden, wurde darauf verzichtet, die Anzahl der Features unnötig zu vergrößern. Stattdessen wurde entschieden, Jahr direkt als unabhängige Variable in das Regressionsmodell aufzunehmen.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Für das lineare Regressionsmodell wurde das in Scikit-Learn bereitgestellte LinearRegression-Objekt verwendet.

            Wenn ein polynomialer Trend berücksichtigt werden sollte, wurde zunächst das PolynomialFeatures-Objekt auf die unabhängigen Variablen angewendet und das transformierte Feature-Set anschließend dem LinearRegression-Modell übergeben.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Bei der Auswahl eines geeigneten Modells zur Schätzung der fehlenden Industrie-Umsatzdaten von Göppingen wurde die Anzahl der Industriebeschäftigten als unterstützender Indikator herangezogen.

            Dementsprechend wurde der standardisierte Wert dieser Beschäftigtendaten als unabhängige Variable verwendet.

            Bevor jedoch das Training mit dem LinearRegression-Objekt durchgeführt wird, ist es notwendig, die oben dargestellte fig7 erneut zu betrachten.

            Insbesondere zeigt ein Blick auf die Werte von Esslingen in denjenigen Jahresbereichen, in denen Göppingen fehlende Daten aufweist, die folgenden abschnittsbezogenen Muster:
            """
        )

        st.space(size='small')

        st.markdown(
            """
            - Abschnitt 1 (2009-2017): Verlauf mit überwiegend quadratischer Tendenz

            - Abschnitt 2 (2018-2020): Verlauf mit überwiegend linearer Tendenz

            - Abschnitt 3 (ab 2021): Verlauf mit überwiegend quadratischer Tendenz
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Um die oben beschriebenen abschnittsbezogenen Muster korrekt abzubilden, wurde für jeden Abschnitt ein eigenes LinearRegression-Modell erstellt und separat trainiert.

            Anschließend konnten die fehlenden Werte von Göppingen geschätzt werden, indem die jeweiligen unabhängigen Variablen dieses Kreises in die trainierten Modelle eingespeist wurden.

            Die resultierenden, imputierten Werte sind in fig8 dargestellt.
            """
        )

        st.space(size='small')

        # subset_df Missing value processing
        subset_df = make_pred_subset(subset_df, lin_pre, lin_pan, lin_post, Gopp_umsatz_scaler)

        # Visualization
        fig_verg_umsatz_2 = px.line(subset_df, x='Jahr', y='Gesamtumsatz_norm', color='DName', markers=True, title='fig8. Gesamtumsatz im Vergleich (z-Score)')
        st.plotly_chart(fig_verg_umsatz_2)

        st.space(size='small')

        st.markdown(
            """
            #### vi) Lösung der fehlenden Umsatzdaten - Teil 2

            Ein Vergleich der fig7 und fig8 zeigt, dass die fehlenden Werte von Göppingen plausibel rekonstruiert wurden.

            Da das lineare Regressionsmodell mit den Daten aus Esslingen trainiert wurde, spiegelt es den Umsatztrend dieser Region wider.

            Werden jedoch die unabhängigen Variablen von Göppingen in das Modell eingespeist, so werden die fehlenden Werte entsprechend der regionalen Gegebenheiten von Göppingen geschätzt.

            Die in fig8 dargestellten Werte liegen allerdings in z-Score-Form vor.

            Daher ist eine inverse Transformation des StandardScaler erforderlich, um die endgültigen Umsatzwerte im jeweiligen regionalen Größenmaßstab wiederherzustellen.

            Das Ergebnis dieser Rücktransformation ist in fig9 dargestellt.
            """
        )

        fig_verg_umsatz_3 = px.line(subset_df, x='Jahr', y='Gesamtumsatz', color='DName', markers=True, title='fig9. Gesamtumsatz im Vergleich (in 1.000 EUR)')
        st.plotly_chart(fig_verg_umsatz_3)




    with sub_tabs_a[1]:
        """
        ### 2. Umgang mit fehlenden Werten II

        In diesem Abschnitt wird erläutert, wie kurze oder nicht zusammenhängende Fehlwertbereiche, die im Rahmen der Datenerhebung dieses Projekts identifiziert wurden, behandelt wurden.

        Dazu ist es zunächst erforderlich, die charakteristischen Merkmale der Industriedaten von Baden-Württemberg näher zu betrachten.
        """

        st.space(size='small')

        st.plotly_chart(fig_indust_total, key='fig_indust_total_2')

        st.space(size='small')

        st.markdown(
            """
            Die fig.1 oben zeigt die standardisierten Werte (z-Score) des jährlichen Industrieumsatzes und der Beschäftigtenzahlen in Baden-Württemberg.

            Die deutlichsten Veränderungen im Zeitverlauf treten zwischen 2008 und 2009 sowie zwischen 2019 und 2020 auf.

            Diese Zeiträume entsprechen der globalen Finanzkrise (2008/2009) bzw. der COVID-19-Pandemie, also Ereignissen, die den Industriemarkt stark beeinflusst haben.

            In den übrigen Zeitabschnitten sind hingegen keine vergleichbar abrupten Veränderungen zu erkennen.

            Zwar gibt es über verschiedene Jahresbereiche hinweg wechselnde Tendenzen der Zu- und Abnahme, jedoch ohne ausgeprägte sprunghafte Ausschläge.

            Für eine detailliertere Analyse wäre es notwendig, die jährlichen Änderungsraten sowie deren Extremwerte zu untersuchen.

            Im Rahmen dieses Projekts wurde jedoch keine weiterführende Analyse dieser Art vorgenommen. Stattdessen wurde aufgrund der beobachteten Muster davon ausgegangen, dass sich die Daten in den wesentlichen Zeitabschnitten einer übergeordneten Trendstruktur folgen, und es wurde folgende Annahme getroffen:
            """
        )

        st.space(size='small')

        st.markdown(
            """
            - **Annahme: Die Daten folgen dem Trend der benachbarten Jahresintervalle.**
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Ein Beispiel für diese Art der Fehlwertbehandlung ist im Folgenden dargestellt.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            ### :rocket: Beispiel : Behandlung der fehlenden Gesamtumsatzdaten von Heidenheim (Industrie) 
            """
        )

        st.space(size='small')

        st.markdown(
            """
            #### i) Überprüfung der fehlenden Werte

            Zunächst werden die fehlenden Werte überprüft.

            Die fehlenden Umsatzdaten von Heidenheim können im folgenden DataFrame eingesehen werden.
            """
        )
        st.space(size='small')

        st.dataframe(Indust_df[(Indust_df['DName']=='Heidenheim')&(Indust_df['Gesamtumsatz'].isna())])

        st.space(size='small')

        st.markdown(
            """
            Die im oben gezeigten DataFrame erkennbaren fehlenden Umsatzwerte lassen sich in fig2 gemeinsam mit dem Trend der benachbarten Jahre visualisieren.
            """
        )

        st.space(size='small')

        fig_heiden_umsatz_1 = px.line(Indust_df[Indust_df['DName']=='Heidenheim'], x='Jahr', y='Gesamtumsatz', color='DName', markers=True, title='fig2. Industrieumsatz in Heidenheim (in 1.000 EUR)')
        st.plotly_chart(fig_heiden_umsatz_1)

        st.space(size='small')

        st.markdown(
            """
            #### ii) Interpolation der fehlenden Werte

            Aus fig2 lässt sich erkennen, dass die fehlenden Umsatzdaten von Heidenheim in einem kurzen und nicht zusammenhängenden Abschnitt auftreten.

            Solche Fehlwerte werden auf Grundlage der zuvor formulierten Annahme behandelt.

            Das bedeutet, dass der in diesem Bereich zu erwartende Trend ermittelt und anschließend zur Interpolation der fehlenden Werte genutzt wird.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Für die trendbasierte Interpolation ist zunächst zu bestimmen, welcher Verlauf in dem Abschnitt mit fehlenden Werten plausibel ist.

            Aus der in fig2 dargestellten Form der Daten lässt sich erkennen, dass im Zeitraum von 1998 bis 2003 ein quadratischer Trend für die Umsatzdaten angemessen ist.

            Dementsprechend kann mithilfe der in Pandas verfügbaren Methode .interpolate() eine Interpolation mit quadratischer Trendform durchgeführt werden.

            Das Ergebnis dieser Interpolation ist in fig3 dargestellt.
            """
        )

        st.space(size='small')

        subset_df_2 = make_subset_2(Indust_df)

        fig_heiden_umsatz_2 = px.line(subset_df_2, x='Jahr', y='Gesamtumsatz', color='DName', markers=True, title='fig3. Industrieumsatz in Heidenheim (in 1.000 EUR)')
        st.plotly_chart(fig_heiden_umsatz_2)


    with sub_tabs_a[2]:
        st.markdown(
            """
            ### 3. Analyse der Multikollinearität

            In den im Rahmen dieses Projekts entwickelten Vorhersagemodellen kann zwischen den verwendeten Features potenziell Multikollinearität auftreten.

            In diesem Tab werden mögliche Formen der Multikollinearität zwischen den Features untersucht und die Maßnahmen beschrieben, die ergriffen wurden, um daraus resultierende Probleme zu vermeiden.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Um das mögliche Vorliegen von Multikollinearität zu überprüfen, sollten zunächst die folgenden grundlegenden Kennzahlen betrachtet werden:
            """
        )

        st.space(size='small')

        st.markdown(
            """
            - die Korrelationskoeffizienten zwischen den Features
            - der VIF (Variance Inflation Factor)
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Darüber hinaus ist es notwendig, anhand von Streudiagrammen zu prüfen, ob die beobachteten Korrelationen zwischen den Features tatsächlich aussagekräftig sind.

            Im Folgenden werden daher zunächst die Korrelationskoeffizienten zwischen den Features untersucht.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            #### i) Korrelationskoeffizienten der Features

            Eine Zusammenfassung der wichtigsten im Projekt verwendeten Features sowie deren Korrelationskoeffizienten ist in der folgenden Abbildung (fig1) dargestellt.
            """
        )

        st.space(size='small')

        corr_mat = make_corr_mat(combined_df)

        fig_corr_hitmap = px.imshow(corr_mat,
            text_auto=True,
            color_continuous_scale='RdBu_r',
            title='fig1. Korrelations-Heatmap der Features im gesamten Datensatz Baden-Württembergs',
            width=700,
            height=700
        )

        st.plotly_chart(fig_corr_hitmap)

        st.space(size='small')

        st.markdown(
            """
            Wie in fig1 zu erkennen ist, kann die Verwendung aller verfügbaren Features als unabhängige Variablen für das Modell zur Vorhersage des industriellen Stromverbrauchs die Modellleistung sogar beeinträchtigen.

            Daher ist es sinnvoll, vor der Überprüfung der Multikollinearität zunächst die inhaltliche Bedeutung der einzelnen Features zu beurteilen und solche auszuschließen, die voraussichtlich keinen zusätzlichen Nutzen bieten.

            So steht beispielsweise die Anzahl der Beschäftigten in einem engen Zusammenhang mit der Wohnbevölkerung sowie der industriellen Beschäftigungsquote.

            Folglich kann man, wenn das Feature Beschäftigte als unabhängige Variable ausgewählt wird, die Merkmale Wohnbevölkerung und Beschäftigungsquote aus der Kandidatenliste für die unabhängigen Variablen ausschließen.

            Unter Berücksichtigung dieser Aspekte lassen sich die für das Vorhersagemodell des industriellen Stromverbrauchs relevanten und sinnvoll einsetzbaren Features wie folgt zusammenfassen:
            """
        )

        st.space(size='small')

        st.markdown(
            """
            - Features : Betriebe, Beschäftigte, Gesamtumsatz, Investitionen
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Wenn man die Korrelationskoeffizienten der ausgewählten Features in der Heatmap von fig1 betrachtet, zeigt sich, dass zwischen Beschäftigte, Gesamtumsatz und Investitionen jeweils sehr hohe Korrelationen bestehen.

            Auch für das Feature Betriebe lässt sich eine deutliche Korrelation mit Beschäftigte erkennen.

            Diese Beobachtungen deuten darauf hin, dass zwischen den jeweiligen Features potenziell Multikollinearität vorliegen könnte.

            Im nächsten Schritt werden wir anhand von Streudiagrammen prüfen, ob diese hohen Korrelationswerte tatsächlich von inhaltlicher Bedeutung sind.
            """
        )

        st.space(size='small')
        
        st.markdown(
            """
            #### ii) Streudiagramme

            Im Folgenden werden die Streudiagramme zwischen den zuvor identifizierten Features mit hohen Korrelationskoeffizienten dargestellt.
            """
        )

        st.space(size='small')

        fig_cols1 = st.columns(2)

        with fig_cols1[0]:
            fig_bes_bet = px.scatter(combined_df, x='Beschäftigte', y='Betriebe', title='fig2. Betriebe vs. Beschäftigte', color='DN_DT')
            st.plotly_chart(fig_bes_bet)

            fig_bes_umsatz = px.scatter(combined_df, x='Beschäftigte', y='Gesamtumsatz', title='fig4. Gesamtumsatz vs. Beschäftigte', color='DN_DT')
            st.plotly_chart(fig_bes_umsatz)

        with fig_cols1[1]:
            fig_bes_invest = px.scatter(combined_df, x='Beschäftigte', y='Investitionen', title='fig3. Investitionen vs. Beschäftigte', color='DN_DT')
            st.plotly_chart(fig_bes_invest)

            fig_invest_umsatz = px.scatter(combined_df, x='Gesamtumsatz', y='Investitionen', title='fig5. Investitionen vs. Gesamtumsatz', color='DN_DT')
            st.plotly_chart(fig_invest_umsatz)
        
        st.space(size='small')

        st.markdown(
            """
            Beim Betrachten der Streudiagramme in fig2 bis fig5 lässt sich feststellen, dass die stark korrelierten Features - abgesehen von einigen Ausreißerregionen - eine hohe gegenseitige Erklärungskraft aufweisen.

            Dies stellt einen deutlichen Hinweis darauf dar, dass zwischen den Features ausgeprägte Multikollinearität bestehen könnte.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            #### iii) VIF

            Durch die Analyse der Korrelationskoeffizienten sowie der Streudiagramme wurde bereits ein mögliches Vorliegen von Multikollinearität zwischen den Features festgestellt.

            Für eine präzisere Beurteilung wurde im Anschluss der Variance Inflation Factor (VIF) berechnet, um das Ausmaß der Multikollinearität für jedes Feature quantitativ zu bewerten.

            Die berechneten VIF-Werte sind in Table1 dargestellt.
            """
        )

        st.space(size='small')

        # VIF
        vif_df = make_vif(combined_df)
        container_for_table1 = st.container(border=True, horizontal_alignment='center', vertical_alignment='center')
        with container_for_table1:
            st.write('Table1. VIF')
            st.dataframe(vif_df, width=300, hide_index=True)

        st.space(size='small')

        st.markdown(
            """
            Beim Blick auf die in Table1 dargestellten VIF-Werte zeigt sich, dass insgesamt ein gewisses Maß an Multikollinearität vermutet werden kann.

            Insbesondere für das Feature Beschäftigte deutet ein deutlich erhöhter VIF darauf hin, dass hier ein ausgeprägtes Multikollinearitätsproblem vorliegt.

            Daher wäre es grundsätzlich empfehlenswert, dieses Feature aus den unabhängigen Variablen zu entfernen.

            Bei näherer Betrachtung dieses Features ließ sich jedoch eine auffällige Besonderheit feststellen : 
            Der Korrelationskoeffizient zwischen Beschäftigte und Stromverbrauch (Industrie) liegt nahezu auf demselben Niveau wie der Korrelationskoeffizient zwischen Betriebe und Stromverbrauch (Industrie).

            Um diese beiden Zusammenhänge genauer zu untersuchen, wurden die folgenden Streudiagramme erstellt.
            """
        )

        st.space(size='small')

        fig_cols2 = st.columns(2)
        with fig_cols2[0]:
            fig_bet_strom = px.scatter(combined_df, x='Betriebe', y='Stromverbrauch(Industrie)', color='DN_DT', title='fig6. Stromverbrauch(Industrie) vs. Betriebe')
            st.plotly_chart(fig_bet_strom)

        with fig_cols2[1]:
            fig_bes_strom = px.scatter(combined_df, x='Beschäftigte', y='Stromverbrauch(Industrie)', color='DN_DT', title='fig7. Stromverbrauch(Industrie) vs. Beschäftigte')
            st.plotly_chart(fig_bes_strom)

        st.space(size='small')

        st.markdown(
            """
            Beim Vergleich von fig6 und fig7 wird deutlich, dass die Verteilungen der beiden Streudiagramme sehr ähnlich sind.

            Insbesondere die Positionen der gemeinsamen Ausreißerregionen verdeutlichen diese Ähnlichkeit noch stärker.

            Dies weist darauf hin, dass beide Features den Stromverbrauch (Industrie) in vergleichbarer Weise erklären.

            Das bedeutet, dass - sofern zur Lösung des Multikollinearitätsproblems die Entfernung des Features Beschäftigte in Betracht gezogen wird - ebenso auch die Entfernung des Features Betriebe als Option geprüft werden könnte.

            Zwar ist es unter Berücksichtigung der VIF-Werte und der Korrelationskoeffizienten grundsätzlich naheliegend, Beschäftigte aus den unabhängigen Variablen zu entfernen.

            Allerdings umfasst das Feature Beschäftigte auch Informationen zu Bevölkerung insgesamt sowie zur Beschäftigungsquote.

            Mit anderen Worten : Der Wert Beschäftigte enthält eine breitere und differenziertere Beschreibung der Struktur eines Kreises.

            Aus diesem Grund wurde in diesem Projekt, zur Reduzierung der Multikollinearität im Vorhersagemodell, nicht Beschäftigte, sondern das Feature Betriebe als unabhängige Variable ausgeschlossen.
            """
        )

        container_for_hinweis = st.container(border=True)
        with container_for_hinweis:
            st.markdown(
                """
                #### :exclamation: Hinweis :exclamation:

                In diesem Projekt wurde das Feature Betriebe ausgeschlossen und Beschäftigte beibehalten. Allerdings weist das Feature Betriebe im Vergleich zu Beschäftigte einen deutlich niedrigeren VIF auf und zeigt zudem geringere Korrelationen zu den übrigen Features.

                Dies bedeutet, dass Betriebe als unabhängige Variable grundsätzlich einen eigenständigen Mehrwert bieten kann und daher ebenfalls sinnvoll in das Vorhersagemodell aufgenommen werden könnte.

                Dieser Aspekt wird in einem zukünftigen Update des Modells näher untersucht werden.
                """
            )

        st.space(size='small')

        st.markdown(
            """
            #### iv) Auswahl des linearen Regressionsmodells

            Obwohl durch die Reduktion der Feature-Anzahl die Multikollinearität verringert werden konnte, ist das Problem damit noch nicht vollständig gelöst.

            Der VIF-Wert des Features Beschäftigte bleibt weiterhin deutlich erhöht, und auch die Korrelationskoeffizienten zu den anderen Features sind nach wie vor relativ hoch.

            Um dieses Problem angemessen zu adressieren, können für das Vorhersagemodell verschiedene lineare Regressionsverfahren in Betracht gezogen werden.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            - Lasso
            - Ridge
            - ElasticNet
            """
        )

        st.markdown(
            """
            Alle drei oben genannten Modelle können durch den Einsatz eines Regularisierungsterms die Größe der Regressionskoeffizienten kontrollieren und dadurch Probleme der Multikollinearität abmildern.

            Da jedoch sowohl das Lasso- als auch das ElasticNet-Modell einzelne Koeffizienten auf 0 setzen können und in diesem Projekt nur eine relativ geringe Anzahl an Features verwendet wird, könnte ein solches „Ausschalten“ von Variablen die Stabilität des Modells beeinträchtigen.

            Aus diesem Grund wurde für die Vorhersage des industriellen Stromverbrauchs das Ridge-Modell ausgewählt.

            Weitere Details hierzu finden sich im Tab [II. Modellierung & Bewertung].
            """
        )


    with sub_tabs_a[3]:
        st.markdown(
            """
            ### 4. Datenquellen

            Die Quellen der in diesem Projekt verwendeten Daten sind wie folgt:
            """
        )

        st.space(size='small')

        link_df = pd.DataFrame(
            {
                'Data':[
                    'Verwaltungsgebietsbezeichnungen',
                    'Geodatenpaket für die Deutschlandkarte (für GeoPandas)',
                    'Endenergieverbrauch der Haushalte und sonstigen Verbraucher',
                    'EN-EB_verbrauchHaushalte',
                    'Verarbeitenden Gewerbe, Betriebe, Beschäftigte, Gesamtumsatz',
                    'Investitionsdaten der Industrie',
                    'Energieverbrauch der Industrie seit 2003',
                    'Bevölkerungsdaten'
                ],
                'URL':[
                    'https://www-genesis.destatis.de/datenbank/online/statistic/11111/table/11111-0002',
                    'https://gdz.bkg.bund.de/index.php/default/digitale-geodaten/verwaltungsgebiete/verwaltungsgebiete-1-5-000-000-stand-31-12-vg5000-12-31.html',
                    'https://www.statistik-bw.de/Energie/Energiebilanz/LRt1005.jsp',
                    'https://www.statistik-bw.de/Energie/Energiebilanz/EN-EB_verbrauchHaushalte.jsp',
                    'https://www.statistik-bw.de/Industrie/Struktur/06015150.tab?R=LA',
                    'https://www.statistik-bw.de/Industrie/Struktur/06013212.tab?R=RV43',
                    'https://www.statistik-bw.de/Energie/ErzeugVerwend/EV-Industrie.jsp',
                    'https://www.statistik-bw.de/BevoelkGebiet/Bevoelkerung/01515020.tab?R=LA'
                ]
            }
        )

        st.dataframe(link_df, hide_index=True, column_config={'URL':st.column_config.LinkColumn()})

        st.space(size='small')

        container_for_hinweis_2 = st.container(border=True)
        with container_for_hinweis_2:
            st.markdown(
                """
                #### :exclamation: Hinweis :exclamation:

                Die unten aufgeführten URLs stammen aus der früheren Version der Website des Statistischen Landesamts Baden-Württemberg und waren zum Zeitpunkt des Datenbezugs gültig.
                
                Nach dem Website-Relaunch im Jahr 2025 werden diese Adressen jedoch nicht mehr zu den ursprünglichen Themenseiten weitergeleitet, sondern führen lediglich auf die neu strukturierte Hauptseite des Statistikportals. Die entsprechenden Themen müssen dort manuell erneut recherchiert werden.

                ## :pensive:
                """
            )


with main_tabs[1]:
    st.markdown(
        """
        ## II. Modellierung & Bewertung

        In diesem Bereich können Sie die im Projekt verwendeten Vorhersagemodelle einsehen.

        Bitte wählen Sie unten einen der Tabs aus.
        """
    )

    sub_tabs_b = st.tabs(['1.Auswahl der Modellierungsstrategie', '2.Bewertung des Gesamtmodells', '3.Bewertung der regionalen Modelle'])

    with sub_tabs_b[0]:
        st.markdown(
            """
            ### 1.Auswahl der Modellierungsstrategie

            In diesem Projekt wurde zur Abschwächung der Multikollinearitätsproblematik das Ridge-Regressionsmodell eingesetzt.

            Mit dem in scikit-learn verfügbaren Ridge-Objekt lässt sich dieses Modell direkt erstellen.

            Für eine bessere Modellperformance ist es jedoch notwendig, die Beziehungen zwischen den Features genauer zu untersuchen.

            Insbesondere die Zusammenhänge zwischen den drei ausgewählten unabhängigen Variablen und der abhängigen Variable können anhand der folgenden Streudiagramme dargestellt werden.
            """
        )

        st.space(size='small')

        fig_scatter_1 = px.scatter(combined_df, x='Beschäftigte', y='Stromverbrauch(Industrie)', color='DN_DT', title='fig1. Beschäftigte vs. Stromverbrauch(Industrie)')
        st.plotly_chart(fig_scatter_1)

        fig_scatter_2 = px.scatter(combined_df, x='Gesamtumsatz', y='Stromverbrauch(Industrie)', color='DN_DT', title='fig2. Gesamtumsatz vs. Stromverbrauch(Industrie)')
        st.plotly_chart(fig_scatter_2)

        fig_scatter_3 = px.scatter(combined_df, x='Investitionen', y='Stromverbrauch(Industrie)', color='DN_DT', title='fig3. Investitionen vs. Stromverbrauch(Industrie)')
        st.plotly_chart(fig_scatter_3)

        st.space(size='small')

        st.markdown(
            """
            Beim Betrachten der drei oben dargestellten Streudiagramme lässt sich erkennen, dass - abgesehen von einigen Ausreißerregionen - die unabhängigen Variablen (Beschäftigte, Gesamtumsatz, Investitionen) den abhängigen Wert (Stromverbrauch (Industrie)) gut und weitgehend linear erklären.

            Aus dieser Perspektive erscheint es sinnvoll, für jene Regionen, in denen die unabhängigen Variablen eine hohe Erklärungskraft aufweisen, ein gemeinsames lineares Regressionsmodell zu verwenden.

            Für die übrigen Regionen hingegen kann ein jeweils regionsspezifisches Modell zu einer insgesamt stabileren Modellleistung führen.

            Daher lassen sich die linearen Regressionsmodelle zur Vorhersage des industriellen Stromverbrauchs im Wesentlichen in zwei Kategorien einteilen:
            """
        )

        st.space(size='small')

        st.markdown(
            """
            - Gesamtmodell

            - Regionale Modelle
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Für den Großteil der Kreise in Baden-Württemberg wird in diesem Projekt das Gesamtmodell verwendet.

            Für jene Regionen jedoch, die in den Streudiagrammen deutliche Ausreißer gezeigt haben, wurden stattdessen regionale Modelle angewendet.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            - **Die Kreise : Alb-Donau-Kreis, Böblingen, Karlsruhe(kreisfreie Stadt), Mannheim, Ortenaukreis, Rastatt, Stuttgart, Waldshut**
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Da die Ausreißerstrukturen in den Regionen, für die regionale Modelle verwendet wurden, je nach Kreis deutlich unterschiedliche Muster aufwiesen, wurde für jeden Kreis ein eigener Ridge-Regressor erstellt und separat trainiert.

            Die regionalen Modelle können dadurch zwar ein erhöhtes Risiko für Overfitting aufweisen, jedoch konnte dieses Problem durch die Anpassung des jeweiligen alpha-Werts des Ridge-Modells in gewissem Maße reduziert werden.

            Zudem wurden zur Optimierung der Modellleistung für jeden Kreis unterschiedliche alpha-Werte verwendet.
            """
        )


    with sub_tabs_b[1]:
        st.markdown(
            """
            ### 2.Bewertung des Gesamtmodells

            In diesem Tab können die wichtigsten Bewertungskennzahlen des Gesamtmodells eingesehen werden.

            Für das Gesamtmodell wurde ein Ridge-Modell verwendet, wobei ein Alpha-Wert von 10 angewendet wurde.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            #### i) Feature-Importance
            """
        )

        st.space(size='small')

        # evalution for Visualization
        y_test, y_pred, residuals, importance_df, eval_results_df, res_df = test_gesamtmodell(combined_df, scaler_for_normal_reg, normal_ridge_model)

        # Visualization
        fig_importance = px.bar(
            importance_df,
            x='feature',
            y='importance',
            title='fig1. Ridge Regression Feature Importance (Coefficient Magnitude)',
            text='importance'
        )

        st.plotly_chart(fig_importance)

        st.space(size='small')

        st.markdown(
            """
            ### ii) Residuenverteilung von Prognose- und Istwerten - Histogramm
            """
        )

        st.space(size='small')

        fig_res_his = px.histogram(
            x=residuals,
            nbins=30,
            title='fig2. Residual Distribution',
            labels={'x': 'Residual'},
        )

        fig_res_his.add_vline(x=0, line_width=2, line_dash='dash', line_color='red')

        st.plotly_chart(fig_res_his)

        st.space(size='small')

        st.markdown(
            """
            ### iii) Residuenverteilung von Prognose- und Istwerten - Streudiagramm
            """
        )

        st.space(size='small')

        fig_res_scatt = px.scatter(
            res_df,
            x='Predicted',
            y='Residual',
            title='fig3. Residual Plot (Predicted vs Residual)',
        )

        fig_res_scatt.add_hline(y=0, line_width=2, line_dash="dash", line_color="red")

        st.plotly_chart(fig_res_scatt)

        st.space(size='small')

        st.markdown(
            """
            ### iv) Verteilung von Ist- und Prognosewerten - Streudiagramm
            """
        )

        st.space(size='small')

        # scatter-object for Visualization
        fig_test_pred_scatt = go.Figure()

        # scatter-object trace
        fig_test_pred_scatt.add_trace(go.Scatter(
            x=y_test,
            y=y_pred,
            mode='markers',
            name='Actual vs Predicted'
        ))

        # y = x line
        fig_test_pred_scatt.add_trace(go.Scatter(
            x=y_test,
            y=y_test,
            mode='lines',
            name='Ideal Fit',
            line=dict(dash='dash', color='red')
        ))

        fig_test_pred_scatt.update_layout(
            title='fig4. Actual vs Predicted Scatter Plot',
            xaxis_title='Actual',
            yaxis_title='Predicted',
            width=700,
            height=500
        )

        st.plotly_chart(fig_test_pred_scatt)


    with sub_tabs_b[2]:
        st.markdown(
            """
            ### 3.Bewertung der regionalen Modelle

            In diesem Tab können die wichtigsten Bewertungskennzahlen der regionalen Modelle eingesehen werden.
            """
        )

        st.space(size='small')

        st.markdown(
            """
            Die Regionen, für die die regionalen Modelle angewendet werden, weisen jeweils unterschiedliche Verteilungen und Charakteristika der Features auf. Daher ist es schwierig, die Zielvariable anhand einer einheitlichen Regel zu erklären.

            Aus diesem Grund wurde für jeden Kreis ein separates Regressionsmodell erstellt.

            Allerdings entsteht bei der Erstellung kreisspezifischer Modelle das Problem, dass das Modell ausschließlich mit den Daten des jeweiligen Kreises trainiert werden kann, wodurch die verfügbare Datenmenge deutlich reduziert wird.

            Um die dadurch mögliche Instabilität der Modelle zu verringern, wurden alle verfügbaren Daten ohne Aufteilung in Trainings- und Testdaten zum Training verwendet. Die Modelle wurden anschließend mittels K-Fold-Cross-Validation bewertet.

            Darüber hinaus wurde für jedes kreisspezifische Ridge-Modell ein optimaler Alpha-Wert ermittelt und angewendet.

            Die folgende DataFrame zeigt die Ergebnisse der K-Fold-Cross-Validation für die einzelnen Kreis-Modelle.
            """
        )

        st.space(size='small')

        # k-fold cv
        result_for_s_model_df = kfold_result(combined_df, scaler_for_special_reg)

        container_for_kfold = st.container(border=True)
        with container_for_kfold:
            st.write('Table1. K-Fold-CV-Ergebnisse und zentrale Kennzahlen nach Region')
            st.dataframe(result_for_s_model_df, hide_index=True)

        st.space(size='small')

        st.markdown(
            """
            Beim Blick auf die obige Ergebnistabelle ist zu erkennen, dass einige regionale Modelle sehr niedrige R²-Werte aufweisen, was darauf hindeutet, dass deren Erklärungsleistung nicht ausreichend ist.

            Allerdings zeigte sich, dass sich die R²-Werte deutlich verändern, wenn der random_state-Parameter der für die Bewertung verwendeten K-Fold-Methode variiert wurde. Diese Instabilität scheint darauf zurückzuführen zu sein, dass die jeweiligen Modelle nur über eine relativ geringe Anzahl an Trainingsdaten verfügen.

            Ein Vergleich der RMSE-Werte mit den durchschnittlichen Werten der unabhängigen Variablen in den einzelnen Kreisen zeigt jedoch, dass der Vorhersagebereich der Modelle insgesamt relativ stabil ist.
            """
        )

        st.space(size='small')

        container_for_hinweis_3 = st.container(border=True)
        with container_for_hinweis_3:
            st.markdown(
                """
                #### :exclamation: Hinweis :exclamation:

                Die in diesem Tab behandelte Leistungsfähigkeit der regionalen Modelle wird in einer zukünftigen Version weiter verbessert werden.
                """
            )
