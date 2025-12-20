# Prognosen Projekt :
# Stromverbrauch in Baden-Württemberg

## Projektübersicht

Dieses Projekt beschäftigt sich mit der Prognose des Stromverbrauchs auf Kreisebene im Bundesland Baden-Württemberg.
Ziel ist es, zukünftige Entwicklungen des Strombedarfs unter Berücksichtigung verschiedener Szenarien zu analysieren und visuell darzustellen.


## Motivation & Zielsetzung

Der steigende Strombedarf stellt hohe Anforderungen an eine zuverlässige Prognose des Energieverbrauchs. Dieses Projekt wurde entwickelt, um regionale Unterschiede sichtbar zu machen und eine datenbasierte Grundlage für zukünftige Planungen zu schaffen.


## Datengrundlage

Die Analyse basiert auf öffentlich zugänglichen Stromverbrauchsdaten 
sowie ergänzenden regionalen Strukturdaten (z. B. Haushalte, Beschäftigte).
Die Rohdaten wurden bereinigt, aggregiert und für die Modellierung aufbereitet.


## Methodik / Modellansatz

Für die Prognose wurde ein regressionsbasierter Modellansatz verwendet.
Zur Vermeidung von Multikollinearität wurden Ridge-Regressionsmodelle auf Basis vorangegangener Korrelationsanalysen eingesetzt.
Die Modellgüte wurde auf Kreisebene anhand von RMSE und Standardabweichungen evaluiert.


## Ergebnisse & Visualisierung

Die Ergebnisse zeigen deutliche regionale Unterschiede im prognostizierten Stromverbrauch.
Zur besseren Interpretation wurden die Resultate in Karten- und Zeitreihenvisualisierungen aufbereitet.

[Streamlit-App ansehen](https://stromverbrauchprojekt-fjewapmyugzcbt2khgsmwp.streamlit.app/)


## Einschränkungen & Ausblick

Einige Kreise weisen höhere Prognoseabweichungen auf, was auf unterschiedliche Verbrauchsmuster hindeutet.
Zukünftige Erweiterungen könnten eine alternative Clusterbildung sowie zusätzliche erklärende Variablen berücksichtigen.


## Technologien

- Python
- pandas, Geopandas, Numpy, scikit-learn
- Streamlit
- Plotly