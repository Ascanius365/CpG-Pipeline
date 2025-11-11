Master-Regulatoren im Alter und Krebs 

Dieses Projekt führt eine bioinformatische Analyse von DNA-Methylierungsdaten (CpG-Stellen) aus dem TCGA-DLBC (Krebs) und einem GSE-Datensatz (Alterung) durch. Das Hauptziel ist die Identifizierung von Master-Regulatoren. Das sind CpG-Stellen, die zwei Kriterien erfüllen: Sie zeigen eine signifikante Korrelation mit dem Alter und sie zeigen eine hohe Korrelation mit dem Krebs-Status (hier dargestellt durch den mittleren Methylierungswert im DLBC-Datensatz). Anschließend wird ein Trade-Off zwischen der regulatorischen Stärke (partielle Korrelation) und dem Stabilitätsverlust im Alter (Varianz-Ratio) quantifiziert und visualisiert. 

Übersicht des Workflows

* Datenvorbereitung & Integration: Laden und Bereinigen von Methylierungsdaten (DLBC.cct, computage_bench_data_GSE131989.parquet) und Metadaten.

* LIMMA-Analyse: Identifizierung von CpGs, die signifikant mit dem Alter (GSE-Datensatz) und dem Krebs-Status (DLBC-Datensatz) korrelieren.

Top-CpG-Auswahl: Auswahl der Top 100 CpGs basierend auf einem kombinierten Effekt (Produkt der Alters-Steigung und des Krebs-Mittelwerts).

Partielle Korrelation: Berechnung der partiellen Korrelation (A, B | Alter) zwischen den Top-CpGs, um die direkten epigenetischen Kopplungen zu isolieren, die unabhängig vom Alter sind. 

Varianz-Analyse (Stabilität): F-Test und Berechnung des Varianz-Ratios (Alt / Jung) für die identifizierten Master-Regulatoren. 

Visualisierung & Netzwerk-Analyse: Erstellung eines Trade-Off-Plots (Stärke vs. Stabilitätsverlust). Erstellung eines funktionalen Interkorrelations-Netzwerks mit Pathway-Annotationen (KEGG/Reactome) über mygene.py.
