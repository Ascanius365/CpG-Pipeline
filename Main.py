from huggingface_hub import snapshot_download
from numpy.linalg import pinv
from scipy.stats import f as f_dist, pearsonr
import seaborn as sns
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mygene
import warnings
import matplotlib.colors as mcolors


# Harte Werte

N_TOP_CPGS = 100
N_PAIRS_TO_SELECT = 10

# DLBC Datenbank laden

try:
    cpg_data_dlbc = pd.read_csv('DLBC.cct', sep='\t', index_col=0)

    if cpg_data_dlbc.empty:
        raise ValueError("Datei wurde leer geladen. Prüfen Sie das Trennzeichen.")

except FileNotFoundError:
    print("Fehler: Die Datei 'DLBC.cct' wurde nicht im aktuellen Verzeichnis gefunden.")

except Exception as e:
    print(f"Fehler beim Laden der Datei: {e}")

# Ausschnitt definieren (Die ersten 5 Proben und 10 CpGs)

if cpg_data_dlbc.shape[0] > cpg_data_dlbc.shape[1]:
    cpg_data_dlbc = cpg_data_dlbc.T

# Wähle die ersten 5 Zeilen (Proben) und die ersten 10 Spalten (CpGs)
sample_data_display = cpg_data_dlbc.iloc[:5, :10]

# Heatmap erstellen

plt.figure(figsize=(10, 5))
sns.heatmap(
    sample_data_display,
    annot=True,
    fmt=".3f", # 3 Dezimalstellen für die Beta-Werte (0.000 bis 1.000)
    cmap='viridis',
    cbar_kws={'label': 'Methylierungsgrad (Beta-Wert)'},
    linewidths=.5,
    linecolor='white'
)

plt.title('Methylierungsmuster: Ausschnitt der ersten 5 TCGA-DLBC Proben und 10 CpGs', fontsize=14)
plt.xlabel('CpG-Stelle (ID)')
plt.ylabel('Proben-ID')
plt.yticks(rotation=0)
plt.savefig('Plot5.png')
plt.show()

import pandas as pd
import numpy as np

# Methylierungsdaten (DLBC.cct) laden
try:
    # Laden der Methylierungs-Beta-Werte (CpG-Stellen in Spalten, Proben in Zeilen)
    cpg_data_dlbc = pd.read_csv('DLBC.cct', sep='\t', index_col=0)

    # Transponieren, damit Proben in Zeilen und CpGs in Spalten sind (Standardformat)
    if cpg_data_dlbc.shape[0] > cpg_data_dlbc.shape[1]:
        cpg_data_dlbc = cpg_data_dlbc.T

    cpg_data_dlbc.index.name = 'Case_ID'

except Exception as e:
    print(f"Fehler beim Laden der Methylierungsdaten (DLBC.cct): {e}")
    cpg_data_dlbc = pd.DataFrame() # Erzeuge leeres DF bei Fehler

# Klinische Metadaten (Annotation.tsi) laden
try:
    # Laden der Metadaten mit korrigierter Struktur
    annotation_data = pd.read_csv(
        'Annotation.tsi',
        sep='\t',
        header=0,
        index_col=0,
        low_memory=False
    ).T

    # Setze den Indexnamen korrekt
    annotation_data.index.name = 'Case_ID'

    # Normalisiere Spaltennamen (Kleinschreibung, Punkte entfernen)
    annotation_data.columns = annotation_data.columns.str.lower().str.replace('.', '_', regex=False)

except Exception as e:
    try:
        annotation_data = pd.read_csv(
            'Annotation.tsi',
            sep='\s+',
            header=0,
            index_col=0,
            low_memory=False
        ).T
        annotation_data.index.name = 'Case_ID'
        annotation_data.columns = annotation_data.columns.str.lower().str.replace('.', '_', regex=False)
    except Exception as e_retry:
        print(f"Fehler beim Laden der Metadaten (Annotation.tsi): {e_retry}")
        annotation_data = pd.DataFrame()

# Zusammenfassung ausgeben

print("\n=====================================================================")
print("ZUSAMMENFASSUNG GELADENER TCGA-DLBC DATEN")
print("=====================================================================")

print(f"Methylierungsdaten (DLBC.cct):")
print(f" Shape (Proben x CpGs): {cpg_data_dlbc.shape}")
if not cpg_data_dlbc.empty:
    print(f" Erste 5 Spalten (CpGs): {cpg_data_dlbc.columns[:5].tolist()}")
    print(f" Erste 5 Proben (IDs): {cpg_data_dlbc.index[:5].tolist()}")

print("\nKlinische Metadaten (Annotation.tsi):")
print(f" Shape (Proben x Metadaten): {annotation_data.shape}")
if not annotation_data.empty:
    print(f" Wichtige Spalten (Ausschnitt): {annotation_data.columns[:5].tolist()}")
    print(f" Erste 5 Proben (IDs): {annotation_data.index[:5].tolist()}")

print("\n--- Nächster Schritt ---")
print("Prüfen Sie, ob die Proben-IDs in beiden Datensätzen übereinstimmen. ")
print("Danach können Sie die Daten über die 'Case_ID' zusammenführen.")


# Illumina Datenbank laden

ANNOTATION_FILE_PATH = 'humanmethylation450_15017482_v1-2.csv'

# Gene-Map Erstellung
try:
    annotation_df = pd.read_csv(
        ANNOTATION_FILE_PATH,
        sep=',',
        skiprows=7,
        low_memory=False
    )

    CpG_ID_COL = 'IlmnID'
    GEN_INFO_COL = 'UCSC_RefGene_Name'
    annotation_df.rename(columns={CpG_ID_COL: 'CpG-ID'}, inplace=True)
    annotation_df['Gen'] = annotation_df[GEN_INFO_COL].astype(str).apply(
        lambda x: str(x).split(';')[0].strip() if str(x) and str(x) != 'nan' else 'Unannotated'
    )

    gene_map = annotation_df[annotation_df['Gen'].str.len() > 1].set_index('CpG-ID')['Gen'].to_dict()

    print("Erfolg: Vollständige Illumina Gen-Annotation geladen und gene_map mit", len(gene_map),
          "Einträgen erstellt.")

except FileNotFoundError:
    print(f"FEHLER: Die Annotationsdatei '{ANNOTATION_FILE_PATH}' wurde nicht gefunden.")

except KeyError as e:
    print(f"FEHLER: Spalte {e} nicht gefunden.")

# Prüfung und Ausgabe der Annotation
if 'annotation_df' in locals() and not annotation_df.empty:
    print("\n" + "=" * 80)
    print("PRÜFUNG: ERSTE 10 ANNOTIERTE EINTRÄGE")
    print("=" * 80)

    # Erstellen eines DataFrames für die Anzeige:
    display_df = pd.DataFrame({
        'CpG-ID': annotation_df['CpG-ID'].head(10),
        'Chrom': annotation_df['CHR'].head(10),
        'Gen_Zuordnung': annotation_df['Gen'].head(10)
    })

    # Erste 10 Zeilen ausgeben
    print(display_df.to_string(index=False))

    print(
        f"\nÜberprüfung abgeschlossen")
else:
    print("FEHLER: Das DataFrame 'annotation_df' konnte nicht geladen werden.")

print("=" * 80)

# Datenbank laden
# df = pd.read_parquet('data/benchmark/computage_bench_data_GSE137594.parquet').T
df = pd.read_parquet('data/benchmark/computage_bench_data_GSE131989.parquet').T
meta = pd.read_csv('computage_bench_meta.tsv', sep='\t', index_col=0)

merged_df = df.merge(meta, left_index=True, right_index=True, how='left')
correlation_data = merged_df.dropna(subset=['Age']).copy()
cpg_columns = df.columns.tolist()

# Beispiele aus Datenbank ausgeben

# Subset für Heatmap
subset_cpgs = merged_df.iloc[:20, :50] # 20 Proben, 50 CpG-Stellen

# Metadaten der gleichen Proben in der Konsole ausgeben
meta_subset = merged_df.loc[subset_cpgs.index, ["Age", "Condition"]]
print("Metadaten der angezeigten Proben:\n")
print(meta_subset)

# Heatmap zeichnen
sns.heatmap(subset_cpgs, cmap="viridis")
plt.title("DNA-Methylierungswerte (Beta) – Beispielsubmatrix")
plt.xlabel("CpG-Stellen")
plt.ylabel("Proben")
plt.savefig('Plot1.png')
plt.show()


cpg_data = correlation_data[cpg_columns]

# HILFSDEFINITIONEN (Aus dem oben stehenden Code)
cpg_data_gse = df.copy()
age_series_gse = correlation_data['Age']

# =========================================================================
# KORREKTUR: Entfernen des Gen-Präfixes in DLBC
# =========================================================================

if not cpg_data_dlbc.empty:
    # Beispiel: 'RBL2_cg00000029' wird zu 'cg00000029'
    print(f"Bereinige DLBC-CpG-Namen (von z.B. '{cpg_data_dlbc.columns[0]}' zu...)")
    cpg_data_dlbc.columns = cpg_data_dlbc.columns.str.split('_', n=1).str[-1].str.lower().str.strip()
    print(f"   ...zu z.B. '{cpg_data_dlbc.columns[0]}'")

    # Auch im GSE-Datensatz sicherstellen, dass alles klein geschrieben ist
    cpg_data_gse.columns = cpg_data_gse.columns.str.lower().str.strip()

import pandas as pd
import numpy as np
import os
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import default_converter, FloatVector, IntVector
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.pandas2ri import converter

# =========================================================================
# DATENVORBEREITUNG
# =========================================================================

R_LIBRARY_PATH = '/home/benito/R/x86_64-pc-linux-gnu-library/4.5'
os.environ['R_LIBS_USER'] = R_LIBRARY_PATH

# VORBEREITUNG DER R-UMGEBUNG
try:
    limma = importr('limma')
    base = importr('base')

except Exception as e:
    print(f"FEHLER beim Laden der R-Pakete: {e}")
    print("Stellen Sie sicher, dass 'limma' in R installiert ist und der Pfad R_LIBS_USER korrekt ist.")
    exit()

# VORBEREITUNG DER GEMEINSAMEN CPGS
common_cpgs = list(set(cpg_data_gse.columns) & set(cpg_data_dlbc.columns))
print(f"Nach Bereinigung wurden {len(common_cpgs)} übereinstimmende CpG-Stellen gefunden.")

cpg_gse_common = cpg_data_gse[common_cpgs]
cpg_dlbc_common = cpg_data_dlbc[common_cpgs]

# =========================================================================
# LIMMA-ANALYSE 1: ALTERS-EFFEKT (GSE-Datensatz)
# =========================================================================

print("\n LIMMA-ANALYSE 1: Alterskorrelation (GSE)")

# Daten ausrichten und Matrix transponieren
cpg_gse_common_aligned = cpg_gse_common.loc[age_series_gse.index.intersection(cpg_gse_common.index)]
age_series_aligned_gse = age_series_gse.loc[cpg_gse_common_aligned.index]
r_cpg_matrix_gse = cpg_gse_common_aligned.T
flat_data = r_cpg_matrix_gse.values.flatten(order='F')
r_vector = FloatVector(flat_data)
num_rows_gse = r_cpg_matrix_gse.shape[0]
num_cols_gse = r_cpg_matrix_gse.shape[1]
r_cpg_matrix_gse_R = ro.r.matrix(r_vector, nrow=num_rows_gse, ncol=num_cols_gse, byrow=False)

# Design-Matrix erstellen
with localconverter(default_converter + converter):
    ro.globalenv['Age'] = age_series_aligned_gse
    ro.r('DESIGN_GSE <- model.matrix(~Age)')

# limma-Fit und eBayes ausführen
with localconverter(default_converter + converter):
    ro.globalenv['R_CPG_MATRIX_GSE'] = r_cpg_matrix_gse_R
    ro.r('FIT_GSE_FINAL <- limma::eBayes(limma::lmFit(R_CPG_MATRIX_GSE, DESIGN_GSE))')

    ro.globalenv['NUM_ROWS_GSE'] = IntVector([num_rows_gse])
    tt_gse = ro.r('limma::topTable(FIT_GSE_FINAL, number=NUM_ROWS_GSE, coef=2)')

# Ergebnisse konvertieren
with localconverter(default_converter + converter):
    results_gse_df = base.as_data_frame(tt_gse)

# Duplikate entfernen VOR Index-Setzung (V4)
results_gse_df['ID'] = r_cpg_matrix_gse.index.tolist()
results_gse_df = results_gse_df.drop_duplicates(subset=['ID'], keep='first')
results_gse = results_gse_df.set_index('ID')
results_gse = results_gse[~results_gse.index.duplicated(keep='first')]
vector_age_limma = results_gse['logFC']
print("Alters-Korrelation (Koeffizienten und FDR) durch limma berechnet.")

# =========================================================================
# LIMMA-ANALYSE 2: KREBS-EFFEKT (DLBC-Datensatz)
# =========================================================================

print("\n LIMMA-ANALYSE 2: Krebs-Effekt (DLBC)")

# Daten transponieren
r_cpg_matrix_dlbc = cpg_dlbc_common.T
flat_data_dlbc = r_cpg_matrix_dlbc.values.flatten(order='F')
r_vector_dlbc = FloatVector(flat_data_dlbc)
num_rows_dlbc = r_cpg_matrix_dlbc.shape[0]
num_cols_dlbc = r_cpg_matrix_dlbc.shape[1]
r_cpg_matrix_dlbc_R = ro.r.matrix(r_vector_dlbc, nrow=num_rows_dlbc, ncol=num_cols_dlbc, byrow=False)

# Design-Matrix erstellen
ro.globalenv['DUMMY_DATA_DLBC'] = ro.r.rep(1, num_cols_dlbc)
ro.r('DESIGN_DLBC <- model.matrix(~1, data=data.frame(DUMMY_DATA_DLBC))')

# limma-Fit ausführen
with localconverter(default_converter + converter):
    ro.globalenv['R_CPG_MATRIX_DLBC'] = r_cpg_matrix_dlbc_R
    ro.r('FIT_DLBC_FINAL <- limma::lmFit(R_CPG_MATRIX_DLBC, DESIGN_DLBC)')

# Koeffizienten extrahieren
mean_meth_limma_r = ro.r('FIT_DLBC_FINAL$coefficients')

with localconverter(default_converter + converter):
    mean_meth_limma_df = base.as_data_frame(mean_meth_limma_r)

# Duplikate im Index bereinigen (V4)
mean_meth_limma_df.index = r_cpg_matrix_dlbc.index.tolist()
mean_meth_limma_df = mean_meth_limma_df[~mean_meth_limma_df.index.duplicated(keep='first')]
vector_mean_meth_limma = mean_meth_limma_df.iloc[:, 0]
print("Krebs-Effekt (gewichteter Mittelwert) durch limma berechnet.")

# =========================================================================
# KORRELATION DER LIMMA-ERGEBNISSE UND FILTERUNG
# =========================================================================

print("\n KORRELATION DER LIMMA-ERGEBNISSE")

# Füge die zwei limma-Ergebnis-Vektoren in ein DataFrame zusammen
limma_comparison = pd.DataFrame({
    'Corr_Age_GSE_Limma': vector_age_limma,
    'Mean_Meth_DLBC_Limma': vector_mean_meth_limma
}).dropna()

# Kombinierter Effekt
combined_limma_effect = (
        limma_comparison['Corr_Age_GSE_Limma'] * limma_comparison['Mean_Meth_DLBC_Limma']
)

# FDR-Signifikanz hinzufügen
top_results = limma_comparison.copy()
top_results['Combined_Effect'] = combined_limma_effect

# Sicherstellen, dass nur die Indizes verwendet werden, die in limma_comparison enthalten sind
results_gse_filtered = results_gse.loc[limma_comparison.index]
top_results['Adj_P_Value_Age'] = results_gse_filtered['adj.P.Val']

print("\n=====================================================================")
print("LIMMA: Alters-Effekt (Steigung) vs. Krebs-Mittelwert (Koeffizient)")
print("=====================================================================")
print(f"Anzahl gemeinsamer und limma-bereinigter CpGs: {limma_comparison.shape[0]}")

# Auswahl der Top CpGs
sorted_correlations_abs = top_results['Combined_Effect'].abs().sort_values(ascending=False)
top_cpgs_names = sorted_correlations_abs.head(N_TOP_CPGS).index.tolist()

print("\nTop 5 CpG-Stellen mit dem stärksten kombinierten Effekt (positiv):")
# Filtere nach Signifikanz des Alters-Effekts (FDR < 0.05)
top_positive = top_results[top_results['Adj_P_Value_Age'] < 0.05].sort_values('Combined_Effect', ascending=False).head(
    5)
print(top_positive[['Combined_Effect', 'Adj_P_Value_Age']])

print("\nTop 5 CpG-Stellen mit dem stärksten entgegengesetzten Effekt (negativ):")
top_negative = top_results[top_results['Adj_P_Value_Age'] < 0.05].sort_values('Combined_Effect', ascending=True).head(5)
print(top_negative[['Combined_Effect', 'Adj_P_Value_Age']])

# =========================================================================
# KORRELATION DER LIMMA-ERGEBNISSE & FILTERUNG (KONSOLIDIERT)
# =========================================================================

# Auswahl der Top CpGs basierend auf dem kombinierten Effekt (Absolutwert)
sorted_correlations_abs = top_results['Combined_Effect'].abs().sort_values(ascending=False)
top_cpgs_prelim = sorted_correlations_abs.head(N_TOP_CPGS).index.tolist()

print(f"\nLIMMA-Pipeline abgeschlossen. Starte Netzwerk-Vorbereitung mit {len(top_cpgs_prelim)} CpGs.")

# =========================================================================
# PARTIELLE KORRELATION (NETZWERK-VORBEREITUNG)
# =========================================================================

print("\n" + "=" * 80)
print("1. Regulatoren heraussuchen (Korrelation A, B | Alter)")
print("=" * 80)

# Datenvorbereitung

cpgs_in_data = correlation_data.columns.tolist()

# Nur CpGs behalten, die in der finalen Datenbasis (correlation_data) existieren.
valid_top_cpgs_names = [cpg for cpg in top_cpgs_prelim if cpg in cpgs_in_data]
top_cpgs_names = valid_top_cpgs_names

column_list = top_cpgs_names + ['Age']
analysis_data = correlation_data[column_list].copy()
cpg_subset_data = correlation_data[top_cpgs_names]

# Berechne die Korrelationsmatrix der Stellen untereinander
print(f" Berechne die {N_TOP_CPGS}x{N_TOP_CPGS} Interkorrelationsmatrix...")
inter_correlation_matrix = cpg_subset_data.corr(method='pearson')

# Heatmap-Erstellung
plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(12, 10))
plt.title(f'Interkorrelation (Heatmap) der Top {N_TOP_CPGS} Alter- und Krebs-Korrelierten CpG-Stellen', fontsize=16)

# Maske erstellen, um die redundante obere Hälfte der Matrix auszublenden
mask = np.triu(inter_correlation_matrix)

# Heatmap visualisieren
sns.heatmap(inter_correlation_matrix,
            mask=mask,
            cmap='coolwarm', # Farbskala von blau (negativ) über weiß (keine) zu rot (positiv)
            center=0,
            vmax=1, vmin=-1,
            cbar_kws={'label': 'Pearson Korrelation'},
            square=True,
            xticklabels=False, yticklabels=False
)

print("\n" + "=" * 80)
print(f"Die Heatmap der Interkorrelation ({N_TOP_CPGS}x{N_TOP_CPGS}) wurde erstellt.")
print("Hinweis: Helle Bereiche (rot/blau) deuten auf funktionale Gruppen von Stellen hin.")
print("=" * 80)

plt.savefig('Plot2.png')
plt.show()

# 1. Regulatoren heraussuchen

print("\n" + "=" * 80)
print("1. Regulatoren heraussuchen (Korrelation A, B | Alter)")
print("=" * 80)

# 1.2 Partielle Korrelationsmatrix berechnen
full_corr_matrix = analysis_data.corr().values
N_VARS = full_corr_matrix.shape[0]

try:
    inv_corr_matrix = pinv(full_corr_matrix)
except np.linalg.LinAlgError:
    inv_corr_matrix = pinv(full_corr_matrix + np.eye(N_VARS) * 1e-6)

partial_corr_matrix_np = np.zeros((N_TOP_CPGS, N_TOP_CPGS))
for i in range(N_TOP_CPGS):
    for j in range(N_TOP_CPGS):
        num = -inv_corr_matrix[i, j]
        den = np.sqrt(inv_corr_matrix[i, i] * inv_corr_matrix[j, j])
        partial_corr_matrix_np[i, j] = num / den

num_cpgs_final = len(top_cpgs_names)
partial_corr_matrix_np = partial_corr_matrix_np[:num_cpgs_final, :num_cpgs_final]
partial_corr_matrix = pd.DataFrame(partial_corr_matrix_np, index=top_cpgs_names, columns=top_cpgs_names)

# Visualisierung der Partiellen Korrelation (Heatmap)

plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(12, 10))
plt.title(f'Partielle Korrelation (A, B | Alter) der Top {N_TOP_CPGS} CpG-Stellen', fontsize=16)

# Maske, um die redundante obere Hälfte der Matrix auszublenden
mask = np.triu(partial_corr_matrix)

sns.heatmap(partial_corr_matrix,
            mask=mask,
            cmap='coolwarm',
            center=0,
            vmax=1, vmin=-1,
            cbar_kws={'label': 'Partielle Korrelation (r)'},
            square=True,
            xticklabels=False, yticklabels=False
            )

print("\n" + "=" * 80)
print(f"Die Partielle Korrelations-Heatmap (100x100) wurde erfolgreich erstellt (Matrixinversion).")
print("Diese Blöcke zeigen die direkten epigenetischen Kopplungen, die NICHT vom Alter erklärt werden.")
print("=" * 80)

plt.savefig('Plot3.png')
plt.show()



# 1.3 Top N_PAIRS_TO_SELECT stärkste Paare finden
corr_pairs = partial_corr_matrix.stack().reset_index()
corr_pairs.columns = ['CpG_A', 'CpG_B', 'Partielle_r']
corr_pairs = corr_pairs[corr_pairs['CpG_A'] != corr_pairs['CpG_B']]
corr_pairs['sorted_pair'] = [tuple(sorted(t)) for t in zip(corr_pairs.CpG_A, corr_pairs.CpG_B)]
corr_pairs.drop_duplicates(subset=['sorted_pair'], inplace=True)

top_pos_pairs = corr_pairs.sort_values(by='Partielle_r', ascending=False).head(N_PAIRS_TO_SELECT)
master_regulator_cpgs = np.unique(top_pos_pairs[['CpG_A', 'CpG_B']].values.flatten()).tolist()

master_regulator_genes = [gene_map.get(cpg, cpg) for cpg in master_regulator_cpgs]

num_unannotated = master_regulator_genes.count('NEUES GEN')
print(
    f"{len(master_regulator_cpgs)} Regulatoren gefunden. {num_unannotated} Gene wurden durch ihre CpG-ID ersetzt.")
print(f"   Gene: {master_regulator_genes}")

# 2. Variationsanalyse (Verlust der epigenetischen Präzision)

print("\n" + "=" * 80)
print("2. Analyse der Stabilität (Varianz Ratio Alt/Jung) und F-Test")
print("=" * 80)

# 2.1 Definition der Altersgruppen
young_data = correlation_data[correlation_data['Age'] <= 40][cpg_columns]
old_data = correlation_data[correlation_data['Age'] >= 60][cpg_columns]

# 2.2 Varianzberechnung für die Regulatoren und F-Test
if young_data.empty or old_data.empty or young_data.shape[0] < 2 or old_data.shape[0] < 2:
    print("Nicht genügend Proben in einer der Altersgruppen für Varianzanalyse.")
    # Initialisiere results_df mit allen benötigten Spalten für TEIL 4
    results_df = pd.DataFrame(columns=['CpG-ID', 'Varianz Ratio (Alt/Jung)', 'Gen', 'p-Wert (F-Test)', 'Signifikanz'])
else:
    young_subset = young_data[master_regulator_cpgs]
    old_subset = old_data[master_regulator_cpgs]

    var_young = young_subset.var()
    var_old = old_subset.var()

    # Freiheitsgrade für den F-Test
    df_old = len(old_subset) - 1
    df_young = len(young_subset) - 1

    variance_ratio_regulators = var_old / var_young
    variance_ratio_regulators.replace([np.inf, -np.inf], np.nan, inplace=True)
    variance_ratio_regulators.dropna(inplace=True)

    p_values = {}
    for cpg_id, ratio in variance_ratio_regulators.items():
        # F-Test (zweiseitig)
        if ratio >= 1:
            p_val = f_dist.sf(ratio, df_old, df_young)
        else:
            p_val = f_dist.sf(1 / ratio, df_young, df_old)
        p_values[cpg_id] = p_val * 2  # Zweiseitiger p-Wert

    sorted_ratios = variance_ratio_regulators.sort_values(ascending=False)

    # 2.3 Ausgabe der Ergebnisse
    results_df = pd.DataFrame({
        'CpG-ID': sorted_ratios.index,
        'Varianz Ratio (Alt/Jung)': sorted_ratios.values,
        'Gen': [gene_map.get(cpg, cpg) for cpg in sorted_ratios.index],
        # Hinzufügen des p-Wertes
        'p-Wert (F-Test)': [p_values.get(cpg, np.nan) for cpg in sorted_ratios.index]
    })

    # Erstellung der Signifikanz-Spalte
    results_df['Signifikanz'] = results_df['p-Wert (F-Test)'].apply(
        lambda p: '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else '')))

    print("-" * 80)
    print("Ratio > 1.0 deutet auf Verlust der epigenetischen Stabilität hin (Signifikanz durch F-Test).")

    # Korrigierte Ausgabe, die den p-Wert und die Signifikanz enthält:
    print(results_df[['Gen', 'Varianz Ratio (Alt/Jung)', 'p-Wert (F-Test)', 'Signifikanz']].to_string(index=False))

    print("=" * 80)

# 3. Visualisierung (Stärke vs. Stabilität)

print("\nErstelle die kombinierte Grafik (Regulatorische Stärke vs. Stabilitätsverlust)...")

# 3.1 Maximale Kopplungsstärke herausfinden (Partielle Korrelation)
max_corr_data = []

for cpg_id in results_df['CpG-ID']:
    if cpg_id in partial_corr_matrix.index:
        subset_corr = partial_corr_matrix.loc[cpg_id, master_regulator_cpgs].abs()
        # Entferne die Korrelation mit sich selbst (1.0) und finde das Maximum
        max_coupling = subset_corr[subset_corr.index != cpg_id].max()
    else:
        max_coupling = np.nan

    max_corr_data.append({
        'CpG-ID': cpg_id,
        'Max_Partielle_r': max_coupling
    })

coupling_df = pd.DataFrame(max_corr_data)

# 3.2 Zusammenführen der Stärke und Stabilität
combined_df = pd.merge(results_df, coupling_df, on='CpG-ID', how='inner').dropna()

# 3.3 Erstellung des Streudiagramms
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(12, 10))

sns.scatterplot(data=combined_df,
                x='Max_Partielle_r',
                y='Varianz Ratio (Alt/Jung)',
                s=200,
                color='darkred',
                alpha=0.8)

# Beschriftung der einzelnen Punkte mit den Gen-Namen
for i in range(combined_df.shape[0]):
    # Verwenden Sie die Gen-Spalte für die Beschriftung
    plt.text(combined_df['Max_Partielle_r'].iloc[i] + 0.005,
             combined_df['Varianz Ratio (Alt/Jung)'].iloc[i],
             combined_df['Gen'].iloc[i],
             fontsize=10,
             alpha=0.8)

plt.title(
    f'Langlebigkeitskandidaten: Regulatorische Stärke vs. Stabilitätsverlust (Top {len(master_regulator_cpgs)} CPGs)',
    fontsize=16)
plt.xlabel('Regulatorische Stärke (Max. Partielle Korrelation | Alter)', fontsize=14)
plt.ylabel('Stabilitätsverlust (Varianz Ratio Alt / Jung)', fontsize=14)

# Hervorheben der kritischen Schwelle (Stabilität)
plt.axhline(y=1.0, color='blue', linestyle='--', linewidth=1.2, label='Stabilitätsschwelle (Ratio=1.0)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

print("\n" + "=" * 80)
print("Die kombinierte Grafik (Stärke vs. Stabilität) wurde erfolgreich erstellt.")
print("Die Gene in der OBEREN RECHTEN Ecke sind die wichtigsten Master-Regulatoren.")
print("=" * 80)

plt.savefig('Plot4.png')
plt.show()

# 4. Quantifizierung und Visualisierung des Trade-Offs

print("\n 4. Quantifiziere und Visualisiere den Trade-Off")

# 4.1 Maximale Kopplungsstärke herausfinden
max_corr_data = []
for cpg_id in results_df['CpG-ID']:
    if cpg_id in partial_corr_matrix.index:
        subset_corr = partial_corr_matrix.loc[cpg_id, master_regulator_cpgs].abs()
        max_coupling = subset_corr[subset_corr.index != cpg_id].max()
    else:
        max_coupling = np.nan

    max_corr_data.append({
        'CpG-ID': cpg_id,
        'Max_Partielle_r': max_coupling
    })

coupling_df = pd.DataFrame(max_corr_data)
combined_df = pd.merge(results_df, coupling_df, on='CpG-ID', how='inner').dropna()

# 4.2 Korrelation zwischen den Achsen quantifizieren
if len(combined_df) > 1:
    corr, p_value_corr = pearsonr(combined_df['Max_Partielle_r'], combined_df['Varianz Ratio (Alt/Jung)'])

    print("\n" + "=" * 50)
    print(f"Maßkorrelationskoeffizient (Stärke vs. Stabilität):")
    print(f"Pearson's r = {corr:.4f}")
    print(f"p-Wert (Korrelation) = {p_value_corr:.4f}")
    print("Interpretation: Negativer r-Wert stützt die Hypothese (wichtige Gene stabiler).")
    print("=" * 50)


warnings.filterwarnings('ignore', category=FutureWarning)

# Liste der Gene
GENE_LIST = master_regulator_genes


# DATEN-FETCHING: ABFRUGE VON PATHWAY-DATEN (KEGG/Reactome)

def fetch_pathway_data(gene_list):
    # Holt Pathway-Daten (KEGG/Reactome) für die Genliste über den mygene.py Wrapper
    print("Starte API-Abfrage über mygene.py: Pathway-Daten (KEGG/Reactome)...")

    try:
        mg = mygene.MyGeneInfo()

        list_results = mg.querymany(
            gene_list,
            scopes='symbol',
            fields='symbol,pathway',
            species='human',
            as_dataframe=False,
            verbose=False
        )

        records = []
        for result in list_results:
            if result.get('notfound', False):
                continue

            gene_name = result.get('symbol')
            pathway_data = result.get('pathway')

            if gene_name and isinstance(pathway_data, dict):
                # KEGG und Reactome sind die wichtigsten Pathway-Quellen
                for source in ['kegg', 'reactome']:
                    source_data = pathway_data.get(source)

                    if source_data:
                        if isinstance(source_data, dict):
                            source_data = [source_data]

                        for item in source_data:
                            # Wir nehmen den Pathway-Namen
                            pathway_name = item.get('name', item.get('id', 'Unbekannt'))
                            if pathway_name != 'Unbekannt':
                                records.append({
                                    'Gene name': gene_name,
                                    'Pathway name': pathway_name,
                                    'Source': source
                                })

        df_pathways = pd.DataFrame(records)
        df_pathways = df_pathways.dropna(subset=['Gene name', 'Pathway name'])

        if df_pathways.empty:
            print("API lieferte keine validen Pathway-Daten.")
            return pd.DataFrame(columns=['Gene name', 'Pathway name'])

        print(f"Pathway-Daten für {df_pathways['Gene name'].nunique()} Gene erfolgreich abgerufen.")
        return df_pathways

    except Exception as e:
        print(f"FEHLER beim Abrufen der Pathway-Daten über mygene.py: {e}")
        return pd.DataFrame(columns=['Gene name', 'Pathway name'])


def assign_specific_pathways(df_pathways, target_genes):

    gene_pathways = {}

    if df_pathways.empty:
        return {gene: 'Unbekannter Pathway' for gene in target_genes}

        # Finde die häufigsten Pathways, die geteilt werden, oder alle Pathways des Gens.
    for gene_name in target_genes:
        # Finde alle Pathways für dieses spezielle Gen
        gene_pathway_names = df_pathways[df_pathways['Gene name'] == gene_name]['Pathway name'].tolist()

        if not gene_pathway_names:
            gene_pathways[gene_name] = 'Unbekannter Pathway'
            continue

        # Wähle den ersten verfügbaren Pathway als Repräsentant
        gene_pathways[gene_name] = gene_pathway_names[0]

    return gene_pathways


# DATENVERARBEITUNG UND NETZWERK-ERSTELLUNG

# Pathway-Daten holen
df_pathways = fetch_pathway_data(GENE_LIST)

# Spezifische Pathways zuweisen
gene_pathway_assignment = assign_specific_pathways(df_pathways, GENE_LIST)

# Dynamische Farb-Map definieren
unique_pathways = list(set(gene_pathway_assignment.values()))
num_pathways = len(unique_pathways)

# Generiere unterschiedliche Farben (bis zu 12, dann wiederholt es sich)
colors = plt.cm.get_cmap('tab10', max(num_pathways, 2)).colors
color_list = [mcolors.to_hex(c) for c in colors]  # Farben in Hex-Format

# Erstelle die dynamische Map
color_map = {'Unbekannter Pathway': 'gray'}
for i, path in enumerate(unique_pathways):
    if path != 'Unbekannter Pathway':
        # Verwende Modulo, falls mehr als 10 Pathways vorhanden sind
        color_map[path] = color_list[i % len(color_list)]

top_genes_names = list(gene_pathway_assignment.keys())
node_colors = [color_map.get(gene_pathway_assignment.get(gene, 'Unbekannter Pathway'), 'gray') for gene in
               top_genes_names]

# Netzwerk-Setup
threshold = 0.3
G = nx.Graph()
G.add_nodes_from(top_genes_names)

# Kanten hinzufügen
for i in range(len(top_genes_names)):
    for j in range(i + 1, len(top_genes_names)):
        gene_a = top_genes_names[i]
        gene_b = top_genes_names[j]
        weight = inter_correlation_matrix.iloc[i, j]

        if abs(weight) >= threshold:
            edge_color = 'red' if weight > 0 else 'blue'
            G.add_edge(gene_a, gene_b, weight=weight, color=edge_color)

# VISUALISIERUNG

plt.figure(figsize=(14, 12))
plt.title(f'Funktionales Interkorrelations-Netzwerk (Spezifische Pathways) - Threshold >|{threshold}|', fontsize=16)

pos = nx.spring_layout(G, k=0.8, seed=42)

edge_weights = [abs(G[u][v]['weight']) * 5 for u, v in G.edges()]
edge_colors = [G[u][v]['color'] for u, v in G.edges()]

# Knoten zeichnen
nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors, alpha=0.9, linewidths=2, edgecolors='black')

# Kanten und Labels zeichnen
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Legende erstellen
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=12)
    for k, v in color_map.items()
]

# Legende in 2 Spalten aufteilen, falls nötig
plt.legend(handles=legend_elements, title="Spezifischer Pathway", loc='upper right', ncol=1, frameon=True)

plt.axis('off')
plt.savefig('Plot6.png')
plt.show()


def assign_broad_pathway_categories(df_pathways, target_genes):
    # Weist Genen Pathway-Kategorien zu (höhere Hierarchie)

    if df_pathways.empty:
        return {gene: 'Unbekannter Pathway' for gene in target_genes}

    # Definition der Oberkategorien und ihrer Schlüsselwörter
    pathway_keywords = {
        'Zelluläre Signalübertragung': ['signaling pathway', 'receptor', 'transduction'],
        'Zellzyklus & DNA-Reparatur': ['cell cycle', 'DNA repair', 'mitosis', 'p53'],
        'Stoffwechsel & Biosynthese': ['metabolism', 'biosynthesis', 'synthesis of', 'amino acid', 'lipid'],
        'Transkription & Regulation': ['transcription', 'regulation of gene expression'],
        'Immunsystem & Entzündung': ['immune', 'leukocyte', 'inflammation', 'T cell']
    }

    gene_pathways = {}

    for gene_name in target_genes:
        # Sammle alle spezifischen Pathway-Namen für das Gen
        pathway_names = df_pathways[df_pathways['Gene name'] == gene_name]['Pathway name'].astype(
            str).str.lower().tolist()
        assigned_category = 'Unbekannter Pathway'

        # Finde die beste Übereinstimmung mit den breiten Keywords
        for category, keywords in pathway_keywords.items():
            if any(kw in p_name for p_name in pathway_names for kw in keywords):
                assigned_category = category
                break

        gene_pathways[gene_name] = assigned_category

    return gene_pathways

# DATENVERARBEITUNG UND NETZWERK-ERSTELLUNG

# Pathway-Daten holen
df_pathways = fetch_pathway_data(GENE_LIST)

# Spezifische Pathways zuweisen
gene_pathway_assignment = assign_broad_pathway_categories(df_pathways, GENE_LIST)

# Dynamische Farb-Map definieren
unique_pathways = list(set(gene_pathway_assignment.values()))
num_pathways = len(unique_pathways)

# Generiere unterschiedliche Farben (bis zu 12, dann wiederholt es sich)
colors = plt.cm.get_cmap('tab10', max(num_pathways, 2)).colors
color_list = [mcolors.to_hex(c) for c in colors]  # Farben in Hex-Format

# Erstelle die dynamische Map
color_map = {'Unbekannter Pathway': 'gray'}
for i, path in enumerate(unique_pathways):
    if path != 'Unbekannter Pathway':
        # Verwende Modulo, falls mehr als 10 Pathways vorhanden sind
        color_map[path] = color_list[i % len(color_list)]

top_genes_names = list(gene_pathway_assignment.keys())
node_colors = [color_map.get(gene_pathway_assignment.get(gene, 'Unbekannter Pathway'), 'gray') for gene in
               top_genes_names]

# Netzwerk-Setup
threshold = 0.3
G = nx.Graph()
G.add_nodes_from(top_genes_names)

# Kanten hinzufügen
for i in range(len(top_genes_names)):
    for j in range(i + 1, len(top_genes_names)):
        gene_a = top_genes_names[i]
        gene_b = top_genes_names[j]
        weight = inter_correlation_matrix.iloc[i, j]

        if abs(weight) >= threshold:
            edge_color = 'red' if weight > 0 else 'blue'
            G.add_edge(gene_a, gene_b, weight=weight, color=edge_color)

# VISUALISIERUNG

plt.figure(figsize=(14, 12))
plt.title(f'Funktionales Interkorrelations-Netzwerk (Spezifische Pathways) - Threshold >|{threshold}|', fontsize=16)

pos = nx.spring_layout(G, k=0.8, seed=42)

edge_weights = [abs(G[u][v]['weight']) * 5 for u, v in G.edges()]
edge_colors = [G[u][v]['color'] for u, v in G.edges()]

# Knoten zeichnen
nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors, alpha=0.9, linewidths=2, edgecolors='black')

# Kanten und Labels zeichnen
nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_weights, alpha=0.7)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

# Legende erstellen
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label=k, markerfacecolor=v, markersize=12)
    for k, v in color_map.items()
]

# Legende in 2 Spalten aufteilen, falls nötig
plt.legend(handles=legend_elements, title="Spezifischer Pathway", loc='upper right', ncol=1, frameon=True)

plt.axis('off')
plt.savefig('Plot7.png')
plt.show()