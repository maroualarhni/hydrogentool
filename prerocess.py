# -*- coding: utf-8 -*-
"""
preprocessing/preprocess.py
Prétraitement scientifique — H2 Morocco
Basé sur : Pr. ESSWIDI AYOUB — Analyse des Données TDI 2024-2025
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = r"C:\Users\HP 840 G8\Downloads\H2Morocco222_Outputs"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DOUBLONS — Détection uniquement (pas de suppression)
# ─────────────────────────────────────────────────────────────────────────────
def detecter_doublons(df, nom_table=""):

    nb_total = df.shape[0]

    doublons = df[df.duplicated()]
    nb_doublons = doublons.shape[0]

    if nb_doublons == 0:
        print(f"  [Doublons] {nom_table} : aucun ✓")

    else:
        print(f"  [Doublons] {nom_table} : {nb_doublons} détectés (conservés)")
        print(f"               {nb_total-nb_doublons} lignes uniques")
        print(doublons.head())

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. VALEURS MANQUANTES
# ─────────────────────────────────────────────────────────────────────────────
def gerer_valeurs_manquantes(df, strategie='median', nom_table=""):

    cols_num = df.select_dtypes(include='number').columns
    nb_total = df[cols_num].isna().sum().sum()

    if nb_total == 0:
        print(f"  [Manquants] {nom_table} : aucune valeur manquante ✓")
        return df

    cols_vides = [c for c in cols_num if df[c].isna().all()]
    cols_valides = [c for c in cols_num if not df[c].isna().all()]

    if cols_vides:
        print(f"  [Manquants] {nom_table} : {len(cols_vides)} colonnes vides supprimées")
        df = df.drop(columns=cols_vides)

    if cols_valides:

        nb_valides = df[cols_valides].isna().sum().sum()

        if nb_valides > 0:
            imputer = SimpleImputer(strategy=strategie)
            df[cols_valides] = imputer.fit_transform(df[cols_valides])

            print(f"  [Manquants] {nom_table} : {nb_valides} valeurs remplacées par {strategie}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONVERSION DES UNITÉS
# ─────────────────────────────────────────────────────────────────────────────
def convertir_unites(df, nom_table=""):

    conversions = 0

    if "distance_miles" in df.columns:
        df["distance_km"] = (df["distance_miles"] * 1.60934).round(2)
        conversions += 1

    if "energie_MWh" in df.columns:
        df["energie_kWh"] = df["energie_MWh"] * 1000
        conversions += 1

    if "cout_USD_kg" in df.columns:
        df["cout_EUR_kg"] = df["cout_USD_kg"] * 0.92
        conversions += 1

    if conversions > 0:
        print(f"  [Unités] {nom_table} : {conversions} conversions appliquées")
    else:
        print(f"  [Unités] {nom_table} : aucune conversion nécessaire")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 4. VALEURS ABERRANTES (IQR)
# ─────────────────────────────────────────────────────────────────────────────
def detecter_aberrantes_IQR(df, colonnes, action='clip', nom_table=""):

    for col in colonnes:

        if col not in df.columns:
            continue

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        borne_inf = Q1 - 1.5 * IQR
        borne_sup = Q3 + 1.5 * IQR

        aberrantes = df[(df[col] < borne_inf) | (df[col] > borne_sup)]

        if aberrantes.empty:
            continue

        print(f"  [Aberrantes] {nom_table} — '{col}' : {len(aberrantes)} valeurs corrigées")

        if action == 'clip':
            df[col] = df[col].clip(borne_inf, borne_sup)

        elif action == 'remove':
            df = df[(df[col] >= borne_inf) & (df[col] <= borne_sup)]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 5. DONNÉES INCOHÉRENTES
# ─────────────────────────────────────────────────────────────────────────────
def corriger_incoherences(df, nom_table=""):

    corrections = {
        'Jorf_Lasfar': 'Jorf Lasfar',
        'jorf lasfar': 'Jorf Lasfar',
        'DAKHLA': 'Dakhla',
        'dakhla': 'Dakhla',
        'Tanger ': 'Tanger',
        'faible': 'Faible',
        'Très faible': 'Très Faible',
        'tres faible': 'Très Faible',
        'bonne': 'Bonne',
        'BONNE': 'Bonne',
        'excellente': 'Excellente',
        'EXCELLENTE': 'Excellente',
    }

    cols_texte = df.select_dtypes(include='object').columns

    nb = 0

    for col in cols_texte:

        for incorrect, correct in corrections.items():

            mask = df[col].astype(str).str.strip() == incorrect

            if mask.any():

                df.loc[mask, col] = correct
                nb += mask.sum()

    if nb > 0:
        print(f"  [Incohérences] {nom_table} : {nb} corrections")
    else:
        print(f"  [Incohérences] {nom_table} : aucune ✓")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 6. ENCODAGE DES VARIABLES CATÉGORIELLES
# ─────────────────────────────────────────────────────────────────────────────
def encoder_categorielles(df, nom_table=""):

    mapping_ordinal = {
        'disponibilite_eau': {'Très Faible': 1, 'Faible': 2, 'Bonne': 3},
        'connexion_reseau_elec': {'Bonne': 1, 'Excellente': 2},
    }

    for col, mapping in mapping_ordinal.items():

        if col not in df.columns:
            continue

        df[col + '_encoded'] = df[col].map(mapping)

        print(f"  [Encodage Ordinal] {nom_table} — '{col}'")

    cols_label = ['technologie', 'type', 'mode_optimal', 'vecteur_H2']

    for col in cols_label:

        if col not in df.columns:
            continue

        df[col + '_encoded'] = LabelEncoder().fit_transform(df[col].astype(str))

        print(f"  [Encodage Label] {nom_table} — '{col}'")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────
def normaliser(df, colonnes, methode='minmax', nom_table=""):

    cols = [c for c in colonnes if c in df.columns]

    if not cols:
        return df

    scalers = {
        'minmax': (MinMaxScaler(), '_norm'),
        'standard': (StandardScaler(), '_std'),
        'robust': (RobustScaler(), '_robust'),
    }

    scaler, suffix = scalers.get(methode, (MinMaxScaler(), '_norm'))

    scaled = pd.DataFrame(
        scaler.fit_transform(df[cols]),
        columns=[c + suffix for c in cols],
        index=df.index
    )

    df = pd.concat([df, scaled], axis=1)

    print(f"  [Normalisation {methode}] {nom_table} : {len(cols)} colonnes")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLET
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_database(df, nom_table="", normalisation=False):

    print(f"\n{'═'*60}")
    print(f"  PRÉTRAITEMENT : {nom_table} | {df.shape}")
    print(f"{'═'*60}")

    df = detecter_doublons(df, nom_table)

    df = gerer_valeurs_manquantes(df, 'median', nom_table)

    df = corriger_incoherences(df, nom_table)

    df = convertir_unites(df, nom_table)

    df = detecter_aberrantes_IQR(df, [
        'LCOE_solaire_USD_kWh',
        'LCOE_eolien_USD_kWh',
        'CF_solaire_PV_pct',
        'CF_eolien_pct',
        'CF_hybride_pct',
        'valeur_min',
        'valeur_mode',
        'valeur_max',
        'cout_transport_USD_kg_min',
        'cout_transport_USD_kg_max',
    ], action='clip', nom_table=nom_table)

    df = encoder_categorielles(df, nom_table)

    if normalisation:

        df = normaliser(df, [
            'CF_hybride_pct',
            'score_potentiel_H2',
            'LCOE_hybride_USD_kWh',
            'distance_port_km',
        ], methode='minmax', nom_table=nom_table)

    print(f"\n  ✅ {nom_table} terminée → {df.shape}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION SUR TOUTES LES TABLES
# ─────────────────────────────────────────────────────────────────────────────
def run_all_preprocessing():

    print("\n" + "="*60)
    print("  PRÉTRAITEMENT COMPLET — H2 MOROCCO")
    print("="*60)

    output_preproc = os.path.join(OUTPUT_DIR, "csv", "preprocessed")
    os.makedirs(output_preproc, exist_ok=True)

    tables = [

        ("T1_ressources_energetiques.csv", "T1_Ressources", True),
        ("T2_technologies_production.csv", "T2_Production", False),
        ("T3_technologies_stockage.csv", "T3_Stockage", False),
        ("T4_corridors_transport.csv", "T4_Transport", False),
        ("T5_parametres_economiques.csv", "T5_Economique", False),
        ("T6a_demande_nationale.csv", "T6a_Demande", False),
        ("T6b_benchmark_competiteurs.csv", "T6b_Benchmark", False),
        ("T7a_emissions_CO2.csv", "T7a_Emissions", False),
        ("T8_projets_reference_maroc.csv", "T8_Projets", False),
        ("T9_scenarios_temporels.csv", "T9_Scenarios", False),

    ]

    resultats = {}

    for fichier, nom, norm in tables:

        chemin = os.path.join(OUTPUT_DIR, "csv", fichier)

        if not os.path.exists(chemin):

            print(f"\n⚠️ {nom} : fichier introuvable")

            continue

        df = pd.read_csv(chemin, encoding='utf-8-sig')

        print(f"\n📂 {nom} chargée : {df.shape}")

        df_clean = preprocess_database(df, nom, norm)

        chemin_out = os.path.join(
            output_preproc,
            fichier.replace(".csv", "_clean.csv")
        )

        df_clean.to_csv(chemin_out, index=False, encoding='utf-8-sig')

        print(f"💾 Sauvegardée : {chemin_out}")

        resultats[nom] = df_clean

    print("\n" + "="*60)
    print("PRÉTRAITEMENT TERMINÉ")
    print("="*60)

    return resultats


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION DIRECTE
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    run_all_preprocessing()