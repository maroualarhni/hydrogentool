# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:41:30 2026

@author: HP 840 G8
"""

# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  H2 MOROCCO — MODÈLE DE CALCUL PRODUCTION v4                               ║
║                                                                              ║
║  Sorties du modèle :                                                         ║
║    1. Production H2 physique  [kg/h, t/an, GWh_LHV/an]                     ║
║    2. LCOE [€/kWh, €/MWh]    ← seul indicateur économique                  ║
║                                                                              ║
║  Ce qui N'est PAS calculé ici : LCOH, LCOS, LCOT                           ║
║                                                                              ║
║  Cost Function Rozzi appliquée sur les CAPEX ENR :                          ║
║    • CAPEX_PV(S,t)  = cost_function_eq4(S_sol)  × learning_curve_eq1_3(t)  ║
║    • CAPEX_EOL(S,t) = cost_function_eq4(S_eol)  × learning_curve_eq1_3(t)  ║
║                                                                              ║
║  Variables de décision (intervalles BDD T1–T6) :                            ║
║    • CAPEX_PV [min,mode,max]  → T3                                          ║
║    • CAPEX_EOL [min,mode,max] → T3                                          ║
║    • WACC [min,mode,max]      → T5                                          ║
║    • efficacité EL            → T2 (pour production H2 uniquement)          ║
║    • CF hybride               → T1                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES PHYSIQUES IMMUABLES
# ══════════════════════════════════════════════════════════════════════════════
LHV_H2_kWh_kg = 33.33
HHV_H2_kWh_kg = 39.41
HOURS_YEAR    = 8760

TURBINE = {
    "v_cut_in" : 3.0,
    "v_rated"  : 12.0,
    "v_cut_out": 25.0,
    "loss_coef": 0.05,
}

DEGRADATION_EL = {"PEM": 0.015, "AEL": 0.010, "SOEC": 0.020}


# ══════════════════════════════════════════════════════════════════════════════
# VARIABLES DE DÉCISION
# Paramètres lus depuis les intervalles [min, mode, max] de la BDD
# qui varient selon le scénario et affectent la production et le LCOE
# ══════════════════════════════════════════════════════════════════════════════
VARIABLES_DECISION = {

    # ── ENR (source T3) — entrent dans LCOE via cost_function + learning_curve
    "capex_pv": {
        "description"  : "CAPEX solaire PV [USD/kW]",
        "unite"        : "USD/kW",
        "source_table" : "T3",  "technologie_t3": "PV_solaire",
        "col_min"      : "capex_min",
        "col_mode"     : "capex_USD_kW",
        "col_max"      : "capex_max",
        "sens"         : "normal",   # faible = optimiste
        "rozzi_eq"     : "eq.(4) effet d'échelle + eq.(1-3) learning curve",
        "impact_lcoe"  : "CAPEX_PV_ann = cost_function(S_sol, t) × CRF(WACC, LT_PV)",
    },
    "capex_eol": {
        "description"  : "CAPEX éolien [USD/kW]",
        "unite"        : "USD/kW",
        "source_table" : "T3",  "technologie_t3": "Eolien",
        "col_min"      : "capex_min",
        "col_mode"     : "capex_USD_kW",
        "col_max"      : "capex_max",
        "sens"         : "normal",
        "rozzi_eq"     : "eq.(4) effet d'échelle + eq.(1-3) learning curve",
        "impact_lcoe"  : "CAPEX_EOL_ann = cost_function(S_eol, t) × CRF(WACC, LT_EOL)",
    },
    "opex_pv": {
        "description"  : "OPEX solaire [USD/kW/an]",
        "unite"        : "USD/kW/an",
        "source_table" : "T3",  "technologie_t3": "PV_solaire",
        "col_mode"     : "opex_USD_kW_an",
        "variation_pct": 0.20,
        "sens"         : "normal",
        "impact_lcoe"  : "OPEX_PV_ann = opex_pv × S_sol",
    },
    "opex_eol": {
        "description"  : "OPEX éolien [USD/kW/an]",
        "unite"        : "USD/kW/an",
        "source_table" : "T3",  "technologie_t3": "Eolien",
        "col_mode"     : "opex_USD_kW_an",
        "variation_pct": 0.20,
        "sens"         : "normal",
        "impact_lcoe"  : "OPEX_EOL_ann = opex_eol × S_eol",
    },

    # ── FINANCIER (source T5)
    "wacc": {
        "description"  : "WACC [%]",
        "unite"        : "%",
        "source_table" : "T5",
        "col_min"      : "WACC_min_pct",
        "col_mode"     : "WACC_pct",
        "col_max"      : "WACC_max_pct",
        "sens"         : "normal",
        "impact_lcoe"  : "CRF = r(1+r)^n / ((1+r)^n-1)",
    },

    # ── PRODUCTION H2 (sources T1, T2) — n'entrent pas dans LCOE
    "efficacite_el": {
        "description"  : "Consommation spécifique EL [kWh/kgH2]",
        "unite"        : "kWh/kgH2",
        "source_table" : "T2",
        "col_min"      : "eff_min",
        "col_mode"     : "efficacite_kWh_kg",
        "col_max"      : "eff_max",
        "sens"         : "normal",
        "impact_lcoe"  : "Aucun — variable de production H2 uniquement",
    },
    "cf_hybride": {
        "description"  : "CF hybride ENR [%]",
        "unite"        : "%",
        "source_table" : "T1",
        "col_mode"     : "CF_hybride_pct",
        "variation_pct": 0.10,
        "sens"         : "inverse",   # haut = optimiste
        "impact_lcoe"  : "Indirect : énergie produite = CF × S_total × 8760",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# COST FUNCTION ROZZI — appliquée aux CAPEX des sources ENR (PV + Éolien)
# ══════════════════════════════════════════════════════════════════════════════

# Coefficients Rozzi Table 4 pour les sources ENR
# Note : Rozzi donne ces coefficients pour les électrolyseurs.
# Pour PV et éolien, on utilise les mêmes paramètres structurels (exp=0.85)
# car les données ENR IRENA suivent le même modèle d'échelle.
ROZZI_ENR_COEFFS = {
    #              C_ref    S_ref_kW   exp_echelle
    "PV_solaire": {"C_ref": 550.0,  "S_ref_kW": 1000.0,  "exp": 0.85},
    "Eolien"    : {"C_ref": 1200.0, "S_ref_kW": 1000.0,  "exp": 0.90},
}


def cost_function_rozzi_eq4(S_kW: float, tech_enr: str,
                             capex_bdd: float = None) -> float:
    """
    [Rozzi et al. 2024 — Eq.(4)] Effet d'échelle sur le CAPEX d'une source ENR.

    CAPEX(S) = C_ref × S_ref × (S / S_ref)^exp / S     [USD/kW]

    Économies d'échelle : plus S est grand, plus le coût unitaire [USD/kW] est faible.

    Exemple PV (C_ref=550, S_ref=1 MW, exp=0.85) :
      S =   1 MW  →  550 × 1000 × (1/1)^0.85   / 1000  =  550 USD/kW
      S =  10 MW  →  550 × 1000 × (10)^0.85    / 10000 =  396 USD/kW  (-28%)
      S = 100 MW  →  550 × 1000 × (100)^0.85   /100000 =  285 USD/kW  (-48%)
      S = 700 MW  →  550 × 1000 × (700)^0.85   /700000 =  178 USD/kW  (-68%)

    Paramètres :
        S_kW      : capacité installée [kW]  ← variable de conception
        tech_enr  : "PV_solaire" | "Eolien"
        capex_bdd : si fourni, remplace C_ref (utilise la valeur de la BDD T3)
    """
    c = ROZZI_ENR_COEFFS.get(tech_enr)
    if c is None or S_kW <= 0:
        return capex_bdd or 550.0

    C_ref = capex_bdd if capex_bdd is not None else c["C_ref"]
    S_ref = c["S_ref_kW"]
    exp   = c["exp"]

    return (C_ref * S_ref * (S_kW / S_ref) ** exp) / S_kW


def learning_curve_rozzi_eq1_3(capex_ref: float, lr_pct_an: float,
                                annee: int, annee_ref: int = 2024) -> float:
    """
    [Rozzi et al. 2024 — Eq.(1-3)] Actualisation temporelle du CAPEX.

    CAPEX(t) = CAPEX_ref × (1 + lr)^(t - t_ref)

    Exemple PV (lr = -8%/an, CAPEX_2024 = 550 USD/kW) :
      2024 → 550 USD/kW
      2030 → 550 × (0.92)^6  = 328 USD/kW  (-40%)
      2040 → 550 × (0.92)^16 = 147 USD/kW  (-73%)
      2050 → 550 × (0.92)^26 =  66 USD/kW  (-88%)
    """
    dt = max(0, annee - annee_ref)
    return capex_ref * (1.0 + lr_pct_an / 100.0) ** dt


def capex_enr_actualise(tech_enr: str, S_kW: float,
                        capex_bdd_USD_kW: float, lr_pct_an: float,
                        annee: int, usd_eur: float = 0.92) -> dict:
    """
    Combine cost_function_rozzi_eq4 + learning_curve_rozzi_eq1_3
    pour obtenir le CAPEX ENR [€/kW] actualisé en taille ET en temps.

    Chaîne de calcul :
        CAPEX_BDD (T3) [USD/kW]
            │
            ├─► [Rozzi eq.4]   effet d'échelle sur S_kW installés
            │       CAPEX_ech = cost_function(S, tech_enr, capex_bdd)
            │
            ├─► [Rozzi eq.1-3] facteur learning curve temporel
            │       facteur_lc = (1 + lr)^(annee - 2024)
            │
            └─► CAPEX_final = CAPEX_ech × facteur_lc × usd_eur   [€/kW]

    Paramètres :
        tech_enr       : "PV_solaire" | "Eolien"
        S_kW           : capacité installée [kW]
        capex_bdd_USD_kW: CAPEX brut depuis T3 pour ce scénario [USD/kW]
        lr_pct_an      : taux learning depuis T3 [%/an]
        annee          : année de calcul
        usd_eur        : taux de change (depuis T5)

    Retourne :
        dict {
          "CAPEX_BDD_USD_kW"   : valeur brute T3,
          "CAPEX_ech_USD_kW"   : après effet d'échelle eq.(4),
          "facteur_lc"         : réduction temporelle eq.(1-3),
          "CAPEX_final_USD_kW" : CAPEX_ech × facteur_lc,
          "CAPEX_final_EUR_kW" : converti en EUR,
          "CAPEX_total_EUR"    : CAPEX_final_EUR × S_kW,
        }
    """
    # Étape 1 — Effet d'échelle [eq.4]
    capex_ech = cost_function_rozzi_eq4(S_kW, tech_enr, capex_bdd_USD_kW)

    # Étape 2 — Facteur learning curve [eq.1-3]
    # On calcule le facteur de réduction depuis la valeur de référence BDD
    capex_lc_val = learning_curve_rozzi_eq1_3(capex_bdd_USD_kW, lr_pct_an, annee)
    facteur_lc   = capex_lc_val / capex_bdd_USD_kW if capex_bdd_USD_kW > 0 else 1.0

    # Étape 3 — CAPEX final
    capex_final_usd = capex_ech * facteur_lc
    capex_final_eur = capex_final_usd * usd_eur

    return {
        "CAPEX_BDD_USD_kW"   : round(capex_bdd_USD_kW, 1),
        "CAPEX_ech_USD_kW"   : round(capex_ech, 1),
        "facteur_lc"         : round(facteur_lc, 4),
        "CAPEX_final_USD_kW" : round(capex_final_usd, 1),
        "CAPEX_final_EUR_kW" : round(capex_final_eur, 1),
        "CAPEX_total_EUR"    : round(capex_final_eur * S_kW, 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CRF
# ══════════════════════════════════════════════════════════════════════════════
def crf(r: float, n: float) -> float:
    """CRF = r(1+r)^n / ((1+r)^n - 1)"""
    if r <= 1e-9:
        return 1.0 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
class ProductionModel:
    """
    Modèle de calcul :
      - Production H2 physique (dispatch ENR → EL → H2)
      - LCOE des sources ENR avec Cost Function Rozzi eq.(4) + Learning Curve eq.(1-3)
      - Intervalles [min, mode, max] propagés depuis T1–T6 pour 3 scénarios
    """

    SCENARIOS = ["optimiste", "central", "pessimiste"]

    def __init__(self, output_dir: str, csv_dir: str = None):
        self.output_dir  = output_dir
        self.csv_dir     = csv_dir or os.path.join(output_dir, "csv")
        self.dir_results = os.path.join(output_dir, "csv", "production_results")
        self.dir_figs    = os.path.join(output_dir, "figures", "production")
        for d in [self.dir_results, self.dir_figs]:
            os.makedirs(d, exist_ok=True)
        self.T1 = self.T2 = self.T3 = self.T5 = self.T6 = None

    # ─────────────────────────────────────────────────────────────────────────
    # A. CHARGEMENT BDD
    # ─────────────────────────────────────────────────────────────────────────
    def load_database(self):
        print("\n" + "═"*65)
        print("  CHARGEMENT BASE DE DONNÉES")
        print("═"*65)
        self.T1 = self._load("T1",
            ["T1_avec_source.csv", "T1_ressources_energetiques.csv"],
            self._default_T1())
        self.T2 = self._load("T2",
            ["T2_avec_source.csv", "T2_technologies_production.csv"],
            self._default_T2())
        self.T3 = self._load("T3",
            ["T3_avec_source.csv", "T3_technologies_ener.csv"],
            self._default_T3())
        self.T5 = self._load("T5",
            ["T5_avec_source.csv", "T5_parametres_economiques.csv"],
            self._default_T5())
        self.T6 = self._load("T6",
            ["T6_avec_source.csv", "T6_learning_curves.csv"],
            self._default_T6())
        print("  ✅ BDD chargée\n")
        return self

    def _load(self, nom, fichiers, default):
        for f in fichiers:
            p = os.path.join(self.csv_dir, f)
            if os.path.exists(p):
                df = pd.read_csv(p, encoding="utf-8-sig")
                print(f"  ✓ {nom} ← {f}  ({len(df)} lignes)")
                return df
        print(f"  ⚠ {nom} non trouvée → valeurs par défaut")
        return default

    # ─────────────────────────────────────────────────────────────────────────
    # B. HELPERS BDD
    # ─────────────────────────────────────────────────────────────────────────
    def _t3(self, tech: str) -> pd.Series:
        m = self.T3["technologie"] == tech
        if not m.any():
            raise ValueError(f"'{tech}' absent de T3")
        return self.T3[m].iloc[0]

    def _t2(self, tech: str) -> pd.Series:
        m = self.T2["technologie"] == tech
        if not m.any():
            raise ValueError(f"'{tech}' absent de T2")
        return self.T2[m].iloc[0]

    def _t1(self, region: str) -> pd.Series:
        m = self.T1["region"] == region
        if not m.any():
            raise ValueError(f"'{region}' absent de T1. "
                             f"Disponibles : {self.T1['region'].tolist()}")
        return self.T1[m].iloc[0]

    def _t5(self, param: str, fallback: float = None) -> float:
        m = self.T5["parametre"] == param
        return float(self.T5[m].iloc[0]["valeur"]) if m.any() else fallback

    def _t6_lr(self, tech_enr: str) -> float:
        """Learning rate ENR depuis T3 (stocké dans T3 pour PV et éolien)."""
        row = self._t3(tech_enr)
        return float(row["learning_rate_pct_an"])

    def _iv(self, row, col_min, col_mode, col_max,
            sens="normal", var_pct=None) -> dict:
        """Extrait l'intervalle [optimiste, central, pessimiste] depuis une ligne."""
        def _v(c):
            return float(row[c]) if c in row.index and pd.notna(row[c]) else float(row[col_mode])

        vmin, vmode, vmax = _v(col_min), _v(col_mode), _v(col_max)

        # Si pas d'intervalle distinct dans la BDD → créer via ±var_pct
        if var_pct and abs(vmin - vmax) < 1e-9:
            vmin = vmode * (1 - var_pct)
            vmax = vmode * (1 + var_pct)

        if sens == "normal":
            return {"optimiste": vmin, "central": vmode, "pessimiste": vmax}
        else:
            return {"optimiste": vmax, "central": vmode, "pessimiste": vmin}

    # ─────────────────────────────────────────────────────────────────────────
    # C. CONSTRUCTION DES VARIABLES DE DÉCISION PAR SCÉNARIO
    # ─────────────────────────────────────────────────────────────────────────
    def get_variables(self, tech_el: str, region_nom: str,
                      sol_cap_kW: float, eol_cap_kW: float,
                      annee: int, scenario: str = "central") -> dict:
        """
        Construit toutes les variables de décision pour un scénario donné.

        Pour chaque variable définie dans VARIABLES_DECISION :
          → extrait l'intervalle depuis la table BDD correspondante
          → retourne la valeur du scénario demandé

        Les CAPEX ENR sont immédiatement actualisés avec :
          cost_function_rozzi_eq4(S)  ×  learning_curve_rozzi_eq1_3(t)

        Paramètres :
            tech_el    : technologie électrolyseur
            region_nom : région Maroc
            sol_cap_kW : capacité solaire installée [kW]  ← entrée cost_function
            eol_cap_kW : capacité éolienne installée [kW] ← entrée cost_function
            annee      : année de calcul
            scenario   : "optimiste" | "central" | "pessimiste"
        """
        usd_eur = self._t5("taux_USD_EUR", 0.92)
        row_pv  = self._t3("PV_solaire")
        row_eol = self._t3("Eolien")
        row_t2  = self._t2(tech_el)
        row_t1  = self._t1(region_nom)

        # ── WACC (T5)
        wacc_iv = {
            "optimiste" : self._t5("WACC_min_pct", 6.0)  / 100,
            "central"   : self._t5("WACC_pct",     8.0)  / 100,
            "pessimiste": self._t5("WACC_max_pct", 12.0) / 100,
        }

        # ── CAPEX PV (T3) — intervalle [capex_min, capex_USD_kW, capex_max]
        capex_pv_iv = self._iv(row_pv, "capex_min", "capex_USD_kW", "capex_max",
                                sens="normal")
        lr_pv  = float(row_pv["learning_rate_pct_an"])
        lt_pv  = int(row_pv["duree_vie_ans"])

        # ── CAPEX éolien (T3)
        capex_eol_iv = self._iv(row_eol, "capex_min", "capex_USD_kW", "capex_max",
                                 sens="normal")
        lr_eol = float(row_eol["learning_rate_pct_an"])
        lt_eol = int(row_eol["duree_vie_ans"])

        # ── OPEX ENR (T3) — variation ±20%
        opex_pv_iv  = {"optimiste": float(row_pv["opex_USD_kW_an"])  * 0.80,
                       "central"  : float(row_pv["opex_USD_kW_an"]),
                       "pessimiste": float(row_pv["opex_USD_kW_an"]) * 1.20}
        opex_eol_iv = {"optimiste": float(row_eol["opex_USD_kW_an"]) * 0.80,
                       "central"  : float(row_eol["opex_USD_kW_an"]),
                       "pessimiste": float(row_eol["opex_USD_kW_an"])* 1.20}

        # ── Efficacité EL (T2) — pour production H2 uniquement
        eff_iv = self._iv(row_t2, "eff_min", "efficacite_kWh_kg", "eff_max",
                          sens="normal")
        lt_el  = float(row_t2["duree_vie_ans"])

        # ── CF hybride (T1) — variation ±10% incertitude météo
        cf_h = float(row_t1["CF_hybride_pct"]) / 100
        cf_iv = {"optimiste": cf_h*1.10, "central": cf_h, "pessimiste": cf_h*0.90}

        # ── PPA (T1) — variation ±20%
        ppa  = float(row_t1["PPA_USD_kWh"])
        ppa_iv = {"optimiste": ppa*0.80, "central": ppa, "pessimiste": ppa*1.20}

        # ────────────────────────────────────────────────────────────────────
        # APPLICATION COST FUNCTION + LEARNING CURVE SUR CAPEX ENR
        # Pour le scénario demandé, on actualise les deux sources ENR
        # ────────────────────────────────────────────────────────────────────
        pv_actu = capex_enr_actualise(
            tech_enr         = "PV_solaire",
            S_kW             = sol_cap_kW,               # taille → effet d'échelle
            capex_bdd_USD_kW = capex_pv_iv[scenario],    # valeur BDD du scénario
            lr_pct_an        = lr_pv,                    # depuis T3
            annee            = annee,
            usd_eur          = usd_eur,
        )

        eol_actu = capex_enr_actualise(
            tech_enr         = "Eolien",
            S_kW             = eol_cap_kW,
            capex_bdd_USD_kW = capex_eol_iv[scenario],
            lr_pct_an        = lr_eol,
            annee            = annee,
            usd_eur          = usd_eur,
        )

        return {
            # Identification
            "scenario"              : scenario,
            "tech_el"               : tech_el,
            "region"                : region_nom,
            "annee"                 : annee,
            "sol_cap_kW"            : sol_cap_kW,
            "eol_cap_kW"            : eol_cap_kW,
            "usd_eur"               : usd_eur,

            # ── WACC
            "wacc"                  : wacc_iv[scenario],
            "wacc_iv"               : wacc_iv,

            # ── CAPEX PV actualisé [eq.4 + eq.1-3]
            "PV_CAPEX_BDD_USD_kW"   : pv_actu["CAPEX_BDD_USD_kW"],
            "PV_CAPEX_ech_USD_kW"   : pv_actu["CAPEX_ech_USD_kW"],
            "PV_facteur_lc"         : pv_actu["facteur_lc"],
            "PV_CAPEX_fin_EUR_kW"   : pv_actu["CAPEX_final_EUR_kW"],
            "PV_CAPEX_total_EUR"    : pv_actu["CAPEX_total_EUR"],
            "lt_pv"                 : lt_pv,
            "lr_pv"                 : lr_pv,

            # ── CAPEX EOL actualisé [eq.4 + eq.1-3]
            "EOL_CAPEX_BDD_USD_kW"  : eol_actu["CAPEX_BDD_USD_kW"],
            "EOL_CAPEX_ech_USD_kW"  : eol_actu["CAPEX_ech_USD_kW"],
            "EOL_facteur_lc"        : eol_actu["facteur_lc"],
            "EOL_CAPEX_fin_EUR_kW"  : eol_actu["CAPEX_final_EUR_kW"],
            "EOL_CAPEX_total_EUR"   : eol_actu["CAPEX_total_EUR"],
            "lt_eol"                : lt_eol,
            "lr_eol"                : lr_eol,

            # ── OPEX ENR [EUR/kW/an]
            "OPEX_PV_EUR_kW_an"     : round(opex_pv_iv[scenario]  * usd_eur, 2),
            "OPEX_EOL_EUR_kW_an"    : round(opex_eol_iv[scenario] * usd_eur, 2),

            # ── Efficacité EL (T2) — production H2 uniquement
            "eff_kWh_kg"            : eff_iv[scenario],
            "eff_iv"                : eff_iv,
            "lt_el"                 : lt_el,

            # ── Ressources (T1)
            "cf_hybride"            : cf_iv[scenario],
            "cf_iv"                 : cf_iv,
            "ppa_USD_kWh"           : ppa_iv[scenario],
            "ppa_iv"                : ppa_iv,
            "w_eolien"              : float(row_t1["w_eolien"]),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # D. PROFILS ENR 8760H
    # ─────────────────────────────────────────────────────────────────────────
    def _wind_cf(self, v: np.ndarray) -> np.ndarray:
        v  = np.asarray(v, dtype=float)
        cf = np.zeros_like(v)
        vc, vr, vo = TURBINE["v_cut_in"], TURBINE["v_rated"], TURBINE["v_cut_out"]
        m1 = (v >= vc) & (v < vr)
        m2 = (v >= vr) & (v < vo)
        cf[m1] = ((v[m1] - vc) / (vr - vc)) ** 3
        cf[m2] = 1.0
        return cf * (1 - TURBINE["loss_coef"])

    def generate_enr_8760(self, region_nom: str,
                          total_enr_kW: float, seed: int = 42) -> dict:
        """Génère profils solaire + éolien 8760h depuis T1."""
        np.random.seed(seed)
        row = self._t1(region_nom)
        h   = np.arange(HOURS_YEAR)
        hj, ja = h % 24, h // 24

        cf_sol = row["CF_solaire_PV_pct"] / 100
        cf_eol = row["CF_eolien_pct"]     / 100
        w_eol  = row["w_eolien"]
        sol_cap = total_enr_kW * (1 - w_eol)
        eol_cap = total_enr_kW * w_eol

        # Solaire
        sol = (np.maximum(0, np.sin(np.pi*(hj-6)/12))
               * (1 + 0.3*np.cos(2*np.pi*(ja-172)/365))
               * np.random.beta(2, 2, HOURS_YEAR))
        if sol.mean() > 1e-9:
            sol = np.clip(sol * cf_sol / sol.mean(), 0, 1)

        # Éolien
        v_m  = np.clip(TURBINE["v_rated"]*(cf_eol**(1/3))*1.2,
                       TURBINE["v_cut_in"]+0.5, TURBINE["v_rated"]*1.5)
        v_ms = np.clip(
            v_m/0.8862 * np.random.weibull(2.0, HOURS_YEAR)
            * (1+0.20*np.sin(2*np.pi*(hj-6)/24))
            * (1+0.12*np.cos(2*np.pi*(ja-60)/365)),
            0, TURBINE["v_cut_out"])
        eol = self._wind_cf(v_ms)
        if eol.mean() > 1e-9:
            eol = np.clip(eol * cf_eol / eol.mean(), 0, 1)

        return {
            "solar_kW": sol * sol_cap,
            "wind_kW" : eol * eol_cap,
            "total_kW": sol * sol_cap + eol * eol_cap,
            "sol_cap" : sol_cap,
            "eol_cap" : eol_cap,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # E. DISPATCH ENR → H2 (production physique)
    # ─────────────────────────────────────────────────────────────────────────
    def dispatch_h2(self, total_kW: np.ndarray, v: dict,
                    el_cap_kW: float,
                    el_min_load: float = 0.10,
                    annee_install: int = 2024) -> dict:
        """
        Dispatch horaire : ENR → Électrolyseur → H2

        Variables utilisées :
          v["eff_kWh_kg"] : efficacité EL initiale (T2, variable de décision)
          v["tech_el"]    : pour taux de dégradation
          v["annee"]      : âge de l'électrolyseur

        H2 [kg/h] = P_EL [kW] / eff_dégradée [kWh/kgH2]
        """
        seuil = el_min_load * el_cap_kW
        ramp  = 0.20 * el_cap_kW
        el_kW = np.zeros(HOURS_YEAR)

        for t in range(HOURS_YEAR):
            tgt = min(total_kW[t], el_cap_kW) if total_kW[t] >= seuil else 0.0
            if t > 0:
                tgt = el_kW[t-1] + np.clip(tgt - el_kW[t-1], -ramp, ramp)
            el_kW[t] = max(0.0, tgt)

        age     = max(0, v["annee"] - annee_install)
        eff_deg = v["eff_kWh_kg"] * (1 + DEGRADATION_EL.get(v["tech_el"], 0.015)) ** age
        h2_kg_h = el_kW / eff_deg
        curt_kW = np.maximum(total_kW - el_cap_kW, 0)

        return {
            "el_kW"       : el_kW,
            "h2_kg_h"     : h2_kg_h,
            "curt_kW"     : curt_kW,
            "H2_ann_kg"   : h2_kg_h.sum(),
            "H2_ann_t"    : h2_kg_h.sum() / 1000,
            "H2_ann_GWh"  : h2_kg_h.sum() * LHV_H2_kWh_kg / 1e6,
            "CF_EL_pct"   : el_kW.mean() / el_cap_kW * 100,
            "heures_ON"   : int((el_kW > 0).sum()),
            "demarrages"  : int(np.diff((el_kW>0).astype(int)).clip(0).sum()),
            "curtail_pct" : curt_kW.sum() / max(total_kW.sum(), 1) * 100,
            "eff_deg"     : eff_deg,
            "age_ans"     : age,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # F. CALCUL LCOE AVEC COST FUNCTION ROZZI
    #
    #  LCOE = (CAPEX_PV_ann + OPEX_PV_ann + CAPEX_EOL_ann + OPEX_EOL_ann)
    #         / Énergie_ENR_totale_annuelle
    #         [€/kWh]
    #
    #  CAPEX_PV_ann  = CAPEX_PV_total_EUR  × CRF(WACC, LT_PV)
    #  CAPEX_EOL_ann = CAPEX_EOL_total_EUR × CRF(WACC, LT_EOL)
    #
    #  CAPEX_PV_total_EUR  = cost_function_eq4(S_sol) × learning_curve_eq1_3(t) × S_sol
    #  CAPEX_EOL_total_EUR = cost_function_eq4(S_eol) × learning_curve_eq1_3(t) × S_eol
    #  (déjà calculés dans get_variables() → v["PV_CAPEX_total_EUR"])
    # ─────────────────────────────────────────────────────────────────────────
    def calc_lcoe(self, enr: dict, v: dict) -> dict:
        """
        LCOE [€/kWh] des sources ENR.

        Les CAPEX utilisés ici sont déjà actualisés par cost_function_rozzi_eq4
        + learning_curve_rozzi_eq1_3 (calculés dans get_variables()).

        Retourne :
            dict avec LCOE_EUR_kWh, LCOE_EUR_MWh et décomposition des coûts
        """
        r       = v["wacc"]
        sol_cap = v["sol_cap_kW"]
        eol_cap = v["eol_cap_kW"]
        energie = max(enr["solar_kW"].sum() + enr["wind_kW"].sum(), 1.0)  # [kWh/an]

        crf_pv  = crf(r, v["lt_pv"])
        crf_eol = crf(r, v["lt_eol"])

        # Annuités CAPEX [€/an] — CAPEX_total déjà actualisé via cost_function + LC
        capex_pv_ann  = v["PV_CAPEX_total_EUR"]  * crf_pv
        capex_eol_ann = v["EOL_CAPEX_total_EUR"] * crf_eol

        # OPEX annuels [€/an]
        opex_pv_ann  = v["OPEX_PV_EUR_kW_an"]  * sol_cap
        opex_eol_ann = v["OPEX_EOL_EUR_kW_an"] * eol_cap

        cout_total = capex_pv_ann + opex_pv_ann + capex_eol_ann + opex_eol_ann

        lcoe_eur_kWh = cout_total / energie

        # CRF moyen pondéré par capacité (pour métriques)
        total_cap = sol_cap + eol_cap + 1e-9
        crf_moy   = (crf_pv * sol_cap + crf_eol * eol_cap) / total_cap

        return {
            # LCOE principal
            "LCOE_EUR_kWh"       : round(lcoe_eur_kWh, 5),
            "LCOE_EUR_MWh"       : round(lcoe_eur_kWh * 1000, 2),
            "LCOE_USD_kWh"       : round(lcoe_eur_kWh / v["usd_eur"], 5),

            # Décomposition du LCOE [€/kWh]
            "terme_CAPEX_PV"     : round(capex_pv_ann  / energie, 5),
            "terme_OPEX_PV"      : round(opex_pv_ann   / energie, 5),
            "terme_CAPEX_EOL"    : round(capex_eol_ann / energie, 5),
            "terme_OPEX_EOL"     : round(opex_eol_ann  / energie, 5),

            # Coûts annuels bruts [€/an]
            "CAPEX_PV_ann_EUR"   : round(capex_pv_ann, 0),
            "CAPEX_EOL_ann_EUR"  : round(capex_eol_ann, 0),
            "OPEX_PV_ann_EUR"    : round(opex_pv_ann, 0),
            "OPEX_EOL_ann_EUR"   : round(opex_eol_ann, 0),
            "cout_total_ann_EUR" : round(cout_total, 0),

            # Métriques financières
            "CRF_PV"             : round(crf_pv, 4),
            "CRF_EOL"            : round(crf_eol, 4),
            "CRF_moyen"          : round(crf_moy, 4),
            "energie_ann_GWh"    : round(energie / 1e6, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # G. ANALYSE COMPLÈTE
    # ─────────────────────────────────────────────────────────────────────────
    def run_analysis(self, region_nom: str = "Dakhla",
                     tech_el: str = "PEM",
                     total_enr_kW: float = 1_000_000,
                     el_cap_kW: float = 600_000,
                     annees: list = None) -> pd.DataFrame:
        """
        Analyse complète : production H2 + LCOE pour une région × technologie.

        Pour chaque année × scénario :
          1. get_variables()       → intervalles BDD + cost_function + learning curve
          2. generate_enr_8760()   → profils 8760h depuis T1
          3. dispatch_h2()         → production H2 physique
          4. calc_lcoe()           → LCOE avec CAPEX actualisés Rozzi
        """
        if annees is None:
            annees = [2024, 2030, 2035, 2040, 2050]

        print("═"*65)
        print(f"  ANALYSE — {region_nom} | {tech_el}")
        print(f"  Sorties : Production H2 + LCOE")
        print(f"  Cost Function : Rozzi eq.(4) + Learning Curve eq.(1-3)")
        print("═"*65)

        rows        = []
        enr_cache   = {}
        row_t1      = self._t1(region_nom)
        w_eol       = float(row_t1["w_eolien"])
        sol_cap     = total_enr_kW * (1 - w_eol)
        eol_cap     = total_enr_kW * w_eol

        for annee in annees:
            if annee not in enr_cache:
                enr_cache[annee] = self.generate_enr_8760(
                    region_nom, total_enr_kW, seed=42)
            enr = enr_cache[annee]

            for sc in self.SCENARIOS:
                # Variables de décision avec intervalles BDD + cost function
                v  = self.get_variables(
                    tech_el, region_nom,
                    sol_cap_kW=sol_cap, eol_cap_kW=eol_cap,
                    annee=annee, scenario=sc)

                # Production H2 physique
                h2 = self.dispatch_h2(enr["total_kW"], v, el_cap_kW,
                                      annee_install=2024)

                # LCOE avec Cost Function Rozzi
                lc = self.calc_lcoe(enr, v)

                rows.append({
                    # Identification
                    "region"             : region_nom,
                    "tech_el"            : tech_el,
                    "annee"              : annee,
                    "scenario"           : sc,

                    # Variables de décision retenues pour ce scénario
                    "VAR_eff_kWh_kg"     : round(v["eff_kWh_kg"], 2),
                    "VAR_wacc_pct"       : round(v["wacc"]*100, 1),
                    "VAR_cf_hybride_pct" : round(v["cf_hybride"]*100, 2),
                    "VAR_ppa_USD_kWh"    : round(v["ppa_USD_kWh"], 4),

                    # CAPEX ENR actualisés (eq.4 + eq.1-3)
                    "PV_CAPEX_BDD_USD_kW"  : v["PV_CAPEX_BDD_USD_kW"],
                    "PV_CAPEX_ech_USD_kW"  : v["PV_CAPEX_ech_USD_kW"],
                    "PV_facteur_lc"        : v["PV_facteur_lc"],
                    "PV_CAPEX_fin_EUR_kW"  : v["PV_CAPEX_fin_EUR_kW"],
                    "EOL_CAPEX_BDD_USD_kW" : v["EOL_CAPEX_BDD_USD_kW"],
                    "EOL_CAPEX_ech_USD_kW" : v["EOL_CAPEX_ech_USD_kW"],
                    "EOL_facteur_lc"       : v["EOL_facteur_lc"],
                    "EOL_CAPEX_fin_EUR_kW" : v["EOL_CAPEX_fin_EUR_kW"],

                    # Production H2 physique
                    "H2_ann_tonne"       : round(h2["H2_ann_t"], 1),
                    "H2_ann_GWh_LHV"     : round(h2["H2_ann_GWh"], 2),
                    "CF_EL_pct"          : round(h2["CF_EL_pct"], 1),
                    "heures_EL_ON"       : h2["heures_ON"],
                    "nb_demarrages"      : h2["demarrages"],
                    "curtail_pct"        : round(h2["curtail_pct"], 1),
                    "eff_deg_kWh_kg"     : round(h2["eff_deg"], 2),
                    "energie_EL_GWh"     : round(h2["el_kW"].sum()/1e6, 2),

                    # LCOE [€/kWh et €/MWh]
                    "LCOE_EUR_kWh"       : lc["LCOE_EUR_kWh"],
                    "LCOE_EUR_MWh"       : lc["LCOE_EUR_MWh"],
                    "LCOE_USD_kWh"       : lc["LCOE_USD_kWh"],
                    "terme_CAPEX_PV"     : lc["terme_CAPEX_PV"],
                    "terme_OPEX_PV"      : lc["terme_OPEX_PV"],
                    "terme_CAPEX_EOL"    : lc["terme_CAPEX_EOL"],
                    "terme_OPEX_EOL"     : lc["terme_OPEX_EOL"],
                    "CRF_moyen"          : lc["CRF_moyen"],
                    "energie_ENR_GWh"    : lc["energie_ann_GWh"],

                    # Traçabilité
                    "eq_capex_enr" : "Rozzi 2024 eq.(4) + eq.(1-3) — T3 + T5",
                })

        df = pd.DataFrame(rows).sort_values(["annee","scenario"]).reset_index(drop=True)

        nom    = f"production_{region_nom}_{tech_el}.csv"
        chemin = os.path.join(self.dir_results, nom)
        df.to_csv(chemin, index=False, encoding="utf-8-sig")
        print(f"  ✅ CSV → {chemin}")

        self._print_summary(df)
        self._plot_main(df, region_nom, tech_el)
        self._plot_capex_decomposition(df, region_nom, tech_el)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # H. COMPARAISON MULTI-RÉGIONS
    # ─────────────────────────────────────────────────────────────────────────
    def run_all_regions(self, tech_el: str = "PEM",
                        total_enr_kW: float = 1_000_000,
                        el_cap_kW: float = 600_000,
                        annee: int = 2030) -> pd.DataFrame:
        dfs = []
        for reg in self.T1["region"].tolist():
            try:
                dfs.append(self.run_analysis(
                    reg, tech_el, total_enr_kW, el_cap_kW, [annee]))
            except Exception as e:
                print(f"  ⚠ {reg} : {e}")
        if not dfs:
            return pd.DataFrame()
        df_all = pd.concat(dfs, ignore_index=True)
        nom    = f"comparaison_{tech_el}_{annee}.csv"
        df_all.to_csv(os.path.join(self.dir_results, nom),
                      index=False, encoding="utf-8-sig")
        self._plot_regions(df_all, tech_el, annee)
        return df_all

    # ─────────────────────────────────────────────────────────────────────────
    # I. VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────────────
    C = {"optimiste":"#27ae60", "central":"#2980b9", "pessimiste":"#c0392b"}
    S = {"optimiste":"--",      "central":"-",       "pessimiste":":"}

    def _plot_main(self, df, region, tech):
        """Production H2 + LCOE avec bandes d'incertitude."""
        annees = sorted(df["annee"].unique())
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle(
            f"Production H2 + LCOE — {region} | {tech}\n"
            f"LCOE : Cost Function Rozzi eq.(4) + Learning Curve eq.(1-3)",
            fontsize=12, fontweight="bold")

        for ax, col, title, ylabel in zip(
            axes.flat,
            ["H2_ann_tonne", "LCOE_EUR_MWh",
             "PV_CAPEX_fin_EUR_kW", "CF_EL_pct"],
            ["Production H2 [t/an]",
             "LCOE [€/MWh]  ← Rozzi eq.(4) + eq.(1-3)",
             "CAPEX PV actualisé [€/kW]",
             "Facteur de capacité EL [%]"],
            ["t/an", "€/MWh", "€/kW", "%"],
        ):
            for sc in self.SCENARIOS:
                sub = df[df["scenario"]==sc].sort_values("annee")
                ax.plot(sub["annee"], sub[col],
                        color=self.C[sc], ls=self.S[sc],
                        marker="o", lw=2, label=sc.capitalize())
            opt = df[df["scenario"]=="optimiste"].sort_values("annee")
            pes = df[df["scenario"]=="pessimiste"].sort_values("annee")
            ax.fill_between(opt["annee"].values,
                            opt[col].values, pes[col].values,
                            alpha=0.10, color="#2980b9")
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.set_xlabel("Année")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(annees)

        fig.tight_layout()
        p = os.path.join(self.dir_figs, f"Fig1_Production_{region}_{tech}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Fig1 → {p}")

    def _plot_capex_decomposition(self, df, region, tech):
        """
        Barres : BDD brut → après eq.(4) échelle → après eq.(1-3) learning
        Pour PV et éolien séparément.
        """
        df_c   = df[df["scenario"] == "central"].sort_values("annee")
        annees = df_c["annee"].values
        x = np.arange(len(annees))
        w = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Décomposition CAPEX ENR — {region} | {tech} | Scénario central\n"
            f"BDD (T3)  →  Rozzi eq.(4) taille  →  Learning curve eq.(1-3)",
            fontsize=11, fontweight="bold")

        for ax, prefix, titre in zip(
            axes,
            ["PV", "EOL"],
            ["Solaire PV", "Éolien"],
        ):
            ax.bar(x-w,  df_c[f"{prefix}_CAPEX_BDD_USD_kW"],  w,
                   label="① BDD brut T3 [USD/kW]",
                   color="#bdc3c7", alpha=0.9, edgecolor="white")
            ax.bar(x,    df_c[f"{prefix}_CAPEX_ech_USD_kW"],  w,
                   label="② Après effet d'échelle eq.(4) [USD/kW]",
                   color="#e67e22", alpha=0.9, edgecolor="white")
            ax.bar(x+w,  df_c[f"{prefix}_CAPEX_fin_EUR_kW"],  w,
                   label="③ Après learning curve eq.(1-3) [EUR/kW]",
                   color="#2980b9", alpha=0.9, edgecolor="white")

            # Annotation réduction totale
            for i, (_, row) in enumerate(df_c.iterrows()):
                bdd = row[f"{prefix}_CAPEX_BDD_USD_kW"]
                fin = row[f"{prefix}_CAPEX_fin_EUR_kW"]
                red = (1 - fin / max(bdd, 1)) * 100
                ax.text(x[i]+w, fin+2, f"-{abs(red):.0f}%",
                        ha="center", fontsize=8, color="#2980b9", fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(annees)
            ax.set_title(titre, fontsize=10)
            ax.set_ylabel("USD ou EUR / kW")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        p = os.path.join(self.dir_figs, f"Fig2_CAPEX_{region}_{tech}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Fig2 CAPEX décomposition → {p}")

    def _plot_regions(self, df_all, tech, annee):
        df_c    = df_all[(df_all["annee"]==annee) & (df_all["scenario"]=="central")]
        regions = df_c["region"].values
        x = np.arange(len(regions))
        w = 0.25

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Comparaison Régions — {tech} | {annee}",
                     fontsize=12, fontweight="bold")

        for ax, col, title in zip(
            axes,
            ["H2_ann_tonne", "LCOE_EUR_MWh"],
            ["Production H2 [t/an]", "LCOE [€/MWh]"],
        ):
            for j, sc in enumerate(self.SCENARIOS):
                sub  = df_all[(df_all["annee"]==annee) & (df_all["scenario"]==sc)]
                vals = [sub[sub["region"]==r][col].values[0]
                        if len(sub[sub["region"]==r]) > 0 else 0 for r in regions]
                ax.bar(x+(j-1)*w, vals, w, label=sc.capitalize(),
                       color=self.C[sc], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(regions, rotation=30, ha="right")
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        p = os.path.join(self.dir_figs, f"Fig3_Regions_{tech}_{annee}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Fig3 régions → {p}")

    def _print_summary(self, df):
        print(f"\n  {'Année':>6} {'Scénario':>12} {'H2[t/an]':>9} "
              f"{'LCOE€/MWh':>10} {'CAPEX_PV€/kW':>13} {'CAPEX_EOL€/kW':>14} {'Fact.LC_PV':>11}")
        print("  " + "─"*80)
        for _, r in df[df["annee"].isin([2024,2030,2050])].iterrows():
            print(f"  {int(r['annee']):>6} {r['scenario']:>12} "
                  f"{r['H2_ann_tonne']:>9,.0f} "
                  f"{r['LCOE_EUR_MWh']:>10.2f} "
                  f"{r['PV_CAPEX_fin_EUR_kW']:>13.1f} "
                  f"{r['EOL_CAPEX_fin_EUR_kW']:>14.1f} "
                  f"{r['PV_facteur_lc']:>11.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # J. VALEURS PAR DÉFAUT
    # ─────────────────────────────────────────────────────────────────────────
    def _default_T1(self):
        return pd.DataFrame([
            {"region":"Dakhla",      "CF_solaire_PV_pct":24.0,"CF_eolien_pct":55.0,
             "CF_hybride_pct":35.84,"w_eolien":0.687,"PPA_USD_kWh":0.025},
            {"region":"Ouarzazate",  "CF_solaire_PV_pct":24.5,"CF_eolien_pct":12.0,
             "CF_hybride_pct":20.13,"w_eolien":0.107,"PPA_USD_kWh":0.032},
            {"region":"Laayoune",    "CF_solaire_PV_pct":22.0,"CF_eolien_pct":43.0,
             "CF_hybride_pct":27.88,"w_eolien":0.662,"PPA_USD_kWh":0.028},
            {"region":"Tanger",      "CF_solaire_PV_pct":18.0,"CF_eolien_pct":27.0,
             "CF_hybride_pct":15.21,"w_eolien":0.600,"PPA_USD_kWh":0.035},
            {"region":"Jorf Lasfar", "CF_solaire_PV_pct":18.0,"CF_eolien_pct":15.0,
             "CF_hybride_pct":14.63,"w_eolien":0.450,"PPA_USD_kWh":0.038},
            {"region":"Guelmim",     "CF_solaire_PV_pct":21.0,"CF_eolien_pct":22.0,
             "CF_hybride_pct":16.65,"w_eolien":0.510,"PPA_USD_kWh":0.030},
        ])

    def _default_T2(self):
        return pd.DataFrame([
            {"technologie":"PEM",  "efficacite_kWh_kg":55.0,"eff_min":47.0,
             "eff_max":66.0,"duree_vie_ans":20},
            {"technologie":"AEL",  "efficacite_kWh_kg":52.0,"eff_min":45.0,
             "eff_max":60.0,"duree_vie_ans":25},
            {"technologie":"SOEC", "efficacite_kWh_kg":40.0,"eff_min":35.0,
             "eff_max":45.0,"duree_vie_ans":20},
        ])

    def _default_T3(self):
        return pd.DataFrame([
            {"technologie":"PV_solaire","capex_USD_kW":550,"capex_min":380,
             "capex_max":900,"opex_USD_kW_an":12,"duree_vie_ans":30,
             "learning_rate_pct_an":-8.0},
            {"technologie":"Eolien",    "capex_USD_kW":1200,"capex_min":900,
             "capex_max":1800,"opex_USD_kW_an":35,"duree_vie_ans":25,
             "learning_rate_pct_an":-3.0},
        ])

    def _default_T5(self):
        return pd.DataFrame([
            {"parametre":"WACC_pct",     "valeur":8.0},
            {"parametre":"WACC_min_pct", "valeur":6.0},
            {"parametre":"WACC_max_pct", "valeur":12.0},
            {"parametre":"taux_USD_EUR", "valeur":0.92},
        ])

    def _default_T6(self):
        return pd.DataFrame([
            {"technologie":"PEM",  "learning_rate_pct_an":-5.6},
            {"technologie":"AEL",  "learning_rate_pct_an":-4.5},
            {"technologie":"SOEC", "learning_rate_pct_an":-5.6},
        ])


# ══════════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    OUTPUT_DIR = r"C:\Users\HP 840 G8\Downloads\H2Morocco222_Outputs"

    model = ProductionModel(output_dir=OUTPUT_DIR)
    model.load_database()

    # Dakhla × PEM — 2024 → 2050
    df1 = model.run_analysis(
        region_nom="Dakhla",   tech_el="PEM",
        total_enr_kW=1_000_000, el_cap_kW=600_000,
        annees=[2024, 2030, 2035, 2040, 2050])

    # Ouarzazate × AEL
    df2 = model.run_analysis(
        region_nom="Ouarzazate", tech_el="AEL",
        total_enr_kW=1_000_000,  el_cap_kW=600_000,
        annees=[2024, 2030, 2035, 2040, 2050])

    # Comparaison toutes régions en 2030
    df_all = model.run_all_regions(
        tech_el="PEM", total_enr_kW=1_000_000,
        el_cap_kW=600_000, annee=2030)