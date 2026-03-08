# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 15:04:13 2026

@author: HP 840 G8
"""

# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  H2 MOROCCO — MODÈLE DE PRODUCTION v6                                      ║
║                                                                              ║
║  Combinaisons supportées :                                                  ║
║    Source ENR  × Technologie EL  =  9 cas possibles                        ║
║    ─────────────────────────────────────────────────                        ║
║    PV seul      × PEM                                                       ║
║    PV seul      × AEL                                                       ║
║    PV seul      × SOEC                                                      ║
║    Éolien seul  × PEM                                                       ║
║    Éolien seul  × AEL                                                       ║
║    Éolien seul  × SOEC                                                      ║
║    Hybride      × PEM                                                       ║
║    Hybride      × AEL                                                       ║
║    Hybride      × SOEC                                                      ║
║                                                                              ║
║  CAPEX actualisés pour TOUTES les technologies :                            ║
║    ENR  : PV, Éolien        → cost_function_eq4(S) × learning_eq1_3(t)     ║
║    EL   : PEM, AEL, SOEC    → cost_function_eq4(S) × learning_eq1_3(t)     ║
║                                                                              ║
║  Variables de dimensionnement :                                             ║
║    pv_size_kW   [kW]  — uniquement si source contient PV                   ║
║    wind_size_kW [kW]  — uniquement si source contient Éolien               ║
║    el_size_kW   [kW]  — toujours requis                                    ║
║                                                                              ║
║  Chaîne de calcul :                                                         ║
║    BDD (T1–T6)                                                              ║
║       ↓                                                                      ║
║    Variables de décision                                                    ║
║       ↓                                                                      ║
║    Production ENR 8760h  (selon source : PV / Éolien / Hybride)            ║
║       ↓                                                                      ║
║    Dispatch électrolyseur (PEM / AEL / SOEC)                               ║
║       ↓                                                                      ║
║    Production H2                                                            ║
║       ↓                                                                      ║
║    LCOE  (CAPEX actualisés ENR + EL via Rozzi)                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings
from itertools import product as iterproduct
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES PHYSIQUES
# ══════════════════════════════════════════════════════════════════════════════
LHV_H2     = 33.33   # kWh/kg
HHV_H2     = 39.41   # kWh/kg
HOURS_YEAR = 8760
ANNEE_REF  = 2024

TURBINE = {"v_cut_in": 3.0, "v_rated": 12.0, "v_cut_out": 25.0, "loss_coef": 0.05}

DEGR_EL = {"PEM": 0.015, "AEL": 0.010, "SOEC": 0.020}


# ══════════════════════════════════════════════════════════════════════════════
# SOURCES ENR ET TECHNOLOGIES EL SUPPORTÉES
# ══════════════════════════════════════════════════════════════════════════════

SOURCES_ENR = {
    "PV"     : {"composants": ["PV"],           "description": "Solaire PV uniquement"},
    "Eolien" : {"composants": ["Eolien"],        "description": "Éolien uniquement"},
    "Hybride": {"composants": ["PV", "Eolien"],  "description": "Hybride PV + Éolien"},
}

TECH_EL = ["PEM", "AEL", "SOEC"]

# Toutes les combinaisons possibles source × technologie EL
COMBINAISONS = [
    (src, tel)
    for src in SOURCES_ENR
    for tel in TECH_EL
]
# → [("PV","PEM"), ("PV","AEL"), ("PV","SOEC"),
#    ("Eolien","PEM"), ..., ("Hybride","SOEC")]  = 9 combinaisons


# ══════════════════════════════════════════════════════════════════════════════
# COST FUNCTION ROZZI — pour ENR et EL
# ══════════════════════════════════════════════════════════════════════════════

# Coefficients pour chaque technologie ENR et EL (Rozzi Table 4)
ROZZI_COEFFS = {
    # ── Sources ENR ──────────────────────────────────────
    "PV"    : {"C_ref":  550.0, "S_ref_kW":  1_000.0, "exp": 0.85},
    "Eolien": {"C_ref": 1200.0, "S_ref_kW":  1_000.0, "exp": 0.90},
    # ── Électrolyseurs ───────────────────────────────────
    "PEM"   : {"C_ref": 1300.0, "S_ref_kW":    897.0, "exp": 0.90},
    "AEL"   : {"C_ref": 1039.0, "S_ref_kW":    890.0, "exp": 0.90},
    "SOEC"  : {"C_ref": 2500.0, "S_ref_kW":    950.0, "exp": 0.88},
}


def cost_function_eq4(S_kW: float, tech: str, capex_bdd: float) -> float:
    """
    [Rozzi eq.(4)] Effet d'échelle.

    CAPEX(S) = capex_bdd × S_ref × (S / S_ref)^exp / S     [USD/kW]

    Valable pour :
      • Sources ENR : tech = "PV" | "Eolien"
      • Électrolyseurs : tech = "PEM" | "AEL" | "SOEC"

    Plus S est grand → CAPEX unitaire plus faible (économies d'échelle).
    """
    c = ROZZI_COEFFS.get(tech)
    if c is None or S_kW <= 0:
        return capex_bdd
    return (capex_bdd * c["S_ref_kW"] * (S_kW / c["S_ref_kW"]) ** c["exp"]) / S_kW


def learning_curve_eq1_3(capex_ref: float, lr_pct_an: float, annee: int) -> float:
    """
    [Rozzi eq.(1-3)] Actualisation temporelle.

    CAPEX(t) = CAPEX_ref × (1 + lr)^(t - 2024)

    Valable pour toutes les technologies ENR et EL.
    """
    return capex_ref * (1.0 + lr_pct_an / 100.0) ** max(0, annee - ANNEE_REF)


def capex_actualise(tech: str, S_kW: float,
                    capex_bdd: float, lr_pct_an: float,
                    annee: int, usd_eur: float = 0.92) -> dict:
    """
    CAPEX actualisé pour une technologie (ENR ou EL) :
      Étape 1 — [eq.4]   effet d'échelle sur S_kW
      Étape 2 — [eq.1-3] facteur learning curve temporel
      Étape 3 — conversion USD → EUR

    Paramètres :
        tech       : "PV" | "Eolien" | "PEM" | "AEL" | "SOEC"
        S_kW       : taille installée [kW]        ← variable de dimensionnement
        capex_bdd  : CAPEX brut BDD [USD/kW]      ← depuis T3 (ENR) ou T2 (EL)
        lr_pct_an  : learning rate [%/an]          ← depuis T3 (ENR) ou T6 (EL)
        annee      : année de calcul
        usd_eur    : taux de change (T5)

    Retourne :
        dict avec CAPEX_BDD, CAPEX_echelle, facteur_lc, CAPEX_final [EUR/kW et EUR total]
    """
    # Étape 1 — Effet d'échelle [eq.4]
    c_ech = cost_function_eq4(S_kW, tech, capex_bdd)

    # Étape 2 — Facteur learning curve [eq.1-3]
    c_lc     = learning_curve_eq1_3(capex_bdd, lr_pct_an, annee)
    fact_lc  = c_lc / capex_bdd if capex_bdd > 0 else 1.0

    # Étape 3 — CAPEX final [EUR/kW]
    c_fin    = c_ech * fact_lc * usd_eur

    return {
        "CAPEX_BDD_USD_kW"  : round(capex_bdd, 1),
        "CAPEX_ech_USD_kW"  : round(c_ech, 1),
        "facteur_lc"        : round(fact_lc, 4),
        "CAPEX_fin_EUR_kW"  : round(c_fin, 1),
        "CAPEX_total_EUR"   : round(c_fin * S_kW, 0),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CRF
# ══════════════════════════════════════════════════════════════════════════════
def crf(r: float, n: float) -> float:
    if r <= 1e-9:
        return 1.0 / n
    return (r * (1 + r) ** n) / ((1 + r) ** n - 1)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
class ProductionModel:
    """
    Modèle de production H2 avec :
      - 9 combinaisons Source ENR × Technologie EL
      - CAPEX actualisés pour TOUTES les technologies via Rozzi eq.(4) + eq.(1-3)
      - Dimensionnement variable : pv_size_kW, wind_size_kW, el_size_kW
    """

    SCENARIOS   = ["optimiste", "central", "pessimiste"]
    SOURCES     = list(SOURCES_ENR.keys())     # ["PV", "Eolien", "Hybride"]
    TECH_EL_LST = TECH_EL                      # ["PEM", "AEL", "SOEC"]

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
        for nom, fichiers, default in [
            ("T1", ["T1_avec_source.csv","T1_ressources_energetiques.csv"], self._default_T1()),
            ("T2", ["T2_avec_source.csv","T2_technologies_production.csv"],  self._default_T2()),
            ("T3", ["T3_avec_source.csv","T3_technologies_ener.csv"],         self._default_T3()),
            ("T5", ["T5_avec_source.csv","T5_parametres_economiques.csv"],   self._default_T5()),
            ("T6", ["T6_avec_source.csv","T6_learning_curves.csv"],           self._default_T6()),
        ]:
            setattr(self, nom, self._load(nom, fichiers, default))
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
    def _t1(self, region):
        m = self.T1["region"] == region
        if not m.any():
            raise ValueError(f"'{region}' absent de T1. Disponibles : {self.T1['region'].tolist()}")
        return self.T1[m].iloc[0]

    def _t2(self, tech_el):
        m = self.T2["technologie"] == tech_el
        if not m.any():
            raise ValueError(f"'{tech_el}' absent de T2. Disponibles : {self.T2['technologie'].tolist()}")
        return self.T2[m].iloc[0]

    def _t3(self, tech_enr):
        # Accepte "PV" et "PV_solaire" comme alias
        alias = {"PV": "PV_solaire", "Eolien": "Eolien"}
        tech  = alias.get(tech_enr, tech_enr)
        m = self.T3["technologie"] == tech
        if not m.any():
            raise ValueError(f"'{tech}' absent de T3. Disponibles : {self.T3['technologie'].tolist()}")
        return self.T3[m].iloc[0]

    def _t5(self, param, fallback=None):
        m = self.T5["parametre"] == param
        return float(self.T5[m].iloc[0]["valeur"]) if m.any() else fallback

    def _t6_lr(self, tech_el):
        m = self.T6["technologie"] == tech_el
        return float(self.T6[m].iloc[0]["learning_rate_pct_an"]) if m.any() else -5.0

    def _iv(self, row, col_min, col_mode, col_max, sens="normal", var_pct=None):
        def _v(c):
            return float(row[c]) if c in row.index and pd.notna(row[c]) else float(row[col_mode])
        vmin, vmode, vmax = _v(col_min), _v(col_mode), _v(col_max)
        if var_pct and abs(vmin - vmax) < 1e-9:
            vmin, vmax = vmode*(1-var_pct), vmode*(1+var_pct)
        return ({"optimiste": vmin, "central": vmode, "pessimiste": vmax}
                if sens == "normal"
                else {"optimiste": vmax, "central": vmode, "pessimiste": vmin})

    # ─────────────────────────────────────────────────────────────────────────
    # C. VARIABLES DE DÉCISION
    #    Lit T1–T6, applique cost_function + learning_curve sur TOUS les CAPEX
    # ─────────────────────────────────────────────────────────────────────────
    def get_variables(self,
                      source_enr:   str,
                      pv_size_kW:   float,
                      wind_size_kW: float,
                      el_size_kW:   float,
                      tech_el:      str,
                      region_nom:   str,
                      annee:        int,
                      scenario:     str = "central") -> dict:
        """
        Construit le vecteur complet des variables de décision.

        Paramètres :
            source_enr   : "PV" | "Eolien" | "Hybride"
            pv_size_kW   : capacité PV installée [kW]   (0 si source sans PV)
            wind_size_kW : capacité éolienne     [kW]   (0 si source sans éolien)
            el_size_kW   : capacité EL           [kW]
            tech_el      : "PEM" | "AEL" | "SOEC"
            region_nom   : région Maroc (T1)
            annee        : année de calcul
            scenario     : "optimiste" | "central" | "pessimiste"

        Retourne :
            dict plat contenant toutes les variables de décision
            avec CAPEX actualisés pour TOUTES les technologies présentes.
        """
        composants = SOURCES_ENR[source_enr]["composants"]
        usd_eur    = self._t5("taux_USD_EUR", 0.92)
        row_t1     = self._t1(region_nom)
        row_t2     = self._t2(tech_el)

        # ── WACC (T5)
        wacc_iv = {
            "optimiste" : self._t5("WACC_min_pct", 6.0)  / 100,
            "central"   : self._t5("WACC_pct",     8.0)  / 100,
            "pessimiste": self._t5("WACC_max_pct", 12.0) / 100,
        }

        # ── Efficacité EL (T2) — pour dispatch H2
        eff_iv = self._iv(row_t2, "eff_min","efficacite_kWh_kg","eff_max", "normal")
        lt_el  = float(row_t2["duree_vie_ans"])
        lr_el  = self._t6_lr(tech_el)

        # ── CAPEX EL actualisé [eq.4 + eq.1-3] — pour TOUTES les technologies EL
        capex_el_iv = self._iv(row_t2, "capex_min","capex_USD_kW","capex_max", "normal")
        el_actu = capex_actualise(
            tech     = tech_el,
            S_kW     = el_size_kW,
            capex_bdd = capex_el_iv[scenario],
            lr_pct_an = lr_el,
            annee    = annee,
            usd_eur  = usd_eur,
        )

        # ── CF et PPA (T1) — avec incertitude météo
        cf_sol = float(row_t1["CF_solaire_PV_pct"])
        cf_eol = float(row_t1["CF_eolien_pct"])
        cf_hyb = float(row_t1["CF_hybride_pct"])
        ppa    = float(row_t1["PPA_USD_kWh"])
        ppa_iv = {"optimiste": ppa*0.80, "central": ppa, "pessimiste": ppa*1.20}

        v = {
            # ── Identification
            "source_enr"            : source_enr,
            "composants_enr"        : composants,
            "tech_el"               : tech_el,
            "region"                : region_nom,
            "annee"                 : annee,
            "scenario"              : scenario,
            "usd_eur"               : usd_eur,

            # ── Dimensionnement
            "pv_size_kW"            : pv_size_kW   if "PV"     in composants else 0,
            "wind_size_kW"          : wind_size_kW if "Eolien" in composants else 0,
            "el_size_kW"            : el_size_kW,

            # ── WACC et financier
            "wacc"                  : wacc_iv[scenario],
            "lt_el"                 : lt_el,
            "lr_el"                 : lr_el,

            # ── CAPEX EL actualisé (PEM ou AEL ou SOEC) [eq.4 + eq.1-3]
            f"{tech_el}_CAPEX_BDD_USD_kW"  : el_actu["CAPEX_BDD_USD_kW"],
            f"{tech_el}_CAPEX_ech_USD_kW"  : el_actu["CAPEX_ech_USD_kW"],
            f"{tech_el}_facteur_lc"        : el_actu["facteur_lc"],
            f"{tech_el}_CAPEX_fin_EUR_kW"  : el_actu["CAPEX_fin_EUR_kW"],
            f"{tech_el}_CAPEX_total_EUR"   : el_actu["CAPEX_total_EUR"],

            # ── Efficacité EL
            "eff_kWh_kg"            : eff_iv[scenario],

            # ── Ressources (T1)
            "CF_sol_pct"            : cf_sol,
            "CF_eol_pct"            : cf_eol,
            "CF_hyb_pct"            : cf_hyb,
            "ppa_USD_kWh"           : ppa_iv[scenario],
        }

        # ── CAPEX PV actualisé — seulement si source contient PV
        if "PV" in composants and pv_size_kW > 0:
            row_pv = self._t3("PV")
            capex_pv_iv = self._iv(row_pv, "capex_min","capex_USD_kW","capex_max", "normal")
            lr_pv = float(row_pv["learning_rate_pct_an"])
            lt_pv = int(row_pv["duree_vie_ans"])
            pv_actu = capex_actualise(
                tech      = "PV",
                S_kW      = pv_size_kW,
                capex_bdd = capex_pv_iv[scenario],
                lr_pct_an = lr_pv,
                annee     = annee,
                usd_eur   = usd_eur,
            )
            v.update({
                "PV_CAPEX_BDD_USD_kW"  : pv_actu["CAPEX_BDD_USD_kW"],
                "PV_CAPEX_ech_USD_kW"  : pv_actu["CAPEX_ech_USD_kW"],
                "PV_facteur_lc"        : pv_actu["facteur_lc"],
                "PV_CAPEX_fin_EUR_kW"  : pv_actu["CAPEX_fin_EUR_kW"],
                "PV_CAPEX_total_EUR"   : pv_actu["CAPEX_total_EUR"],
                "OPEX_PV_EUR_kW"       : round(float(row_pv["opex_USD_kW_an"]) * usd_eur, 2),
                "lt_pv"                : lt_pv,
                "lr_pv"                : lr_pv,
            })
        else:
            v.update({
                "PV_CAPEX_BDD_USD_kW": 0, "PV_CAPEX_ech_USD_kW": 0,
                "PV_facteur_lc": 0, "PV_CAPEX_fin_EUR_kW": 0,
                "PV_CAPEX_total_EUR": 0, "OPEX_PV_EUR_kW": 0,
                "lt_pv": 30, "lr_pv": -8.0,
            })

        # ── CAPEX Éolien actualisé — seulement si source contient Éolien
        if "Eolien" in composants and wind_size_kW > 0:
            row_eol = self._t3("Eolien")
            capex_eol_iv = self._iv(row_eol, "capex_min","capex_USD_kW","capex_max", "normal")
            lr_eol = float(row_eol["learning_rate_pct_an"])
            lt_eol = int(row_eol["duree_vie_ans"])
            eol_actu = capex_actualise(
                tech      = "Eolien",
                S_kW      = wind_size_kW,
                capex_bdd = capex_eol_iv[scenario],
                lr_pct_an = lr_eol,
                annee     = annee,
                usd_eur   = usd_eur,
            )
            v.update({
                "EOL_CAPEX_BDD_USD_kW" : eol_actu["CAPEX_BDD_USD_kW"],
                "EOL_CAPEX_ech_USD_kW" : eol_actu["CAPEX_ech_USD_kW"],
                "EOL_facteur_lc"       : eol_actu["facteur_lc"],
                "EOL_CAPEX_fin_EUR_kW" : eol_actu["CAPEX_fin_EUR_kW"],
                "EOL_CAPEX_total_EUR"  : eol_actu["CAPEX_total_EUR"],
                "OPEX_EOL_EUR_kW"      : round(float(row_eol["opex_USD_kW_an"]) * usd_eur, 2),
                "lt_eol"               : lt_eol,
                "lr_eol"               : lr_eol,
            })
        else:
            v.update({
                "EOL_CAPEX_BDD_USD_kW": 0, "EOL_CAPEX_ech_USD_kW": 0,
                "EOL_facteur_lc": 0, "EOL_CAPEX_fin_EUR_kW": 0,
                "EOL_CAPEX_total_EUR": 0, "OPEX_EOL_EUR_kW": 0,
                "lt_eol": 25, "lr_eol": -3.0,
            })

        return v

    # ─────────────────────────────────────────────────────────────────────────
    # D. PRODUCTION ENR 8760H
    #    Génère les profils selon la source : PV / Éolien / Hybride
    # ─────────────────────────────────────────────────────────────────────────
    def _wind_cf_curve(self, v_ms: np.ndarray) -> np.ndarray:
        v  = np.asarray(v_ms, dtype=float)
        cf = np.zeros_like(v)
        vc, vr, vo = TURBINE["v_cut_in"], TURBINE["v_rated"], TURBINE["v_cut_out"]
        cf[(v>=vc)&(v<vr)] = ((v[(v>=vc)&(v<vr)] - vc) / (vr - vc)) ** 3
        cf[(v>=vr)&(v<vo)] = 1.0
        return cf * (1 - TURBINE["loss_coef"])

    def generate_enr_8760(self, v: dict, seed: int = 42) -> dict:
        """
        Génère les profils ENR 8760h selon la source_enr.

        Source = "PV"     → profil solaire uniquement
        Source = "Eolien" → profil éolien uniquement
        Source = "Hybride"→ profil solaire + éolien
        """
        np.random.seed(seed)
        h  = np.arange(HOURS_YEAR)
        hj = h % 24
        ja = h // 24

        pv_cap   = v["pv_size_kW"]
        wind_cap = v["wind_size_kW"]
        cf_sol   = v["CF_sol_pct"] / 100
        cf_eol   = v["CF_eol_pct"] / 100

        # ── Profil solaire PV
        if pv_cap > 0:
            sol = (np.maximum(0, np.sin(np.pi*(hj-6)/12))
                   * (1 + 0.3*np.cos(2*np.pi*(ja-172)/365))
                   * np.random.beta(2, 2, HOURS_YEAR))
            if sol.mean() > 1e-9:
                sol = np.clip(sol * cf_sol / sol.mean(), 0, 1)
            solar_kW = sol * pv_cap
        else:
            solar_kW = np.zeros(HOURS_YEAR)

        # ── Profil éolien
        if wind_cap > 0:
            v_m  = np.clip(TURBINE["v_rated"]*(cf_eol**(1/3))*1.2,
                           TURBINE["v_cut_in"]+0.5, TURBINE["v_rated"]*1.5)
            v_ms = np.clip(
                v_m/0.8862 * np.random.weibull(2.0, HOURS_YEAR)
                * (1+0.20*np.sin(2*np.pi*(hj-6)/24))
                * (1+0.12*np.cos(2*np.pi*(ja-60)/365)),
                0, TURBINE["v_cut_out"])
            eol = self._wind_cf_curve(v_ms)
            if eol.mean() > 1e-9:
                eol = np.clip(eol * cf_eol / eol.mean(), 0, 1)
            wind_kW = eol * wind_cap
        else:
            wind_kW = np.zeros(HOURS_YEAR)

        total_kW = solar_kW + wind_kW
        total_cap = max(pv_cap + wind_cap, 1)

        return {
            "solar_kW"   : solar_kW,
            "wind_kW"    : wind_kW,
            "total_kW"   : total_kW,
            "CF_sol_sim" : round(solar_kW.mean() / max(pv_cap,1)   * 100, 2) if pv_cap   > 0 else 0.0,
            "CF_eol_sim" : round(wind_kW.mean()  / max(wind_cap,1) * 100, 2) if wind_cap > 0 else 0.0,
            "CF_mix_sim" : round(total_kW.mean() / total_cap        * 100, 2),
            "E_sol_GWh"  : round(solar_kW.sum()  / 1e6, 2),
            "E_eol_GWh"  : round(wind_kW.sum()   / 1e6, 2),
            "E_tot_GWh"  : round(total_kW.sum()  / 1e6, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # E. DISPATCH ÉLECTROLYSEUR → H2
    # ─────────────────────────────────────────────────────────────────────────
    def dispatch_h2(self, enr: dict, v: dict,
                    el_min_load: float = 0.10,
                    annee_install: int = 2024) -> dict:
        """
        Dispatch ENR → EL → H2.

        Fonctionne pour PEM, AEL et SOEC (via DEGR_EL).
        """
        el_size = v["el_size_kW"]
        eff_0   = v["eff_kWh_kg"]
        tech    = v["tech_el"]
        annee   = v["annee"]
        total   = enr["total_kW"]

        seuil = el_min_load * el_size
        ramp  = 0.20 * el_size
        el_kW = np.zeros(HOURS_YEAR)

        for t in range(HOURS_YEAR):
            tgt = min(total[t], el_size) if total[t] >= seuil else 0.0
            if t > 0:
                tgt = el_kW[t-1] + np.clip(tgt - el_kW[t-1], -ramp, ramp)
            el_kW[t] = max(0.0, tgt)

        age     = max(0, annee - annee_install)
        eff_deg = eff_0 * (1 + DEGR_EL.get(tech, 0.015)) ** age
        h2_kg_h = el_kW / eff_deg
        curt_kW = np.maximum(total - el_size, 0)

        return {
            "el_kW"       : el_kW,
            "h2_kg_h"     : h2_kg_h,
            "H2_ann_kg"   : h2_kg_h.sum(),
            "H2_ann_t"    : h2_kg_h.sum() / 1000,
            "H2_ann_GWh"  : h2_kg_h.sum() * LHV_H2 / 1e6,
            "CF_EL_pct"   : el_kW.mean() / el_size * 100,
            "heures_ON"   : int((el_kW > 0).sum()),
            "demarrages"  : int(np.diff((el_kW>0).astype(int)).clip(0).sum()),
            "curtail_pct" : curt_kW.sum() / max(total.sum(), 1) * 100,
            "curtail_GWh" : round(curt_kW.sum() / 1e6, 2),
            "eff_deg"     : eff_deg,
            "age_ans"     : age,
            "E_EL_GWh"    : round(el_kW.sum() / 1e6, 2),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # F. CALCUL LCOE
    #    Inclut CAPEX PV + CAPEX Éolien + CAPEX EL — tous actualisés
    # ─────────────────────────────────────────────────────────────────────────
    def calc_lcoe(self, enr: dict, v: dict) -> dict:
        """
        LCOE = (Σ CAPEX_ann_i + Σ OPEX_ann_i) / Énergie_ENR_totale

        Composants inclus selon la source :
          • PV seul    : CAPEX_PV + OPEX_PV  + CAPEX_EL + OPEX_EL
          • Éolien     : CAPEX_EOL + OPEX_EOL + CAPEX_EL + OPEX_EL
          • Hybride    : CAPEX_PV + OPEX_PV + CAPEX_EOL + OPEX_EOL + CAPEX_EL + OPEX_EL

        Tous les CAPEX sont déjà actualisés via eq.(4)+eq.(1-3) dans get_variables().
        """
        r       = v["wacc"]
        tech_el = v["tech_el"]
        energie = max(enr["E_tot_GWh"] * 1e6, 1.0)  # [kWh/an]

        # CRF par composant
        crf_pv  = crf(r, v["lt_pv"])
        crf_eol = crf(r, v["lt_eol"])
        crf_el  = crf(r, v["lt_el"])

        # CAPEX annualisés ENR [€/an]
        capex_pv_ann  = v["PV_CAPEX_total_EUR"]                           * crf_pv
        capex_eol_ann = v["EOL_CAPEX_total_EUR"]                          * crf_eol
        capex_el_ann  = v[f"{tech_el}_CAPEX_total_EUR"]                   * crf_el

        # OPEX annuels ENR [€/an]
        opex_pv_ann   = v["OPEX_PV_EUR_kW"]  * v["pv_size_kW"]
        opex_eol_ann  = v["OPEX_EOL_EUR_kW"] * v["wind_size_kW"]

        # OPEX EL — % du CAPEX total EL
        opex_el_pct  = self._t2(tech_el).get("opex_pct_capex", 0.03) if hasattr(self._t2(tech_el), 'get') else float(self._t2(tech_el)["opex_pct_capex"]) if "opex_pct_capex" in self._t2(tech_el).index else 0.03
        opex_el_ann  = opex_el_pct * v[f"{tech_el}_CAPEX_total_EUR"]

        cout_tot = (capex_pv_ann + opex_pv_ann +
                    capex_eol_ann + opex_eol_ann +
                    capex_el_ann + opex_el_ann)
        lcoe = cout_tot / energie

        return {
            "LCOE_EUR_kWh"      : round(lcoe, 5),
            "LCOE_EUR_MWh"      : round(lcoe * 1000, 2),
            "LCOE_USD_kWh"      : round(lcoe / v["usd_eur"], 5),
            # Décomposition
            "terme_CAPEX_PV"    : round(capex_pv_ann  / energie, 5),
            "terme_OPEX_PV"     : round(opex_pv_ann   / energie, 5),
            "terme_CAPEX_EOL"   : round(capex_eol_ann / energie, 5),
            "terme_OPEX_EOL"    : round(opex_eol_ann  / energie, 5),
            "terme_CAPEX_EL"    : round(capex_el_ann  / energie, 5),
            "terme_OPEX_EL"     : round(opex_el_ann   / energie, 5),
            # CRF
            "CRF_PV"            : round(crf_pv, 4),
            "CRF_EOL"           : round(crf_eol, 4),
            "CRF_EL"            : round(crf_el, 4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # G. RUN_CASE — calcul d'un cas unique
    # ─────────────────────────────────────────────────────────────────────────
    def run_case(self,
                 source_enr:   str,
                 pv_size_kW:   float,
                 wind_size_kW: float,
                 el_size_kW:   float,
                 tech_el:      str,
                 region_nom:   str,
                 annee:        int   = 2030,
                 scenario:     str   = "central",
                 annee_install:int   = 2024,
                 seed:         int   = 42) -> dict:
        """
        Calcule un cas unique pour une combinaison (source_enr × tech_el).

        Chaîne complète :
          BDD → Variables (CAPEX actualisés) → ENR 8760h → Dispatch → H2 → LCOE
        """
        # Validation combinaison
        if source_enr not in SOURCES_ENR:
            raise ValueError(f"source_enr='{source_enr}' invalide. "
                             f"Valeurs : {list(SOURCES_ENR.keys())}")
        if tech_el not in TECH_EL:
            raise ValueError(f"tech_el='{tech_el}' invalide. Valeurs : {TECH_EL}")

        # Ajuster les tailles selon la source
        composants = SOURCES_ENR[source_enr]["composants"]
        _pv   = pv_size_kW   if "PV"     in composants else 0.0
        _wind = wind_size_kW if "Eolien" in composants else 0.0

        # Étape 1 — Variables de décision
        v = self.get_variables(
            source_enr, _pv, _wind, el_size_kW,
            tech_el, region_nom, annee, scenario)

        # Étape 2 — Production ENR 8760h
        enr = self.generate_enr_8760(v, seed=seed)

        # Étape 3 — Dispatch EL → H2
        h2 = self.dispatch_h2(enr, v, annee_install=annee_install)

        # Étape 4 — LCOE
        lc = self.calc_lcoe(enr, v)

        total_enr = _pv + _wind
        return {
            # Identification combinaison
            "source_enr"              : source_enr,
            "tech_el"                 : tech_el,
            "combinaison"             : f"{source_enr} × {tech_el}",
            "region"                  : region_nom,
            "annee"                   : annee,
            "scenario"                : scenario,

            # Dimensionnement
            "pv_size_MW"              : round(_pv   / 1000, 1),
            "wind_size_MW"            : round(_wind / 1000, 1),
            "el_size_MW"              : round(el_size_kW / 1000, 1),
            "total_enr_MW"            : round(total_enr / 1000, 1),
            "ratio_EL_ENR"            : round(el_size_kW / max(total_enr, 1), 3),

            # Production ENR
            "CF_sol_sim_pct"          : enr["CF_sol_sim"],
            "CF_eol_sim_pct"          : enr["CF_eol_sim"],
            "CF_mix_sim_pct"          : enr["CF_mix_sim"],
            "E_sol_GWh"               : enr["E_sol_GWh"],
            "E_eol_GWh"               : enr["E_eol_GWh"],
            "E_total_GWh"             : enr["E_tot_GWh"],

            # Production H2
            "H2_ann_tonne"            : round(h2["H2_ann_t"], 1),
            "H2_ann_GWh_LHV"          : round(h2["H2_ann_GWh"], 2),
            "CF_EL_pct"               : round(h2["CF_EL_pct"], 1),
            "heures_EL_ON"            : h2["heures_ON"],
            "nb_demarrages"           : h2["demarrages"],
            "curtail_pct"             : round(h2["curtail_pct"], 1),
            "curtail_GWh"             : h2["curtail_GWh"],
            "eff_deg_kWh_kg"          : round(h2["eff_deg"], 2),

            # CAPEX actualisés — ENR
            "PV_CAPEX_BDD_USD_kW"     : v["PV_CAPEX_BDD_USD_kW"],
            "PV_CAPEX_ech_USD_kW"     : v["PV_CAPEX_ech_USD_kW"],
            "PV_facteur_lc"           : v["PV_facteur_lc"],
            "PV_CAPEX_fin_EUR_kW"     : v["PV_CAPEX_fin_EUR_kW"],
            "EOL_CAPEX_BDD_USD_kW"    : v["EOL_CAPEX_BDD_USD_kW"],
            "EOL_CAPEX_ech_USD_kW"    : v["EOL_CAPEX_ech_USD_kW"],
            "EOL_facteur_lc"          : v["EOL_facteur_lc"],
            "EOL_CAPEX_fin_EUR_kW"    : v["EOL_CAPEX_fin_EUR_kW"],

            # CAPEX actualisé — EL (PEM / AEL / SOEC)
            f"{tech_el}_CAPEX_BDD_USD_kW" : v[f"{tech_el}_CAPEX_BDD_USD_kW"],
            f"{tech_el}_CAPEX_ech_USD_kW" : v[f"{tech_el}_CAPEX_ech_USD_kW"],
            f"{tech_el}_facteur_lc"       : v[f"{tech_el}_facteur_lc"],
            f"{tech_el}_CAPEX_fin_EUR_kW" : v[f"{tech_el}_CAPEX_fin_EUR_kW"],

            # LCOE
            "LCOE_EUR_kWh"            : lc["LCOE_EUR_kWh"],
            "LCOE_EUR_MWh"            : lc["LCOE_EUR_MWh"],
            "LCOE_USD_kWh"            : lc["LCOE_USD_kWh"],
            "terme_CAPEX_PV"          : lc["terme_CAPEX_PV"],
            "terme_CAPEX_EOL"         : lc["terme_CAPEX_EOL"],
            "terme_CAPEX_EL"          : lc["terme_CAPEX_EL"],
            "terme_OPEX_EOL"          : lc["terme_OPEX_EOL"],
        }

    # ─────────────────────────────────────────────────────────────────────────
    # H. RUN_ALL_COMBINAISONS — toutes les 9 combinaisons source × tech_el
    # ─────────────────────────────────────────────────────────────────────────
    def run_all_combinaisons(self,
                              pv_size_kW:   float = 500_000,
                              wind_size_kW: float = 500_000,
                              el_size_kW:   float = 600_000,
                              region_nom:   str   = "Dakhla",
                              annee:        int   = 2030,
                              scenario:     str   = "central") -> pd.DataFrame:
        """
        Lance les 9 combinaisons : source_enr × tech_el.

        Source  × Tech  → 9 cas
        PV      × PEM
        PV      × AEL
        PV      × SOEC
        Éolien  × PEM
        Éolien  × AEL
        Éolien  × SOEC
        Hybride × PEM
        Hybride × AEL
        Hybride × SOEC
        """
        print("\n" + "═"*65)
        print(f"  9 COMBINAISONS Source ENR × Technologie EL")
        print(f"  Région={region_nom} | Année={annee} | Scénario={scenario}")
        print("═"*65)

        rows = []
        for src, tel in COMBINAISONS:
            try:
                r = self.run_case(
                    source_enr   = src,
                    pv_size_kW   = pv_size_kW,
                    wind_size_kW = wind_size_kW,
                    el_size_kW   = el_size_kW,
                    tech_el      = tel,
                    region_nom   = region_nom,
                    annee        = annee,
                    scenario     = scenario,
                )
                rows.append(r)
                print(f"  ✓ {src:<8} × {tel:<5}  "
                      f"H2={r['H2_ann_tonne']:>8,.0f} t/an  "
                      f"LCOE={r['LCOE_EUR_MWh']:>6.2f} €/MWh  "
                      f"CF_EL={r['CF_EL_pct']:>5.1f}%")
            except Exception as e:
                print(f"  ✗ {src} × {tel} → {e}")

        df = pd.DataFrame(rows)
        nom    = f"combinaisons_{region_nom}_{annee}_{scenario}.csv"
        chemin = os.path.join(self.dir_results, nom)
        df.to_csv(chemin, index=False, encoding="utf-8-sig")
        print(f"\n  ✅ CSV → {chemin}")

        self._plot_combinaisons(df, region_nom, annee, scenario)
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # I. RUN_SIZING — balayage dimensionnement pour une combinaison
    # ─────────────────────────────────────────────────────────────────────────
    def run_sizing(self,
                   source_enr:    str,
                   tech_el:       str,
                   pv_sizes_kW:   list,
                   wind_sizes_kW: list,
                   el_sizes_kW:   list,
                   region_nom:    str  = "Dakhla",
                   annee:         int  = 2030,
                   scenario:      str  = "central") -> pd.DataFrame:
        """
        Balayage de dimensionnement pour une combinaison (source × tech_el).
        """
        composants = SOURCES_ENR[source_enr]["composants"]
        pvs   = pv_sizes_kW   if "PV"     in composants else [0]
        winds = wind_sizes_kW if "Eolien" in composants else [0]
        combos = list(iterproduct(pvs, winds, el_sizes_kW))

        print(f"\n  BALAYAGE {source_enr} × {tech_el} — {len(combos)} combinaisons")

        rows = []
        for pv, wind, el in combos:
            r = self.run_case(source_enr, pv, wind, el,
                              tech_el, region_nom, annee, scenario)
            rows.append(r)

        df = pd.DataFrame(rows).sort_values("LCOE_EUR_MWh").reset_index(drop=True)

        il = df["LCOE_EUR_MWh"].idxmin()
        ih = df["H2_ann_tonne"].idxmax()
        print(f"  Min LCOE : PV={df.loc[il,'pv_size_MW']:.0f} MW | "
              f"Wind={df.loc[il,'wind_size_MW']:.0f} MW | EL={df.loc[il,'el_size_MW']:.0f} MW "
              f"→ {df.loc[il,'LCOE_EUR_MWh']:.2f} €/MWh")
        print(f"  Max H2   : PV={df.loc[ih,'pv_size_MW']:.0f} MW | "
              f"Wind={df.loc[ih,'wind_size_MW']:.0f} MW | EL={df.loc[ih,'el_size_MW']:.0f} MW "
              f"→ {df.loc[ih,'H2_ann_tonne']:,.0f} t/an")

        nom    = f"sizing_{source_enr}_{tech_el}_{region_nom}_{annee}.csv"
        df.to_csv(os.path.join(self.dir_results, nom), index=False, encoding="utf-8-sig")
        print(f"  ✅ CSV → {nom}")
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # J. VISUALISATIONS
    # ─────────────────────────────────────────────────────────────────────────
    def _plot_combinaisons(self, df, region, annee, scenario):
        """
        Comparaison des 9 combinaisons source × tech_el :
          1. LCOE groupé par source ENR
          2. Production H2 groupé par source ENR
          3. CAPEX EL actualisé par technologie EL
        """
        sources = ["PV", "Eolien", "Hybride"]
        techs   = ["PEM", "AEL", "SOEC"]
        colors  = {"PEM": "#2980b9", "AEL": "#27ae60", "SOEC": "#e67e22"}
        x       = np.arange(len(sources))
        w       = 0.25

        fig, axes = plt.subplots(1, 3, figsize=(16, 6))
        fig.suptitle(
            f"9 Combinaisons Source ENR × Technologie EL\n"
            f"Région={region} | Année={annee} | Scénario={scenario}",
            fontsize=12, fontweight="bold")

        for ax, col, title, ylabel in zip(
            axes,
            ["LCOE_EUR_MWh", "H2_ann_tonne", "CF_EL_pct"],
            ["LCOE [€/MWh]", "Production H2 [t/an]", "CF Électrolyseur [%]"],
            ["€/MWh", "t/an", "%"],
        ):
            for j, tel in enumerate(techs):
                sub  = df[df["tech_el"] == tel]
                vals = []
                for src in sources:
                    row = sub[sub["source_enr"] == src]
                    vals.append(float(row[col].values[0]) if len(row) > 0 else 0)
                ax.bar(x + (j-1)*w, vals, w, label=tel,
                       color=colors[tel], alpha=0.85, edgecolor="white")

            ax.set_xticks(x)
            ax.set_xticklabels(sources)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=9, title="Tech EL")
            ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        p = os.path.join(self.dir_figs, f"Fig_9combinaisons_{region}_{annee}.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Fig 9 combinaisons → {p}")

        # Figure 2 — CAPEX actualisés par technologie (EL + ENR)
        fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
        fig2.suptitle(
            f"CAPEX actualisés — Rozzi eq.(4) + eq.(1-3)\n"
            f"Année={annee} | {region} | {scenario}",
            fontsize=11, fontweight="bold")

        # CAPEX EL par technologie
        el_capex_cols = {tel: f"{tel}_CAPEX_fin_EUR_kW" for tel in techs}
        el_vals_bdd   = {}
        el_vals_fin   = {}
        for tel in techs:
            sub = df[df["tech_el"]==tel]
            if len(sub):
                el_vals_bdd[tel] = float(sub[f"{tel}_CAPEX_BDD_USD_kW"].mean())
                el_vals_fin[tel] = float(sub[el_capex_cols[tel]].mean())

        x2 = np.arange(len(techs))
        axes2[0].bar(x2-0.2, [el_vals_bdd.get(t,0) for t in techs], 0.35,
                     label="BDD brut T2 [USD/kW]",  color="#bdc3c7", alpha=0.9)
        axes2[0].bar(x2+0.2, [el_vals_fin.get(t,0) for t in techs], 0.35,
                     label="Actualisé Rozzi [EUR/kW]", color=[colors[t] for t in techs], alpha=0.85)
        axes2[0].set_xticks(x2) ; axes2[0].set_xticklabels(techs)
        axes2[0].set_title("CAPEX Électrolyseurs\n(eq.4 taille + eq.1-3 learning)")
        axes2[0].set_ylabel("USD ou EUR / kW")
        axes2[0].legend(fontsize=8) ; axes2[0].grid(True, alpha=0.3, axis="y")

        # CAPEX ENR par source (scénario central)
        sub_hyb = df[df["source_enr"]=="Hybride"]
        if len(sub_hyb):
            enr_bdd  = [float(sub_hyb["PV_CAPEX_BDD_USD_kW"].mean()),
                        float(sub_hyb["EOL_CAPEX_BDD_USD_kW"].mean())]
            enr_fin  = [float(sub_hyb["PV_CAPEX_fin_EUR_kW"].mean()),
                        float(sub_hyb["EOL_CAPEX_fin_EUR_kW"].mean())]
            x3 = np.arange(2)
            axes2[1].bar(x3-0.2, enr_bdd, 0.35, label="BDD brut T3 [USD/kW]",
                         color="#bdc3c7", alpha=0.9)
            axes2[1].bar(x3+0.2, enr_fin, 0.35, label="Actualisé Rozzi [EUR/kW]",
                         color=["#f39c12","#1abc9c"], alpha=0.85)
            axes2[1].set_xticks(x3) ; axes2[1].set_xticklabels(["PV", "Éolien"])
            axes2[1].set_title("CAPEX Sources ENR\n(eq.4 taille + eq.1-3 learning)")
            axes2[1].set_ylabel("USD ou EUR / kW")
            axes2[1].legend(fontsize=8) ; axes2[1].grid(True, alpha=0.3, axis="y")

        fig2.tight_layout()
        p2 = os.path.join(self.dir_figs, f"Fig_CAPEX_Actualises_{region}_{annee}.png")
        fig2.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  ✅ Fig CAPEX actualisés → {p2}")

    # ─────────────────────────────────────────────────────────────────────────
    # K. VALEURS PAR DÉFAUT
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
            {"technologie":"PEM",  "capex_USD_kW":1413,"capex_min": 900,"capex_max":2000,
             "efficacite_kWh_kg":55.0,"eff_min":47.0,"eff_max":66.0,
             "opex_pct_capex":0.030,"duree_vie_ans":20},
            {"technologie":"AEL",  "capex_USD_kW":1130,"capex_min": 500,"capex_max":1800,
             "efficacite_kWh_kg":52.0,"eff_min":45.0,"eff_max":60.0,
             "opex_pct_capex":0.020,"duree_vie_ans":25},
            {"technologie":"SOEC", "capex_USD_kW":2717,"capex_min":1500,"capex_max":4000,
             "efficacite_kWh_kg":40.0,"eff_min":35.0,"eff_max":45.0,
             "opex_pct_capex":0.035,"duree_vie_ans":20},
        ])

    def _default_T3(self):
        return pd.DataFrame([
            {"technologie":"PV_solaire","capex_USD_kW":550,"capex_min":380,
             "capex_max":900, "opex_USD_kW_an":12,"duree_vie_ans":30,
             "learning_rate_pct_an":-8.0},
            {"technologie":"Eolien",   "capex_USD_kW":1200,"capex_min":900,
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

    # ── 1. Toutes les 9 combinaisons source × tech_el (dimensionnement fixe)
    df_combos = model.run_all_combinaisons(
        pv_size_kW   = 500_000,
        wind_size_kW = 500_000,
        el_size_kW   = 600_000,
        region_nom   = "Dakhla",
        annee        = 2030,
        scenario     = "central",
    )

    # ── 2. Balayage dimensionnement pour Hybride × PEM
    df_sizing = model.run_sizing(
        source_enr    = "Hybride",
        tech_el       = "PEM",
        pv_sizes_kW   = [200_000, 400_000, 600_000, 800_000],
        wind_sizes_kW = [200_000, 400_000, 600_000, 800_000],
        el_sizes_kW   = [300_000, 500_000, 700_000],
        region_nom    = "Dakhla",
        annee         = 2030,
        scenario      = "central",
    )

    # ── 3. Cas unique — PV seul × SOEC (exemple)
    r = model.run_case(
        source_enr   = "PV",
        pv_size_kW   = 800_000,
        wind_size_kW = 0,
        el_size_kW   = 500_000,
        tech_el      = "SOEC",
        region_nom   = "Ouarzazate",
        annee        = 2035,
        scenario     = "optimiste",
    )
    print(f"\n  PV×SOEC Ouarzazate 2035 optimiste :")
    print(f"  H2={r['H2_ann_tonne']:,.0f} t/an | LCOE={r['LCOE_EUR_MWh']:.2f} €/MWh | "
          f"CAPEX_SOEC={r['SOEC_CAPEX_fin_EUR_kW']:.1f} €/kW")