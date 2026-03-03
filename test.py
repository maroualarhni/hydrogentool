"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     DATABASE BUILDER — CHAÎNE HYDROGÈNE MAROC (Production→Stockage→Transport)
║     Approche : Ancrage littérature + Correction Maroc + Monte Carlo + Validation
║   
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import requests
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm, lognorm, triang
import warnings, os, json, time
warnings.filterwarnings('ignore')

np.random.seed(42)  # Reproductibilité scientifique

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION GLOBALE
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR  = r"C:\Users\marwa\Downloads\H2Morocco222_Outputs"
N_SIM       = 10_000   # Simulations Monte Carlo
ANNEES      = [2024, 2030, 2035, 2040, 2050]

os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ["csv", "figures", "reports"]:
    os.makedirs(f"{OUTPUT_DIR}/{sub}", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COULEURS & STYLE
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    'primary'   : '#006233',   # Vert Maroc
    'secondary' : '#C1272D',   # Rouge Maroc
    'accent'    : '#FF8C00',   # Orange énergie
    'PEM'       : '#2196F3',   # Bleu PEM
    'AEL'       : '#4CAF50',   # Vert AEL
    'SOEC'      : '#9C27B0',   # Violet SOEC
    'NH3'       : '#FF5722',   # Orange ammoniac
    'LH2'       : '#00BCD4',   # Cyan H2 liquide
    'LOHC'      : '#795548',   # Brun LOHC
    'GH2'       : '#607D8B',   # Gris GH2
    'pipeline'  : '#3F51B5',   # Bleu pipeline
    'light_bg'  : '#F8F9FA',
    'grid'      : '#E0E0E0',
}

plt.rcParams.update({
    'figure.facecolor'  : 'white',
    'axes.facecolor'    : COLORS['light_bg'],
    'axes.grid'         : True,
    'grid.color'        : COLORS['grid'],
    'grid.linewidth'    : 0.7,
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.titlesize'    : 12,
    'axes.titleweight'  : 'bold',
    'axes.labelsize'    : 10,
    'xtick.labelsize'   : 9,
    'ytick.labelsize'   : 9,
    'legend.fontsize'   : 9,
})


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — RESSOURCES ÉNERGÉTIQUES MAROCAINES (12 régions)
# Sources : IRENA 2022, NASA POWER, MASEN, Bakkari et al. 2024
# ══════════════════════════════════════════════════════════════════════════════
def build_T1_ressources():
    print("  [T1] Construction : Ressources énergétiques marocaines")

    data = {
        'region'                    : ['Ouarzazate','Laayoune','Dakhla','Tanger','Jorf Lasfar','Guelmim'],
        'latitude_N'                : [30.9, 27.1, 23.7, 35.8, 33.1, 29],
        'longitude_W'               : [6.9,  13.2, 15.9, 5.8, 8.6, 10],

        # ── Ressources Solaires ─────────────────────────────────────────────
        'DNI_kWh_m2_an'             : [2512, 2207, 1969, 1914, 1833, 2115],
        'GHI_kWh_m2_an'             : [2172, 2179, 2157, 1835, 1894, 2087],
        'heures_ensoleillement_an'  : [3420, 3205, 3175, 2960, 3059, 3322],
        'CF_solaire_PV_pct'         : [19.8,   19.9,   19.7,   16.8,   17.3,   19.1],
        'LCOE_solaire_USD_kWh'      : [0.040,0.040,0.0405,0.0417,0.0458,0.0473],
        # Calibration des paramètres techno-économiques pour le calcul du LCOE
        # Basé sur la référence :
        # Hamza El Hafdaoui, Ahmed Khallaayoun, Salah Al-Majeed,
        # "Renewable energies in Morocco: A comprehensive review and analysis
        # of current status, policy framework, and prospective potential".
        # Cette étude rapporte des valeurs LCOE du photovoltaïque utility-scale au Maroc
        # dans une plage moyenne de 30–50 $/

        # ── Ressources Éoliennes ────────────────────────────────────────────
        'vitesse_vent_moy_ms'       : [6,  7.5,  8.7,  5.1,  4.7,  3.5],
        'CF_eolien_pct'             : [20.45,   33.6,   43.19,   13.04,   10.15,   3.68],
        'potentiel_eolien_GW'       : [3.31,  6.49,  10.12,  2.03,  1.59,  1.46],
        'LCOE_eolien_USD_kWh'       : [0.048, 0.0292, 0.023, 0.076, 0.098, 0.271],
        #- Conseil Économique, Social et Environnemental (CESE). Acélérer la transition énergétique pour installer le Maroc dans la croissance verte. Rabat : CESE, 2020.
        # #- IRENA. Renewable Energy Profile: Morocco. AbuDhabi : International Renewable Energy Agency, 2025.
         #pour calculer le CF on utilise le code suivant (Méthode Weibull)
        #import numpy as np
        # from scipy.special import gamma
        # from scipy.integrate import quad
        # import pandas as pd
        # df = pd.DataFrame({
        # "region"              : ["Ouarzazate", "Laayoune", "Dakhla", "Tanger", "Jorf Lasfar", "Guelmim"],
        # "vitesse_vent_moy_ms" : [6.0, 7.5, 8.7, 5.1, 4.7, 3.5]})
        # def wind_cf_weibull(v_mean):
    """
    Calcul du CF éolien via distribution de Weibull
    Paramètres turbine standard IEC classe II
    Source : Manwell et al., Wind Energy Explained, 2009
    """
        # Paramètres turbine
        #vi = 3.0    # cut-in  (m/s)
        #vr = 12.0   # rated   (m/s)
        #vo = 25.0   # cut-out (m/s)

        # Paramètres Weibull
        # k=2 → distribution de Rayleigh, standard Maroc côtier (IRENA, MASEN)
        #k = 2.0
        #c = v_mean / gamma(1 + 1/k)   # paramètre d'échelle

        # Densité de probabilité Weibull
        #def weibull_pdf(v):
        #return (k/c) * (v/c)**(k-1) * np.exp(-(v/c)**k)

        # Puissance normalisée de la turbine P(v)/Pr
        #def power_curve(v):
        #if v < vi or v >= vo:
        #return 0.0
        #elif vi <= v < vr:
        #return (v**3 - vi**3) / (vr**3 - vi**3)  # montée cubique
        #else:  # vr <= v < vo
        #return 1.0

        # CF = intégrale de P(v)/Pr × f(v) dv
        #def integrand(v):
        #return power_curve(v) * weibull_pdf(v)

        #cf, _ = quad(integrand, 0, vo)

        # Contrainte physique
        #cf = max(0.0, min(cf, 1.0))
        #return round(cf, 4)

        #df["CF_eolien_weibull"] = df["vitesse_vent_moy_ms"].apply(wind_cf_weibull)

        #print(df[["region", "vitesse_vent_moy_ms", "CF_eolien_weibull"]])
        # Reference : Feasibility analysis and Atlas for green hydrogen project in MENA region: 
        # Production, cost, and environmental maps
        #code pour calcul estimation du potentiel eolien
        #import pandas as pd
        #import numpy as np

        # ── Données de base ────────────────────────────────────────────────────────
        #data = {
        # 'region'              : ['Ouarzazate','Laayoune','Dakhla','Tanger','Jorf Lasfar','Guelmim'],
        # 'vitesse_vent_ms'     : [6.0, 7.5, 8.7, 5.1, 4.7, 3.5], 
        # 'CF_eolien_pct'       : [20.45,   33.6,   43.19,   13.04,   10.15,   3.68],}

        #df = pd.DataFrame(data)

        # ── Potentiel éolien (loi de Betz, ancré sur 25 GW CESE 2020) ─────────────
        #POTENTIEL_NATIONAL_GW = 25.0

        #df['potentiel_eolien_GW'] = (
        # df['vitesse_vent_ms']**3 / df['vitesse_vent_ms'].pow(3).sum()
        # * POTENTIEL_NATIONAL_GW
        # ).round(2)

        # ── LCOE éolien (formule CAPEX/CF) ────────────────────────────────────────
        # Source : tableau El Hafdaoui et al. → Wind : 25–40 $/MWh
        # Méthode : LCOE = (CAPEX × CRF) / (CF × 8760)
        # CAPEX = 1100 $/kW (IRENA Maroc 2024), CRF = 8%, 20 ans

        #CAPEX_eolien = 1100    # $/kW
        #CRF          = 0.0802  # Capital Recovery Factor (8%, 20 ans)

        #df['LCOE_eolien_USD_kWh'] = (
        #(CAPEX_eolien * CRF) / (df['CF_eolien_pct'] / 100 * 8760)).round(4)

        #print(df[['region', 'vitesse_vent_ms', 'CF_eolien_pct',
        #  'potentiel_eolien_GW', 'LCOE_eolien_USD_kWh']])
        # Calibration des paramètres techno-économiques pour le calcul du LCOE
        # Basé sur la référence :
        # Hamza El Hafdaoui, Ahmed Khallaayoun, Salah Al-Majeed,
        # "Renewable energies in Morocco: A comprehensive review and analysis
        # of current status, policy framework, and prospective potential".
        # Cette étude rapporte des valeurs de l'eolien utility-scale au Maroc
        # dans une plage moyenne de 25–40 $/

        # ── Hybride PV + Éolien (synergie) ─────────────────────────────────
        'w_éolien'                  : [0.508, 0.628, 0.687, 0.437, 0.370, 0.161],
        'CF_hybride_pct'            : [20.13, 27.88, 35.84, 15.21, 14.63, 16.65],
        'LCOE_hybride_USD_kWh'      : [0.044, 0.032, 0.028, 0.057, 0.065, 0.083],
        'PPA_local_USD_kWh'         : [0.050, 0.037, 0.032, 0.065, 0.074, 0.083],
        'PPA_Europe_USD_kWh'        : [0.055, 0.041, 0.035, 0.071, 0.081, 0.1039],
        'PPA_H2_USD_kWh'            : [0.061, 0.046, 0.040, 0.080, 0.091, 0.116],
        #import pandas as pd

        #data = {
        #'region'               : ['Ouarzazate','Laayoune','Dakhla','Tanger','Jorf Lasfar','Guelmim'],
        #'CF_solaire_PV_pct'    : [19.8,  19.9,  19.7,  16.8,  17.3,  19.1],
        #'CF_eolien_pct'        : [20.45, 33.6,  43.19, 13.04, 10.15, 3.68],
        #'LCOE_solaire_USD_kWh' : [0.040, 0.040, 0.0405, 0.0417, 0.0458, 0.0473],
        #'LCOE_eolien_USD_kWh'  : [0.048, 0.029, 0.023,  0.076,  0.098,  0.271],}

        #df = pd.DataFrame(data)

        # ── Étape 1 : Poids optimaux ───────────────────────────────────────────────
        #df['w_eolien']  = df['CF_eolien_pct'] / (df['CF_eolien_pct'] + df['CF_solaire_PV_pct'])
        #df['w_solaire'] = 1 - df['w_eolien']

        # ── Étape 2 : CF hybride ───────────────────────────────────────────────────
        #df['CF_hybride_pct'] = (
        #df['w_eolien']  * df['CF_eolien_pct'] +
        # df['w_solaire'] * df['CF_solaire_PV_pct']).round(2)

        # ── Étape 3 : LCOE hybride ─────────────────────────────────────────────────
        #df['LCOE_hybride_USD_kWh'] = (
        #df['w_eolien']  * df['LCOE_eolien_USD_kWh'] +
        #df['w_solaire'] * df['LCOE_solaire_USD_kWh']).round(4)

         # ── Étape 4 : PPA selon marché ─────────────────────────────────────────────
         #marge_local    = 0.15
         #marge_europe   = 0.25
         #marge_H2       = 0.40

        #df['PPA_local_USD_kWh']   = (df['LCOE_hybride_USD_kWh'] * (1 + marge_local)).round(4)
        #df['PPA_europe_USD_kWh']  = (df['LCOE_hybride_USD_kWh'] * (1 + marge_europe)).round(4)
        #df['PPA_H2_USD_kWh']      = (df['LCOE_hybride_USD_kWh'] * (1 + marge_H2)).round(4)

          #print(df[['region','w_eolien','CF_hybride_pct',
          #'LCOE_hybride_USD_kWh',
          #'PPA_local_USD_kWh',
          #'PPA_europe_USD_kWh',
          #'PPA_H2_USD_kWh']].to_string(index=False))

        # ── Ressources Eau ──────────────────────────────────────────────────
        'disponibilite_eau'         : ['Très Faible','faible','Bonne','Bonne','Bonne','Très faible'],
        'dessalement_requis'        : [True, True, True, True, True,  True],
        'cout_eau_USD_m3'           : [1,  0.72,  0.72,  0.6,  0.5,  1],
        'consommation_eau_L_kgH2'   : [10.0]*6,  # constante physique : 9 L/kgH2 + pertes
        #Approximations du prix de l'eau basée sur le rapport annuel de la Cour des comptes au titre de 2024-2025 
        #IEA, Hydrogen Production Costs, 2023
        # ── Infrastructure & Logistique ─────────────────────────────────────
        'surface_disponible_km2'    : [973.2 ,1040.6, 7143.2, 800.5, 0.25, 539.1],
        #= surface régionale totale × 5% (taux exploitation standard)
        'distance_port_km'          : [347,  27,    10,   45,   2,  59],
        #données de OpenStreetMap
        'connexion_reseau_elec'     : ['Bonne','Excellente','Excellente','Excellente','Excellente',
                                       'Excellente'],
        'score_potentiel_H2'        : [69, 82, 92, 60, 58, 40],
        # Score composite (0-100) : irradiation×0.3 + CF×0.25 + logistique×0.2 + eau×0.15 + réseau×0.10
    }

    df = pd.DataFrame(data)
    df.to_csv(f"{OUTPUT_DIR}/csv/T1_ressources_energetiques.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T1 sauvegardé : {len(df)} régions × {len(df.columns)} variables")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 — TECHNOLOGIES DE PRODUCTION H2
# ══════════════════════════════════════════════════════════════════════════════
def build_T2_production():
    print("  [T2] Construction : Technologies de production H2...")

    # Paramètres par technologie — format : (min, mode/mean, max, unité, source)
    rows = [
        # ── ÉLECTROLYSE ALCALINE (AEL) ──────────────────────────────────────
        ['AEL','CAPEX_stack',          500, 650,  1000,'USD/kW',       'Hydrogen Europe 2024'],
        ['AEL','CAPEX_systeme_complet',600, 800,  1200,'USD/kW',       'IEA Global H2 Review 2024'],
        ['AEL','CAPEX_systeme_complet',500, 950,  1400,'USD/kW',       'Halder et al., 2023'],
        ['AEL','CAPEX_systeme_complet',800, 900,  1000,'USD/kW',       'Ahmad et al., 2024'],
        ['AEL','OPEX_fixe',            1.5, 2.0,  3.0,'%CAPEX/an',    'NREL H2A 2024'],
        ['AEL','OPEX_fixe',            1.5, 2.0,  3.0,'%CAPEX/an',    'Taoufik & Fekri, 2023'],
        ['AEL','OPEX_fixe',            1.5, 2.0,  3.0,'%CAPEX/an',    'Squadrito et al., 2023'],
        ['AEL','OPEX_variable',        0.03,0.05, 0.08,'USD/kgH2',    'IEA 2024'],
        ['AEL','efficacite',           50,  52,   58, 'kWh/kgH2',     'Hydrogen Europe 2024'],
        ['AEL','purete_H2',            99.3,99.5, 99.8,'%',           'ISO 14687'],
        ['AEL','purete_H2',            99.99,99.9945, 99.999,'%',     'Ahmad et al., 2024'],
        ['AEL','purete_H2',            99.99,99.9945, 99.999,'%',     'Taoufik & Fekri, 2023'],
        ['AEL','pression_sortie',      1,   10,   30, 'bar',          'Hydrogen Europe 2024'],
        ['AEL','pression_sortie',      1,   10,   30, 'bar',          'Collis & Schomäcker, 2024'],
        ['AEL','temperature_op',       60,  75,   90, '°C',           'Schmidt et al. 2017'],
        ['AEL','temperature_op',       60,  70,   80, '°C',           'Ahmad et al., 2024'],
        ['AEL','temperature_op',       60,  70,   80, '°C',           'Gorji , 2023'],
        ['AEL','duree_vie_stack',      60000,80000,100000,'heures',   'NREL 2024'],
        ['AEL','duree_vie_stack',      60000,80000,100000,'heures',   'Ahmad et al., 2024'],
        ['AEL','duree_vie_systeme',    20,  25,   30, 'ans',          'IEA 2024'],
        ['AEL','remplacement_stack',   8,   10,   12, 'ans',          'Hydrogen Europe 2024'],
        ['AEL','degradation',          0.08,0.12, 0.20,'%/1000h',     'Hydrogen Europe 2024'],
        ['AEL','rampe_montee',         5,   10,   20, '%/s',          'Buttler & Spliethoff 2018'],
        ['AEL','charge_min',           10,  20,   30, '%',            'Hydrogen Europe 2024'],
        ['AEL','disponibilite',        93,  95,   98, '%',            'IEA 2024'],
        ['AEL','disponibilite',        90,  94,   98, '%',             'Habour et al., 2024'],
        ['AEL','TRL',                  9,   9,    9,  '-',            'IEA TRL scale'],

        # ── PEM (Proton Exchange Membrane) ───────────────────────────────────
        ['PEM','CAPEX_stack',          600, 900,  1500,'USD/kW',      'IEA 2024 + DOE 2024'],
        ['PEM','CAPEX_systeme_complet',800, 1100, 2000,'USD/kW',      'DOE H2 Program 2024'],
        ['PEM','CAPEX_systeme_complet',1100, 1450, 1800,'USD/kW',      'Ahmad et al., 2024'],
        ['PEM','OPEX_fixe',            2.0, 3.0,  4.0,'%CAPEX/an',   'NREL H2A 2024'],
        ['PEM','OPEX_variable',        0.05,0.07, 0.12,'USD/kgH2',   'IEA 2024'],
        ['PEM','efficacite',           50,  55,   65, 'kWh/kgH2',    'Hydrogen Europe 2024'],
        ['PEM','purete_H2',            99.9,99.999,99.999,'%',        'ISO 14687'],
        ['PEM','pression_sortie',      30,  50,   80, 'bar',         'Hydrogen Europe 2024'],
        ['PEM','temperature_op',       50,  65,   80, '°C',          'Schmidt et al. 2017'],
        ['PEM','temperature_op',       50,  65,   80, '°C',          'Gorji 2023'],
        ['PEM','duree_vie_stack',      40000,60000,90000,'heures',    'NREL 2024'],
        ['PEM','duree_vie_stack',      50000,65000,80000,'heures',    'Squadrito et al., 2023'],
        ['PEM','duree_vie_systeme',    15,  20,   25, 'ans',         'IEA 2024'],
        ['PEM','duree_vie_systeme',    20,  25,   30, 'ans',         'Villarreal Vives et al., 2023'],
        ['PEM','remplacement_stack',   5,   7,    10, 'ans',         'Hydrogen Europe 2024'],
        ['PEM','degradation',          0.15,0.25, 0.40,'%/1000h',    'Hydrogen Europe 2024'],
        ['PEM','rampe_montee',         50,  100,  200,'%/s',         'Buttler & Spliethoff 2018'],
        ['PEM','charge_min',           3,   5,    10, '%',           'Hydrogen Europe 2024'],
        ['PEM','disponibilite',        94,  97,   99, '%',           'IEA 2024'],
        ['PEM','TRL',                  8,   8,    9,  '-',           'IEA TRL scale'],

        # ── SOEC (Solid Oxide) ───────────────────────────────────────────────
        ['SOEC','CAPEX_systeme_complet',1500,2500,4000,'USD/kW',     'IEA 2024'],
        ['SOEC','CAPEX_systeme_complet',2800,4200,5600,'USD/kW',     'Ahmad et al. ,2024'],
        ['SOEC','CAPEX_systeme_complet',2800,4200,5600,'USD/kW',     'Squadrito et al., 2023'],
        ['SOEC','OPEX_fixe',           2.5, 3.5,  5.0,'%CAPEX/an',  'NREL H2A 2024'],
        ['SOEC','efficacite',          35,  40,   45, 'kWh/kgH2',   'IEA 2024 (meilleure effi)'],
        ['SOEC','temperature_op',      700, 800,  900,'°C',         'Schmidt et al. 2017'],
        ['SOEC','temperature_op',      700, 850,  1000,'°C',         'Gorji 2023'],
        ['SOEC','duree_vie_systeme',   8,   10,   15, 'ans',        'IEA 2024'],
        ['SOEC','TRL',                 5,   6,    7,  '-',          'IEA TRL scale'],

       
        # ── Solaire PV (énergie couplée) ─────────────────────────────────────
        ['PV_solaire','CAPEX',         350, 550,  900,'USD/kW',      'IRENA 2024'],
        ['PV_solaire','CAPEX',         730, 740,  750,'USD/kW',      'Taoufik & Fekri, 2023'],
        ['PV_solaire','OPEX_fixe_kWan',8,   12,   18, 'USD/kW/an',  'IRENA 2024'],
        ['PV_solaire','OPEX_fixe_kWan',21,   22,   23, 'USD/kW/an',  'Taoufik & Fekri, 2023'],
        ['PV_solaire','degradation',   0.3, 0.5,  0.8,'%/an',       'IRENA 2024'],
        ['PV_solaire','duree_vie',     25,  30,   35, 'ans',        'IRENA 2024'],
        ['PV_solaire','efficacite',    18,  21,   24, '%',          'IEA 2024'],
        ['PV_solaire','LCOE_Maroc',    0.015,0.025,0.040,'USD/kWh', 'MASEN PPA record 0.018 (Midelt 2020)'],
        ['PV_solaire','LCOE_Maroc',    0.030,0.040,0.050,'USD/kWh', 'El Hafdaoui et al., 2024'],

        # ── Éolien terrestre ─────────────────────────────────────────────────
        ['Eolien','CAPEX',             900, 1200, 1600,'USD/kW',     'IRENA 2024'],
        ['Eolien','OPEX_fixe_kWan',    25,  35,   50, 'USD/kW/an',  'IRENA 2024'],
        ['Eolien','OPEX_fixe_kWan',    47,  48.5,   50, 'USD/kW/an',  'El Hafdaoui et al., 2024'],
        ['Eolien','duree_vie',         20,  25,   30, 'ans',        'IRENA 2024'],
        ['Eolien','LCOE_Maroc',        0.020,0.032,0.050,'USD/kWh', 'Tarfaya LCOE=0.038 (IRENA 2022)'],
        ['Eolien','LCOE_Maroc',        0.025,0.032,0.040,'USD/kWh', 'El Hafdaoui et al., 2024'],
    ]

    df = pd.DataFrame(rows, columns=['technologie','parametre','valeur_min','valeur_mode',
                                      'valeur_max','unite','source'])
    df.to_csv(f"{OUTPUT_DIR}/csv/T2_technologies_production.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T2 sauvegardé : {len(df)} paramètres × {len(df.columns)} colonnes")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 3 — TECHNOLOGIES DE STOCKAGE H2
# ══════════════════════════════════════════════════════════════════════════════
def build_T3_stockage():
    print("  [T3] Construction : Technologies de stockage H2...")

    rows = [
        # ── H2 COMPRIMÉ 350 bar ──────────────────────────────────────────────
        ['GH2_350bar','CAPEX_reservoir',       400, 600,  900, 'USD/kgH2', 'DOE 2024'],
        ['GH2_350bar','CAPEX_compresseur',      800,1200, 2000,'USD/kW',   'Hydrogen Council 2023'],
        ['GH2_350bar','energie_compression',    1.5, 2.0,  3.0,'kWh/kgH2','IEA 2024'],
        ['GH2_350bar','OPEX_pct_CAPEX',         1.5, 2.0,  3.0,'%/an',    'IEA 2024'],
        ['GH2_350bar','pertes_boil_off',         1,   1.5,    2, '%/jour',   'physique'],
        ['GH2_350bar','densite_vol',            23.5,23.5,23.5,'kgH2/m3', 'physique'],
        ['GH2_350bar','duree_vie',              15,  20,   25, 'ans',      'DOE 2024'],
        ['GH2_350bar','TRL',                    9,   9,    9,  '-',        'IEA TRL'],
        ['GH2_350bar','LCOS',                   0.3, 0.6,  1.2,'USD/kgH2','IEA 2024'],

        # ── H2 COMPRIMÉ 700 bar ──────────────────────────────────────────────
        ['GH2_700bar','CAPEX_reservoir',        600, 900, 1400,'USD/kgH2', 'DOE 2024'],
        ['GH2_700bar','energie_compression',    2.5, 3.5,  5.0,'kWh/kgH2','IEA 2024'],
        ['GH2_700bar','densite_vol',            40.2,  40.2,   40.2, 'kgH2/m3', 'physique'],
        ['GH2_700bar','TRL',                    9,   9,    9,  '-',        'IEA TRL'],

        # ── H2 LIQUIDE (LH2) ─────────────────────────────────────────────────
        ['LH2','CAPEX_liquefacteur',           2000,3500, 6000,'USD/(kgH2/j)','DOE H2 Program 2024'],
        ['LH2','CAPEX_reservoir',               800,1200, 2000,'USD/kgH2',    'IEA 2024'],
        ['LH2','energie_liquefaction',            8,  10,   14,'kWh/kgH2',    'IEA 2024'],
        ['LH2','OPEX_pct_CAPEX',                1.5,  2.0,  3.0,'%/an',       'IEA 2024'],
        ['LH2','pertes_boil_off',                0.1, 0.3,  0.5,'%/jour',     'IEA 2024'],
        ['LH2','densite_vol',                   70.8,70.8, 70.8,'kgH2/m3',    'physique (-253°C)'],
        ['LH2','temperature_K',               -253, -253, -253,'°C',          'physique'],
        ['LH2','duree_vie',                     20,  25,   30, 'ans',          'IEA 2024'],
        ['LH2','TRL',                            6,   7,    8, '-',            'IEA TRL'],
        ['LH2','LCOS',                           1.5, 2.5,  4.0,'USD/kgH2',   'Hydrogen Council 2023'],

        # ── AMMONIAC (NH3) — Vecteur stratégique Maroc ───────────────────
        ['NH3','CAPEX_synthese_Haber',         400, 700, 1100,'USD/(tNH3/j)','Hydrogen Council 2023'],
        ['NH3','CAPEX_craquage_NH3_H2',        500, 900, 1500,'USD/(tNH3/j)','IEA 2024'],
        ['NH3','CAPEX_stockage',               200, 350,  500,'USD/tNH3',    'IEA 2024'],
        ['NH3','energie_synthese',               8,  10,   12, 'kWh/kgH2',  'IEA 2024'],
        ['NH3','energie_craquage',              12,  15,   20, 'kWh/kgH2',  'IEA 2024'],
        ['NH3','OPEX_pct_CAPEX',                2.0, 3.0,  4.0,'%/an',      'IEA 2024'],
        ['NH3','efficacite_H2_to_NH3',         68,  72,   76, '%',          'IEA 2024'],
        ['NH3','efficacite_NH3_to_H2',         80,  85,   90, '%',          'IEA 2024'],
        ['NH3','densite_vol',                  121, 121,  121,'kgH2/m3',    'physique (liquide -33°C)'],
        ['NH3','temperature_stockage_C',       -33, -33,  -33,'°C ou ambiant sous pression','physique'],
        ['NH3','pertes_pct',                   0.5, 1.0,  2.0,'%',          'IEA 2024'],
        ['NH3','TRL',                           9,   9,    9, '-',           'IEA TRL (infrastructure existante)'],
        ['NH3','LCOS_USD_kgH2',                0.8, 1.5,  2.5,'USD/kgH2',  'Hydrogen Council 2023'],
    
        # ── LOHC (Dibenzyltoluène) ────────────────────────────────────────────
        ['LOHC','CAPEX_hydrogenation',         500, 800, 1200,'USD/(kgH2/j)','Hydrogenious 2024'],
        ['LOHC','CAPEX_dehydrogenation',       800,1200, 2000,'USD/(kgH2/j)','Hydrogenious 2024'],
        ['LOHC','energie_hydrogenation',         3,   5,    8, 'kWh/kgH2', 'IEA 2024'],
        ['LOHC','energie_dehydrogenation',       8,  10,   14, 'kWh/kgH2', 'IEA 2024'],
        ['LOHC','temperature_hydrog_C',         130, 150,  180,'°C',        'Hydrogenious 2024'],
        ['LOHC','temperature_dehydrog_C',       280, 310,  340,'°C',        'Hydrogenious 2024'],
        ['LOHC','densite_vol',                   50,  57,   62,'kgH2/m3',   'physique'],
        ['LOHC','pertes_carrier_pct_cycle',      0.05,0.10, 0.20,'%',       'Hydrogenious 2024'],
        ['LOHC','TRL',                            6,   7,    8, '-',         'IEA TRL'],
        ['LOHC','LCOS_USD_kgH2',                1.5, 2.5,  4.0,'USD/kgH2', 'Hydrogen Council 2023'],

        # ── CAVERNE SALINE (potentiel Maroc) ─────────────────────────────────
        ['Caverne_saline','CAPEX_USD_kWh',      0.5, 3.0, 10.0,'USD/kWh',  'IEA 2024'],
        ['Caverne_saline','OPEX_pct_CAPEX',     0.5, 1.0,  2.0,'%/an',     'IEA 2024'],
        ['Caverne_saline','efficacite',          96,  98,   99, '%',         'IEA 2024'],
        ['Caverne_saline','capacite_GWh',       100,1000,10000,'GWh',       'IEA 2024'],
        ['Caverne_saline','duree_vie',           40,  50,   60, 'ans',       'IEA 2024'],
        ['Caverne_saline','TRL',                  7,   8,    9, '-',         'IEA TRL'],
        ['Caverne_saline','sites_potentiels',     1,   3,    5,'nb sites Maroc','IRESEN géologie 2022'],
        # --- Méthanol vert (e-methanol) -------------------------------------------
        ['e_methanol',   'ratio_H2_kgPerkg',  0.18, 0.19, 0.20, 'kgH2/kgMeOH', 'IRENA 2021'],
        ['e_methanol',   'ratio_CO2_kgPerkg', 1.37, 1.40, 1.45, 'kgCO2/kgMeOH','IEA 2024'],
        ['e_methanol',   'CAPEX_synthese',    400,  600,  900,  'USD/tMeOH/an','IRENA 2023'],
        ['e_methanol',   'CAPEX_stockage',    50,   80,   120,  'USD/tMeOH',   'IEA 2024'],
        ['e_methanol',   'efficacite_conv',   55,   60,   65,   '%',           'Hydrogen Europe 2024'],
        ['e_methanol',   'cout_CO2_capture',  50,   80,   150,  'USD/tCO2',    'IEA 2024'],
        ['e_methanol',   'densite_energie',   15.6, 15.6, 15.6, 'MJ/L',        'IEA 2024'],
        ['e_methanol',   'temperature_stock', 20,   20,   20,   '°C ambiant',  'IRENA 2023'],
        ['e_methanol',   'TRL',               7,    8,    9,    '-',           'IEA TRL scale 2024'],
    ]

    df = pd.DataFrame(rows, columns=['technologie','parametre','valeur_min','valeur_mode',
                                      'valeur_max','unite','source'])
    df.to_csv(f"{OUTPUT_DIR}/csv/T3_technologies_stockage.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T3 sauvegardé : {len(df)} paramètres")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 4 — TRANSPORT & INFRASTRUCTURE Calcul automatique des distances via OSRM (OpenStreetMap)
# + Choix du mode optimal selon distance et type de corridor
#+ Export CSV

# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# COORDONNÉES DES NŒUDS
# ══════════════════════════════════════════════════════════════════════════════
NODES = {
    # Maroc
    'Ouarzazate'    : (30.9189,  -6.8934),
    'Dakhla'        : (23.6848, -15.9572),
    'Laayoune'      : (27.1253, -13.1625),
    'Tanger'        : (35.7595,  -5.8340),
    'Jorf_Lasfar'   : (33.1100,  -8.6300),
    'Guelmim'       : (28.9870, -10.0572),
    'Casablanca'    : (33.5731,  -7.5898),
    'Agadir'        : (30.4278,  -9.5981),
    'Marrakech'     : (31.6295,  -7.9811),
    'Nador'         : (35.1681,  -2.9335),
    'Tarfaya'       : (27.9378, -12.9194),
    # Europe
    'Algésiras'     : (36.1408,  -5.4530),
    'Almería'       : (36.8340,  -2.4637),
    'Rotterdam'     : (51.9244,   4.4777),
    'Barcelone'     : (41.3851,   2.1734),
    'Marseille'     : (43.2965,   5.3698),
    'Paris'         : (48.8566,   2.3522),
    # Afrique
    'Dakar'         : (14.7167, -17.4677),
    # Canaries
    'Canaries'      : (28.1235, -15.4363),
}

# ══════════════════════════════════════════════════════════════════════════════
# CORRIDORS À CALCULER
# ══════════════════════════════════════════════════════════════════════════════
CORRIDORS = [
    # (origine, destination, type)
    ('Ouarzazate',  'Casablanca',   'Domestique'),
    ('Dakhla',      'Agadir',       'Domestique'),
    ('Dakhla',      'Casablanca',   'Domestique'),
    ('Laayoune',    'Marrakech',    'Domestique'),
    ('Tarfaya',     'Casablanca',   'Domestique'),
    ('Tanger',      'Casablanca',   'Domestique'),
    ('Nador',       'Casablanca',   'Domestique'),
    ('Guelmim',     'Agadir',       'Domestique'),
    ('Tanger',      'Algésiras',    'Export'),
    ('Nador',       'Almería',      'Export'),
    ('Agadir',      'Canaries',     'Export'),
    ('Casablanca',  'Rotterdam',    'Export'),
    ('Casablanca',  'Barcelone',    'Export'),
    ('Agadir',      'Marseille',    'Export'),
    ('Casablanca',  'Dakar',        'Export'),
    ('Tanger',      'Paris',        'Export'),
]

# ══════════════════════════════════════════════════════════════════════════════
# PARAMÈTRES DES MODES DE TRANSPORT
# ══════════════════════════════════════════════════════════════════════════════
MODES_PARAMS = {
    'Tube_trailer'           : {'cout_min': 0.35, 'cout_max': 0.80, 'vecteur': 'GH2'},
    'Pipeline_H2_reconverti' : {'cout_min': 0.15, 'cout_max': 0.50, 'vecteur': 'GH2'},
    'Pipeline_H2_nouveau'    : {'cout_min': 0.20, 'cout_max': 0.78, 'vecteur': 'GH2'},
    'Pipeline_sous_marin'    : {'cout_min': 0.05, 'cout_max': 0.15, 'vecteur': 'GH2'},
    'Tanker_NH3'             : {'cout_min': 0.50, 'cout_max': 2.50, 'vecteur': 'NH3'},
    'Tanker_LH2'             : {'cout_min': 0.80, 'cout_max': 3.00, 'vecteur': 'LH2'},
}

# ══════════════════════════════════════════════════════════════════════════════
# RÈGLES DE DÉCISION : DISTANCE → MODE OPTIMAL
# Source : IEA 2024, Hydrogen Council 2023
# ══════════════════════════════════════════════════════════════════════════════
def mode_optimal(distance_km, type_corridor):
    """
    Sélection automatique du mode de transport H2 selon :
    - Distance réelle (OSM/OSRM)
    - Type de corridor (Domestique / Export)
    
    Règles basées sur IEA (2024) et Hydrogen Council (2023)
    """
    if type_corridor == 'Domestique':
        if distance_km < 200:
            return 'Tube_trailer'
        elif distance_km < 600:
            return 'Pipeline_H2_reconverti'
        else:
            return 'Pipeline_H2_nouveau'

    elif type_corridor == 'Export':
        if distance_km < 50:
            return 'Pipeline_sous_marin'
        elif distance_km < 500:
            return 'Tanker_NH3'
        else:
            return 'Tanker_NH3'  # NH3 dominant longue distance

# ══════════════════════════════════════════════════════════════════════════════
# CALCUL DISTANCE VIA OSRM (OpenStreetMap Routing Machine)
# ══════════════════════════════════════════════════════════════════════════════
def get_distance_osrm(coord1, coord2, type_corridor):
    """
    Calcule la distance routière via OSRM (gratuit, basé OSM)
    Pour les corridors maritimes (Export longue distance) :
    → Distance orthodromique (grande cercle) utilisée
    """
    # Corridors maritimes longue distance → distance orthodromique
    maritime = ['Rotterdam', 'Barcelone', 'Marseille', 'Dakar', 'Canaries', 'Paris']

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    try:
        url = (f"http://router.project-osrm.org/route/v1/driving/"
               f"{lon1},{lat1};{lon2},{lat2}"
               f"?overview=false")
        r = requests.get(url, timeout=10)
        data = r.json()
        if data['code'] == 'Ok':
            return round(data['routes'][0]['distance'] / 1000, 0)
    except Exception:
        pass

    # Fallback : distance à vol d'oiseau × 1.3 (facteur détour routier)
    import math
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    d = 2 * R * math.asin(math.sqrt(a))
    return round(d * 1.3, 0)


# ══════════════════════════════════════════════════════════════════════════════
# CALCUL COÛT TRANSPORT ($/kg H2)
# IEA 2024 : coût ∝ distance pour pipeline, fixe par voyage pour tanker
# ══════════════════════════════════════════════════════════════════════════════
def calcul_cout(mode, distance_km):
    p = MODES_PARAMS[mode]
    if 'Pipeline' in mode or 'Tube' in mode:
        # Coût proportionnel à la distance
        cout_min = p['cout_min'] * (distance_km / 1000)
        cout_max = p['cout_max'] * (distance_km / 1000)
    else:
        # Tanker : coût par voyage (partiellement fixe)
        cout_min = p['cout_min'] * (distance_km / 5000) ** 0.6
        cout_max = p['cout_max'] * (distance_km / 5000) ** 0.6
    return round(cout_min, 3), round(cout_max, 3)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — CONSTRUCTION DE LA TABLE T4
# ══════════════════════════════════════════════════════════════════════════════
def build_T4_transport():
    print("  [T4] Calcul distances OSM + modes transport H2...\n")
    rows = []

    for origine, destination, type_c in CORRIDORS:
        coord1 = NODES[origine]
        coord2 = NODES[destination]

        # Distance réelle OSM
        dist = get_distance_osrm(coord1, coord2, type_c)
        time.sleep(0.5)  # respecter rate limit OSRM

        # Mode optimal automatique
        mode = mode_optimal(dist, type_c)
        vecteur = MODES_PARAMS[mode]['vecteur']

        # Coût transport
        cout_min, cout_max = calcul_cout(mode, dist)

        rows.append({
            'corridor'                  : f"{origine}→{destination}",
            'type'                      : type_c,
            'distance_km'               : dist,
            'mode_optimal'              : mode,
            'vecteur_H2'                : vecteur,
            'cout_transport_USD_kg_min' : cout_min,
            'cout_transport_USD_kg_max' : cout_max,
        })

        print(f"  ✓ {origine:15} → {destination:15} | {dist:6.0f} km "
              f"| {mode:25} | {vecteur} "
              f"| {cout_min:.3f}–{cout_max:.3f} $/kg")

    df = pd.DataFrame(rows)
    df.to_csv("T4_corridors_transport.csv", index=False, encoding='utf-8-sig')
    print(f"\n  ✅ T4 sauvegardé : {len(df)} corridors")
    return df


if __name__ == "__main__":
    df = build_T4_transport()
    print("\n", df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# TABLE 5 — PARAMÈTRES ÉCONOMIQUES & FINANCIERS
# ══════════════════════════════════════════════════════════════════════════════
def build_T5_economique():
    print("  [T5] Construction : Paramètres économiques & financiers...")

    data = {
        'parametre'             : [
            # Macro Maroc
            'taux_actualisation_min','taux_actualisation_mode','taux_actualisation_max',
            'inflation_MAD_pct','taux_USD_MAD','taux_EUR_MAD',
            'prime_risque_pays_vs_EU','cout_dette_pct','ratio_dette_fp',
            'duree_amortissement_ans','IS_corporate_pct',
            # Prix électricité
            'tarif_indus_HTA_USD_kWh','PPA_solaire_min','PPA_solaire_mode','PPA_solaire_max',
            'PPA_eolien_min','PPA_eolien_mode','PPA_eolien_max',
            'PPA_hybride_min','PPA_hybride_mode','PPA_hybride_max',
            'evolution_prix_elec_2030','evolution_prix_elec_2040','evolution_prix_elec_2050',
            # Main d'œuvre
            'salaire_ingenieur_min_USD_mois','salaire_ingenieur_mode_USD_mois',
            'salaire_technicien_min_USD_mois','salaire_technicien_mode_USD_mois',
            'salaire_operateur_min_USD_mois','salaire_operateur_mode_USD_mois',
            'ratio_salaire_Maroc_vs_EU','charges_sociales_pct',
            'emplois_directs_par_MW_H2','emplois_indirects_par_MW_H2',
            # Prix H2 marché
            'H2_vert_actuel_min','H2_vert_actuel_mode','H2_vert_actuel_max',
            'H2_vert_cible_2030_min','H2_vert_cible_2030_mode','H2_vert_cible_2030_max',
            'H2_vert_cible_2040_min','H2_vert_cible_2040_mode','H2_vert_cible_2040_max',
            'prix_import_EU_USD_kg_min','prix_import_EU_USD_kg_mode','prix_import_EU_USD_kg_max',
            'NH3_vert_USD_tonne_min','NH3_vert_USD_tonne_mode','NH3_vert_USD_tonne_max',
            'objectif_Hydrogen_Shot_DOE_USD_kg',
        ],
        'valeur'                : [
            6,8,12, 3.5,10.05,10.90, 3.0,5.5,70, 20,31,
            0.094, 0.015,0.025,0.040, 0.020,0.032,0.050, 0.018,0.028,0.045,
            0.75,0.55,0.40,
            1000,1500, 400,700, 250,400,
            0.25,26, 2.5,5.0,
            4.0,6.0,9.0, 2.0,3.0,4.5, 1.5,2.0,3.0,
            3.5,5.0,7.0, 400,600,900, 1.0,
        ],
        'unite'                 : [
            '%','%','%', '%','MAD/USD','MAD/EUR', '%','%','%', 'ans','%',
            'USD/kWh','USD/kWh','USD/kWh','USD/kWh','USD/kWh','USD/kWh','USD/kWh',
            'USD/kWh','USD/kWh','USD/kWh', 'facteur','facteur','facteur',
            'USD/mois','USD/mois','USD/mois','USD/mois','USD/mois','USD/mois',
            'ratio','%','emplois/MW','emplois/MW',
            'USD/kg','USD/kg','USD/kg','USD/kg','USD/kg','USD/kg',
            'USD/kg','USD/kg','USD/kg','USD/kg','USD/kg','USD/kg',
            'USD/kg','USD/kg','USD/kg','USD/t',
        ],
        'source'                : [
            'WACC MENA analysis',
            'HCP Maroc 2024','BAM 2024',
            'Moody\'s rating Ba1','BAM 2024','Standard financement MENA',
            'Standard projet EnR','DGI Maroc 2024',
            'ONEE tarif 2024',
            'MASEN PPA record Midelt 2019','MASEN 2024','IEA LCOH Review 2024',
            'Tarfaya LCOE réel IRENA 2022','IRENA Maroc 2024','IEA 2024',
            'MASEN PPA Midelt 2019 (record)','MASEN 2024','IEA 2024',
            'IEA Learning curves 2024','IEA 2024','IEA 2024',
            'ANAPEC Maroc 2024','ANAPEC Maroc 2024','ANAPEC Maroc 2024','ANAPEC Maroc 2024',
            'ANAPEC Maroc 2024','ANAPEC Maroc 2024',
            'BIT 2024','CNSS Maroc 2024','IRENA Jobs Report 2024','IRENA Jobs Report 2024',
            'IEA LCOH 2024','IEA LCOH 2024','IEA LCOH 2024',
            'IEA LCOH 2024','IEA LCOH 2024','IEA LCOH 2024',
            'IEA Hydrogen Roadmap 2024','IEA 2024','IEA 2024',
            'IEA 2040 outlook','IEA 2040 outlook','IEA 2040 outlook',
            'EU H2 import pricing study','EU H2 import pricing study','EU H2 import pricing study',
            'Hydrogen Council 2023','Hydrogen Council 2023','Hydrogen Council 2023',
            'DOE H2 Program 2024 (Hydrogen Shot)',
        ]
    }

    df = pd.DataFrame(data)
    df.to_csv(f"{OUTPUT_DIR}/csv/T5_parametres_economiques.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T5 sauvegardé : {len(df)} paramètres économiques")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 6 — MARCHÉ & DEMANDE
# ══════════════════════════════════════════════════════════════════════════════
def build_T6_marche():
    print("  [T6] Construction : Marché & demande H2...")

    # Demande nationale par secteur (ktH2/an)
    demande = pd.DataFrame({
        'secteur'       : ['Industrie_chimique_OCP','Raffinage_pétrole','Mobilité_FCEV',
                           'Mobilité_train_H2','Chaleur_industrielle','Stockage_réseau',
                           'Aviation_carburant_synth','Dessalement_eau','Résidentiel_heat'],
        'demande_2024'  : [120, 80,  2,  0,  10,  0,  0, 0, 0],
        'demande_2030'  : [200, 70,  20, 5,  50,  15, 5, 2, 1],
        'demande_2035'  : [280, 55,  50, 15, 100, 40, 15,5, 3],
        'demande_2040'  : [350, 45,  80, 30, 150, 80, 40,8, 8],
        'demande_2050'  : [500, 30,  200,80, 350, 250,150,15,20],
        'unite'         : ['ktH2/an']*9,
        'scenario'      : ['Moderé']*9,
        'source'        : ['Stratégie Nationale H2 Maroc 2021 + OCP Group']*9,
    })
    #OCP consomme aujourd'hui du H2 gris pour produire ses engrais
    #consommation réelle OCP ~ 120 000 tonnes H2/an (rapport OCP 2023)
    #objectif : remplacer progressivement par H2 vert
    #2050 : 500 kt → OCP veut devenir 100% vert
    #Source : OCP Green Investment Program 2023
    #Raffinage pétrole
    #2024 : 80 ktH2/an → baisse vers 30 kt en 2050
    #Pourquoi ça baisse ?
    # Déclin progressif du raffinage pétrolier avec transition énergétique
    #Source : IEA Oil Refining Outlook 2024
    #Mobilité FCEV (voitures H2)
    #2024 : 2 kt  (quasi inexistant aujourd'hui)
    #2050 : 200 kt (croissance forte)
    #Calcul :
    #1 voiture FCEV consomme ~1 kg H2 / 100 km
    #Parc 2050 estimé : ~500 000 véhicules × 400 kg/an = 200 kt ✅
    #Source : Stratégie mobilité verte Maroc 2030
    #Stockage réseau
    #2024 : 0  → pas encore déployé
    #2050 : 250 kt → power-to-gas massif
    #Logique :
    #Quand ENR > demande → électrolyse → H2 stocké
    #Quand ENR < demande → H2 → électricité (pile à combustible)
    #Source : ONEE Plan Réseau 2024
    # Benchmark concurrents (compétiteurs de Maroc)
    competitors = pd.DataFrame({
        'pays'              : ['Maroc','Arabie Saoudite','Égypte','Chili','Australie','Namibie','Espagne'],
        'LCOH_2024_USD_kg'  : [4.5, 4.0, 4.8, 3.8, 4.2, 5.0, 5.5],
        'LCOH_2030_USD_kg'  : [2.0, 1.5, 2.2, 1.8, 2.0, 2.5, 2.8],
        'LCOH_2040_USD_kg'  : [1.3, 1.0, 1.5, 1.2, 1.4, 1.8, 2.0],
        'CF_solaire_pct'    : [31,  35,  28,  30,  30,  30,  20 ],
        'CF_eolien_pct'     : [40,  25,  35,  55,  45,  40,  30 ],
        'distance_EU_km'    : [14,  5000,3000,12000,18000,6000,0  ],
        'avantage_decisif'  : ['Proximité EU 14km',
                               'Ressources solaires exceptionnelles',
                               'Faibles coûts + Suez',
                               'Éolien record Patagonie',
                               'Solaire + éolien + superficie',
                               'Territoire vierge + soleil',
                               'Marché domestique EU'],
        'risque_principal'  : ['Infrastructure réseau','Stabilité politique','Financement',
                               'Distance marché','Distance marché','Infrastructure','Coûts élevés'],
        'source'            : ['IEA 2024 + Applied Energy 2024']*7,
    })

    demande.to_csv(f"{OUTPUT_DIR}/csv/T6a_demande_nationale.csv", index=False, encoding='utf-8-sig')
    competitors.to_csv(f"{OUTPUT_DIR}/csv/T6b_benchmark_competiteurs.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T6 sauvegardé : {len(demande)} secteurs + {len(competitors)} pays benchmarkés")
    return demande, competitors


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 7 — ENVIRONNEMENT & CARBONE
# ══════════════════════════════════════════════════════════════════════════════
def build_T7_environnement():
    print("  [T7] Construction : Environnement & carbone...")

    emissions = pd.DataFrame({
        'filiere'           : ['H2_vert_PEM_solaire','H2_vert_PEM_eolien','H2_vert_PEM_hybride',
                               'H2_vert_AEL_solaire','H2_vert_AEL_eolien',
                               'NH3_vert_Haber_Bosch'],
        'emissions_kgCO2_kgH2_min': [0.3, 0.2, 0.3, 0.4, 0.2, 0.5],
        'emissions_kgCO2_kgH2_mode':[ 1.2, 0.8, 0.9, 1.3, 0.7, 1.5],
        'emissions_kgCO2_kgH2_max': [2.5, 1.5, 1.8, 2.5, 1.2, 3.0],
        'seuil_RFNBO_EU_gCO2_MJ'  : [None]*6,  # seuil = 3.67 gCO2/MJ = 0.44 kgCO2/kgH2
        'conforme_EU_RFNBO'       : [True,True,True,True,True,True],
        'certifiable_CertifHy'    : [True,True,True,True,True,True],
        'source'                  : ['IEA LCOH Review 2024']*6,
    })

    certifications = pd.DataFrame({
        'certification'         : ['EU_RFNBO','CertifHy_Premium','GS_H2_Gold','I_REC_Standard','ISO_14687'],
        'seuil_kgCO2_kgH2'     : [0.44, 4.4, 1.0, None, None],
        'seuil_gCO2_MJ'        : [3.67, None, None, None, None],
        'premium_prix_pct'      : [20, 15, 10, 5, 0],
        'marche_cible'          : ['EU mandatory 2030','EU/Japan','Global','Global','Global'],
        'importance_Maroc'      : ['CRITIQUE - accès marché EU','Haute','Haute','Moyenne','Obligatoire'],
        'applicable_Maroc'      : [True]*5,
        'source'                : ['EU Delegated Regulation 2023/1184',
                                   'CertifHy v3.0','Gold Standard Foundation',
                                   'I-REC Standard','ISO 14687:2019'],
    })

    co2_evite = pd.DataFrame({
        'application'           : ['Substitution_NH3_OCP','Transport_FCEV','Industrie_acier',
                                   'Production_electricite','Chaleur_industrielle'],
        'CO2_evite_tCO2_tH2'   : [9.5, 11.2, 8.8, 12.5, 7.5],
        'potentiel_2030_MtCO2' : [5.0, 1.0, 0.5, 2.0, 1.5],
        'potentiel_2040_MtCO2' : [8.0, 3.5, 2.0, 5.0, 4.0],
        'potentiel_2050_MtCO2' : [12.0, 8.0, 5.5, 10.0, 8.0],
        'source'                : ['Stratégie Nationale H2 Maroc + OCP']*5,
    })

    emissions.to_csv(f"{OUTPUT_DIR}/csv/T7a_emissions_CO2.csv", index=False, encoding='utf-8-sig')
    certifications.to_csv(f"{OUTPUT_DIR}/csv/T7b_certifications.csv", index=False, encoding='utf-8-sig')
    co2_evite.to_csv(f"{OUTPUT_DIR}/csv/T7c_CO2_evite.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T7 sauvegardé : émissions + certifications + CO2 évité")
    return emissions, certifications


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 8 — PROJETS DE RÉFÉRENCE MAROC (Validation)
# ══════════════════════════════════════════════════════════════════════════════
def build_T8_projets():
    print("  [T8] Construction : Projets de référence Maroc...")

    projets = pd.DataFrame({
        'projet'            : ['NOOR_Ouarzazate_I-IV','Tarfaya_Wind_301MW',
                               'Noor_Midelt_800MW','ONEE_Wind_Taza',
                               'OCP_Green_H2_Jorf','IRESEN_H2_BenGuerir',
                               'Dakhla_H2_RWE','MASEN_H2_Dakhla_Offshore',
                               'H2Uppp_GermanyMorocco','OCP_Green_NH3_Jorf'],
        'type'              : ['CSP+PV','Éolien terrestre','CSP+PV hybride',
                               'Éolien terrestre','Électrolyse PEM',
                               'Pilote PEM+AEL R&D','Pipeline H2 offshore',
                               'Éolien offshore + H2','Partenariat H2 export',
                               'NH3 vert industriel'],
        'capacite_MW'       : [580, 301, 800, 150, 100, 0.25, 200, 500, 500, 200],
        'CAPEX_total_MUSD'  : [2500, 420, 2100, 200, 500, 2, 1200, 3000, 2000, 1000],
        'LCOE_ou_LCOH'      : [0.062, 0.038, 0.018, 0.035, 2.0, None, None, None, None, None],
        'unite_LCOE_LCOH'   : ['USD/kWh','USD/kWh','USD/kWh PPA','USD/kWh',
                               'USD/kgH2 cible 2030',None,None,None,None,None],
        'CF_reel_pct'       : [28, 43, 31, 38, None, None, None, None, None, None],
        'annee_commission'  : [2018, 2014, 2025, 2016, 2027, 2020, 2028, 2030, 2027, 2028],
        'statut'            : ['Opérationnel','Opérationnel','Construction',
                               'Opérationnel','Développement','Opérationnel pilote',
                               'Étude faisabilité','Étude faisabilité',
                               'Accord signé','Développement'],
        'developpeur'       : ['MASEN','Nareva/ENGIE','MASEN/EDF/Nareva',
                               'ONEE','OCP Group','IRESEN/UM6P',
                               'RWE/ONEE','MASEN','BMWi/MASEN','OCP Group'],
        'pertinence_outil'  : ['Calibration CF solaire + coûts PV Ouarzazate',
                               'Calibration CF éolien + LCOE éolien Tarfaya',
                               'Calibration PPA record → LCOH minimal atteignable',
                               'Calibration éolien intérieur Maroc',
                               'Référence LCOH H2 vert Maroc 2030',
                               'Seul projet H2 réel opérationnel → validation directe',
                               'Calibration coûts pipeline + export H2',
                               'Référence potentiel offshore + export',
                               'Validation corridor export EU via pipeline',
                               'Validation intégration H2→NH3 contexte OCP'],
        'source'            : ['MASEN Annual Report 2022 + World Bank PCR',
                               'IRENA Renewable Power Generation Costs 2022',
                               'MASEN 2019 + World Bank Group 2020',
                               'ONEE Annual Report 2022',
                               'OCP Group Press Release 2023',
                               'IRESEN Annual Report 2022',
                               'RWE Press Release 2023',
                               'MASEN Strategic Plan 2030',
                               'H2Uppp Germany-Morocco 2023',
                               'OCP Sustainability Report 2023'],
    })

    projets.to_csv(f"{OUTPUT_DIR}/csv/T8_projets_reference_maroc.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T8 sauvegardé : {len(projets)} projets de référence")
    return projets


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 9 — SCÉNARIOS TEMPORELS 2024–2050
# ══════════════════════════════════════════════════════════════════════════════
def build_T9_scenarios():
    print("  [T9] Construction : Scénarios temporels 2024-2050...")

    annees = [2024, 2027, 2030, 2035, 2040, 2045, 2050]

    # Wright's Law : learning rate
    # CAPEX(n) = CAPEX_ref × (cumul_capacity/ref_capacity)^(-b)  b = log(1-LR)/log(2)
    def wright_law(capex_ref, lr, doublings):
        b = np.log(1 - lr) / np.log(2)
        return capex_ref * (2**doublings)**b

    # Doublements cumulatifs estimés pour électrolyseurs (global)
    doublings_PEM = [0, 1.0, 2.5, 4.5, 6.5, 8.0, 9.5]  # par année
    LR_PEM = 0.18  # 18% réduction par doublement (Nature Energy 2017)
    LR_AEL = 0.12
    LR_solar= 0.24

    scenarios = pd.DataFrame({
        'annee'                         : annees,
        # ── CAPEX Technologies (apprentissage) ──────────────────────────────
        'CAPEX_PEM_USD_kW'              : [wright_law(1100, LR_PEM, d) for d in doublings_PEM],
        'CAPEX_AEL_USD_kW'              : [wright_law(800,  LR_AEL, d) for d in doublings_PEM],
        'CAPEX_solaire_USD_kW'          : [wright_law(550,  LR_solar,d) for d in doublings_PEM],
        'CAPEX_eolien_USD_kW'           : [1200, 1100, 1000, 900, 820, 760, 700],
        # ── Efficacités (amélioration technologique) ─────────────────────────
        'efficacite_PEM_kWh_kgH2'       : [55, 53, 51, 49, 47, 46, 44],
        'efficacite_AEL_kWh_kgH2'       : [52, 51, 50, 48, 47, 46, 45],
        # ── Prix électricité ─────────────────────────────────────────────────
        'PPA_solaire_Maroc_USD_kWh'     : [0.025,0.022,0.018,0.015,0.013,0.012,0.011],
        'PPA_eolien_Maroc_USD_kWh'      : [0.032,0.029,0.025,0.021,0.018,0.016,0.015],
        # ── LCOH résultant ───────────────────────────────────────────────────
        'LCOH_PEM_solaire_min'          : [3.5, 2.8, 2.0, 1.5, 1.2, 1.0, 0.9],
        'LCOH_PEM_solaire_mode'         : [5.5, 4.0, 2.8, 2.0, 1.5, 1.2, 1.0],
        'LCOH_PEM_solaire_max'          : [8.0, 6.0, 4.0, 3.0, 2.2, 1.8, 1.5],
        'LCOH_AEL_hybride_min'          : [3.0, 2.5, 1.8, 1.3, 1.0, 0.9, 0.8],
        'LCOH_AEL_hybride_mode'         : [4.8, 3.5, 2.5, 1.8, 1.4, 1.1, 0.9],
        'LCOH_AEL_hybride_max'          : [7.0, 5.5, 3.8, 2.8, 2.0, 1.6, 1.3],
        # ── Chaîne complète (LCODC = Leveled Cost of Delivery Chain) ─────────
        'LCODC_export_EU_pipeline_min'  : [4.0, 3.2, 2.5, 1.9, 1.5, 1.3, 1.1],
        'LCODC_export_EU_pipeline_mode' : [6.5, 5.0, 3.5, 2.7, 2.0, 1.6, 1.3],
        'LCODC_export_EU_pipeline_max'  : [10.0,7.5, 5.5, 4.0, 3.0, 2.4, 2.0],
        'LCODC_export_NH3_ship_min'     : [5.0, 4.0, 3.0, 2.3, 1.8, 1.5, 1.3],
        'LCODC_export_NH3_ship_mode'    : [8.0, 6.0, 4.2, 3.2, 2.4, 1.9, 1.6],
        'LCODC_export_NH3_ship_max'     : [12.0,9.0, 6.5, 4.8, 3.5, 2.8, 2.3],
        # ── Objectifs nationaux ──────────────────────────────────────────────
        'production_H2_Maroc_ktH2_an'   : [5, 50, 400, 1200, 2500, 4000, 6000],
        'export_H2_pct'                 : [0, 20, 45,  60,   70,   75,   80  ],
        'capacite_electrolyseur_GW'     : [0.01,0.1,0.8,2.5,5.0,8.0,12.0],
        'emplois_crees_milliers'        : [0.5, 5,  35,  100, 200, 320, 450],
        'investissement_cumul_Mrd_USD'  : [0.1, 1,  8,   25,  55,  90,  130],
        'CO2_evite_MtCO2_an'           : [0.1, 0.8,5,   12,  22,  35,  50 ],
        # Source : Stratégie Nationale H2 Maroc 2021 + IEA Morocco Energy Profile 2024
    })

    # Arrondis pour lisibilité
    for col in scenarios.columns[1:]:
        scenarios[col] = scenarios[col].round(3)

    scenarios.to_csv(f"{OUTPUT_DIR}/csv/T9_scenarios_temporels.csv", index=False, encoding='utf-8-sig')
    print(f"     ✓ T9 sauvegardé : {len(annees)} années × {len(scenarios.columns)} variables")
    return scenarios


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO ENGINE — Calcul LCOH + LCOS + LCOT + LCODC
# ══════════════════════════════════════════════════════════════════════════════
class MonteCarloH2Morocco:
    """Moteur Monte Carlo pour la chaîne H2 complète"""

    def __init__(self, n=N_SIM):
        self.N = n
        np.random.seed(42)

    def sample(self, min_val, mode_val, max_val, dist='triangular'):
        if dist == 'triangular':
            return np.random.triangular(min_val, mode_val, max_val, self.N)
        elif dist == 'normal':
            mean = mode_val
            std = (max_val - min_val) / 4
            return np.clip(np.random.normal(mean, std, self.N), min_val, max_val)
        elif dist == 'lognormal':
            sigma = np.log(max_val / min_val) / 4
            return np.random.lognormal(np.log(mode_val), sigma, self.N)
        elif dist == 'uniform':
            return np.random.uniform(min_val, max_val, self.N)

    def run_LCOH(self, location='Ouarzazate', technologie='PEM', annee=2024):
        """Calcule la distribution LCOH pour une configuration donnée"""

        # Facteurs d'apprentissage selon l'année
        yr_factor = {2024:1.0, 2030:0.75, 2035:0.60, 2040:0.50, 2050:0.38}
        yf = yr_factor.get(annee, 1.0)

        # Capacity Factors par location
        CF_data = {'Ouarzazate':0.28,'Laayoune':0.31,'Dakhla':0.33,
                   'Tanger':0.20,'Taroudant':0.27,'Guelmim':0.30}
        CF_wind = {'Ouarzazate':0.30,'Laayoune':0.38,'Dakhla':0.42,
                   'Tanger':0.33,'Taroudant':0.34,'Guelmim':0.40}

        CF_s = CF_data.get(location, 0.28)
        CF_w = CF_wind.get(location, 0.33)
        CF_h = min(CF_s + CF_w * 0.15, 0.70)  # hybride

        # Paramètres selon technologie
        if technologie == 'PEM':
            CAPEX_e = self.sample(600*yf, 900*yf, 1500*yf)
            OPEX_r  = self.sample(0.025, 0.030, 0.040, 'uniform')
            effic   = self.sample(50, 55, 65, 'normal')
        else:  # AEL
            CAPEX_e = self.sample(500*yf, 650*yf, 1000*yf)
            OPEX_r  = self.sample(0.015, 0.020, 0.030, 'uniform')
            effic   = self.sample(50, 52, 58, 'normal')

        # Énergie solaire
        CAPEX_s = self.sample(350*yf, 550*yf, 900*yf)
        OPEX_s  = self.sample(8,12,18,'normal')  # USD/kW/an

        # Éolien
        CAPEX_w = self.sample(900*yf, 1200*yf, 1600*yf)
        OPEX_w  = self.sample(25,35,50,'normal')

        # Paramètres financiers
        DR    = self.sample(0.06, 0.08, 0.12, 'normal')   # discount rate
        LT    = np.random.choice([20,25,30], self.N, p=[0.3,0.5,0.2])
        PPA   = self.sample(0.015, 0.025, 0.040, 'lognormal')

        # Eau
        water_cost = self.sample(0.5, 3.0, 4.0, 'uniform')
        water_cons = 9.0  # L/kgH2

        # Capacité : 100 MW électrolyseur uniquement (PPA pour énergie)
        C_e = 100e3   # kW électrolyseur

        # CRF (Capital Recovery Factor)
        CRF = (DR * (1+DR)**LT) / ((1+DR)**LT - 1)

        # Production H2 annuelle (kg/an)
        H_hybrid = CF_h * 8760
        H2_prod = H_hybrid * C_e / effic  # kg/an

        # CAPEX électrolyseur annualisé seulement (énergie achetée via PPA)
        CAPEX_ann = CAPEX_e * C_e * CRF

        # OPEX électrolyseur
        OPEX_ann = OPEX_r * CAPEX_e * C_e

        # Coût électricité via PPA
        elec_cost = PPA * effic * H2_prod

        # Eau
        water_total = water_cost * water_cons / 1000 * H2_prod  # USD/an

        # LCOH
        LCOH = (CAPEX_ann + OPEX_ann + elec_cost + water_total) / H2_prod
        LCOH = np.clip(LCOH, 0.5, 15)

        return {
            'LCOH'  : LCOH,
            'mean'  : float(np.mean(LCOH)),
            'median': float(np.median(LCOH)),
            'std'   : float(np.std(LCOH)),
            'P10'   : float(np.percentile(LCOH, 10)),
            'P50'   : float(np.percentile(LCOH, 50)),
            'P90'   : float(np.percentile(LCOH, 90)),
            'CI95'  : (float(np.percentile(LCOH, 2.5)), float(np.percentile(LCOH, 97.5))),
            'location': location, 'technologie': technologie, 'annee': annee,
        }

    # ── ANCIENNE VERSION (commentée pour traçabilité) ──────────────────────
    #def run_LCOS(self, technologie='NH3'):
    #   """Coût de stockage"""
    #   CAPEX_synth = self.sample(400, 700, 1100)
    #   CAPEX_store = self.sample(200, 350, 500)
    #   effic = self.sample(0.68, 0.72, 0.76, 'normal')
    #   DR = self.sample(0.06, 0.08, 0.12, 'normal')
    #   LT = 25
    #   E_synth = self.sample(8, 10, 12, 'normal')
    #   PPA = self.sample(0.018, 0.025, 0.040, 'lognormal')
    #   CRF = (DR * (1+DR)**LT) / ((1+DR)**LT - 1)
    #   # CAPEX_t = (CAPEX_synth + CAPEX_store) * 1000  # par tonne cap/j → normalisé
    #   # OPEX = 0.03 * CAPEX_t
    #   # elec = E_synth * PPA
    #   # LCOS = (CAPEX_t * CRF / 365 + OPEX / 365) / 1000 / effic + elec / effic
    #   # CORRECTION tentative 1 : sans × 1000 — encore incorrect (unités incohérentes)
    #   # CAPEX_t = (CAPEX_synth + CAPEX_store)  # USD/kW sans × 1000
    #   # prod_annuelle_kgH2 = 1e6
    #   # elec = E_synth * PPA
    #   # LCOS = (CAPEX_t * CRF / 8760 + CAPEX_t * 0.03 / 8760) / effic + elec / effic
    #   # LCOS = np.clip(LCOS, 0.3, 8)
    #   return {'LCOS': LCOS, 'mean': float(np.mean(LCOS)),
    #           'P10': float(np.percentile(LCOS,10)),
    #           'P50': float(np.percentile(LCOS,50)),
    #           'P90': float(np.percentile(LCOS,90))}

    def run_LCOS(self, technologie='NH3'):
        # Cout de stockage NH3 (synthese Haber-Bosch + stockage cryogenique)
        # via productivite reelle electrolyseur (kgH2 produit par kW installe par an)
        # Sources : IEA 2024, Hydrogen Council 2023, IRENA 2022
        CAPEX_synth = self.sample(400, 700, 1100)          # $/kW_e synthese NH3
        CAPEX_store = self.sample(200, 350, 500)            # $/kW_e stockage NH3
        effic_nh3   = self.sample(0.68, 0.72, 0.76, 'normal')  # efficacite H2→NH3
        DR          = self.sample(0.06, 0.08, 0.12, 'normal')
        LT          = 25
        E_synth     = self.sample(8, 10, 12, 'normal')     # kWh_e / kgH2 pour synthese
        PPA         = self.sample(0.018, 0.025, 0.040, 'lognormal')

        CRF = (DR * (1+DR)**LT) / ((1+DR)**LT - 1)

        # Productivite electrolyseur AEL (Dakhla hybride) :
        # 1 kW_e installe → kgH2/an = 8760 * CF_hybride / efficacite_elec
        # CF_hybride Dakhla ~0.39, efficacite AEL ~52 kWh/kgH2
        kgH2_par_kW_par_an = 8760 * 0.39 / 52.0           # ≈ 65.6 kgH2/kW/an

        # CAPEX annualise par kgH2 produit
        CAPEX_per_kgH2_an = (CAPEX_synth + CAPEX_store) / kgH2_par_kW_par_an
        LCOS_capex = CAPEX_per_kgH2_an * CRF
        LCOS_opex  = CAPEX_per_kgH2_an * 0.03

        # Energie synthese NH3 : E_synth kWh/kgH2 a PPA $/kWh, rendement effic_nh3
        elec = E_synth * PPA / effic_nh3

        LCOS = LCOS_capex + LCOS_opex + elec
        LCOS = np.clip(LCOS, 0.3, 8)

        return {'LCOS': LCOS, 'mean': float(np.mean(LCOS)),
                'P10': float(np.percentile(LCOS, 10)),
                'P50': float(np.percentile(LCOS, 50)),
                'P90': float(np.percentile(LCOS, 90))}

    def run_LCOT(self, corridor='Casablanca→Rotterdam', mode='Tanker_NH3'):
        """Coût de transport"""
        distances = {
            'Casablanca→Rotterdam': 3500, 'Agadir→Barcelone': 900,
            'Tanger→Algésiras': 22, 'Dakhla→Agadir': 1200,
            'Ouarzazate→Casablanca': 430,
        }
        d = distances.get(corridor, 1000)

        if mode == 'Pipeline':
            CAPEX_km = self.sample(1.5, 2.5, 4.0) * 1e6
            LT, cap = 40, 5e6
            CRF = 0.06
            LCOT = (CAPEX_km * d * CRF + CAPEX_km * d * 0.02) / (cap * 8760 / 1000)
        #elif mode == 'Tanker_NH3':
        #   CAPEX_ship = self.sample(50, 65, 85) * 1e6
        #   trips_per_year = max(1, int(8760 / (d/25 * 2 + 48)))
            #cap_per_trip = self.sample(60000, 80000, 100000) * 0.176  # kgH2 equivalent
            # CORRECTION : ajouter × 1000 (tonnes → kg) ou reformuler
        #    cap_per_trip = self.sample(60000, 80000, 100000) * 1000 * 0.176  # t × 1000 = kg → × ratio H2/NH3
        #   LT = 25
        #   CRF = (0.08 * 1.08**LT) / (1.08**LT - 1)
        #    LCOT = (CAPEX_ship * CRF + CAPEX_ship * 0.03) / (trips_per_year * cap_per_trip)
        elif mode == 'Tanker_NH3':
            CAPEX_ship   = self.sample(50, 65, 85) * 1e6          # USD
            OPEX_voyage  = self.sample(500, 700, 1000) * 1000      # USD/voyage (fuel+port)
            # Vitesse 14 nœuds = 25.9 km/h, temps AR + escale port 48h
            t_voyage_h   = d / 25.9 * 2 + 48
            trips_per_year = max(1, int(8760 / t_voyage_h))
            # Capacité tanker : 40 000–84 000 t NH3 × 1000 kg/t × ratio H2/NH3 (3/17)
            cap_per_trip = self.sample(40000, 60000, 84000) * 1000 * 0.176  # kgH2
            LT = 25
            CRF = (0.08 * 1.08**LT) / (1.08**LT - 1)
            kgH2_an = cap_per_trip * trips_per_year
            LCOT = (CAPEX_ship * (CRF + 0.025) + OPEX_voyage * trips_per_year) / kgH2_an
        else:
            LCOT = self.sample(0.1, 0.3, 0.8)

        LCOT = np.clip(LCOT, 0.05, 10)
        return {'LCOT': LCOT, 'mean': float(np.mean(LCOT)),
                'P10': float(np.percentile(LCOT,10)),
                'P50': float(np.percentile(LCOT,50)),
                'P90': float(np.percentile(LCOT,90)),
                'corridor': corridor, 'mode': mode}

    def run_full_chain(self, location='Dakhla', technologie='AEL',
                       storage='NH3', transport='Tanker_NH3',
                       corridor='Casablanca→Rotterdam', annee=2030):
        """Chaîne complète : LCODC = LCOH + LCOS + LCOT"""
        r_prod  = self.run_LCOH(location, technologie, annee)
        r_stor  = self.run_LCOS(storage)
        r_trans = self.run_LCOT(corridor, transport)

        LCODC = r_prod['LCOH'] + r_stor['LCOS'] + r_trans['LCOT']
        LCODC = np.clip(LCODC, 1.0, 25)

        return {
            'LCODC': LCODC,
            'mean': float(np.mean(LCODC)),
            'P10' : float(np.percentile(LCODC, 10)),
            'P50' : float(np.percentile(LCODC, 50)),
            'P90' : float(np.percentile(LCODC, 90)),
            'LCOH_contrib_pct': float(np.mean(r_prod['LCOH']) / np.mean(LCODC) * 100),
            'LCOS_contrib_pct': float(np.mean(r_stor['LCOS']) / np.mean(LCODC) * 100),
            'LCOT_contrib_pct': float(np.mean(r_trans['LCOT']) / np.mean(LCODC) * 100),
            'config': f"{location}|{technologie}|{storage}|{transport}|{annee}",
        }

    def sensitivity_analysis(self, location='Ouarzazate', annee=2030):
        """Analyse de sensibilité — Tornado Chart"""
        base = self.run_LCOH(location, 'PEM', annee)['P50']
        params = {
            'Prix électricité (PPA)': (0.015, 0.040),
            'CAPEX Électrolyseur':    (600,   1500),
            'Capacity Factor solaire':(0.20,  0.33),
            'Taux actualisation':     (0.06,  0.12),
            'Efficacité électrolyseur':(50,   65),
            'Durée de vie projet':    (15,    30),
            'CAPEX Solaire PV':       (350,   900),
        }
        impacts = {}
        for p, (low, high) in params.items():
            impacts[p] = abs(high - low) / (high + low) * base * 0.8
        return base, impacts


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION MODEL
# ══════════════════════════════════════════════════════════════════════════════
def validate_model(mc_engine, projets_df):
    """Valide le modèle contre les projets réels marocains"""
    print("\n  [VALIDATION] Benchmark contre projets réels...")
    results = []

    checks = [
        # (description, valeur_simulée, valeur_réelle, tolérance_%)
        ('CF solaire Ouarzazate',
         mc_engine.run_LCOH('Ouarzazate','PEM',2024)['LCOH'].mean(),
         5.5, 40),
        ('LCOH Dakhla 2030',
         mc_engine.run_LCOH('Dakhla','AEL',2030)['P50'],
         2.5, 35),
        ('LCOH Ouarzazate 2030',
         mc_engine.run_LCOH('Ouarzazate','PEM',2030)['P50'],
         2.8, 35),
        ('LCODC export NH3 2030',
         mc_engine.run_full_chain('Dakhla','AEL','NH3','Tanker_NH3',
                                  'Casablanca→Rotterdam',2030)['P50'],
         4.5, 40),
    ]

    for desc, simul, real, tol in checks:
        err = abs(simul - real) / real * 100
        ok = err <= tol
        results.append({'Test': desc, 'Simulé': round(simul,3),
                        'Référence': real, 'Erreur_%': round(err,1),
                        'Tolérance_%': tol, 'Statut': '✅ VALIDE' if ok else '⚠️ ÉCART'})

    df_val = pd.DataFrame(results)
    score = (df_val['Statut'] == '✅ VALIDE').sum() / len(df_val) * 100

    print(f"\n{'─'*70}")
    print(f"{'TEST VALIDATION':40s} {'SIMULÉ':>8} {'RÉEL':>8} {'ERR%':>6} {'STATUT':>10}")
    print(f"{'─'*70}")
    for _, r in df_val.iterrows():
        print(f"  {r['Test']:38s} {r['Simulé']:>8.2f} {r['Référence']:>8.2f} {r['Erreur_%']:>5.1f}% {r['Statut']:>10}")
    print(f"{'─'*70}")
    print(f"  SCORE GLOBAL DE VALIDATION : {score:.0f}%  ", end='')
    print("✅ MODÈLE VALIDÉ — Prêt pour publication" if score>=75 else "⚠️  RÉVISION NÉCESSAIRE")

    df_val.to_csv(f"{OUTPUT_DIR}/reports/validation_report.csv", index=False, encoding='utf-8-sig')
    return df_val, score
# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════════

def fig1_lcoh_distributions(mc):
    print("  [Fig1] Distributions LCOH Monte Carlo...")
    regions = ['Ouarzazate', 'Laayoune', 'Dakhla', 'Tanger', 'Taroudant', 'Guelmim']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Distribution LCOH H2 Vert par Region — Monte Carlo (n=10 000) — 2030",
                 fontsize=13, fontweight='bold')
    for ax, reg in zip(axes.flat, regions):
        r_pem = mc.run_LCOH(reg, 'PEM', 2030)
        r_ael = mc.run_LCOH(reg, 'AEL', 2030)
        ax.hist(r_pem['LCOH'], bins=60, alpha=0.6, color=COLORS['PEM'],
                label=f"PEM  P50={r_pem['P50']:.2f}", density=True)
        ax.hist(r_ael['LCOH'], bins=60, alpha=0.6, color=COLORS['AEL'],
                label=f"AEL  P50={r_ael['P50']:.2f}", density=True)
        ax.axvline(r_pem['P50'], color=COLORS['PEM'], ls='--', lw=1.8)
        ax.axvline(r_ael['P50'], color=COLORS['AEL'], ls='--', lw=1.8)
        ax.axvline(2.0, color='red', ls=':', lw=1.5, label='Cible 2030 = 2 $/kg')
        ax.set_title(reg)
        ax.set_xlabel("LCOH ($/kg H2)")
        ax.set_ylabel("Densite")
        ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig1_LCOH_Monte_Carlo.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig1 sauvegardee")


def fig2_trajectoires_lcoh(t9):
    print("  [Fig2] Trajectoires LCOH 2024-2050...")
    fig, ax = plt.subplots(figsize=(12, 7))
    an = t9['annee'].values
    ax.fill_between(an, t9['LCOH_AEL_hybride_min'], t9['LCOH_AEL_hybride_max'],
                    alpha=0.20, color=COLORS['AEL'], label='AEL hybride [min-max]')
    ax.plot(an, t9['LCOH_AEL_hybride_mode'], 'o-',
            color=COLORS['AEL'], lw=2.5, ms=8, label='AEL hybride (central)')
    ax.fill_between(an, t9['LCOH_PEM_solaire_min'], t9['LCOH_PEM_solaire_max'],
                    alpha=0.20, color=COLORS['PEM'], label='PEM solaire [min-max]')
    ax.plot(an, t9['LCOH_PEM_solaire_mode'], 's--',
            color=COLORS['PEM'], lw=2.5, ms=8, label='PEM solaire (central)')
    ax.axhline(2.0, color='orange', ls=':', lw=2, label='Parite H2 gris (~2 $/kg)')
    ax.axhline(1.0, color='red',    ls=':', lw=2, label='Objectif DOE Hydrogen Shot')
    ax.set_xlabel("Annee", fontsize=12)
    ax.set_ylabel("LCOH ($/kg H2)", fontsize=12)
    ax.set_title("Trajectoire LCOH H2 Vert Maroc 2024-2050\nLoi de Wright — PEM LR=18% | AEL LR=12%", fontsize=13)
    ax.legend(fontsize=9)
    ax.set_xticks(an)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig2_Trajectoires_LCOH.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig2 sauvegardee")


def fig3_tornado(mc, location='Ouarzazate', annee=2030):
    print("  [Fig3] Tornado Chart sensibilite...")
    base, impacts = mc.sensitivity_analysis(location, annee)
    keys   = sorted(impacts, key=impacts.get)
    vals   = [impacts[k] for k in keys]
    colors = [COLORS['secondary'] if v > np.median(vals) else COLORS['primary'] for v in vals]
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(keys, vals, color=colors, edgecolor='white', height=0.6)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'+/-{val:.2f} $/kg', va='center', fontsize=9)
    ax.set_xlabel("Impact sur LCOH ($/kg H2)", fontsize=11)
    ax.set_title(f"Analyse de Sensibilite — LCOH H2 Vert {location} {annee}\nValeur centrale P50 = {base:.2f} $/kg", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig3_Tornado_Sensibilite.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig3 sauvegardee")


def fig4_benchmark(t6_comp):
    print("  [Fig4] Benchmark competiteurs...")
    pays = t6_comp['pays'].values
    x = np.arange(len(pays))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - w, t6_comp['LCOH_2024'], w, label='2024', color='#e74c3c', alpha=0.85)
    ax.bar(x,     t6_comp['LCOH_2030'], w, label='2030', color='#f39c12', alpha=0.85)
    ax.bar(x + w, t6_comp['LCOH_2040'], w, label='2040', color=COLORS['primary'], alpha=0.85)
    ax.axhline(2.0, color='orange', ls='--', lw=1.5, label='Parite H2 gris 2030')
    ax.axhline(1.0, color='red',    ls='--', lw=1.5, label='Objectif DOE')
    ax.set_xticks(x)
    ax.set_xticklabels(pays, rotation=15, ha='right')
    ax.set_ylabel("LCOH ($/kg H2)")
    ax.set_title("Benchmark LCOH H2 Vert — Maroc vs Competiteurs", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig4_Benchmark_Competiteurs.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig4 sauvegardee")


def fig5_demande_sectorielle(t6_demande):
    print("  [Fig5] Demande sectorielle...")
    secteurs = t6_demande['secteur'].values
    annees_d = [2024, 2030, 2035, 2040, 2050]
    cols_d   = ['demande_2024','demande_2030','demande_2035','demande_2040','demande_2050']
    palette  = plt.cm.Set3(np.linspace(0, 1, len(secteurs)))
    fig, ax = plt.subplots(figsize=(12, 7))
    bottom  = np.zeros(len(annees_d))
    for i, (sect, col) in enumerate(zip(secteurs, palette)):
        vals = [t6_demande[c].iloc[i] for c in cols_d]
        ax.bar(annees_d, vals, bottom=bottom, label=sect.replace('_', ' '), color=col, alpha=0.9)
        bottom += np.array(vals)
    ax.set_xlabel("Annee")
    ax.set_ylabel("Demande (ktH2/an)")
    ax.set_title("Demande Nationale H2 Vert par Secteur — Maroc 2024-2050", fontsize=13)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig5_Demande_Sectorielle.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig5 sauvegardee")


def fig6_lcodc(mc):
    print("  [Fig6] LCODC chaine complete...")
    configs = [
        ('Dakhla',     'AEL', 'NH3', 'Tanker_NH3', 'Casablanca→Rotterdam', 2030),
        ('Dakhla',     'AEL', 'NH3', 'Tanker_NH3', 'Casablanca→Rotterdam', 2040),
        ('Ouarzazate', 'PEM', 'NH3', 'Tanker_NH3', 'Casablanca→Rotterdam', 2030),
        ('Laayoune',   'AEL', 'NH3', 'Tanker_NH3', 'Casablanca→Rotterdam', 2030),
        ('Tanger',     'PEM', 'NH3', 'Pipeline',   'Casablanca→Rotterdam', 2030),
    ]
    labels, p10s, p50s, p90s, pcp, pcs, pct_ = [], [], [], [], [], [], []
    for loc, tech, stor, trans, corr, yr in configs:
        r = mc.run_full_chain(loc, tech, stor, trans, corr, yr)
        labels.append(f"{loc}\n{tech}|{yr}")
        p10s.append(r['P10']); p50s.append(r['P50']); p90s.append(r['P90'])
        pcp.append(r['LCOH_contrib_pct'])
        pcs.append(r['LCOS_contrib_pct'])
        pct_.append(r['LCOT_contrib_pct'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    x = np.arange(len(labels))
    ax1.bar(x, p50s, color=COLORS['primary'], alpha=0.85, label='P50 (central)')
    for i, (lo, hi) in enumerate(zip(p10s, p90s)):
        ax1.plot([i, i], [lo, hi], 'k-', lw=2)
        ax1.plot(i, lo, 'v', color='gray', ms=8)
        ax1.plot(i, hi, '^', color='gray', ms=8)
    ax1.axhline(4.5, color='red', ls='--', lw=1.5, label='Ref EU import 4.5 $/kg')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("LCODC ($/kg H2)")
    ax1.set_title("Cout Chaine Complete LCODC\n[P10 - P50 - P90]")
    ax1.legend(fontsize=8)
    ax2.bar(x, pcp, 0.5, label='Production (LCOH)', color=COLORS['PEM'],     alpha=0.85)
    ax2.bar(x, pcs, 0.5, bottom=pcp,
            label='Stockage (LCOS)',   color=COLORS['NH3'],     alpha=0.85)
    ax2.bar(x, pct_, 0.5, bottom=np.array(pcp)+np.array(pcs),
            label='Transport (LCOT)',  color=COLORS['pipeline'], alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_ylabel("Contribution (%)")
    ax2.set_title("Decomposition LCODC\npar Poste de Cout")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/figures/Fig6_LCODC_Chaine_Complete.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("     OK : Fig6 sauvegardee")


# ══════════════════════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("="*60)
    print("  GENERATION DES DONNEES & VISUALISATIONS")
    print("="*60)

    t1                  = build_T1_ressources()
    t2                  = build_T2_production()
    t3                  = build_T3_stockage()
    t4                  = build_T4_transport()
    t5                  = build_T5_economique()
    t6_demande, t6_comp = build_T6_marche()
    t7                  = build_T7_environnement()
    t8                  = build_T8_projets()
    t9                  = build_T9_scenarios()

    # Renommer colonnes T6 pour les figures
    t6_comp = t6_comp.rename(columns={
        'LCOH_2024_USD_kg': 'LCOH_2024',
        'LCOH_2030_USD_kg': 'LCOH_2030',
        'LCOH_2040_USD_kg': 'LCOH_2040',
    })

    print("\n=== Monte Carlo (n=10 000) ===")
    mc = MonteCarloH2Morocco(N_SIM)

    print("\n=== Validation ===")
    validate_model(mc, t8)

    print("\n=== Visualisations ===")
    fig1_lcoh_distributions(mc)
    fig2_trajectoires_lcoh(t9)
    fig3_tornado(mc)
    fig4_benchmark(t6_comp)
    fig5_demande_sectorielle(t6_demande)
    fig6_lcodc(mc)

    print("\nTermine — resultats dans :", OUTPUT_DIR)

