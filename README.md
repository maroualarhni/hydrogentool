# 🌍 Projet H₂ Vert Maroc — Analyse Géospatiale & Potentiel 🔋

<div align="center">
  <img alt="Earth Engine" src="https://img.shields.io/badge/Google%20Earth%20Engine-JavaScript-green?style=for-the-badge&logo=google" />
  <img alt="GIS" src="https://img.shields.io/badge/GIS-Spatial%20Analysis-blue?style=for-the-badge" />
  <img alt="Status" src="https://img.shields.io/badge/Status-Final%20V6.0-brightgreen?style=for-the-badge" />
</div>

<br>

Bienvenue sur le dépôt du script complet (**Version 6.0**) dédié à l'évaluation et l'analyse du potentiel de production d'hydrogène vert (H₂) au Maroc (incluant les Provinces du Sud). 

Ce travail est destiné à des fins de recherche et vise à faire l'objet d'une publication scientifique ultérieure.

---

## 🎯 Objectif Principal

Le script identifie, classe et évalue les zones les plus propices à l'installation d'infrastructures de production d'hydrogène vert. L'analyse génère des cartes d'aptitude normalisées de **0 à 100%** pour trois catégories :
- ☀️ **Aptitude Solaire**
- 🌬️ **Aptitude Éolien**
- ⚡ **Aptitude Hybride** (Moyenne Solaire + Éolien)

---

## 🔬 Données et Méthodologie

L'analyse est entièrement basée sur le cloud computing via **Google Earth Engine (GEE)**. Elle intègre et pondère **9 critères géo-environnementaux et d'infrastructure** en utilisant la méthode d'Analyse Hiérarchique des Procédés (AHP).

### 📖 Consulter la Méthodologie Détaillée
> Pour des détails approfondis sur les formules, le traitement des données ERA5 (modélisation du rayonnement et extrapolation du vent) et la matrice de pondération AHP, veuillez consulter le document dédié :
> 👉 **[Lien vers la Méthodologie (METHODOLOGIE.md)](./METHODOLOGIE.md)**

### Les Critères Étudiés :
1. **Météorologie** : Irradiation Solaire GHI, Vitesse du Vent (extrapolée à 200m), Température (LST).
2. **Géographie** : Pente du terrain, occupation des sols, distance à la mer.
3. **Infrastructures** : Proximité des réseaux électriques, routes, centres urbains, rivières et barrages.

---

## 📍 12 Sites Stratégiques

En plus de l'analyse globale, l'étude extrait des données statistiques précises pour **12 sites majeurs** :
1. Dakhla | 2. Tarfaya | 3. Tan-Tan | 4. Agadir | 5. Safi | 6. Jorf Lasfar
7. Kénitra | 8. Tanger | 9. Meknès | 10. Nador | 11. Béni Mellal | 12. Ouarzazate

---

## 🚀 Guide d'Utilisation sur Earth Engine

1. Connectez-vous à votre compte [Google Earth Engine](https://code.earthengine.google.com/).
2. Copiez l'intégralité du fichier `script.js` et collez-le dans l'éditeur de code (Code Editor).
3. Adaptez le chemin de la variable `regions` avec votre propre *FeatureCollection* (pour les statistiques régionales).
4. Cliquez sur **Run**.
5. Allez dans l'onglet **Tasks** (à droite). Vous y verrez 16 tâches prêtes à être exportées vers votre Google Drive.

---

## 📂 Exports Générés

L'exécution des tâches produira le dossier `H2_Vert_Maroc_Final_v6` contenant :

| Type | Fichier | Description |
|:---:|:---|:---|
| 📊 **CSV** | `15_Extraction_12_Sites_H2.csv` | Valeurs exactes (scores et aptitudes) pour les 12 sites focaux. |
| 📊 **CSV** | `16_Stats_Regions_H2.csv` | Statistiques moyennes des potentiels pour chaque région. |
| 🗺️ **TIF** | `01_Aptitude_Solaire...` | Carte finale (0-100%) du potentiel Solaire. |
| 🗺️ **TIF** | `02_Aptitude_Eolien...` | Carte finale (0-100%) du potentiel Éolien. |
| 🗺️ **TIF** | `03_Aptitude_Hybride...` | Carte finale (0-100%) du potentiel Hybride. |
| 🧩 **TIF** | `04_` à `14_` | Cartes des sous-critères (GHI, Vent, Pente, Villes, Masques, etc.). |

---

## 📝 Auteurs et Citation

Ce script a été structuré en vue de la préparation d'une publication académique.
*(Les informations de citation et les noms des auteurs seront mis à jour au moment de la publication officielle).*

⚠️ **Licence :** Veuillez contacter l'auteur avant toute réutilisation académique ou commerciale de ce code.
