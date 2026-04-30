# 📖 Méthodologie Détaillée du Projet H₂ Vert Maroc

Ce document annexe détaille la méthodologie scientifique et technique employée dans le cadre de l'évaluation du potentiel de l'hydrogène vert au Maroc (Solaire, Éolien et Hybride).

---

## 1. Origine et Traitement des Données Climatologiques (Météo)

Les données météorologiques utilisées dans ce projet proviennent de la réanalyse climatique **ERA5-Land**, développée par le Centre Européen pour les Prévisions Météorologiques à Moyen Terme (ECMWF). ERA5-Land fournit des données climatiques cohérentes sur plusieurs décennies avec une résolution spatiale de 9 km, en combinant des observations mondiales avec un modèle de prévision numérique (IFS).

### A. Potentiel Solaire : Irradiation Solaire (GHI)
- **Source** : `ECMWF/ERA5_LAND` (Moyenne annuelle sur la période 2018-2022).
- **Paramètre** : *Surface solar radiation downwards* (Somme de l'irradiation solaire descendante).
- **Méthodologie ERA5** : L'estimation de ce paramètre repose sur la modélisation du transfert radiatif atmosphérique en ondes courtes. Le modèle simule avec précision la manière dont le rayonnement solaire interagit avec l'atmosphère avant de toucher le sol. Il intègre de manière dynamique l'atténuation, la diffusion et l'absorption causées par :
  - La couverture nuageuse.
  - Les aérosols.
  - La vapeur d'eau.
  - L'ozone stratosphérique.

### B. Potentiel Éolien : Vitesse du Vent et Extrapolation (Vecteurs U et V)
- **Source** : `ECMWF/ERA5_LAND` (Moyenne sur la période 2018-2022).
- **Paramètres Initiaux** : Composantes vectorielles `U` et `V` du vent à 10 mètres du sol.
- **Méthodologie ERA5 et Traitement** : 
  1. **Vecteurs du Vent** : Dans le modèle ERA5, le vent n'est pas fourni directement comme une simple vitesse, mais décomposé en deux vecteurs directionnels :
     - La composante **U (zonale)** : représente le vent soufflant d'Ouest en Est.
     - La composante **V (méridienne)** : représente le vent soufflant du Nord au Sud.
  2. **Calcul de la Vitesse Scalaire** : Pour obtenir la vitesse absolue du vent (indépendamment de sa direction), le script combine ces deux vecteurs en utilisant le théorème de Pythagore : 
     $$Vitesse = \sqrt{U^2 + V^2}$$
  3. **Extrapolation à 200m** : Les éoliennes modernes de grande capacité ont des hauteurs de moyeu qui dépassent largement les 10 mètres. Pour évaluer le potentiel réel, cette vitesse à 10m est extrapolée à **200 mètres** en utilisant la **loi logarithmique du profil des vents**. 
  4. **Formule utilisée** : $V_{200} = V_{10} \times \frac{\ln(200 / Z_0)}{\ln(10 / Z_0)}$
  5. **Rugosité ($Z_0$)** : Une longueur de rugosité de `0.03 m` a été choisie, ce qui est caractéristique des terrains ouverts et plats, très présents dans les régions étudiées.

### C. Correction Thermique (LST)
- **Source** : `MODIS MOD11A2` (2010-2023).
- **Objectif** : La température de surface diurne (LST) est utilisée pour pénaliser les zones où les températures extrêmes pourraient réduire l'efficacité des panneaux photovoltaïques et des électrolyseurs.

---

## 2. L'Analyse Hiérarchique des Procédés (AHP)

Afin de combiner des critères de natures très différentes (météo, géographie, infrastructures), le modèle utilise la méthode d'Analyse Hiérarchique des Procédés (AHP) pour attribuer un poids spécifique à chaque critère en fonction de son importance.

### A. Reclassification
Toutes les données brutes ont été normalisées et reclassées sur une échelle de **1 à 5** (1 étant le moins favorable, et 5 le plus favorable). 

### B. Pondérations Appliquées
Les poids ont été calculés via une matrice de comparaison par paires. Voici la répartition finale :

**Poids pour l'Aptitude Solaire (Max = 4.596) :**
1. **GHI Solaire** : 35.4 %
2. **Pente du Terrain** : 23.8 %
3. **Température (LST)** : 16.1 %
4. **Proximité des Routes** : 10.9 %
5. **Proximité des Villes** : 7.5 %
6. **Proximité Réseau Électrique** : 2.6 %
7. **Proximité Rivières** : 1.3 %
8. **Proximité Barrages** : 1.3 %
9. **Proximité Côte/Mer** : 1.2 %

**Poids pour l'Aptitude Éolien (Max = 4.606) :**
1. **Vitesse du Vent (200m)** : 40.2 %
2. **Pente du Terrain** : 20.8 %
3. **Proximité des Villes** : 14.0 %
4. **Proximité Réseau Électrique** : 9.1 %
5. **Proximité des Routes** : 9.1 %
6. **Température (LST)** : 2.4 %
7. **Proximité Rivières** : 1.5 %
8. **Proximité Barrages** : 1.5 %
9. **Proximité Côte/Mer** : 1.4 %

### C. Masque d'Exclusion Stricte
Avant de générer les cartes finales, un masque d'exclusion binaire est appliqué. Il élimine toute zone qui n'est physiquement ou légalement pas exploitable :
- Les pentes supérieures à 45°.
- Les plans d'eau permanents et temporaires.
- Les zones urbaines denses.
- Les zones couvertes de neige ou de glace.

Le score final de chaque pixel est ensuite ramené sur une échelle de **0 à 100** pour faciliter la lecture et la comparaison des cartes d'aptitude.
