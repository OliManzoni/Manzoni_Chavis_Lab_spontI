# 🔬 Expert sIPSC Pipeline: Inhibition & Kinetic Analysis
### *Manzoni & Chavis Labs | Synaptic Plasticity & Biophysics*

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19915016.svg)](https://doi.org/10.5281/zenodo.19915016)

*(Le README en français se trouve dans la seconde moitié de ce document / The French README is located in the second half of this document).*

👉 **Online Access: [https://manzonilab-sipsc.streamlit.app/](https://manzonilab-sipsc.streamlit.app/)** *(Update with your actual link)*

---

## 🇬🇧 ENGLISH VERSION

### 1. Scientific Introduction
The analysis of spontaneous inhibitory post-synaptic currents (**sIPSCs**) is crucial for understanding the GABAergic tone and the excitation/inhibition balance in neural networks. 

This Streamlit application is a dedicated workstation for extracting GABA$_A$ receptor-mediated events. Unlike glutamatergic currents, GABAergic currents require specific handling of polarity (depending on the driving force $V_m - E_{Cl}$) and exhibit slower decay kinetics. This pipeline automates denoising, template-matching detection, and precise kinetic calculations from raw electrophysiological data files (`.abf`).

### 2. Key Features
* **Patch Configuration (Polarity Aware)**: Automatically handles both **Outward** currents (e.g., recorded at $0$ mV to isolate inhibition) and **Inward** currents (e.g., recorded at $-70$ mV with high intracellular chloride).
* **GABA-Specific Multi-Template Detection**: Scans traces using an optimized $0.2$ ms rise-time baseline coupled with varying decay time constant templates ($5, 10, 15,$ and $25$ ms) to capture dendritic variability.
* **Bessel Filtering (4th Order)**: Preserves the ultra-fast rising phase of the sIPSC by ensuring a maximally flat group delay, avoiding artifactual ringing.
* **10-90% Kinetic Measurement**: Precisely calculates the rise time via linear interpolation, bypassing the limits of the sampling frequency.
* **Dual Export System**: Generates an "Events" file (individual sIPSCs with their specific IEI, Area, Rise, and Decay) and a comprehensive "Population" visualization dashboard.

### 3. Algorithms and Mathematics

#### Decay Estimation via Charge Integration
Performing non-linear curve fitting on hundreds of noisy, spontaneous GABAergic events often fails due to overlapping currents. Instead, this pipeline uses a robust mathematical approximation based on total charge. Assuming a simple exponential decay for a GABA$_A$ current:
$$ I(t) = I_{max} e^{-t/\tau} $$
The total charge (Area) is calculated using the trapezoidal rule:
$$ Area = \int_{0}^{\infty} I_{max} e^{-t/\tau} dt = I_{max} \cdot \tau $$
Therefore, the decay constant $\tau$ is rapidly and robustly estimated without curve-fitting:
$$ \tau \approx \frac{Area}{Amplitude} $$

### 4. User Guide
1. **Load Data**: Upload your `.abf` file.
2. **Set Polarity**: Select "Outward (> 0 pA)" or "Inward (< 0 pA)" based on your patch-clamp configuration.
3. **Thresholding**: Adjust the Z-Score (typically $2.5$ to $3.5$) and the Amplitude filter ($> 7$ pA) to distinguish true events from background noise.
4. **Visual Inspection**: Use the interactive graph to verify detections (green dots) and check the cross-correlation trace (blue trace).
5. **Export**: Download your data as `.csv` for further statistical analysis.

### 5. Citation
If you use this software in your research, please cite it as follows:

> **Manzoni, O. J. (2026). Manzoni_Chavis_Lab_sIPSC. Zenodo. https://doi.org/10.5281/zenodo.19915016**

---
<br><br>

## 🇫🇷 VERSION FRANÇAISE

### 1. Introduction Scientifique
L'analyse des courants post-synaptiques inhibiteurs spontanés (**sIPSC**) est fondamentale pour comprendre le tonus GABAergique et la balance excitation/inhibition au sein des réseaux neuronaux.

Cette application Streamlit est une station de travail dédiée à l'extraction des événements médiés par les récepteurs GABA$_A$. Contrairement aux courants glutamatergiques, les courants GABAergiques nécessitent une gestion spécifique de la polarité (selon la force électromotrice $V_m - E_{Cl}$) et présentent des cinétiques de désactivation plus lentes. Ce pipeline automatise le débruitage, la détection par modèles (template-matching) et les calculs cinétiques précis à partir de fichiers bruts (`.abf`).

### 2. Fonctionnalités Principales
* **Configuration Patch (Gestion de la Polarité)** : Gère automatiquement les courants **Sortants** (Outward, ex: enregistrement à $0$ mV) et **Entrants** (Inward, ex: enregistrement à $-70$ mV avec une solution riche en chlorure).
* **Détection Multi-Modèles GABA** : Scanne les traces en utilisant une base optimisée avec un temps de montée de $0.2$ ms, couplée à des modèles de décroissance variables ($5, 10, 15,$ et $25$ ms) pour capturer la variabilité dendritique.
* **Filtre de Bessel (4ème Ordre)** : Préserve la phase montante ultra-rapide du sIPSC en appliquant un délai de groupe maximalement plat, évitant ainsi les oscillations artificielles.
* **Mesure Cinétique 10-90%** : Calcule précisément le temps de montée par interpolation linéaire, s'affranchissant des limites de la fréquence d'échantillonnage.
* **Double Export** : Génère un fichier d'événements individuels (avec IEI, Aire, Rise et Decay) et un tableau de bord visuel de la population.

### 3. Algorithmes et Mathématiques

#### Estimation du Decay par Intégration de la Charge
L'ajustement de courbe non-linéaire sur des événements GABAergiques spontanés échoue souvent à cause du chevauchement et du bruit de fond. Ce pipeline utilise une approximation mathématique robuste basée sur la charge totale. En supposant une décroissance exponentielle simple pour un courant GABA$_A$ :
$$ I(t) = I_{max} e^{-t/\tau} $$
La charge totale (Aire) est calculée via la méthode des trapèzes :
$$ Aire = \int_{0}^{\infty} I_{max} e^{-t/\tau} dt = I_{max} \cdot \tau $$
La constante de temps $\tau$ est donc estimée de manière robuste et rapide sans *curve-fitting* :
$$ \tau \approx \frac{Aire}{Amplitude} $$

### 4. Guide d'Utilisation
1. **Chargement** : Uploadez votre fichier `.abf`.
2. **Polarité** : Sélectionnez "Sortant / Outward" ou "Entrant / Inward" selon votre configuration de patch-clamp.
3. **Seuils** : Ajustez le Z-Score (typiquement entre $2.5$ et $3.5$) et le filtre d'Amplitude ($> 7$ pA) pour distinguer les vrais événements du bruit de fond.
4. **Inspection Visuelle** : Utilisez le graphique interactif pour vérifier les détections (points verts) et la trace de corrélation croisée (trace bleue).
5. **Exportation** : Téléchargez vos données au format `.csv` pour vos analyses statistiques.

### 5. Installation Locale
Pour les chercheurs souhaitant modifier le code source localement :

1. **Installer Python 3.9+**
2. **Installer les dépendances** :
   ```bash
   pip install streamlit pyabf numpy matplotlib scipy pandas seaborn
