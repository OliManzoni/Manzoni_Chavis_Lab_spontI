🟣 Expert sIPSC Pipeline: Iterative Template Matching & GABAergic Kinetics
Manzoni & Chavis Labs | Synaptic Plasticity & Biophysics

(Le README en français se trouve dans la seconde moitié de ce document / The French README is located in the second half of this document).

👉 Online Access: https://manzonilab-sipsc.streamlit.app/ (Update with your actual link)
🇬🇧 ENGLISH VERSION
1. Scientific Introduction

The analysis of spontaneous inhibitory post-synaptic currents (sIPSCs) is fundamental to deciphering the GABAergic tone and the excitation/inhibition balance in neural networks (e.g., in FXS or Reeler models).

However, classic threshold-based detection methods suffer from stochastic electrical noise and operator bias, threatening reproducibility. This Streamlit application acts as an expert workstation using an advanced Iterative Template Matching algorithm. It creates a noise-free, cell-specific fingerprint to detect GABA$_A$ events even when buried in noise, and mathematically extracts the exact synaptic charge.
2. Key Features

    Adaptive Polarity (Inward/Outward): Automatically adapts to the chloride driving force (Vm​−ECl​), whether you record outward currents (e.g., at 0 mV) or inward currents (e.g., at −70 mV with high intracellular chloride).

    Tier 1 (Fingerprint Generation): Scans the trace with multiple long-decay templates (5,10,20,30 ms) to capture the slower kinetics of GABA$_A$ receptors. The highest signal-to-noise ratio events are biologically aligned (Biological Snapping) and averaged to build a perfect, noise-free cell fingerprint.

    Tier 2 (Iterative Detection & Scaling): The fingerprint slides across the trace using a robust Z-Score. Detected events are analyzed via Least Squares Scaling, isolating the true quantal amplitude from background noise.

    Dendritic Filtering Analysis: Calculates the 10-90% Rise Time via linear interpolation to estimate synaptic location and dendritic signal attenuation.

    Dual Export System: Exports raw events (Tier 1) and iteratively scaled events (Tier 2) with frequency, amplitude, and synaptic charge, directly formatted for high-impact statistical analysis.

3. Algorithms and Mathematics
Least Squares Amplitude Scaling & Synaptic Charge

Instead of reading the raw peak height—which is highly vulnerable to patch-clamp stochastic noise—the algorithm scales the perfect cellular fingerprint (Template) to optimally fit each localized event.

The scaling factor (s) is calculated using the dot product (Least Squares approximation):
s=∑(Template2)∑(Signal×Template)​

This factor s becomes the true Scaled Amplitude.
Because the Template's area is perfectly defined, the Synaptic Charge (Area under the curve, reflecting the exact number of opened GABA$A$ receptors) is robustly computed as:
$$Charge{scaled} = s \times \int_{0}^{\infty} Template(t) dt$$
4. User Guide

    Upload Data: Load your .abf file.

    Preprocessing: Select the Signal Polarity ("Inward" or "Outward").

    Thresholding (Z-Score): Adjust the robust Z-Score (typically 2.5 to 4.0) to define the strictness of the detection.

    Kinetics Limits: Apply absolute amplitude or rise-time filters to exclude artifacts.

    Visual Inspection: Review the two-pass graphs. The right panel displays the perfect cell fingerprint (Tier 1), while the left panel overlays the Tier 2 scaled detections.

    Export: Download the .csv files containing all metric distributions and scaled events.

5. Citation

If you use this software or its mathematical architecture in your research, please include the following citation and DOI:

    Manzoni Lab (2026). Expert Pipeline: sIPSC Iterative Template Matching.
    DOI: 10.5281/zenodo.19920540
    GitHub: github.com/OliManzoni/Manzoni_Chavis_Lab_Ephys_Suite



🇫🇷 VERSION FRANÇAISE
1. Introduction Scientifique

L'analyse des courants post-synaptiques inhibiteurs spontanés (sIPSC) est fondamentale pour décrypter le tonus GABAergique et la balance excitation/inhibition au sein des réseaux neuronaux (notamment dans les modèles FXS ou Reeler).

Cependant, les méthodes classiques de détection par seuil souffrent du bruit électrique stochastique et des biais de l'opérateur, menaçant la reproductibilité. Cette application Streamlit opère comme une station de travail experte utilisant un algorithme avancé de Template Matching Itératif. Elle crée une "empreinte" cellulaire parfaite et sans bruit pour détecter les événements GABA$_A$, et extrait mathématiquement la charge synaptique exacte.
2. Fonctionnalités Principales

    Polarité Adaptative (Entrant/Sortant) : S'adapte automatiquement à la force électromotrice du chlore (Vm​−ECl​), que vous enregistriez des courants sortants (ex: à 0 mV) ou entrants (ex: à −70 mV avec haut chlore intracellulaire).

    Passe 1 (Création de l'Empreinte) : Scanne la trace avec de longs gabarits de décroissance (5,10,20,30 ms) pour capturer les cinétiques lentes du GABA$_A$. Les événements les plus nets sont alignés (Biological Snapping) et moyennés pour construire l'empreinte parfaite de la cellule.

    Passe 2 (Détection Itérative & Mise à l'échelle) : L'empreinte glisse sur la trace via un Z-Score robuste. Les événements détectés sont analysés par la méthode des Moindres Carrés, isolant l'amplitude quantique réelle du bruit de fond.

    Analyse du Filtrage Dendritique : Calcule le temps de montée 10-90% (Rise Time) par interpolation linéaire pour estimer la localisation synaptique et l'atténuation du signal.

    Double Export : Exporte les événements bruts (Passe 1) et itératifs (Passe 2) avec leur fréquence, amplitude et charge synaptique, prêts pour les analyses statistiques de haut niveau.

3. Algorithmes et Mathématiques
Mise à l'Échelle par Moindres Carrés & Charge Synaptique

Au lieu de lire la hauteur brute du pic (très vulnérable au bruit stochastique du patch-clamp), l'algorithme met à l'échelle l'empreinte cellulaire parfaite (le Gabarit) pour qu'elle épouse au mieux chaque événement.

Le facteur d'échelle (s) est calculé via le produit scalaire (Moindres Carrés) :
s=∑(Gabarit2)∑(Signal×Gabarit)​

Ce facteur s devient l'Amplitude Scaled (mise à l'échelle) véritable.
L'aire du gabarit étant parfaitement définie, la Charge Synaptique (l'aire sous la courbe, reflétant le nombre exact de récepteurs GABA$A$ ouverts) est calculée de manière implacable :
$$Charge{scaled} = s \times \int_{0}^{\infty} Gabarit(t) dt$$
4. Guide d'Utilisation

    Chargement : Uploadez votre fichier .abf.

    Prétraitement : Sélectionnez la polarité du signal ("Entrant" ou "Sortant").

    Seuils (Z-Score) : Ajustez le Z-Score robuste (typiquement entre 2.5 et 4.0) pour définir la rigueur de la détection.

    Filtres Cinétiques : Appliquez des limites d'amplitude absolue ou de temps de montée pour exclure les artefacts.

    Inspection Visuelle : Observez les graphiques de la double passe. Le panneau de droite affiche l'empreinte parfaite (Passe 1), tandis que celui de gauche superpose les détections itératives (Passe 2).

    Exportation : Téléchargez les fichiers .csv contenant les distributions et les métriques des événements mis à l'échelle.

5. Citation

Si vous utilisez ce logiciel ou son architecture mathématique pour vos recherches, merci d'inclure la citation et le DOI suivants :

    Manzoni Lab (2026). Expert Pipeline: sIPSC Iterative Template Matching.
    DOI: 10.5281/zenodo.19920540
    GitHub: github.com/OliManzoni/Manzoni_Chavis_Lab_Ephys_Suite
