import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize, integrate, ndimage
import tempfile
import os
import pandas as pd

try:
    import pyabf
except ImportError:
    st.error("Le module pyabf n'est pas installé. Exécutez : pip install pyabf")

# --- CONFIGURATION ---
st.set_page_config(page_title="sEPSC/sIPSC Pipeline", layout="wide")

# Initialisation de la mémoire (Session State) pour la navigation temporelle
if 'fs_nyquist' not in st.session_state:
    st.session_state.fs_nyquist = 5000.0
if 'x_start' not in st.session_state:
    st.session_state.x_start = 10.0
if 'x_end' not in st.session_state:
    st.session_state.x_end = 11.0

# --- FONCTIONS DE NAVIGATION (Callbacks) ---
def scroll_left():
    window = st.session_state.x_end - st.session_state.x_start
    shift = window * 0.8 # Décalage de 80% de la fenêtre (laisse un chevauchement)
    new_start = max(0.0, st.session_state.x_start - shift)
    st.session_state.x_end = new_start + window
    st.session_state.x_start = new_start

def scroll_right():
    window = st.session_state.x_end - st.session_state.x_start
    shift = window * 0.8
    st.session_state.x_start += shift
    st.session_state.x_end += shift

# --- LANGUAGE SELECTION ---
lang = st.sidebar.selectbox("Language / Langue", ["English", "Français"])

T = {
    "English": {
        "title": "# Expert Electrophysiology Pipeline",
        "sb_preproc": "1. Preprocessing & Polarity",
        "polarity": "Signal Polarity",
        "pol_out": "Outward (Positive)",
        "pol_in": "Inward (Negative)",
        "baseline_method": "Baseline Mode",
        "dyn_detrend": "Dynamic Detrending (Median)",
        "stat_detrend": "Static Global Median",
        "cutoff": "Bessel Cutoff (Hz)",
        "sb_detec": "2. Detection Threshold",
        "threshold": "Z-Score Threshold",
        "sb_viz": "3. Visualization & Navigation",
        "zoom_y": "Zoom Y (pA)",
        "x_start": "Start (s)",
        "x_end": "End (s)",
        "auto_z": "Auto-scale Z-score axis",
        "viz_header": "Visualization & Detection",
        "btn_left": "⬅️ Left",
        "btn_right": "Right ➡️"
    },
    "Français": {
        "title": "# Pipeline Expert Électrophysiologie",
        "sb_preproc": "1. Prétraitement & Polarité",
        "polarity": "Polarité du Signal",
        "pol_out": "Sortant (Positif)",
        "pol_in": "Entrant (Négatif)",
        "baseline_method": "Mode de Ligne de Base",
        "dyn_detrend": "Detrending Dynamique (Médiane)",
        "stat_detrend": "Médiane Globale Statique",
        "cutoff": "Coupure Bessel (Hz)",
        "sb_detec": "2. Seuil de Détection",
        "threshold": "Seuil Z-Score",
        "sb_viz": "3. Visualisation & Navigation",
        "zoom_y": "Zoom Y (pA)",
        "x_start": "Début (s)",
        "x_end": "Fin (s)",
        "auto_z": "Auto-ajustement axe Z",
        "viz_header": "Visualisation & Détection",
        "btn_left": "⬅️ Gauche",
        "btn_right": "Droite ➡️"
    }
}[lang]

st.title(T["title"])
st.divider()

# --- SIDEBAR ---
st.sidebar.header(T["sb_preproc"])
polarity = st.sidebar.radio(T["polarity"], [T["pol_out"], T["pol_in"]])
is_outward = (polarity == T["pol_out"])
baseline_mode = st.sidebar.radio(T["baseline_method"], [T["dyn_detrend"], T["stat_detrend"]], index=0)
use_bessel = st.sidebar.checkbox("Bessel Filter", value=True)
cutoff = st.sidebar.slider(T["cutoff"], 100, int(st.session_state.fs_nyquist), 2000)

st.sidebar.header(T["sb_detec"])
threshold = st.sidebar.slider(T["threshold"], 1.0, 8.0, 2.5)

st.sidebar.header(T["sb_viz"])
y_zoom = st.sidebar.slider(T["zoom_y"], -400, 400, (-80, 50))

auto_z = st.sidebar.checkbox(T["auto_z"], value=True)

# Navigation Temporelle (Boutons + Inputs)
st.sidebar.markdown("**Navigation Temporelle X (s)**")
col_b1, col_b2 = st.sidebar.columns(2)
col_b1.button(T["btn_left"], on_click=scroll_left, use_container_width=True)
col_b2.button(T["btn_right"], on_click=scroll_right, use_container_width=True)

col_x1, col_x2 = st.sidebar.columns(2)
# On lie les widgets directement aux variables de la session via le paramètre "key"
col_x1.number_input(T["x_start"], step=0.1, key="x_start")
col_x2.number_input(T["x_end"], step=0.1, key="x_end")

# Protection mathématique si l'utilisateur inverse début et fin
if st.session_state.x_start >= st.session_state.x_end:
    st.sidebar.error("Le début doit être inférieur à la fin.")
x_zoom = (st.session_state.x_start, st.session_state.x_end)


# --- ANALYSE ---
file = st.file_uploader("Charger .abf", type=["abf"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.abf') as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        abf = pyabf.ABF(tmp_path)
        abf.setSweep(0)
        fs, times, dt = abf.dataRate, abf.sweepX, 1000/abf.dataRate
        st.session_state.fs_nyquist = fs / 2

        # Ligne de base
        if baseline_mode == T["dyn_detrend"]:
            raw_data = ndimage.median_filter(abf.sweepY, size=int(0.5 * fs))
            raw_data = abf.sweepY - raw_data
        else:
            raw_data = abf.sweepY - np.median(abf.sweepY)

        # Filtrage
        f_data = raw_data
        if use_bessel:
            nyq = 0.5 * fs
            b, a = signal.bessel(4, cutoff/nyq, btype='low', analog=False)
            f_data = signal.filtfilt(b, a, raw_data)

        # Détection (Template Matching)
        detect_trace = f_data if is_outward else -f_data
        
        best_corr = np.zeros_like(detect_trace)
        # Adapté pour AMPA/GABA selon la polarité choisie
        decays = [10.0, 20.0, 30.0, 50.0] if is_outward else [2.0, 5.0, 10.0, 15.0]
        
        for d in decays:
            t_tmpl = np.arange(0, 30, dt)
            tmpl = (np.exp(-t_tmpl/d) - np.exp(-t_tmpl/0.5)) 
            tmpl /= np.max(np.abs(tmpl))
            best_corr = np.maximum(best_corr, signal.correlate(detect_trace, tmpl, mode='same'))
            
        corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
        peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.005 * fs))

        # --- PLOTTING ---
        st.subheader(T["viz_header"])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
        
        ax1.plot(times, f_data, color='black', lw=0.5)
        
        # Affichage des événements validés (simplifié pour la visualisation rapide)
        # Dans un pipeline complet, on garde la boucle de filtrage cinétique ici.
        if len(peaks) > 0:
            ax1.plot(times[peaks], f_data[peaks], 'o', color='purple' if is_outward else '#FF8C00', markersize=4)

        ax1.set_ylim(y_zoom)
        ax1.set_xlim(x_zoom)
        ax1.set_ylabel("pA")

        ax2.plot(times, corr_z, color='blue', alpha=0.6)
        ax2.axhline(threshold, color='red', ls='--')
        ax2.set_ylabel("Z-Score")
        
        # LOGIQUE D'AUTO-AJUSTEMENT DE L'AXE Z-SCORE
        if auto_z:
            mask = (times >= st.session_state.x_start) & (times <= st.session_state.x_end)
            if np.any(mask):
                z_local = corr_z[mask]
                z_min, z_max = np.min(z_local), np.max(z_local)
                margin = abs(z_max - z_min) * 0.15 if z_max != z_min else 1.0
                ax2.set_ylim(z_min - margin, z_max + margin)

        st.pyplot(fig)

    except Exception as e: st.error(f"Error: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
