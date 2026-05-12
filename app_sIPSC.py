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
st.set_page_config(page_title="Manzoni Lab - sIPSC GABA Pipeline", layout="wide")

# Initialisation de la mémoire (Session State)
if 'fs_nyquist' not in st.session_state:
    st.session_state.fs_nyquist = 5000.0
if 'x_start' not in st.session_state:
    st.session_state.x_start = 10.0
if 'x_end' not in st.session_state:
    st.session_state.x_end = 11.0

# --- FONCTIONS DE NAVIGATION ---
def scroll_left():
    window = st.session_state.x_end - st.session_state.x_start
    shift = window * 0.8
    new_start = max(0.0, st.session_state.x_start - shift)
    st.session_state.x_end = new_start + window
    st.session_state.x_start = new_start

def scroll_right():
    window = st.session_state.x_end - st.session_state.x_start
    shift = window * 0.8
    st.session_state.x_start += shift
    st.session_state.x_end += shift

# --- LANGUAGE & TRADUCTION ---
lang = st.sidebar.selectbox("Language / Langue", ["Français", "English"])

T = {
    "Français": {
        "title": "# Pipeline Expert : sIPSC / sEPSC",
        "sb_preproc": "1. Prétraitement",
        "polarity": "Type d'événement",
        "pol_out": "GABA Outward (0mV)",
        "pol_in": "GABA/AMPA Inward (-70mV)",
        "sb_detec": "2. Seuil Statistique",
        "threshold": "Seuil Z-Score (Sensibilité)",
        "sb_kinetics": "3. Filtres Cinétiques (Rigueur)",
        "rise_min": "Rise Time Min (ms)",
        "rise_max": "Rise Time Max (ms)",
        "decay_min": "Decay Min (ms)",
        "decay_max": "Decay Max (ms)",
        "amp_filter": "Amplitude Min (pA)",
        "summary_title": "📊 Résumé des Événements Validés",
    },
    "English": {
        "title": "# Expert sIPSC / sEPSC Pipeline",
        "sb_preproc": "1. Preprocessing",
        "polarity": "Event Type",
        "pol_out": "GABA Outward (0mV)",
        "pol_in": "GABA/AMPA Inward (-70mV)",
        "sb_detec": "2. Statistical Threshold",
        "threshold": "Z-Score Threshold (Sensitivity)",
        "sb_kinetics": "3. Kinetic Filters (Stringency)",
        "rise_min": "Min Rise Time (ms)",
        "rise_max": "Max Rise Time (ms)",
        "decay_min": "Min Decay (ms)",
        "decay_max": "Max Decay (ms)",
        "amp_filter": "Min Amplitude (pA)",
        "summary_title": "📊 Validated Events Summary",
    }
}[lang]

st.title(T["title"])

# --- SIDEBAR : RÉGLAGES EXPERTS ---
st.sidebar.header(T["sb_preproc"])
polarity = st.sidebar.radio(T["polarity"], [T["pol_out"], T["pol_in"]], index=0)
is_outward = (polarity == T["pol_out"])

# 1. SEUIL DE DÉTECTION (Plus sensible pour le GABA)
st.sidebar.header(T["sb_detec"])
threshold = st.sidebar.slider(T["threshold"], 1.0, 7.0, 3.0, help="Un score de 3.0 est idéal pour le GABA.")

# 2. FILTRES CINÉTIQUES (Correction des limites pour éviter de rejeter les vrais événements)
st.sidebar.header(T["sb_kinetics"])

col1, col2 = st.sidebar.columns(2)
# Rise Time : On accepte de 0.5 à 5ms par défaut
rise_min = col1.number_input(T["rise_min"], value=0.5, step=0.1)
rise_max = col2.number_input(T["rise_max"], value=5.0, step=0.1)

col3, col4 = st.sidebar.columns(2)
# Decay Time : On baisse le minimum à 2ms pour ne pas rater les événements à 4.8ms !
decay_min = col3.number_input(T["decay_min"], value=2.0, step=1.0)
decay_max = col4.number_input(T["decay_max"], value=80.0, step=5.0)

amp_limit = st.sidebar.number_input(T["amp_filter"], value=5.0, step=1.0)

# --- LOGIQUE D'ANALYSE ---
def calculate_rise_time(segment_y, dt):
    try:
        peak_idx = np.argmax(segment_y)
        if peak_idx < 2: return 0
        rising_limb = segment_y[:peak_idx + 1]
        t_vec = np.arange(len(rising_limb)) * dt
        peak_val = rising_limb[-1]
        t10 = np.interp(0.10 * peak_val, rising_limb, t_vec)
        t90 = np.interp(0.90 * peak_val, rising_limb, t_vec)
        return t90 - t10
    except: return 0

file = st.file_uploader("Charger .abf", type=["abf"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.abf') as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        abf = pyabf.ABF(tmp_path)
        abf.setSweep(0)
        fs, times, dt = abf.dataRate, abf.sweepX, 1000/abf.dataRate
        
        # Prétraitement
        data = abf.sweepY - ndimage.median_filter(abf.sweepY, size=int(0.5 * fs))
        
        # Détection par corrélation (Template Matching)
        detect_trace = data if is_outward else -data
        best_corr = np.zeros_like(detect_trace)
        # Templates adaptés au GABA
        decays = [5.0, 15.0, 30.0, 60.0] 
        for d in decays:
            t_tmpl = np.arange(0, 50, dt)
            tmpl = (np.exp(-t_tmpl/d) - np.exp(-t_tmpl/0.8)) 
            tmpl /= np.max(np.abs(tmpl))
            best_corr = np.maximum(best_corr, signal.correlate(detect_trace, tmpl, mode='same'))
            
        corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
        peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.01 * fs))
        
        valid_ev = []
        for i, p in enumerate(peaks):
            start, end = p - int(0.005*fs), p + int(0.050*fs)
            if start < 0 or end >= len(data): continue
            
            l_base = np.mean(data[p-int(0.005*fs):p-int(0.002*fs)])
            seg = (data[start:end] - l_base) if is_outward else -(data[start:end] - l_base)
            
            amp = np.max(seg)
            rise = calculate_rise_time(seg, dt)
            area = integrate.trapezoid(seg, dx=dt)
            decay = abs(area / amp) if amp > 0 else 0
            
            # --- APPLICATION DES FILTRES CORRIGÉS ---
            if (amp >= amp_limit and 
                rise_min <= rise <= rise_max and 
                decay_min <= decay <= decay_max):
                
                valid_ev.append({
                    'idx': p, 'time': times[p], 'amp': amp, 
                    'rise': rise, 'decay': decay, 'area': abs(area)
                })

        # --- AFFICHAGE & DIAGNOSTIC ---
        st.subheader("📈 Visualisation & Détection")
        
        # Navigation
        col_n1, col_n2, col_n3, col_n4 = st.columns([1,1,2,2])
        col_n1.button("⬅️", on_click=scroll_left)
        col_n2.button("➡️", on_click=scroll_right)
        x_s = col_n3.number_input("Début (s)", value=st.session_state.x_start, key="x_start_in")
        x_e = col_n4.number_input("Fin (s)", value=st.session_state.x_end, key="x_end_in")
        st.session_state.x_start, st.session_state.x_end = x_s, x_e

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios':[2,1]})
        ax1.plot(times, data, color='black', lw=0.5)
        if valid_ev:
            ax1.plot([e['time'] for e in valid_ev], [data[e['idx']] for e in valid_ev], 'o', color='purple', markersize=4)
        
        ax1.set_xlim(st.session_state.x_start, st.session_state.x_end)
        ax1.set_ylim(-100, 100) if not is_outward else ax1.set_ylim(-50, 150)
        ax2.plot(times, corr_z, color='blue', alpha=0.5)
        ax2.axhline(threshold, color='red', ls='--')
        st.pyplot(fig)

        # --- RÉSUMÉ STATISTIQUE (Le fameux "Average") ---
        if valid_ev:
            df = pd.DataFrame(valid_ev)
            st.subheader(T["summary_title"])
            
            # Calcul des moyennes pour vérification
            stats = pd.DataFrame({
                "Paramètre": ["Amplitude (pA)", "Rise Time (ms)", "Decay (ms)"],
                "Moyenne (Average)": [df['amp'].mean(), df['rise'].mean(), df['decay'].mean()],
                "Minimum (Limit)": [amp_limit, rise_min, decay_min],
                "Maximum (Limit)": ["N/A", rise_max, decay_max]
            })
            st.table(stats.style.format(precision=2))
            
            st.download_button("💾 Télécharger CSV", df.to_csv(index=False).encode('utf-8'), "events_gaba.csv")

    except Exception as e: st.error(f"Erreur : {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
