import streamlit as st
import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, integrate, ndimage
import tempfile
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Manzoni Lab - Détection des IPSCs", layout="wide")

# --- GESTION DU BILINGUISME ---
st.sidebar.header("🌍 Language / Langue")
lang = st.sidebar.radio("Select Language:", ["Français", "English"])

# Initialisation de la mémoire (Session State) pour la navigation
if 'x_start' not in st.session_state:
    st.session_state.x_start = 10.0
if 'x_end' not in st.session_state:
    st.session_state.x_end = 11.0

# --- DICTIONNAIRE DE TRADUCTION ---
T = {
    "title": {"Français": "Détections des IPSCs", "English": "IPSC Detection"},
    "subtitle": {"Français": "Analyse des courants synaptiques spontanés | Manzoni Lab", "English": "Spontaneous synaptic currents analysis | Manzoni Lab"},
    "tab_analyse": {"Français": "📈 Analyse & Détection", "English": "📈 Analysis & Detection"},
    "tab_methode": {"Français": "📚 Formalisme & Méthodes", "English": "📚 Formalism & Methods"},
    "tab_export": {"Français": "📥 Exportation", "English": "📥 Export Results"},
    "settings_1": {"Français": "1. Type & Seuil", "English": "1. Type & Threshold"},
    "settings_2": {"Français": "2. Filtres Cinétiques", "English": "2. Kinetic Filters"},
    "settings_3": {"Français": "3. Visualisation", "English": "3. Visualisation"},
    "polarity": {"Français": "Polarité", "English": "Polarity"},
    "pol_out": {"Français": "Outward (GABA 0mV)", "English": "Outward (GABA 0mV)"},
    "pol_in": {"Français": "Inward (AMPA/GABA -70mV)", "English": "Inward (AMPA/GABA -70mV)"},
    "thresh_label": {"Français": "Seuil Z-Score", "English": "Z-Score Threshold"},
    "auto_y": {"Français": "Auto-scale Trace Y", "English": "Auto-scale Trace Y"},
    "auto_z": {"Français": "Auto-scale Z-score", "English": "Auto-scale Z-score"},
}

# --- CALLBACKS DE NAVIGATION ---
def scroll_left():
    window = st.session_state.x_end - st.session_state.x_start
    shift = window * 0.8
    st.session_state.x_start = max(0.0, st.session_state.x_start - shift)
    st.session_state.x_end = st.session_state.x_start + window

def scroll_right():
    window = st.session_state.x_end - st.session_state.x_start
    shift = window * 0.8
    st.session_state.x_start += shift
    st.session_state.x_end += shift

def calculate_rise_time(segment_y, dt):
    try:
        pk = np.argmax(segment_y)
        if pk < 3: return 0
        rising = segment_y[:pk + 1]
        t_vec = np.arange(len(rising)) * dt
        p_val = rising[-1]
        t10 = np.interp(0.10 * p_val, rising, t_vec)
        t90 = np.interp(0.90 * p_val, rising, t_vec)
        return t90 - t10
    except: return 0

# --- EN-TÊTE INSTITUTIONNEL ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: 
        st.image("logo_chavis_final.png", width=300) 
    except: 
        st.info("Manzoni Lab - Neurosciences") 
with col_r:
    st.markdown(f"# {T['title'][lang]}")
    st.markdown(f"### {T['subtitle'][lang]}")

st.divider()

# --- BARRE LATÉRALE ---
st.sidebar.header(T["settings_1"][lang])
polarity = st.sidebar.radio(T["polarity"][lang], [T["pol_out"][lang], T["pol_in"][lang]], index=1)
is_outward = "Outward" in polarity
threshold = st.sidebar.slider(T["thresh_label"][lang], 1.0, 7.0, 3.5)

st.sidebar.header(T["settings_2"][lang])
c1, c2 = st.sidebar.columns(2)
rise_min = c1.number_input("Rise Min (ms)", value=0.20)
rise_max = c2.number_input("Rise Max (ms)", value=5.00)
c3, c4 = st.sidebar.columns(2)
decay_min = c3.number_input("Decay Min (ms)", value=2.00) 
decay_max = c4.number_input("Decay Max (ms)", value=80.00)
amp_min = st.sidebar.number_input("Amplitude Min (pA)", value=5.0)

st.sidebar.header(T["settings_3"][lang])
auto_y = st.sidebar.checkbox(T["auto_y"][lang], value=True)
auto_z = st.sidebar.checkbox(T["auto_z"][lang], value=True)

st.sidebar.markdown("---")
col_n1, col_n2 = st.sidebar.columns(2)
col_n1.button("⬅️ Scroll", on_click=scroll_left, use_container_width=True)
col_n2.button("Scroll ➡️", on_click=scroll_right, use_container_width=True)

# --- CHARGEMENT DU FICHIER ---
uploaded_file = st.file_uploader("📂 Charger un fichier ABF", type=["abf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_filepath = tmp_file.name

    try:
        abf = pyabf.ABF(tmp_filepath)
        abf.setSweep(0)
        times, fs, dt = abf.sweepX, abf.dataRate, 1000/abf.dataRate
        
        # Prétraitement (Filtre Médian)
        f_data = abf.sweepY - ndimage.median_filter(abf.sweepY, size=int(0.5 * fs))
        
        # Détection Template Matching
        trace_detec = f_data if is_outward else -f_data
        best_corr = np.zeros_like(trace_detec)
        for t in [5, 15, 30, 60]:
            t_tmpl = np.arange(0, 50, dt)
            tmpl = (np.exp(-t_tmpl/t) - np.exp(-t_tmpl/0.5))
            tmpl /= np.max(np.abs(tmpl))
            best_corr = np.maximum(best_corr, signal.correlate(trace_detec, tmpl, mode='same'))
            
        corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
        peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.01 * fs))
        
        valid_ev = []
        for p in peaks:
            s, e = p - int(0.005*fs), p + int(0.060*fs)
            if s < 0 or e >= len(f_data): continue
            seg = (f_data[s:e]) if is_outward else -(f_data[s:e])
            seg -= np.mean(seg[:int(0.004*fs)])
            amp, rise = np.max(seg), calculate_rise_time(seg, dt)
            area = integrate.trapezoid(seg, dx=dt)
            decay = abs(area / amp) if amp > 0 else 0
            
            if (amp >= amp_min and rise_min <= rise <= rise_max and decay_min <= decay <= decay_max):
                valid_ev.append({'idx': p, 'time': times[p], 'amp': amp, 'rise': rise, 'decay': decay})

        # --- ONGLETS ---
        tab1, tab2, tab3 = st.tabs([T["tab_analyse"][lang], T["tab_methode"][lang], T["tab_export"][lang]])

        with tab1:
            st.subheader("Visualisation & Diagnostic")
            cx1, cx2 = st.columns(2)
            st.session_state.x_start = cx1.number_input("Start (s)", value=st.session_state.x_start, step=0.1)
            st.session_state.x_end = cx2.number_input("End (s)", value=st.session_state.x_end, step=0.1)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios':[3,1]})
            ax1.plot(times, f_data, color='black', lw=0.6)
            if valid_ev:
                ax1.plot([e['time'] for e in valid_ev], [f_data[e['idx']] for e in valid_ev], 'o', color='purple' if is_outward else '#FF8C00', markersize=5)
            
            mask = (times >= st.session_state.x_start) & (times <= st.session_state.x_end)
            if auto_y and np.any(mask):
                y_loc = f_data[mask]
                ax1.set_ylim(np.min(y_loc)*1.3, np.max(y_loc)*1.3)
            ax1.set_xlim(st.session_state.x_start, st.session_state.x_end)
            ax1.set_ylabel("Amplitude (pA)")
            
            # Ligne rouge de seuil sur le Z-score
            ax2.plot(times, corr_z, color='blue', alpha=0.5)
            ax2.axhline(threshold, color='red', ls='--', label=f"Seuil: {threshold}")
            if auto_z and np.any(mask):
                z_loc = corr_z[mask]
                ax2.set_ylim(-1, np.max(z_loc)*1.2 if np.max(z_loc) > threshold else threshold + 1)
            ax2.set_ylabel("Z-Score")
            ax2.legend(loc='upper right')
            st.pyplot(fig)

        with tab2:
            if lang == "Français":
                st.markdown(r"""
                ### 📄 README & Formalisme Biophysique
                
                **Objectif :** Détection automatique des courants synaptiques (sIPSCs / sEPSCs) par Template Matching.
                
                **1. Prétraitement :**
                La ligne de base est corrigée par un filtre médian (fenêtre de 500ms). Cela élimine les dérives lentes sans altérer la forme des événements rapides.
                
                **2. Détection (Z-Score) :**
                L'algorithme calcule la corrélation entre le signal et plusieurs modèles (templates) de décroissances exponentielles ($\tau = 5, 15, 30, 60 ms$). 
                Le **Z-Score** est défini comme : $Z = \frac{Corr - \mu_{Corr}}{\sigma_{Corr}}$. 
                Un événement est marqué si $Z > Seuil$.
                
                **3. Validation Cinétique :**
                Chaque pic détecté est soumis à des filtres de forme :
                * **Rise Time (10-90%)** : Élimine les bruits haute fréquence (trop rapides) ou les artefacts (trop lents).
                * **Decay Time** : Calculé par le rapport Surface/Amplitude. C'est le critère clé pour identifier les récepteurs GABA_A.
                """)
            else:
                st.markdown(r"""
                ### 📄 README & Biophysical Formalism
                
                **Goal:** Automated detection of synaptic currents (sIPSCs / sEPSCs) via Template Matching.
                
                **1. Preprocessing:**
                Baseline is corrected using a median filter (500ms window). This removes slow drifts without distorting fast event kinetics.
                
                **2. Detection (Z-Score):**
                The algorithm computes the correlation between the signal and several exponential decay templates ($\tau = 5, 15, 30, 60 ms$). 
                The **Z-Score** is defined as: $Z = \frac{Corr - \mu_{Corr}}{\sigma_{Corr}}$. 
                An event is flagged if $Z > Threshold$.
                
                **3. Kinetic Validation:**
                Each detected peak must pass shape-based filters:
                * **Rise Time (10-90%)**: Rejects high-frequency noise (too fast) or artifacts (too slow).
                * **Decay Time**: Calculated as the Area/Amplitude ratio. This is the key criterion for identifying GABA_A receptors.
                """)

        with tab3:
            if valid_ev:
                df = pd.DataFrame(valid_ev)
                st.subheader("📊 Résultats Statistiques")
                st.table(df.describe().loc[['mean', 'min', 'max']])
                st.download_button("💾 Exporter CSV", df.to_csv(index=False).encode('utf-8'), "GABA_Analysis.csv")

    finally:
        if os.path.exists(tmp_filepath): os.remove(tmp_filepath)
