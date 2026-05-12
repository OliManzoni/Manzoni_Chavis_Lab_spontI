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

# Initialisation de la mémoire (Session State) pour la navigation fluide
if 'x_start' not in st.session_state:
    st.session_state.x_start = 10.0
if 'x_end' not in st.session_state:
    st.session_state.x_end = 11.0

# --- CALLBACKS DE NAVIGATION (INDISPENSABLES) ---
def scroll_left():
    window = st.session_state.x_end - st.session_state.x_start
    st.session_state.x_start = max(0.0, st.session_state.x_start - window * 0.75)
    st.session_state.x_end = st.session_state.x_start + window

def scroll_right():
    window = st.session_state.x_end - st.session_state.x_start
    st.session_state.x_start += window * 0.75
    st.session_state.x_end = st.session_state.x_start + window

# --- DICTIONNAIRE MULTILINGUE ---
lang = st.sidebar.selectbox("Language / Langue", ["Français", "English"])
T = {
    "Français": {
        "title": "Pipeline Expert : sIPSC / sEPSC (GABA & Glutamate)",
        "sb_preproc": "1. Prétraitement & Polarité",
        "sb_detec": "2. Détection (Template Matching)",
        "sb_kinetics": "3. Filtres Cinétiques (Acceptation)",
        "summary": "📊 Statistiques des Événements Validés",
        "freq": "Fréquence (Hz)",
        "amp_mean": "Amplitude Moyenne (pA)"
    },
    "English": {
        "title": "Expert sIPSC / sEPSC Pipeline (GABA & Glutamate)",
        "sb_preproc": "1. Preprocessing & Polarity",
        "sb_detec": "2. Detection (Template Matching)",
        "sb_kinetics": "3. Kinetic Filters (Acceptance)",
        "summary": "📊 Validated Events Statistics",
        "freq": "Frequency (Hz)",
        "amp_mean": "Mean Amplitude (pA)"
    }
}[lang]

st.markdown(f"# {T['title']}")

# --- SIDEBAR : RÉGLAGES ---
st.sidebar.header(T["sb_preproc"])
polarity = st.sidebar.radio("Polarité", ["Outward (GABA 0mV)", "Inward (AMPA/GABA -70mV)"])
is_outward = "Outward" in polarity

st.sidebar.header(T["sb_detec"])
threshold = st.sidebar.slider("Seuil Z-Score", 2.0, 6.0, 3.5)

st.sidebar.header(T["sb_kinetics"])
col1, col2 = st.sidebar.columns(2)
# Ajustement GABA : On baisse les MINIMA pour ne plus rejeter les événements à 4.8ms
rise_min = col1.number_input("Rise Min (ms)", value=0.5, step=0.1)
decay_min = col1.number_input("Decay Min (ms)", value=2.5, step=0.5) 
rise_max = col2.number_input("Rise Max (ms)", value=10.0, step=0.5)
decay_max = col2.number_input("Decay Max (ms)", value=100.0, step=5.0)
amp_min = st.sidebar.number_input("Amplitude Min (pA)", value=5.0)

# --- LOGIQUE DE CALCUL ---
def get_kinetics(seg, dt):
    try:
        pk = np.argmax(seg)
        if pk < 2: return 0, 0
        # Rise 10-90%
        v10, v90 = 0.1*seg[pk], 0.9*seg[pk]
        t10 = np.where(seg[:pk] >= v10)[0][0] * dt
        t90 = np.where(seg[:pk] >= v90)[0][0] * dt
        # Decay (Charge / Amplitude)
        area = integrate.trapezoid(seg, dx=dt)
        return t90 - t10, abs(area / seg[pk])
    except: return 0, 0

file = st.file_uploader("Charger un fichier .abf", type=["abf"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        abf = pyabf.ABF(tmp_path)
        abf.setSweep(0)
        fs, dt = abf.dataRate, 1000/abf.dataRate
        
        # Filtrage médian pour la ligne de base (Indispensable)
        baseline = ndimage.median_filter(abf.sweepY, size=int(0.5 * fs))
        f_data = abf.sweepY - baseline
        
        # Template Matching Multi-échelles
        detect_trace = f_data if is_outward else -f_data
        best_corr = np.zeros_like(detect_trace)
        for tau in [5, 15, 30, 60]: # Diversité cinétique GABA
            t_tmpl = np.arange(0, 100, dt)
            tmpl = (np.exp(-t_tmpl/tau) - np.exp(-t_tmpl/(tau/10)))
            tmpl /= np.max(tmpl)
            best_corr = np.maximum(best_corr, signal.correlate(detect_trace, tmpl, mode='same'))
        
        corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
        peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.01 * fs))
        
        valid_ev = []
        for p in peaks:
            # Fenêtre d'analyse
            s, e = p - int(0.005*fs), p + int(0.100*fs)
            if s < 0 or e >= len(f_data): continue
            
            seg = (f_data[s:e]) if is_outward else -(f_data[s:e])
            seg -= np.mean(seg[:int(0.004*fs)]) # Zéro local
            
            amp = np.max(seg)
            rise, decay = get_kinetics(seg, dt)
            
            # Application des filtres corrigés
            if (amp > amp_min and 
                rise_min <= rise <= rise_max and 
                decay_min <= decay <= decay_max):
                valid_ev.append({'idx': p, 'time': abf.sweepX[p], 'amp': amp, 'rise': rise, 'decay': decay})

        # --- INTERFACE DE NAVIGATION (RESTAURÉE) ---
        st.subheader("📈 Visualisation Interactive")
        c_nav1, c_nav2, c_nav3, c_nav4 = st.columns([1,1,2,2])
        c_nav1.button("⬅️ Scroll", on_click=scroll_left)
        c_nav2.button("Scroll ➡️", on_click=scroll_right)
        
        st.session_state.x_start = c_nav3.number_input("Début (s)", value=st.session_state.x_start)
        st.session_state.x_end = c_nav4.number_input("Fin (s)", value=st.session_state.x_end)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios':[3, 1]})
        ax1.plot(abf.sweepX, f_data, color='black', lw=0.6)
        if valid_ev:
            ev_times = [e['time'] for e in valid_ev]
            ev_vals = [f_data[e['idx']] for e in valid_ev]
            ax1.plot(ev_times, ev_vals, 'o', color='purple' if is_outward else 'orange', markersize=4)
        
        ax1.set_xlim(st.session_state.x_start, st.session_state.x_end)
        ax1.set_ylabel("Amplitude (pA)")
        
        ax2.plot(abf.sweepX, corr_z, color='blue', alpha=0.4)
        ax2.axhline(threshold, color='red', ls='--')
        ax2.set_ylabel("Z-Score")
        st.pyplot(fig)

        # --- RÉSUMÉ & EXPORT (DÉTAILLÉ) ---
        if valid_ev:
            df = pd.DataFrame(valid_ev)
            st.divider()
            st.subheader(T["summary"])
            
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric(T["freq"], f"{len(df)/abf.sweepX[-1]:.2f} Hz")
            col_res2.metric(T["amp_mean"], f"{df['amp'].mean():.2f} pA")
            col_res3.metric("Decay Moyen", f"{df['decay'].mean():.2f} ms")
            
            # Tableau comparatif (Indispensable pour le réglage)
            stats_table = pd.DataFrame({
                "Paramètre": ["Amplitude", "Rise Time", "Decay Time"],
                "Moyenne (Average)": [df['amp'].mean(), df['rise'].mean(), df['decay'].mean()],
                "Limite Min (Limit)": [amp_min, rise_min, decay_min],
                "Limite Max (Limit)": ["N/A", rise_max, decay_max]
            })
            st.table(stats_table.style.format(precision=2))
            
            st.download_button("💾 Exporter CSV Complet", df.to_csv(index=False).encode('utf-8'), "GABA_Analysis.csv")

    except Exception as e: st.error(f"Erreur d'analyse : {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
