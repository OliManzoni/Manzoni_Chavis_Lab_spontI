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
st.set_page_config(page_title="Détections des IPSCs", layout="wide")

# Initialisation de la mémoire (Session State) pour la navigation temporelle
if 'x_start' not in st.session_state:
    st.session_state.x_start = 10.0
if 'x_end' not in st.session_state:
    st.session_state.x_end = 11.0

# --- FONCTIONS DE NAVIGATION (Callbacks) ---
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

def calculate_rise_time(segment_y, dt):
    try:
        peak_idx = np.argmax(segment_y)
        if peak_idx < 3: return 0
        rising_limb = segment_y[:peak_idx + 1]
        t_vec = np.arange(len(rising_limb)) * dt
        peak_val = rising_limb[-1]
        t10 = np.interp(0.10 * peak_val, rising_limb, t_vec)
        t90 = np.interp(0.90 * peak_val, rising_limb, t_vec)
        return t90 - t10
    except: return 0

# --- GESTION DU BILINGUISME ---
lang = st.sidebar.selectbox("Language / Langue", ["Français", "English"])

T = {
    "title": {"Français": "Détections des IPSCs", "English": "IPSC Detection"},
    "subtitle": {"Français": "Analyse des courants synaptiques spontanés | Manzoni Lab", "English": "Spontaneous synaptic currents analysis | Manzoni Lab"},
    "sb_preproc": {"Français": "1. Prétraitement", "English": "1. Preprocessing"},
    "sb_detec": {"Français": "2. Détection (Template)", "English": "2. Detection (Template)"},
    "sb_kinetics": {"Français": "3. Filtres Cinétiques", "English": "3. Kinetic Filters"},
}

# --- EN-TÊTE ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: st.image("logo_chavis_final.png", width=300) 
    except: st.info("Manzoni Lab") 
with col_r:
    st.markdown(f"# {T['title'][lang]}")
    st.markdown(f"### {T['subtitle'][lang]}")

st.divider()

# --- SIDEBAR : RÉGLAGES ---
st.sidebar.header(T["sb_preproc"][lang])
polarity = st.sidebar.radio("Polarité", ["Outward (GABA 0mV)", "Inward (-70mV)"], index=1)
is_outward = "Outward" in polarity

bessel_hz = st.sidebar.number_input("Filtre Bessel (Hz)", value=2000)
baseline_method = st.sidebar.selectbox("Méthode de remise à zéro", ["Médian (Median)", "Polynômial (Polyfit)"])

st.sidebar.header(T["sb_detec"][lang])
threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 7.0, 3.5)

st.sidebar.header(T["sb_kinetics"][lang])
# Ajout des bornes MIN et MAX pour Rise et Decay
c1, c2 = st.sidebar.columns(2)
rise_min = c1.number_input("Rise Min (ms)", value=0.1)
rise_max = c2.number_input("Rise Max (ms)", value=5.0)

c3, c4 = st.sidebar.columns(2)
decay_min = c3.number_input("Decay Min (ms)", value=2.0) # Permet d'accepter le 4.8ms
decay_max = c4.number_input("Decay Max (ms)", value=80.0)

area_min = st.sidebar.number_input("Area Min (pA.ms)", value=10.0)
amp_min = st.sidebar.number_input("Amplitude Min (pA)", value=5.0)

st.sidebar.header("4. Visualisation")
auto_y = st.sidebar.checkbox("Auto-scale Trace Y", value=True)
auto_z = st.sidebar.checkbox("Auto-scale Z-score axis", value=True)

col_n1, col_n2 = st.sidebar.columns(2)
col_n1.button("⬅️ Scroll", on_click=scroll_left, use_container_width=True)
col_n2.button("Scroll ➡️", on_click=scroll_right, use_container_width=True)

# --- ANALYSE ---
file = st.file_uploader("📂 Charger un fichier ABF", type=["abf"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.abf') as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        abf = pyabf.ABF(tmp_path)
        abf.setSweep(0)
        times, fs, dt = abf.sweepX, abf.dataRate, 1000/abf.dataRate
        
        # 1. Filtre Bessel
        nyq = 0.5 * fs
        b, a = signal.bessel(4, bessel_hz/nyq, btype='low')
        y_filt = signal.filtfilt(b, a, abf.sweepY)
        
        # 2. Baseline
        if "Médian" in baseline_method:
            base = ndimage.median_filter(y_filt, size=int(0.5 * fs))
        else:
            p = np.polyfit(times, y_filt, 3)
            base = np.polyval(p, times)
        
        f_data = y_filt - base
        
        # 3. Détection
        detect_trace = f_data if is_outward else -f_data
        best_corr = np.zeros_like(detect_trace)
        for t_val in [5, 15, 30, 60]:
            t_tmpl = np.arange(0, 50, dt)
            tmpl = (np.exp(-t_tmpl/t_val) - np.exp(-t_tmpl/0.5))
            tmpl /= np.max(np.abs(tmpl))
            best_corr = np.maximum(best_corr, signal.correlate(detect_trace, tmpl, mode='same'))
            
        corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
        peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.01 * fs))
        
        valid_ev = []
        for p in peaks:
            start, end = p - int(0.005*fs), p + int(0.060*fs)
            if start < 0 or end >= len(f_data): continue
            
            seg = (f_data[start:end]) if is_outward else -(f_data[start:end])
            seg -= np.mean(seg[:int(0.004*fs)]) 
            
            amp = np.max(seg)
            rise = calculate_rise_time(seg, dt)
            area = integrate.trapezoid(seg, dx=dt)
            decay = abs(area / amp) if amp > 0 else 0
            
            # Application des filtres corrigés (Plages)
            if (amp >= amp_min and 
                rise_min <= rise <= rise_max and 
                decay_min <= decay <= decay_max and 
                abs(area) >= area_min):
                valid_ev.append({'idx': p, 'time': times[p], 'amp': amp, 'rise': rise, 'decay': decay, 'area': abs(area)})

        # --- VISUALISATION ---
        tab1, tab2, tab3 = st.tabs(["📈 Analyse", "📚 Méthodes", "📥 Export"])

        with tab1:
            col_x1, col_x2 = st.columns(2)
            st.session_state.x_start = col_x1.number_input("Start (s)", value=st.session_state.x_start, step=0.1)
            st.session_state.x_end = col_x2.number_input("End (s)", value=st.session_state.x_end, step=0.1)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios':[3,1]})
            ax1.plot(times, f_data, color='black', lw=0.6)
            if valid_ev:
                ax1.plot([e['time'] for e in valid_ev], [f_data[e['idx']] for e in valid_ev], 'o', color='purple' if is_outward else '#FF8C00', markersize=5)
            
            mask = (times >= st.session_state.x_start) & (times <= st.session_state.x_end)
            if auto_y and np.any(mask):
                y_loc = f_data[mask]
                ax1.set_ylim(np.min(y_loc)*1.3, np.max(y_loc)*1.3)
            ax1.set_xlim(st.session_state.x_start, st.session_state.x_end)
            
            ax2.plot(times, corr_z, color='blue', alpha=0.5)
            ax2.axhline(threshold, color='red', ls='--')
            if auto_z and np.any(mask):
                ax2.set_ylim(-1, np.max(corr_z[mask])*1.2 if np.max(corr_z[mask]) > threshold else threshold+1)
            
            st.pyplot(fig)

        with tab2:
            st.markdown(r"### 📄 Formalisme" if lang=="Français" else r"### 📄 Formalism")
            st.markdown(r"""
            - **Bessel Filter:** 4th order low-pass filter.
            - **Baseline:** Median or Polyfit subtraction.
            - **Template Matching:** Multi-tau exponential sliding window.
            """)

        with tab3:
            if valid_ev:
                df = pd.DataFrame(valid_ev)
                st.table(df.describe().loc[['mean', 'min', 'max']])
                st.download_button("💾 Export CSV", df.to_csv(index=False).encode('utf-8'), "GABA_events.csv")

    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
