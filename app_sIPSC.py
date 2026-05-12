import streamlit as st
import pyabf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, optimize, integrate, ndimage
import tempfile
import os

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Manzoni Lab - Détections des IPSCs", layout="wide")

# --- GESTION DU BILINGUISME ---
if 'lang' not in st.session_state:
    st.session_state.lang = "Français"

# --- DICTIONNAIRE DE TRADUCTION ---
T = {
    "title": {"Français": "Détections des IPSCs", "English": "IPSC Detection"},
    "subtitle": {"Français": "Analyse des courants synaptiques spontanés | Manzoni Lab", "English": "Spontaneous synaptic currents analysis | Manzoni Lab"},
    "tab_analyse": {"Français": "📈 Analyse & Détection", "English": "📈 Analysis & Detection"},
    "tab_methode": {"Français": "📚 Formalisme & Méthodes", "English": "📚 Formalism & Methods"},
    "tab_export": {"Français": "📥 Exportation", "English": "📥 Export Results"},
    "sb_preproc": {"Français": "1. Prétraitement", "English": "1. Preprocessing"},
    "sb_detec": {"Français": "2. Détection (Template)", "English": "2. Detection (Template)"},
    "sb_kinetics": {"Français": "3. Filtres Cinétiques", "English": "3. Kinetic Filters"},
    "sb_visu": {"Français": "4. Visualisation", "English": "4. Visualization"},
    "bessel_lab": {"Français": "Filtre Bessel (Hz)", "English": "Bessel Filter (Hz)"},
    "zero_meth": {"Français": "Méthode de remise à zéro", "English": "Baseline Method"},
    "thresh_lab": {"Français": "Seuil Z-Score", "English": "Z-Score Threshold"},
}

# --- NAVIGATION TEMPORELLE ---
if 'x_start' not in st.session_state: st.session_state.x_start = 10.0
if 'x_end' not in st.session_state: st.session_state.x_end = 11.0

def scroll_left():
    window = st.session_state.x_end - st.session_state.x_start
    st.session_state.x_start = max(0.0, st.session_state.x_start - window * 0.8)
    st.session_state.x_end = st.session_state.x_start + window

def scroll_right():
    window = st.session_state.x_end - st.session_state.x_start
    st.session_state.x_start += window * 0.8
    st.session_state.x_end = st.session_state.x_start + window

def calculate_rise_time(segment_y, dt):
    try:
        pk = np.argmax(segment_y)
        if pk < 3: return 0
        rising = segment_y[:pk+1]
        t_vec = np.arange(len(rising)) * dt
        p_val = rising[-1]
        t10 = np.interp(0.10 * p_val, rising, t_vec)
        t90 = np.interp(0.90 * p_val, rising, t_vec)
        return t90 - t10
    except: return 0

# --- INTERFACE ---
lang = st.sidebar.radio("Language / Langue", ["Français", "English"])

col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: st.image("logo_chavis_final.png", width=300) 
    except: st.info("Manzoni Lab") 
with col_r:
    st.markdown(f"# {T['title'][lang]}")
    st.markdown(f"### {T['subtitle'][lang]}")

st.divider()

# --- SIDEBAR : LE RETOUR DU CODE D'ORIGINE ---
st.sidebar.header(T["sb_preproc"][lang])
polarity = st.sidebar.radio("Polarité", ["Outward (GABA 0mV)", "Inward (-70mV)"], index=1)
is_outward = "Outward" in polarity

bessel_hz = st.sidebar.number_input(T["bessel_lab"][lang], value=2000)
baseline_method = st.sidebar.selectbox(T["zero_meth"][lang], ["Médian (Median)", "Polynômial (Polyfit)"])

st.sidebar.header(T["sb_detec"][lang])
threshold = st.sidebar.slider(T["thresh_lab"][lang], 1.0, 7.0, 3.5)

st.sidebar.header(T["sb_kinetics"][lang])
col1, col2 = st.sidebar.columns(2)
rise_min = col1.number_input("Rise Min (ms)", value=0.20)
rise_max = col2.number_input("Rise Max (ms)", value=5.00)
decay_min = col1.number_input("Decay Min (ms)", value=2.50) # Pour accepter 4.8ms
decay_max = col2.number_input("Decay Max (ms)", value=80.00)
amp_min = st.sidebar.number_input("Amp Min (pA)", value=5.0)

st.sidebar.header(T["sb_visu"][lang])
auto_y = st.sidebar.checkbox("Auto-scale Trace", value=True)
auto_z = st.sidebar.checkbox("Auto-scale Z-score", value=True)

col_n1, col_n2 = st.sidebar.columns(2)
col_n1.button("⬅️ Scroll", on_click=scroll_left, use_container_width=True)
col_n2.button("Scroll ➡️", on_click=scroll_right, use_container_width=True)

# --- LOGIQUE ANALYSE ---
uploaded_file = st.file_uploader("📂 Charger ABF", type=["abf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp:
        tmp.write(uploaded_file.getvalue()); tmp_path = tmp.name

    try:
        abf = pyabf.ABF(tmp_path); abf.setSweep(0)
        times, fs, dt = abf.sweepX, abf.dataRate, 1000/abf.dataRate
        
        # 1. Filtre Bessel (Scipy)
        nyq = 0.5 * fs
        b, a = signal.bessel(4, bessel_hz/nyq, btype='low')
        y_filt = signal.filtfilt(b, a, abf.sweepY)
        
        # 2. Les 2 méthodes de remise à zéro
        if "Médian" in baseline_method:
            base = ndimage.median_filter(y_filt, size=int(0.5 * fs))
        else:
            p = np.polyfit(times, y_filt, 3)
            base = np.polyval(p, times)
        
        f_data = y_filt - base
        
        # 3. Détection Template Matching
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
            s, e = p - int(0.005*fs), p + int(0.060*fs)
            if s < 0 or e >= len(f_data): continue
            seg = (f_data[s:e]) if is_outward else -(f_data[s:e])
            seg -= np.mean(seg[:int(0.004*fs)]) # Local Zero
            amp, rise = np.max(seg), calculate_rise_time(seg, dt)
            decay = abs(integrate.trapezoid(seg, dx=dt) / amp) if amp > 0 else 0
            
            if (amp >= amp_min and rise_min <= rise <= rise_max and decay_min <= decay <= decay_max):
                valid_ev.append({'idx': p, 'time': times[p], 'amp': amp, 'rise': rise, 'decay': decay})

        # --- ONGLETS ---
        tab1, tab2, tab3 = st.tabs([T["tab_analyse"][lang], T["tab_methode"][lang], T["tab_export"][lang]])

        with tab1:
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
            
            # Z-Score avec ligne de seuil
            ax2.plot(times, corr_z, color='blue', alpha=0.5)
            ax2.axhline(threshold, color='red', ls='--', label=f"Seuil: {threshold}")
            if auto_z and np.any(mask):
                ax2.set_ylim(-1, np.max(corr_z[mask])*1.2 if np.max(corr_z[mask]) > threshold else threshold+1)
            ax2.legend()
            st.pyplot(fig)

        with tab2:
            st.markdown(r"### 📄 README & Formalisme" if lang=="Français" else r"### 📄 README & Formalism")
            st.markdown(r"""
            - **Bessel Filter:** 4th order low-pass filter to reduce high-frequency noise.
            - **Baseline Correction:** Median filter (running baseline) or 3rd order Polynomial fit.
            - **Z-Score Detection:** Template matching using multiple decay constants ($\tau$).
            """)

        with tab3:
            if valid_ev:
                df = pd.DataFrame(valid_ev)
                st.subheader("Stats")
                st.table(df.describe().loc[['mean', 'min', 'max']])
                st.download_button("💾 Export CSV", df.to_csv(index=False).encode('utf-8'), "GABA_Results.csv")

    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
