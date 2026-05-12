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
st.set_page_config(page_title="sIPSC Pipeline", layout="wide")

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
    shift = window * 0.8
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
        "pol_out": "Outward (Positive, e.g. IPSC at 0mV)",
        "pol_in": "Inward (Negative, e.g. EPSC or IPSC high Cl-)",
        "baseline_method": "Baseline Mode",
        "dyn_detrend": "Dynamic Detrending (Median)",
        "stat_detrend": "Static Global Median",
        "cutoff": "Bessel Cutoff (Hz)",
        "sb_detec": "2. Detection Threshold",
        "threshold": "Z-Score Threshold",
        "sb_kinetics": "3. Kinetics Filters",
        "decay_thresh": "Max Decay (ms)",
        "rise_thresh": "Max Rise Time (ms)",
        "amp_filter": "Min Absolute Amplitude (pA)",
        "sb_viz": "4. Visualization & Navigation",
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
        "pol_out": "Sortant (Positif, ex: IPSC à 0mV)",
        "pol_in": "Entrant (Négatif, ex: EPSC ou IPSC haut Cl-)",
        "baseline_method": "Mode de Ligne de Base",
        "dyn_detrend": "Detrending Dynamique (Médiane)",
        "stat_detrend": "Médiane Globale Statique",
        "cutoff": "Coupure Bessel (Hz)",
        "sb_detec": "2. Seuil de Détection",
        "threshold": "Seuil Z-Score",
        "sb_kinetics": "3. Filtres Cinétiques",
        "decay_thresh": "Decay Max (ms)",
        "rise_thresh": "Rise Time Max (ms)",
        "amp_filter": "Amplitude Absolue Min (pA)",
        "sb_viz": "4. Visualisation & Navigation",
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
polarity = st.sidebar.radio(T["polarity"], [T["pol_out"], T["pol_in"]], index=1)
is_outward = (polarity == T["pol_out"])
baseline_mode = st.sidebar.radio(T["baseline_method"], [T["dyn_detrend"], T["stat_detrend"]], index=0)
use_bessel = st.sidebar.checkbox("Bessel Filter", value=True)
cutoff = st.sidebar.slider(T["cutoff"], 100, int(st.session_state.fs_nyquist), 2000)

st.sidebar.header(T["sb_detec"])
threshold = st.sidebar.slider(T["threshold"], 1.0, 8.0, 2.5)

st.sidebar.header(T["sb_kinetics"])
use_amp_filter = st.sidebar.checkbox("Filter Amplitude", value=True)
# CORRECTION ICI : min_value=0.0 et libellé "Amplitude Absolue"
amp_limit = st.sidebar.number_input(T["amp_filter"], min_value=0.0, value=5.0 if is_outward else 7.0, step=1.0)

use_decay_filter = st.sidebar.checkbox("Filter Decay", value=True)
decay_limit = st.sidebar.number_input(T["decay_thresh"], value=50.0 if is_outward else 4.0, step=1.0)

use_rise_filter = st.sidebar.checkbox("Filter Rise Time", value=True)
rise_limit = st.sidebar.number_input(T["rise_thresh"], value=5.0 if is_outward else 1.5, step=0.1)

st.sidebar.header(T["sb_viz"])
y_default = (-50, 200) if is_outward else (-200, 50)
y_zoom = st.sidebar.slider(T["zoom_y"], -400, 400, y_default)

auto_z = st.sidebar.checkbox(T["auto_z"], value=True)

# Navigation Temporelle (Boutons + Inputs)
st.sidebar.markdown("**Navigation Temporelle X (s)**")
col_b1, col_b2 = st.sidebar.columns(2)
col_b1.button(T["btn_left"], on_click=scroll_left, use_container_width=True)
col_b2.button(T["btn_right"], on_click=scroll_right, use_container_width=True)

col_x1, col_x2 = st.sidebar.columns(2)
col_x1.number_input(T["x_start"], step=0.1, key="x_start")
col_x2.number_input(T["x_end"], step=0.1, key="x_end")

if st.session_state.x_start >= st.session_state.x_end:
    st.sidebar.error("Le début doit être inférieur à la fin.")
x_zoom = (st.session_state.x_start, st.session_state.x_end)

def calculate_rise_time_expert(segment_y, dt):
    try:
        peak_idx = np.argmax(segment_y)
        if peak_idx < 3: return 0
        rising_limb = segment_y[:peak_idx + 1]
        t_vec = np.arange(len(rising_limb)) * dt
        peak_val = rising_limb[-1]
        y10, y90 = 0.10 * peak_val, 0.90 * peak_val
        t10 = np.interp(y10, rising_limb, t_vec)
        t90 = np.interp(y90, rising_limb, t_vec)
        return t90 - t10
    except: return 0

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

        # Détection
        detect_trace = f_data if is_outward else -f_data
        
        best_corr = np.zeros_like(detect_trace)
        decays = [10.0, 20.0, 30.0, 50.0] if is_outward else [2.0, 5.0, 10.0, 15.0]
        
        for d in decays:
            t_tmpl = np.arange(0, 30, dt)
            tmpl = (np.exp(-t_tmpl/d) - np.exp(-t_tmpl/0.5)) 
            tmpl /= np.max(np.abs(tmpl))
            best_corr = np.maximum(best_corr, signal.correlate(detect_trace, tmpl, mode='same'))
            
        corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
        peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.005 * fs))
        
        valid_ev = []
        k_trace = f_data
        
        for i, p in enumerate(peaks):
            start, end = p - int(0.005*fs), p + int(0.030*fs)
            if start < 0 or end >= len(k_trace): continue
            
            l_base = np.mean(k_trace[p-int(0.005*fs):p-int(0.002*fs)])
            
            if is_outward:
                seg = k_trace[start:end] - l_base
            else:
                seg = -(k_trace[start:end] - l_base)
            
            amp = np.max(seg)
            rise_1090 = calculate_rise_time_expert(seg, dt)
            area = integrate.trapezoid(seg, dx=dt)
            
            estimated_decay = abs(area / amp) if amp > 0 else 0
            
            # Utilisation de la nouvelle limite absolue
            pass_amp = (not use_amp_filter or amp >= amp_limit)
            pass_decay = (not use_decay_filter or estimated_decay <= decay_limit)
            pass_rise = (not use_rise_filter or rise_1090 <= rise_limit)
            
            if pass_amp and pass_decay and pass_rise:
                ev = {'idx': p, 'time': times[p], 'amp': amp, 'rise': rise_1090, 'area': abs(area), 'decay': estimated_decay}
                ev['iei'] = (times[p] - times[peaks[i-1]])*1000 if i>0 else np.nan
                valid_ev.append(ev)

        # --- PLOTTING ---
        st.subheader(T["viz_header"])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
        
        ax1.plot(times, f_data, color='black', lw=0.5)
        
        if valid_ev:
            ax1.plot([e['time'] for e in valid_ev], [f_data[e['idx']] for e in valid_ev], 'o', color='purple' if is_outward else '#FF8C00', markersize=5)

        ax1.set_ylim(y_zoom)
        ax1.set_xlim(x_zoom)
        ax1.set_ylabel("pA")

        ax2.plot(times, corr_z, color='blue', alpha=0.6)
        ax2.axhline(threshold, color='red', ls='--')
        ax2.set_ylabel("Z-Score")
        
        if auto_z:
            mask = (times >= st.session_state.x_start) & (times <= st.session_state.x_end)
            if np.any(mask):
                z_local = corr_z[mask]
                z_min, z_max = np.min(z_local), np.max(z_local)
                margin = abs(z_max - z_min) * 0.15 if z_max != z_min else 1.0
                ax2.set_ylim(z_min - margin, z_max + margin)

        st.pyplot(fig)
        
        # --- EXPORT ---
        if valid_ev:
            df = pd.DataFrame(valid_ev)
            st.divider()
            
            freq_hz = len(df) / times[-1]
            st.subheader(f"Total Events: {len(valid_ev)} | Freq: {freq_hz:.2f} Hz")
            
            col_exp1, col_exp2 = st.columns(2)
            df_export = df[['time', 'amp', 'rise', 'decay', 'area', 'iei']].copy()
            col_exp1.download_button(label="📁 Download Events (CSV)", data=df_export.to_csv(index=False).encode('utf-8'), file_name='events.csv', mime='text/csv')

    except Exception as e: st.error(f"Error: {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
