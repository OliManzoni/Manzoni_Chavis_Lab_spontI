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

if 'x_start' not in st.session_state:
    st.session_state.x_start = 10.0
if 'x_end' not in st.session_state:
    st.session_state.x_end = 11.0

# --- FONCTIONS DE NAVIGATION (D'origine) ---
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
        if peak_idx < 2: return 0
        rising_limb = segment_y[:peak_idx + 1]
        t_vec = np.arange(len(rising_limb)) * dt
        peak_val = rising_limb[-1]
        t10 = np.interp(0.10 * peak_val, rising_limb, t_vec)
        t90 = np.interp(0.90 * peak_val, rising_limb, t_vec)
        return t90 - t10
    except: return 0

st.title("🟣 Pipeline Expert : sIPSC GABA")

# --- SIDEBAR : FILTRES MODIFIÉS ---
st.sidebar.header("1. Type & Seuil")
polarity = st.sidebar.radio("Polarité", ["Outward (GABA 0mV)", "Inward (-70mV)"])
is_outward = "Outward" in polarity
threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 7.0, 3.5)

st.sidebar.header("2. Filtres Cinétiques (Plages)")
# Rise Time : On accepte de 0.2 à 5ms
col1, col2 = st.sidebar.columns(2)
rise_min = col1.number_input("Rise Min (ms)", value=0.2)
rise_max = col2.number_input("Rise Max (ms)", value=5.0)

# Decay Time : La clé est ici ! Min à 2.0 pour accepter les événements rapides (4.8ms)
col3, col4 = st.sidebar.columns(2)
decay_min = col3.number_input("Decay Min (ms)", value=2.0) 
decay_max = col4.number_input("Decay Max (ms)", value=80.0)

# Surface et Amplitude
area_min = st.sidebar.number_input("Area Min (pA.ms)", value=10.0)
amp_min = st.sidebar.number_input("Amplitude Min (pA)", value=5.0)

# --- ANALYSE (Structure d'origine préservée) ---
file = st.file_uploader("Charger un fichier .abf", type=["abf"])

if file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.abf') as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name

    try:
        abf = pyabf.ABF(tmp_path)
        abf.setSweep(0)
        times = abf.sweepX
        fs = abf.dataRate
        dt = 1000 / fs
        
        # Filtre médian (D'origine)
        f_data = abf.sweepY - ndimage.median_filter(abf.sweepY, size=int(0.5 * fs))
        
        # Template Matching (D'origine)
        detect_trace = f_data if is_outward else -f_data
        best_corr = np.zeros_like(detect_trace)
        taus = [5.0, 15.0, 30.0, 50.0]
        for t in taus:
            t_tmpl = np.arange(0, 100, dt)
            tmpl = (np.exp(-t_tmpl/t) - np.exp(-t_tmpl/(t/10)))
            tmpl /= np.max(tmpl)
            best_corr = np.maximum(best_corr, signal.correlate(detect_trace, tmpl, mode='same'))
            
        corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
        peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.01 * fs))
        
        valid_ev = []
        for p in peaks:
            start, end = p - int(0.005*fs), p + int(0.060*fs)
            if start < 0 or end >= len(f_data): continue
            
            seg = (f_data[start:end]) if is_outward else -(f_data[start:end])
            seg -= np.mean(seg[:int(0.004*fs)]) # Baseline locale
            
            amp = np.max(seg)
            rise = calculate_rise_time(seg, dt)
            area = integrate.trapezoid(seg, dx=dt)
            decay = abs(area / amp) if amp > 0 else 0
            
            # --- LOGIQUE DE FILTRAGE CORRIGÉE ---
            if (amp >= amp_min and 
                rise_min <= rise <= rise_max and 
                decay_min <= decay <= decay_max and 
                abs(area) >= area_min):
                
                valid_ev.append({'idx': p, 'time': times[p], 'amp': amp, 'rise': rise, 'decay': decay, 'area': abs(area)})

        # --- VISUALISATION (D'origine) ---
        c_nav1, c_nav2, c_nav3, c_nav4 = st.columns([1,1,2,2])
        c_nav1.button("⬅️ Scroll", on_click=scroll_left)
        c_nav2.button("Scroll ➡️", on_click=scroll_right)
        st.session_state.x_start = c_nav3.number_input("Start (s)", value=st.session_state.x_start)
        st.session_state.x_end = c_nav4.number_input("End (s)", value=st.session_state.x_end)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios':[3,1]})
        ax1.plot(times, f_data, color='black', lw=0.6)
        if valid_ev:
            ax1.plot([e['time'] for e in valid_ev], [f_data[e['idx']] for e in valid_ev], 'o', color='purple', markersize=4)
        
        ax1.set_xlim(st.session_state.x_start, st.session_state.x_end)
        ax1.set_ylabel("pA")
        
        ax2.plot(times, corr_z, color='blue', alpha=0.4)
        ax2.axhline(threshold, color='red', ls='--')
        ax2.set_ylabel("Z-Score")
        st.pyplot(fig)

        # --- EXPORT & RÉSUMÉ ---
        if valid_ev:
            df = pd.DataFrame(valid_ev)
            st.divider()
            st.subheader("📊 Résumé Statistique")
            
            # Ce tableau vous permet de vérifier la moyenne (Average) par rapport à vos limites
            res = pd.DataFrame({
                "Paramètre": ["Amplitude (pA)", "Rise Time (ms)", "Decay Time (ms)"],
                "Moyenne": [df['amp'].mean(), df['rise'].mean(), df['decay'].mean()],
                "Limite Min": [amp_min, rise_min, decay_min],
                "Limite Max": ["N/A", rise_max, decay_max]
            })
            st.table(res.style.format(precision=2))
            st.download_button("💾 Exporter CSV", df.to_csv(index=False).encode('utf-8'), "GABA_events.csv")

    except Exception as e: st.error(f"Erreur : {e}")
    finally:
        if os.path.exists(tmp_path): os.remove(tmp_path)
