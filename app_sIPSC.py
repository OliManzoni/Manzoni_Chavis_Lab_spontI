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

if 'fs_nyquist' not in st.session_state:
    st.session_state.fs_nyquist = 5000.0

# --- LANGUAGE SELECTION ---
lang = st.sidebar.selectbox("Language / Langue", ["English", "Français"])

# Textes explicatifs (Markdown + LaTeX) adaptés pour les IPSCs
THEORY_EN = """
### 🔬 Biophysical and Mathematical Principles (IPSCs)

#### 1. Signal Polarity
Unlike AMPA-mediated EPSCs, GABAergic **IPSCs** can be outward or inward depending on the recording conditions. When recorded at ~0 mV (to isolate them from glutamatergic currents), they are **outward currents** (positive peaks). If recorded with a high intracellular $Cl^-$ solution at negative holding potentials, they are **inward currents** (negative peaks). The algorithm adjusts its math based on the selected polarity.

#### 2. GABAergic Kinetics & Multi-Scale Detection
$GABA_A$ receptors have significantly slower kinetics than AMPA receptors. The rise time is often >1 ms, and the decay constant ($\\tau$) is typically between 15 and 40 ms. 
The algorithm convolves the trace with multiple bi-exponential templates having slow decay constants (e.g., $\\tau = 10, 20, 30, 50$ ms) to detect heterogenous inhibitory events accurately.

#### 3. Decay Estimation via Charge Integration
Assuming a simple exponential decay for a $GABA_A$ current ($I(t) = I_{max} e^{-t/\\tau}$), the total charge (Area) is the integral:
$$ \\text{Area} = \\int_{0}^{\\infty} I_{max} e^{-t/\\tau} dt = I_{max} \\cdot \\tau $$
Therefore, the decay constant $\\tau$ is rapidly and robustly estimated:
$$ \\tau \\approx \\frac{\\text{Area}}{\\text{Amplitude}} $$
"""

THEORY_FR = """
### 🔬 Principes Biophysiques et Mathématiques (IPSCs)

#### 1. Polarité du Signal
Contrairement aux EPSCs (AMPA), les **IPSCs** GABAergiques peuvent être sortants ou entrants selon les conditions. Enregistrés à ~0 mV (pour les isoler du glutamate), ce sont des **courants sortants** (pics positifs). Avec une solution riche en chlorure ($Cl^-$) à un potentiel négatif, ce sont des **courants entrants** (pics négatifs). L'algorithme s'adapte via le sélecteur de polarité.

#### 2. Cinétique GABAergique & Détection Multi-Échelle
Les récepteurs $GABA_A$ ont une cinétique beaucoup plus lente que les récepteurs AMPA. Le temps de montée (Rise time) est souvent >1 ms, et la constante de décroissance ($\\tau$) se situe généralement entre 15 et 40 ms. 
L'algorithme génère des modèles bi-exponentiels avec des constantes de décroissance lentes (ex: $\\tau = 10, 20, 30, 50$ ms) pour la corrélation croisée.

#### 3. Estimation du Decay par Intégration
En supposant une décroissance exponentielle simple ($I(t) = I_{max} e^{-t/\\tau}$), la charge totale (Aire) est :
$$ \\text{Aire} = \\int_{0}^{\\infty} I_{max} e^{-t/\\tau} dt = I_{max} \\cdot \\tau $$
La constante de temps $\\tau$ est estimée robustement :
$$ \\tau \\approx \\frac{\\text{Aire}}{\\text{Amplitude}} $$
"""

T = {
    "English": {
        "title": "# sIPSC Expert Pipeline: Preprocessing & Kinetics",
        "branding": "Chavis Lab - Biophysics",
        "tab_analysis": "📈 Analysis Pipeline",
        "tab_theory": "📚 Biophysics Theory",
        "theory_text": THEORY_EN,
        "sb_preproc": "1. Preprocessing & Polarity",
        "polarity": "Signal Polarity",
        "pol_out": "Outward (Positive peaks, e.g. 0mV)",
        "pol_in": "Inward (Negative peaks, high Cl-)",
        "baseline_method": "Baseline Mode",
        "dyn_detrend": "Dynamic Detrending (Median)",
        "stat_detrend": "Static Global Median",
        "cutoff": "Bessel Cutoff (Hz)",
        "sb_detec": "2. Multi-Scale Detection (GABA)",
        "threshold": "Z-Score Threshold",
        "sb_kinetics": "3. Kinetics & Filters",
        "decay_thresh": "Decay Max (ms)",
        "rise_thresh": "Rise Time Max (ms)",
        "amp_filter": "Amplitude Filter (>5pA)",
        "sb_viz": "4. Visualization",
        "zoom_y": "Zoom Y (pA)",
        "x_start": "Start (s)",
        "x_end": "End (s)"
    },
    "Français": {
        "title": "# Pipeline Expert sIPSC : Prétraitement & Cinétique",
        "branding": "Chavis Lab - Biophysique",
        "tab_analysis": "📈 Pipeline d'Analyse",
        "tab_theory": "📚 Théorie Biophysique",
        "theory_text": THEORY_FR,
        "sb_preproc": "1. Prétraitement & Polarité",
        "polarity": "Polarité du Signal",
        "pol_out": "Sortant (Pics positifs, ex: 0mV)",
        "pol_in": "Entrant (Pics négatifs, haut Cl-)",
        "baseline_method": "Mode de Ligne de Base",
        "dyn_detrend": "Detrending Dynamique (Médiane)",
        "stat_detrend": "Médiane Globale Statique",
        "cutoff": "Coupure Bessel (Hz)",
        "sb_detec": "2. Détection Multi-Scale (GABA)",
        "threshold": "Seuil Z-Score",
        "sb_kinetics": "3. Cinétique & Filtres",
        "decay_thresh": "Seuil maximal Decay (ms)",
        "rise_thresh": "Seuil maximal Rise Time (ms)",
        "amp_filter": "Filtre Amplitude (>5pA)",
        "sb_viz": "4. Visualisation",
        "zoom_y": "Zoom Y (pA)",
        "x_start": "Début (s)",
        "x_end": "Fin (s)"
    }
}[lang]

col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: st.image("logo_chavis_final.png", width=360) 
    except: st.info(T["branding"]) 
with col_r:
    st.markdown(T["title"])

st.divider()

tab_analysis, tab_theory = st.tabs([T["tab_analysis"], T["tab_theory"]])

with tab_theory:
    st.markdown(T["theory_text"])

with tab_analysis:
    def apply_dynamic_detrending(data, fs, window_ms=500):
        kernel_size = int((window_ms / 1000.0) * fs)
        if kernel_size % 2 == 0: kernel_size += 1
        baseline = ndimage.median_filter(data, size=kernel_size)
        return data - baseline

    def apply_bessel_filter(data, fs, cutoff=1500, order=4):
        nyquist = 0.5 * fs
        effective_cutoff = min(cutoff, nyquist * 0.95)
        normal_cutoff = effective_cutoff / nyquist
        b, a = signal.bessel(order, normal_cutoff, btype='low', analog=False)
        return signal.filtfilt(b, a, data)

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

    st.sidebar.header(T["sb_preproc"])
    polarity = st.sidebar.radio(T["polarity"], [T["pol_out"], T["pol_in"]])
    is_outward = (polarity == T["pol_out"])
    
    baseline_mode = st.sidebar.radio(T["baseline_method"], [T["dyn_detrend"], T["stat_detrend"]], index=0)
    use_bessel = st.sidebar.checkbox("Bessel Filter", value=True)
    cutoff = st.sidebar.slider(T["cutoff"], 100, int(st.session_state.fs_nyquist), 1500)

    st.sidebar.header(T["sb_detec"])
    threshold = st.sidebar.slider(T["threshold"], 1.0, 8.0, 3.0)

    st.sidebar.header(T["sb_kinetics"])
    use_decay_filter = st.sidebar.checkbox("Filter Decay", value=True)
    decay_limit = st.sidebar.number_input(T["decay_thresh"], value=60.0, step=5.0)
    use_rise_filter = st.sidebar.checkbox("Filter Rise Time", value=True)
    rise_limit = st.sidebar.number_input(T["rise_thresh"], value=3.0, step=0.2)
    use_amp_filter = st.sidebar.checkbox(T["amp_filter"], value=True)
    calc_on_raw = st.sidebar.checkbox("Calc on RAW", value=False)

    st.sidebar.header(T["sb_viz"])
    # Zoom Y adapté : centré autour de 0 mais allant plus loin dans les positifs si Outward
    y_default = (-50, 200) if is_outward else (-200, 50)
    y_zoom = st.sidebar.slider(T["zoom_y"], -400, 400, y_default)
    
    col_x1, col_x2 = st.sidebar.columns(2)
    x_start = col_x1.number_input(T["x_start"], value=10.0, step=0.5)
    x_end = col_x2.number_input(T["x_end"], value=11.0, step=0.5)
    x_zoom = (x_start, x_end)

    file = st.file_uploader("Fichier ABF", type=["abf"])

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.abf') as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        try:
            abf = pyabf.ABF(tmp_path)
            abf.setSweep(0)
            fs, times, dt = abf.dataRate, abf.sweepX, 1000/abf.dataRate
            st.session_state.fs_nyquist = fs / 2

            if baseline_mode == T["dyn_detrend"]:
                with st.spinner("Detrending..."):
                    raw_data = apply_dynamic_detrending(abf.sweepY, fs, window_ms=500)
            else:
                raw_data = abf.sweepY - np.median(abf.sweepY)

            f_data = apply_bessel_filter(raw_data, fs, cutoff) if use_bessel else raw_data
            
            # --- LOGIQUE DE POLARITÉ ---
            # Si le signal est sortant (positif), on le garde tel quel pour la détection.
            # S'il est entrant (négatif), on l'inverse pour la détection mathématique (peaks).
            detect_data = f_data if is_outward else -f_data
            
            best_corr = np.zeros_like(detect_data)
            
            # Templates GABA : constantes de temps plus longues
            default_decays_gaba = [10.0, 20.0, 30.0, 50.0]
            
            for d in default_decays_gaba:
                t_tmpl = np.arange(0, 40, dt) # Fenêtre plus longue (40ms)
                tmpl = (np.exp(-t_tmpl/d) - np.exp(-t_tmpl/1.0)) # Rise time à 1.0 ms
                tmpl /= np.max(np.abs(tmpl))
                best_corr = np.maximum(best_corr, signal.correlate(detect_data, tmpl, mode='same'))
            
            corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
            peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.010 * fs)) # Distance min augmentée à 10ms
            
            valid_ev = []
            k_trace = raw_data if calc_on_raw else f_data
            
            for i, p in enumerate(peaks):
                # Fenêtre d'analyse plus large pour les IPSCs (-5ms à +30ms)
                start, end = p - int(0.005*fs), p + int(0.030*fs)
                if start < 0 or end >= len(k_trace): continue
                
                l_base = np.mean(k_trace[p-int(0.005*fs):p-int(0.002*fs)])
                
                # Inversion selon la polarité pour que le calcul cinétique voie toujours un pic positif
                if is_outward:
                    seg = k_trace[start:end] - l_base
                else:
                    seg = -(k_trace[start:end] - l_base)
                
                amp = np.max(seg)
                rise_1090 = calculate_rise_time_expert(seg, dt)
                area = integrate.trapezoid(seg, dx=dt)
                
                estimated_decay = abs(area / amp) if amp > 0 else 0
                
                pass_amp = (not use_amp_filter or amp >= 5)
                pass_decay = (not use_decay_filter or estimated_decay <= decay_limit)
                pass_rise = (not use_rise_filter or rise_1090 <= rise_limit)
                
                if pass_amp and pass_decay and pass_rise:
                    ev = {'idx': p, 'time': times[p], 'amp': amp, 'rise': rise_1090, 'area': abs(area), 'decay': estimated_decay}
                    ev['iei'] = (times[p] - times[peaks[i-1]])*1000 if i>0 else np.nan
                    valid_ev.append(ev)

            # --- AFFICHAGE ---
            st.subheader("Visualisation IPSC")
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
            ax1.plot(times, f_data, color='black', lw=0.4)
            if valid_ev: ax1.plot([e['time'] for e in valid_ev], [f_data[e['idx']] for e in valid_ev], 'o', color='purple', markersize=5) # Changé en violet pour IPSC
            ax1.set_ylim(y_zoom)
            ax1.set_xlim(x_zoom)
            ax2.plot(times, corr_z, color='blue', alpha=0.5)
            ax2.axhline(threshold, color='red', ls='--')
            st.pyplot(fig1)

            if valid_ev:
                df = pd.DataFrame(valid_ev)
                st.divider()
                st.subheader(f"Total Inhibitory Events: {len(valid_ev)} | Freq: {len(df)/times[-1]:.2f} Hz")
                
                fig2, (ha, hb, hc) = plt.subplots(1, 3, figsize=(15, 4))
                ha.hist(df['amp'], bins=25, color='gray', edgecolor='white')
                ha.set_title("Amplitude (pA)")
                hb.hist(df['rise'], bins=25, color='purple', edgecolor='white')
                hb.set_title("Rise Time 10-90% (ms)")
                hc.hist(df['decay'], bins=25, color='teal', edgecolor='white')
                hc.set_title("Estimated Decay (ms)")
                st.pyplot(fig2)

        except Exception as e: st.error(f"Error: {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
