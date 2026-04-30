import streamlit as st
import pyabf
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize, integrate
import tempfile
import os
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="sIPSC Pipeline", layout="wide")

if 'fs_nyquist' not in st.session_state:
    st.session_state.fs_nyquist = 5000.0

# --- LANGUAGE SELECTION ---
lang = st.sidebar.selectbox("Language / Langue", ["English", "Français"])

# Textes explicatifs (Markdown + LaTeX)
THEORY_EN = """
### 🔬 Biophysical and Mathematical Principles: GABAergic Inhibition

#### 1. Bessel Filter & Phase Linearity
To accurately measure the ultra-fast rising phase of GABA$_A$ currents (target $\\sim 0.2$ ms), preserving the signal's phase is critical. The 4th-order **Bessel filter** applies a maximally flat group delay, ensuring no high-frequency artifacts (ringing) distort the kinetic onset of the sIPSC.

#### 2. GABA-A Template Matching & Polarity
GABA$_A$ receptor-mediated currents exhibit slower deactivation kinetics than AMPA currents (decay typically 10-30 ms). The algorithm convolves the trace with multi-scale bi-exponential templates initialized with a **0.2 ms rise time** and varying decay constants ($\\tau = 5, 10, 15, 25$ ms). 
Furthermore, depending on the electrochemical driving force ($V_m - E_{Cl}$), sIPSCs can be recorded as **Outward** (e.g., holding at $0$ mV) or **Inward** (e.g., using high intracellular $[Cl^-]$ at $-70$ mV). The algorithm dynamically mathematically inverts the trace based on the selected polarity prior to cross-correlation.

#### 3. Charge Integration & Decay Estimation
Non-linear Levenberg-Marquardt curve fitting on spontaneous GABAergic events often fails due to overlapping events and noisy baselines. We employ a robust mathematical approximation. For a simple exponential decay:
$$ I(t) = I_{max} e^{-t/\\tau} $$
The total charge (Area) is the integral:
$$ \\text{Area} = \\int_{0}^{\\infty} I_{max} e^{-t/\\tau} dt = I_{max} \\cdot \\tau $$
Thus, the decay constant $\\tau$ is robustly estimated across the population:
$$ \\tau \\approx \\frac{\\text{Area}}{\\text{Amplitude}} $$
"""

THEORY_FR = """
### 🔬 Principes Biophysiques et Mathématiques : Inhibition GABAergique

#### 1. Filtre de Bessel & Linéarité de Phase
Pour mesurer avec précision la phase montante ultra-rapide des courants GABA$_A$ (cible $\\sim 0.2$ ms), la préservation de la phase du signal est critique. Le **filtre de Bessel** de 4ème ordre applique un délai de groupe plat, garantissant qu'aucun artefact haute fréquence ne vient distordre le *Rise Time* du sIPSC.

#### 2. Template Matching GABA-A & Polarité
Les courants médiés par les récepteurs GABA$_A$ présentent des cinétiques de désactivation plus lentes que l'AMPA (decay typique de 10-30 ms). L'algorithme génère des modèles bi-exponentiels avec un **rise time de 0.2 ms** et des temps de décroissance variables ($\\tau = 5, 10, 15, 25$ ms).
De plus, selon la force électromotrice ($V_m - E_{Cl}$), les sIPSCs peuvent être **Sortants** (maintien à $0$ mV) ou **Entrants** (solution riche en $[Cl^-]$ à $-70$ mV). L'algorithme adapte l'inversion de la trace mathématiquement avant d'appliquer la corrélation croisée.

#### 3. Intégration de Charge & Estimation du Decay
L'ajustement de courbe non-linéaire sur des événements GABAergiques spontanés échoue souvent à cause du chevauchement et du bruit. Nous utilisons une approximation robuste. Pour une décroissance exponentielle :
$$ I(t) = I_{max} e^{-t/\\tau} $$
La charge totale (Aire) est l'intégrale :
$$ \\text{Aire} = \\int_{0}^{\\infty} I_{max} e^{-t/\\tau} dt = I_{max} \\cdot \\tau $$
La constante de temps $\\tau$ est donc estimée de manière robuste pour toute la population :
$$ \\tau \\approx \\frac{\\text{Aire}}{\\text{Amplitude}} $$
"""

T = {
    "English": {
        "title": "# sIPSC Expert Pipeline: Inhibition & Kinetics",
        "branding": "Manzoni Lab - Synaptic Plasticity",
        "readme_link": "📖 View README (Documentation)",
        "cite_header": "🎓 Cite this App",
        "cite_text": "If you use this tool, please cite:",
        "tab_analysis": "📈 Analysis Pipeline",
        "tab_theory": "📚 Biophysics & Math Theory",
        "theory_text": THEORY_EN,
        "sb_polarity": "0. Patch Configuration",
        "pol_radio": "Current Direction",
        "pol_out": "Outward (> 0 pA, e.g., 0 mV)",
        "pol_in": "Inward (< 0 pA, e.g., high Cl-)",
        "sb_bessel": "1. Bessel Filter (Denoising)",
        "cutoff": "Cutoff (Hz)",
        "nyquist_warn": "⚠️ Limited by Nyquist frequency",
        "sb_detec": "2. Multi-Scale Detection",
        "threshold": "Z-Score Threshold",
        "sb_kinetics": "3. Kinetics & Filters",
        "decay_thresh": "Decay Threshold (ms)",
        "rise_thresh": "Rise Time Threshold (ms)",
        "calc_raw": "Calculate on RAW trace",
        "amp_filter": "Amplitude Filter (Absolute > 7pA)",
        "sb_viz": "4. Visualization",
        "zoom_y": "Zoom Y (pA)",
        "zoom_x": "Zoom X (s)",
        "uploader": "Upload .abf",
        "viz_header": "Visualization & Detection",
        "export_header": "📥 Export Results",
        "btn_events": "📁 Download Individual Events",
        "btn_summary": "📊 Download Population Analysis",
        "col_time": "Time (s)",
        "col_amp": "Absolute Amplitude (pA)",
        "col_rise": "Rise Time 10-90% (ms)",
        "col_decay": "Estimated Decay (ms)",
        "col_area": "Area (pA.ms)",
        "col_iei": "IEI (ms)"
    },
    "Français": {
        "title": "# Pipeline Expert sIPSC : Inhibition & Cinétique",
        "branding": "Manzoni Lab - Plasticité Synaptique",
        "readme_link": "📖 Voir le README (Documentation)",
        "cite_header": "🎓 Citer cette App",
        "cite_text": "Si vous utilisez cet outil, merci de citer :",
        "tab_analysis": "📈 Pipeline d'Analyse",
        "tab_theory": "📚 Théorie Biophysique & Maths",
        "theory_text": THEORY_FR,
        "sb_polarity": "0. Configuration Patch",
        "pol_radio": "Direction du Courant",
        "pol_out": "Sortant / Outward (> 0 pA, ex: 0 mV)",
        "pol_in": "Entrant / Inward (< 0 pA, ex: Cl- int.)",
        "sb_bessel": "1. Filtre Bessel (Denoising)",
        "cutoff": "Fréquence de coupure (Hz)",
        "nyquist_warn": "⚠️ Limité par la fréquence de Nyquist",
        "sb_detec": "2. Détection Multi-Scale",
        "threshold": "Seuil Z-Score",
        "sb_kinetics": "3. Cinétique & Filtres",
        "decay_thresh": "Seuil maximal Decay (ms)",
        "rise_thresh": "Seuil maximal Rise Time (ms)",
        "calc_raw": "Calculer sur trace BRUTE",
        "amp_filter": "Filtre Amplitude (Absolue > 7pA)",
        "sb_viz": "4. Visualisation",
        "zoom_y": "Zoom Y (pA)",
        "zoom_x": "Zoom X (s)",
        "uploader": "Charger .abf",
        "viz_header": "Visualisation & Détection",
        "export_header": "📥 Exportation des Résultats",
        "btn_events": "📁 Télécharger Événements Individuels",
        "btn_summary": "📊 Télécharger Analyse Population",
        "col_time": "Temps (s)",
        "col_amp": "Amplitude Absolue (pA)",
        "col_rise": "Rise Time 10-90% (ms)",
        "col_decay": "Decay Estimé (ms)",
        "col_area": "Aire (pA.ms)",
        "col_iei": "IEI (ms)"
    }
}[lang]

# --- EN-TÊTE INSTITUTIONNEL ---
col_l, col_r = st.columns([2, 5]) 
with col_l:
    try: st.image("logo_chavis_final.png", width=360) 
    except: st.info(T["branding"]) 
with col_r:
    st.markdown(T["title"])

st.divider()

# --- CRÉATION DES ONGLETS ---
tab_analysis, tab_theory = st.tabs([T["tab_analysis"], T["tab_theory"]])

with tab_theory:
    st.markdown(T["theory_text"])

with tab_analysis:
    # --- FONCTIONS MATHÉMATIQUES ---
    def apply_bessel_filter(data, fs, cutoff=2000, order=4):
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

    # --- SIDEBAR & FILTRES ---
    st.sidebar.header(T["sb_polarity"])
    polarity_selection = st.sidebar.radio(T["pol_radio"], [T["pol_out"], T["pol_in"]])
    # sign_mult is 1 for Outward (positive peaks), -1 for Inward (negative peaks)
    sign_mult = 1 if polarity_selection == T["pol_out"] else -1

    st.sidebar.header(T["sb_bessel"])
    use_bessel = st.sidebar.checkbox("Bessel", value=True)
    cutoff = st.sidebar.slider(T["cutoff"], 100, int(st.session_state.fs_nyquist), 1500)

    st.sidebar.header(T["sb_detec"])
    threshold = st.sidebar.slider(T["threshold"], 1.0, 8.0, 3.0)

    st.sidebar.header(T["sb_kinetics"])
    use_decay_filter = st.sidebar.checkbox("Filter Decay", value=True)
    # Default decay threshold increased for GABA
    decay_limit = st.sidebar.number_input(T["decay_thresh"], value=50.0, step=5.0)

    use_rise_filter = st.sidebar.checkbox("Filter Rise Time", value=True)
    # Default rise threshold slightly adjusted
    rise_limit = st.sidebar.number_input(T["rise_thresh"], value=2.0, step=0.1)

    use_amp_filter = st.sidebar.checkbox(T["amp_filter"], value=True)
    calc_on_raw = st.sidebar.checkbox(T["calc_raw"], value=False)

    st.sidebar.header(T["sb_viz"])
    default_y = (-20, 150) if sign_mult == 1 else (-150, 20)
    y_zoom = st.sidebar.slider(T["zoom_y"], -300, 300, default_y)
    x_zoom = st.sidebar.slider(T["zoom_x"], 0.0, 600.0, (0.0, 2.0), step=0.1)

    # --- LOGIQUE PRINCIPALE ---
    file = st.file_uploader(T["uploader"], type=["abf"])

    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.abf') as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name

        try:
            abf = pyabf.ABF(tmp_path)
            abf.setSweep(0)
            fs, times, dt = abf.dataRate, abf.sweepX, 1000/abf.dataRate
            
            nyquist_limit = fs / 2
            st.session_state.fs_nyquist = nyquist_limit
            if cutoff >= nyquist_limit:
                st.sidebar.warning(T["nyquist_warn"])

            raw_data = abf.sweepY - np.median(abf.sweepY)
            f_data = apply_bessel_filter(raw_data, fs, cutoff) if use_bessel else raw_data
            
            best_corr = np.zeros_like(f_data)
            # GABA multi-scale decays: 5, 10 (base), 15, 25 ms
            default_decays = [5.0, 10.0, 15.0, 25.0] 
            for d in default_decays:
                t_tmpl = np.arange(0, 40, dt) # Extended template window
                # Updated Rise time to 0.2ms per your specification
                tmpl = (np.exp(-t_tmpl/d) - np.exp(-t_tmpl/0.2)) 
                tmpl /= np.max(np.abs(tmpl))
                
                # Apply sign_mult so correlation always targets positive synthetic peaks
                best_corr = np.maximum(best_corr, signal.correlate(f_data * sign_mult, tmpl, mode='same'))
            
            corr_z = (best_corr - np.mean(best_corr)) / np.std(best_corr)
            peaks, _ = signal.find_peaks(corr_z, height=threshold, distance=int(0.010 * fs))
            
            valid_ev = []
            k_trace = raw_data if calc_on_raw else f_data
            
            for i, p in enumerate(peaks):
                # Extended window for slower GABA events
                start, end = p - int(0.005*fs), p + int(0.040*fs) 
                if start < 0 or end >= len(k_trace): continue
                
                l_base = np.mean(k_trace[p-int(0.008*fs):p-int(0.003*fs)])
                
                # Normalize segment to always be a positive peak for math operations
                seg_inv = (k_trace[start:end] - l_base) * sign_mult
                
                amp = np.max(seg_inv)
                rise_1090 = calculate_rise_time_expert(seg_inv, dt)
                area = integrate.trapezoid(seg_inv, dx=dt)
                
                estimated_decay = abs(area / amp) if amp > 0 else 0
                
                pass_amp = (not use_amp_filter or amp >= 7)
                pass_decay = (not use_decay_filter or estimated_decay <= decay_limit)
                pass_rise = (not use_rise_filter or rise_1090 <= rise_limit)
                
                if pass_amp and pass_decay and pass_rise:
                    ev = {'idx': p, 'time': times[p], 'amp': amp, 'rise': rise_1090, 'area': abs(area), 'decay': estimated_decay}
                    ev['iei'] = (times[p] - times[peaks[i-1]])*1000 if i>0 else np.nan
                    valid_ev.append(ev)

            # --- AFFICHAGE GRAPHIQUE ---
            st.subheader(T["viz_header"])
            fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios':[2,1]})
            ax1.plot(times, f_data, color='black', lw=0.4)
            if valid_ev: ax1.plot([e['time'] for e in valid_ev], [f_data[e['idx']] for e in valid_ev], 'o', color='#2ca02c', markersize=5)
            ax1.set_ylim(y_zoom); ax1.set_xlim(x_zoom)
            ax2.plot(times, corr_z, color='teal', alpha=0.5)
            ax2.axhline(threshold, color='red', ls='--')
            st.pyplot(fig1)

            # --- EXPORT & DISTRIBUTIONS ---
            if valid_ev:
                df = pd.DataFrame(valid_ev)
                st.divider()
                
                freq_hz = len(df) / times[-1]
                mean_iei_ms = df['iei'].mean()
                
                df_export = df[['time', 'amp', 'rise', 'decay', 'area', 'iei']].copy()
                df_export.rename(columns={'time': T['col_time'], 'amp': T['col_amp'], 'rise': T['col_rise'], 'decay': T['col_decay'], 'area': T['col_area'], 'iei': T['col_iei']}, inplace=True)
                
                st.subheader(T["export_header"])
                col_exp1, col_exp2 = st.columns(2)
                col_exp1.download_button(label=T["btn_events"], data=df_export.to_csv(index=False).encode('utf-8'), file_name='sIPSC_events.csv', mime='text/csv')
                
                st.divider()
                
                st.subheader(f"Total Events: {len(valid_ev)} | Freq: {freq_hz:.2f} Hz")
                n_bins = 25
                fig2, (ha, hb, hc) = plt.subplots(1, 3, figsize=(15, 4))
                
                counts_amp, bins_amp = np.histogram(df['amp'], bins=n_bins)
                ha.bar((bins_amp[:-1] + bins_amp[1:]) / 2, counts_amp, width=(bins_amp[1]-bins_amp[0])*0.9, color='gray')
                ha.set_title(T["col_amp"])
                
                counts_rise, bins_rise = np.histogram(df['rise'], bins=n_bins)
                hb.bar((bins_rise[:-1] + bins_rise[1:]) / 2, counts_rise, width=(bins_rise[1]-bins_rise[0])*0.9, color='#2ca02c')
                hb.set_title(T["col_rise"])
                
                iei_clean = df['iei'].dropna()
                if not iei_clean.empty:
                    counts_iei, bins_iei = np.histogram(iei_clean, bins=n_bins)
                    hc.bar((bins_iei[:-1] + bins_iei[1:]) / 2, counts_iei, width=(bins_iei[1]-bins_iei[0])*0.9, color='salmon')
                    hc.set_title(T["col_iei"])
                    
                st.pyplot(fig2)

        except Exception as e: st.error(f"Error: {e}")
        finally:
            if os.path.exists(tmp_path): os.remove(tmp_path)
