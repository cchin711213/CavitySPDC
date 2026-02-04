import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Physical Constants & Models ---
def n_ktp(wavelength_m, axis='z'):
    """Sellmeier equations for KTP. Ref: Fan et al. (1987)"""
    wl_um = wavelength_m * 1e6
    # Clipping to avoid numerical instability outside transparency range
    wl_um = np.clip(wl_um, 0.35, 4.0)
    if axis == 'z':
        n_sq = 2.25411 + (1.06543 * wl_um**2) / (wl_um**2 - 0.05486) + (1.11202 * wl_um**2) / (wl_um**2 - 232.5)
    elif axis == 'y':
        n_sq = 2.19229 + (0.83547 * wl_um**2) / (wl_um**2 - 0.04963) + (0.39636 * wl_um**2) / (wl_um**2 - 187.0)
    return np.sqrt(n_sq)

def get_vg(wavelength_m, axis='z'):
    c = 299792458.0
    eps = 1e-10
    n = n_ktp(wavelength_m, axis)
    dn_dl = (n_ktp(wavelength_m + eps, axis) - n_ktp(wavelength_m - eps, axis)) / (2 * eps)
    ng = n - wavelength_m * dn_dl
    return c / ng

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Type-II CE-SPDC Simulator", layout="wide")
st.title("Cavity-Enhanced SPDC (Type-II PPKTP)")

# --- Sidebar Controls ---
st.sidebar.header("System Parameters")
L_cav_cm = st.sidebar.slider("Cavity Length (cm)", 5.0, 15.0, 10.0, step=0.01)
L_cry_cm = st.sidebar.slider("Crystal Length (cm)", 0.5, 2.0, 1.0, step=0.1)
finesse = st.sidebar.slider("Cavity Finesse", 10, 500, 100)

st.sidebar.header("Fine Tuning")
lambda_base = 852.354 
lambda_off = st.sidebar.number_input("Wavelength Offset (nm)", value=0.0021, format="%.5f")
lambda0 = (lambda_base + lambda_off) * 1e-9

# --- Simulation Core ---
c = 299792458.0
f0 = c / lambda0
L_cav = L_cav_cm / 100
L_cry = L_cry_cm / 100
L_air = L_cav - L_cry

# Calculate Indices and FSRs for Output
ny0 = n_ktp(lambda0, 'y')
nz0 = n_ktp(lambda0, 'z')

# Optical lengths
L_opt_y = L_air + (ny0 * L_cry)
L_opt_z = L_air + (nz0 * L_cry)

# Free Spectral Ranges
fsr_y = c / (2 * L_opt_y)
fsr_z = c / (2 * L_opt_z)

# Physical Bandwidth (GVM)
vg_s = get_vg(lambda0, 'y')
vg_i = get_vg(lambda0, 'z')
vg_p = get_vg(lambda0/2, 'z')
gvm = 0.5 * (1/vg_s + 1/vg_i) - 1/vg_p

def get_spectrum(f_detuning):
    fs = f0 + f_detuning
    fi = f0 - f_detuning
    
    # 1. Sinc Envelope
    arg = L_cry * gvm * f_detuning
    sinc_sq = np.sinc(arg)**2

    # 2. Cavity Airy Functions
    def airy(f, axis):
        n = n_ktp(c / f, axis=axis)
        L_opt = L_air + (n * L_cry)
        phi = (2 * np.pi * f / c) * (2 * L_opt)
        coeff = (2 * finesse / np.pi)**2
        return 1 / (1 + coeff * np.sin(phi / 2)**2)

    val = sinc_sq * airy(fs, 'y') * airy(fi, 'z')
    return val, sinc_sq

# --- Plotting ---
# Increase resolution for high-finesse peaks
f_wide = np.linspace(-400e9, 400e9, 250000)
p_wide, sinc_wide = get_spectrum(f_wide)

peaks, _ = find_peaks(p_wide, distance=1000, height=0.005)
cluster_freqs = f_wide[peaks]
f_center_cluster = cluster_freqs[np.argmin(np.abs(cluster_freqs))] if len(cluster_freqs) > 0 else 0

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(f_wide / 1e9, sinc_wide, 'r--', alpha=0.4, label='Crystal Envelope')
    ax1.plot(f_wide / 1e9, p_wide, color='midnightblue', lw=0.7, label='CE-SPDC Clusters')
    ax1.set_title("Macroscopic View: Phase Matching & Clusters")
    ax1.set_xlabel("Detuning (GHz)")
    ax1.set_ylabel("Intensity")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    micro_span = 5 * fsr_y
    f_micro_det = np.linspace(f_center_cluster - micro_span, f_center_cluster + micro_span, 15000)
    p_micro, sinc_micro = get_spectrum(f_micro_det)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot((f_micro_det - f_center_cluster) / 1e6, p_micro, color='darkgreen', lw=1.5, label='Modes')
    ax2.set_title(f"Microscopic View: Cluster at {f_center_cluster/1e9:.2f} GHz")
    ax2.set_xlabel("Offset from Cluster Center (MHz)")
    ax2.set_ylabel("Intensity")
    st.pyplot(fig2)

# --- Calculated Physics Outputs ---
st.write("---")
st.subheader("Optical Properties at Target Wavelength")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("Refractive Index (Signal - Y)", f"{ny0:.5f}")
    st.metric("Refractive Index (Idler - Z)", f"{nz0:.5f}")

with c2:
    st.metric("FSR (Signal - Y)", f"{fsr_y/1e6:.2f} MHz")
    st.metric("FSR (Idler - Z)", f"{fsr_z/1e6:.2f} MHz")

with c3:
    cluster_spacing = abs(fsr_y * fsr_z / (fsr_y - fsr_z))
    st.metric("Mode Delta (ΔFSR)", f"{abs(fsr_y - fsr_z)/1e6:.2f} MHz")
    st.metric("Expected Cluster Spacing", f"{cluster_spacing/1e9:.2f} GHz")

st.info(f"The interaction is Type-II: Signal is Y-polarized, Idler is Z-polarized. Difference in $n$ results in the Vernier effect.")
