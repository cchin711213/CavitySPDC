import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Physical Constants & Models ---
def n_ktp(wavelength_m, axis='z'):
    wl_um = wavelength_m * 1e6
    wl_um = np.clip(wl_um, 0.35, 4.0)
    if axis == 'z':
        n_sq = 2.25411 + (1.06543 * wl_um**2) / (wl_um**2 - 0.05486) + (1.11202 * wl_um**2) / (wl_um**2 - 232.5)
    elif axis == 'y':
        n_sq = 2.19229 + (0.83547 * wl_um**2) / (wl_um**2 - 0.04963) + (0.39636 * wl_um**2) / (wl_um**2 - 187.0)
    return np.sqrt(n_sq)

def get_vg(wavelength_m, axis='z'):
    c_const = 299792458.0
    eps = 1e-10
    n = n_ktp(wavelength_m, axis)
    dn_dl = (n_ktp(wavelength_m + eps, axis) - n_ktp(wavelength_m - eps, axis)) / (2 * eps)
    ng = n - wavelength_m * dn_dl
    return c_const / ng

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Type-II CE-SPDC Simulator", layout="wide")
st.title("Cavity-Enhanced SPDC (Type-II PPKTP)")

# --- Sidebar Controls ---
st.sidebar.header("System Parameters")
L_cav_cm = st.sidebar.slider("Cavity Length (cm)", 5.0, 15.0, 10.0, step=0.01)
L_cry_cm = st.sidebar.slider("Crystal Length (cm)", 0.5, 2.0, 1.0, step=0.1)
finesse = st.sidebar.slider("Cavity Finesse", 10, 500, 100)

st.sidebar.header("Plotting Controls")
micro_range_mhz = st.sidebar.slider("Micro View Range (± MHz)", 100, 5000, 1000)

st.sidebar.header("Fine Tuning")
lambda_base = 852.354 
lambda_off = st.sidebar.number_input("Wavelength Offset (nm)", value=0.0021, format="%.5f")
lambda0 = (lambda_base + lambda_off) * 1e-9

# --- Simulation Core ---
c_const = 299792458.0
f0 = c_const / lambda0
L_cav = L_cav_cm / 100
L_cry = L_cry_cm / 100
L_air = L_cav - L_cry

# 1. Physics Calcs
ny0 = n_ktp(lambda0, 'y')
nz0 = n_ktp(lambda0, 'z')
fsr_y = c_const / (2 * (L_air + ny0 * L_cry))
fsr_z = c_const / (2 * (L_air + nz0 * L_cry))

vg_s = get_vg(lambda0, 'y')
vg_i = get_vg(lambda0, 'z')
vg_p = get_vg(lambda0/2, 'z')
gvm = 0.5 * (1/vg_s + 1/vg_i) - 1/vg_p

def get_spectrum(f_detuning):
    fs = f0 + f_detuning
    fi = f0 - f_detuning
    arg = L_cry * gvm * f_detuning
    sinc_sq = np.sinc(arg)**2 
    
    def airy(f, axis):
        n = n_ktp(c_const / f, axis=axis)
        phi = (2 * np.pi * f / c_const) * (2 * (L_air + n * L_cry))
        coeff = (2 * finesse / np.pi)**2
        return 1 / (1 + coeff * np.sin(phi / 2)**2)

    val = sinc_sq * airy(fs, 'y') * airy(fi, 'z')
    return val, sinc_sq

# --- Generation ---
# High resolution for macro plot to ensure peaks are caught
f_wide = np.linspace(-300e9, 300e9, 300000)
p_wide, sinc_wide = get_spectrum(f_wide)

# Robust Peak Finding
peaks, _ = find_peaks(p_wide, height=np.max(p_wide)*0.05)
if len(peaks) > 0:
    f_center_cluster = f_wide[peaks[np.argmin(np.abs(f_wide[peaks]))]]
else:
    f_center_cluster = 0.0 # Fallback

# --- Layout ---
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(f_wide / 1e9, sinc_wide, 'r--', alpha=0.3, label='Sinc Envelope')
    ax1.plot(f_wide / 1e9, p_wide, color='midnightblue', lw=0.8, label='Clusters')
    ax1.set_title("Macroscopic View")
    ax1.set_xlabel("Detuning (GHz)")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    f_micro = np.linspace(f_center_cluster - (micro_range_mhz * 1e6), 
                          f_center_cluster + (micro_range_mhz * 1e6), 2000)
    p_micro, _ = get_spectrum(f_micro)
    
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.plot((f_micro - f_center_cluster) / 1e6, p_micro, color='darkgreen', lw=1.5)
    ax2.set_title(f"Microscopic View (Zoom: {f_center_cluster/1e9:.2f} GHz)")
    ax2.set_xlabel("Offset (MHz)")
    st.pyplot(fig2)

# --- Reference Tables ---
st.write("---")
st.subheader("Cavity & Material Properties")
c1, c2, c3 = st.columns(3)

with c1:
    st.write("**Refractive Indices**")
    st.write(f"Signal ($n_y$): `{ny0:.6f}`")
    st.write(f"Idler ($n_z$): `{nz0:.6f}`")

with c2:
    st.write("**Free Spectral Ranges**")
    st.write(f"FSR Signal: `{fsr_y/1e6:.2f} MHz`")
    st.write(f"FSR Idler: `{fsr_z/1e6:.2f} MHz`")

with c3:
    delta_fsr = abs(fsr_y - fsr_z)
    cluster_spacing = (fsr_y * fsr_z) / delta_fsr if delta_fsr != 0 else 0
    st.write("**Vernier Characteristics**")
    st.write(f"$\Delta FSR$: `{delta_fsr/1e6:.4f} MHz`")
    st.write(f"Cluster Dist: `{cluster_spacing/1e9:.2f} GHz`")
