import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# --- Physical Constants & Models ---
def n_ktp(wavelength_m, axis='z'):
    """Sellmeier equations for KTP. Ref: Fan et al. (1987)"""
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

# 1. Physical Calculations for Output
ny0 = n_ktp(lambda0, 'y')
nz0 = n_ktp(lambda0, 'z')
L_opt_y = L_air + (ny0 * L_cry)
L_opt_z = L_air + (nz0 * L_cry)
fsr_y = c_const / (2 * L_opt_y)
fsr_z = c_const / (2 * L_opt_z)

# GVM for Sinc Envelope
vg_s = get_vg(lambda0, 'y')
vg_i = get_vg(lambda0, 'z')
vg_p = get_vg(lambda0/2, 'z')
gvm = 0.5 * (1/vg_s + 1/vg_i) - 1/vg_p

def get_spectrum(f_detuning):
    fs = f0 + f_detuning
    fi = f0 - f_detuning
    # Sinc Envelope
    arg = L_cry * gvm * f_detuning
    sinc_sq = np.sinc(arg)**2 
    
    def airy(f, axis):
        n = n_ktp(c_const / f, axis=axis)
        # Fix: Ensure parentheses are balanced and L_opt is recalculated for freq
        phi = (2 * np.pi * f / c_const) * (2 * (L_air + n * L_cry))
        coeff = (2 * finesse / np.pi)**2
        return 1 / (1 + coeff * np.sin(phi / 2)**2)

    val = sinc_sq * airy(fs, 'y') * airy(fi, 'z')
    return val, sinc_sq

# --- Data Generation ---
f_wide = np.linspace(-400e9, 400e9, 200000)
p_wide, sinc_wide = get_spectrum(f_wide)

# Cluster Identification
peaks, _ = find_peaks(p_wide, distance=2000, height=0.005)
cluster_freqs = f_wide[peaks]
f
