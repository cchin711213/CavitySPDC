import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(page_title="CESPDC Cavity Simulator", layout="wide")

st.title("Cavity-Enhanced SPDC Spectral Simulator")
st.markdown("""
This app simulates the **Power Spectral Density** of a doubly-resonant Cavity-Enhanced Spontaneous Parametric Down-Conversion (CESPDC) source, 
following the NIST model (Eq. 8).
""")

# --- Sidebar Controls ---
st.sidebar.header("Cavity Parameters")
l_cav_cm = st.sidebar.slider("Cavity Length (cm)", 5.0, 20.0, 10.0)
l_cry_cm = st.sidebar.slider("Crystal Length (cm)", 0.5, 5.0, 1.0)
finesse = st.sidebar.number_input("Cavity Finesse", value=100)

st.sidebar.header("Material Properties (PPKTP)")
n_s = st.sidebar.number_input("Index (Signal)", value=1.834, format="%.4f")
n_i = st.sidebar.number_input("Index (Idler)", value=1.831, format="%.4f")

st.sidebar.header("Pump Properties")
lambda_p_nm = st.sidebar.number_input("Pump Wavelength (nm)", value=852.354, format="%.3f")

# --- Physical Calculations ---
c = 299792458.0
L_cav = l_cav_cm / 100.0
L_cry = l_cry_cm / 100.0
L_air = L_cav - L_cry
lambda_p = lambda_p_nm * 1e-9
f_p = c / lambda_p
f_s0 = f_p / 2

# Optical Path Lengths and FSRs
L_opt_s = L_air + (n_s * L_cry)
L_opt_i = L_air + (n_i * L_cry)
fsr_s = c / (2 * L_opt_s)
fsr_i = c / (2 * L_opt_i)

# Calculate physical mode offsets
m_s_center = int(np.round(f_s0 / fsr_s))
m_i_center = int(np.round(f_s0 / fsr_i))

def get_spectrum(f_detuning):
    # Phase Matching (Free Space)
    tau_o = (L_cry / c) * (n_i - n_s)
    sinc_sq = np.sinc(f_detuning * tau_o)**2
    
    # Cavity Response (Airy Functions)
    def airy(f_abs, fsr):
        phi = 2 * np.pi * f_abs / fsr
        coeff = (2 * finesse / np.pi)**2
        return 1 / (1 + coeff * np.sin(phi/2)**2)
    
    airy_s = airy(f_s0 + f_detuning, fsr_s)
    airy_i = airy(f_s0 - f_detuning, fsr_i)
    return sinc_sq * airy_s * airy_i, sinc_sq

# --- Data Generation ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Macro View: Clusters & Phase Matching")
    f_wide = np.linspace(-12e12, 12e12, 2000)
    p_wide, sinc_wide = get_spectrum(f_wide)
    
    fig1, ax1 = plt.subplots()
    ax1.plot(f_wide / 1e12, sinc_wide, 'r--', alpha=0.5, label='Sinc Envelope')
    ax1.fill_between(f_wide / 1e12, p_wide, color='midnightblue', alpha=0.7, label='Clusters')
    ax1.set_xlabel("Detuning (THz)")
    ax1.set_ylabel("Intensity")
    ax1.legend()
    st.pyplot(fig1)

with col2:
    st.subheader("Micro View: Longitudinal Modes")
    f_zoom = np.linspace(-5e9, 5e9, 5000)
    p_zoom, _ = get_spectrum(f_zoom)
    
    fig2, ax2 = plt.subplots()
    ax2.plot(f_zoom / 1e9, p_zoom, color='blue')
    ax2.set_xlabel("Detuning (GHz)")
    ax2.set_ylabel("Intensity")
    st.pyplot(fig2)

# --- Display Diagnostics ---
st.divider()
st.subheader("Calculated Values")
d1, d2, d3, d4 = st.columns(4)
d1.metric("FSR Signal", f"{fsr_s/1e6:.2f} MHz")
d2.metric("FSR Idler", f"{fsr_i/1e6:.2f} MHz")
d3.metric("Cluster Spacing", f"{(fsr_s*fsr_i)/abs(fsr_s-fsr_i)/1e12:.2f} THz")
d4.metric("Linewidth", f"{fsr_s/finesse/1e6:.2f} MHz")
