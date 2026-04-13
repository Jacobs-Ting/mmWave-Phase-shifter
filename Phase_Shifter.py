import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# ==========================================
# Force Matplotlib to use the dark theme globally
# ==========================================
plt.style.use('dark_background')

# --- Page Configuration ---
st.set_page_config(page_title="Quantized VM Phase Shifter", layout="wide")
st.title("📡 VM Phase Shifter: DAC Quantization & Calibration Limits")
st.markdown("Exploring how perfect Digital Pre-compensation behaves when constrained by a finite-resolution **DAC (Digital-to-Analog Converter)**, forcing control voltages to 'snap' to a discrete grid and causing parasitic **phase errors** and **amplitude ripples**.")

# --- Session State Management ---
if 'is_calibrated' not in st.session_state:
    st.session_state.is_calibrated = False
if 'calibrated_data' not in st.session_state:
    st.session_state.calibrated_data = None

# --- Sidebar: Environment & Hardware Parameters ---
st.sidebar.header("⚙️ Hidden Hardware Errors (Plant)")
random_seed = st.sidebar.number_input("Chip Batch (Random Seed)", min_value=1, max_value=1000, value=42)
iq_mismatch_percent = st.sidebar.slider("I/Q Gain Imbalance (+/- %)", 0, 80, 40) / 100.0

st.sidebar.divider()
st.sidebar.header("🎛️ Digital-to-Analog Converter (DAC)")
dac_bits = st.sidebar.slider("DAC Resolution (Bits)", min_value=2, max_value=10, value=4, help="Determines the density of the control grid. e.g., 4-bit has 16 levels.")
dac_levels = 2 ** dac_bits
dac_lsb = 2.0 / (dac_levels - 1)
st.sidebar.caption(f"Current LSB Step: **{dac_lsb:.4f} V**")

st.sidebar.divider()
st.sidebar.header("🎯 Target Beam")
target_theta = st.sidebar.slider("Theta $\\theta$ (Elevation)", -60, 60, 45)
target_phi = st.sidebar.slider("Phi $\\phi$ (Azimuth)", 0, 360, 0)

def reset_calibration():
    st.session_state.is_calibrated = False
st.sidebar.button("🔄 Reset Hardware State", on_click=reset_calibration)

# --- Quantization Function ---
def quantize_voltage(v, lsb):
    """Clips voltage to [-1, 1] and snaps it to the nearest LSB grid point"""
    v_clipped = np.clip(v, -1.0, 1.0)
    v_quantized = np.round(v_clipped / lsb) * lsb
    return v_quantized

# --- Core Logic: Plant Modeling & Ideal Calculation ---
np.random.seed(random_seed)
N_x, N_y = 4, 4
num_elements = 16
wavelength = 3e8 / 28e9
d = wavelength / 2

# Inject independent I and Q gain errors
true_gains_I = np.random.uniform(1.0 - iq_mismatch_percent, 1.0 + iq_mismatch_percent, num_elements)
true_gains_Q = np.random.uniform(1.0 - iq_mismatch_percent, 1.0 + iq_mismatch_percent, num_elements)

# Calculate Ideal State
theta_rad = np.deg2rad(target_theta)
phi_rad = np.deg2rad(target_phi)
k = 2 * np.pi / wavelength

ideal_phases_deg = np.zeros(num_elements)
ideal_I = np.zeros(num_elements)
ideal_Q = np.zeros(num_elements)

for i in range(num_elements):
    x_pos = (i % N_x) * d
    y_pos = (i // N_x) * d
    ideal_phase_rad = k * (x_pos * np.sin(theta_rad) * np.cos(phi_rad) + y_pos * np.sin(theta_rad) * np.sin(phi_rad))
    ideal_phases_deg[i] = np.rad2deg(ideal_phase_rad) % 360
    ideal_I[i] = np.cos(ideal_phase_rad)
    ideal_Q[i] = np.sin(ideal_phase_rad)

# --- Auto Calibration Engine ---
col_btn, col_msg = st.columns([1, 4])
with col_btn:
    if st.button("🚀 Execute Quantized Pre-compensation", type="primary", use_container_width=True):
        with st.spinner("Calculating pre-compensation and snapping to DAC grid..."):
            progress_bar = st.progress(0)
            
            exact_I, exact_Q = np.zeros(num_elements), np.zeros(num_elements)
            quantized_I, quantized_Q = np.zeros(num_elements), np.zeros(num_elements)
            actual_amp, actual_phase = np.zeros(num_elements), np.zeros(num_elements)
            
            for i in range(num_elements):
                time.sleep(0.02)
                progress_bar.progress((i + 1) / num_elements)
                req_phase_rad = np.deg2rad(ideal_phases_deg[i])
                
                # 1. Exact floating-point pre-compensation
                exact_I[i] = np.cos(req_phase_rad) / true_gains_I[i]
                exact_Q[i] = np.sin(req_phase_rad) / true_gains_Q[i]
                
                # 2. DAC Quantization Constraints (Rounding Error + Saturation Clipping)
                quantized_I[i] = quantize_voltage(exact_I[i], dac_lsb)
                quantized_Q[i] = quantize_voltage(exact_Q[i], dac_lsb)
                
                # 3. Simulate final RF signal vector emitted
                rf_I = quantized_I[i] * true_gains_I[i]
                rf_Q = quantized_Q[i] * true_gains_Q[i]
                
                actual_amp[i] = np.sqrt(rf_I**2 + rf_Q**2)
                actual_phase[i] = np.rad2deg(np.arctan2(rf_Q, rf_I)) % 360
                
            st.session_state.calibrated_data = {
                'exact_I': exact_I, 'exact_Q': exact_Q,
                'q_I': quantized_I, 'q_Q': quantized_Q,
                'actual_amp': actual_amp, 'actual_phase': actual_phase
            }
            st.session_state.is_calibrated = True
            progress_bar.empty()

# --- Tab Interface Setup ---
tab_intro, tab_const, tab_err, tab_cb = st.tabs([
    "📖 System Architecture & Principles", 
    "📐 Quantization Grid & Constellation", 
    "📊 Phase & Amplitude Errors", 
    "🗂️ Quantized Codebook"
])

# ==========================================
# Tab 0: System Architecture & Principles
# ==========================================
with tab_intro:
    st.header("Deep Dive into mmWave Arrays: Vector Modulator (VM) Phase Shifters")
    st.markdown("""
    In modern high-frequency mmWave communication and radar systems, **Vector Modulation (VM)** architectures are widely adopted in RFICs to achieve high-speed and precise beamforming, replacing traditional passive switched-line phase shifters.
    
    This simulator explores the impact of hardware physical defects and digital control limits on the system. Below are the core principles:
    """)
    
    col_intro1, col_intro2 = st.columns([1, 1])
    
    with col_intro1:
        st.subheader("1. Hardware Architecture & Math Model")
        st.markdown(r"""
        The core physics concept of a VM phase shifter is **orthogonal vector synthesis**. The input RF signal passes through a polyphase filter to generate In-phase (I) and Quadrature (Q) signals with a $90^\circ$ phase difference.
        These two signals enter Variable Gain Amplifiers (VGAs). By altering the VGA gain weights ($Ctrl_I$ and $Ctrl_Q$), the desired phase and amplitude RF vector is synthesized:
        
        $$V_{ideal} = Ctrl_I + j \cdot Ctrl_Q$$
        """)
        
        st.info(r"""
        **⚠️ Fatal Physical Defect: I/Q Mismatch**
        In real semiconductor processes, the gains of I and Q amplifiers are never perfectly identical. With independent hardware gain variations $G_I$ and $G_Q$, the actual emitted microwave vector suffers severe phase distortion:
        $$V_{actual} = (G_I \cdot Ctrl_I) + j \cdot (G_Q \cdot Ctrl_Q)$$
        """)

    with col_intro2:
        st.subheader("2. Digital Pre-compensation & I/Q Calibration")
        st.markdown(r"""
        To compensate for hardware variations, the Baseband DSP extracts the $G_I$ and $G_Q$ characteristics for each antenna during Factory Calibration. 
        *Note: This is static **I/Q Calibration** for linear errors, not non-linear PA Digital Pre-Distortion (DPD).*
        
        During operation, the algorithm inversely scales the control voltages to offset hardware attenuation. The calculated compensation matrix is written into the chip's **NV (Non-Volatile Memory)**, forming the Beam Codebook.
        
        $$Ctrl_I = \frac{\cos(\Phi_{target})}{G_I} \quad , \quad Ctrl_Q = \frac{\sin(\Phi_{target})}{G_Q}$$
        """)
        
    st.divider()
    
    st.subheader("3. The Ultimate Bottleneck: Cartesian Quantization")
    st.markdown("""
    In ideal math, the calculated voltages are continuous floating-point numbers. However, VGAs are driven by finite-resolution **Digital-to-Analog Converters (DACs)**.
    * **Square Grid Constraint:** A 4-bit DAC forces control voltages onto the intersections of a $16 \times 16$ discrete square grid.
    * **Parasitic AM Effect:** To approximate the target phase on this grid, amplitude is often sacrificed. This creates unpredictable **Amplitude Ripple**, destroying pre-defined array tapering (e.g., Taylor Window).
    * **Saturation Clipping:** If the pre-compensation demands a voltage exceeding the DAC's rail limits ($\pm 1.0\text{V}$), the signal is forcibly clipped, leading to beam collapse.
    """)

# ==========================================
# Tab 1: Quantization Grid & Constellation
# ==========================================
with tab_const:
    if not st.session_state.is_calibrated:
        st.info("👆 Please click the 'Execute Quantized Pre-compensation' button on the sidebar.")
    else:
        cal_data = st.session_state.calibrated_data
        st.subheader("📐 VM Phase Shifter: I/Q Cartesian Grid")
        st.markdown("**Blue Dots**: Exact ideal DPD values. **Red Stars**: Quantized voltages forced by DAC limits.")
        
        # Make figure background transparent to blend with Streamlit dark mode
        fig_const, ax_const = plt.subplots(figsize=(8, 8))
        fig_const.patch.set_alpha(0.0) 
        ax_const.patch.set_alpha(0.0)
        
        grid_lines = np.arange(-1.0, 1.0 + dac_lsb/2, dac_lsb)
        for g in grid_lines:
            ax_const.axhline(g, color='lightgray', linestyle=':', alpha=0.3)
            ax_const.axvline(g, color='lightgray', linestyle=':', alpha=0.3)
        
        ax_const.axhline(0, color='white', lw=1.5, alpha=0.7)
        ax_const.axvline(0, color='white', lw=1.5, alpha=0.7)
        ax_const.add_patch(plt.Circle((0, 0), 1.0, color='white', fill=False, linestyle='--', alpha=0.5))
        
        ax_const.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color='red', linestyle='-', lw=2, alpha=0.5)
        
        for i in range(num_elements):
            e_i, e_q = cal_data['exact_I'][i], cal_data['exact_Q'][i]
            q_i, q_q = cal_data['q_I'][i], cal_data['q_Q'][i]
            
            ax_const.plot(e_i, e_q, 'dodgerblue', marker='o', alpha=0.6, markersize=6, linestyle='None')
            ax_const.plot(q_i, q_q, 'crimson', marker='*', markersize=10, linestyle='None')
            ax_const.annotate('', xy=(q_i, q_q), xytext=(e_i, e_q),
                              arrowprops=dict(arrowstyle="->", color="violet", alpha=0.6))
            
            if abs(e_i) > 1.0 or abs(e_q) > 1.0:
                ax_const.text(q_i, q_q, f" TX_{i} Clipped!", color='red', fontsize=8, fontweight='bold')

        limit = max(1.2, np.max(np.abs(cal_data['exact_I'])) * 1.1, np.max(np.abs(cal_data['exact_Q'])) * 1.1)
        ax_const.set_xlim(-limit, limit)
        ax_const.set_ylim(-limit, limit)
        ax_const.set_aspect('equal')
        ax_const.set_xlabel(f"VGA_I Control Voltage (DAC LSB = {dac_lsb:.3f})")
        ax_const.set_ylabel(f"VGA_Q Control Voltage (DAC LSB = {dac_lsb:.3f})")
        st.pyplot(fig_const)

# ==========================================
# Tab 2: Phase & Amplitude Errors
# ==========================================
with tab_err:
    if not st.session_state.is_calibrated:
        st.info("👆 Please click the 'Execute Quantized Pre-compensation' button on the sidebar.")
    else:
        cal_data = st.session_state.calibrated_data
        st.subheader("📊 Residual Errors Caused by DAC Quantization")
        st.markdown("Since the hardware grid cannot provide perfect compensation voltages, this leads to **parasitic amplitude ripples** and **residual phase errors** in the final RF output. This is a primary cause of Side Lobe degradation!")
        
        phase_errors = cal_data['actual_phase'] - ideal_phases_deg
        phase_errors = (phase_errors + 180) % 360 - 180 
        amp_errors = cal_data['actual_amp'] - 1.0
        
        fig_err, (ax_phase, ax_amp) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        fig_err.patch.set_alpha(0.0)
        ax_phase.patch.set_alpha(0.0)
        ax_amp.patch.set_alpha(0.0)
        
        x_positions = np.arange(num_elements)
        
        ax_phase.bar(x_positions, phase_errors, color='mediumorchid', alpha=0.8)
        ax_phase.axhline(0, color='white', lw=1.2, alpha=0.6)
        ax_phase.set_ylabel("Phase Error $\Delta \Phi$ (Degrees)")
        ax_phase.set_title("Final RF Output: Residual Phase Error (Causes Beam Squint)")
        ax_phase.grid(axis='y', linestyle=':', alpha=0.4, color='gray')
        
        ax_amp.bar(x_positions, amp_errors, color='mediumseagreen', alpha=0.8)
        ax_amp.axhline(0, color='white', lw=1.2, alpha=0.6)
        ax_amp.set_ylabel("Amplitude Ripple $\Delta A$ (Linear)")
        ax_amp.set_title("Final RF Output: Parasitic AM (Destroys Tapering, Causes SLL spikes)")
        ax_amp.set_xlabel("Antenna Elements")
        ax_amp.set_xticks(x_positions)
        ax_amp.set_xticklabels([f'TX_{i}' for i in range(num_elements)])
        ax_amp.grid(axis='y', linestyle=':', alpha=0.4, color='gray')
        
        st.pyplot(fig_err)

# ==========================================
# Tab 3: Quantized Codebook (Heatmap Style)
# ==========================================
with tab_cb:
    if not st.session_state.is_calibrated:
        st.info("👆 Please click the 'Execute Quantized Pre-compensation' button on the sidebar.")
    else:
        cal_data = st.session_state.calibrated_data
        st.subheader("🗂️ Quantized Codebook for NV Memory")
        st.markdown("The table below shows the conversion from floating-point exact values to real DAC digital codes, mimicking the thermal/heatmap view.")
        
        comparison_data = []
        for i in range(num_elements):
            comparison_data.append({
                "Antenna": f"TX_{i}",
                "Hardware G_I": round(true_gains_I[i], 6),
                "Hardware G_Q": round(true_gains_Q[i], 6),
                "Exact DPD_I": round(cal_data['exact_I'][i], 6),
                "Quantized DAC_I": round(cal_data['q_I'][i], 6),
                "Exact DPD_Q": round(cal_data['exact_Q'][i], 6),
                "Quantized DAC_Q": round(cal_data['q_Q'][i], 6)
            })
        
        df_comp = pd.DataFrame(comparison_data)
        
        # Apply the gradient styling to match the screenshot aesthetics
        styled_df = df_comp.style.background_gradient(subset=['Quantized DAC_I'], cmap='Oranges') \
                                 .background_gradient(subset=['Quantized DAC_Q'], cmap='Greys')
        
        st.dataframe(styled_df, use_container_width=True, height=600)