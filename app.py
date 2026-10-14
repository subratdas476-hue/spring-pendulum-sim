import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================================================
# 1. NEON THEME PAGE CONFIGURATION & CUSTOM CSS
# =========================================================
st.set_page_config(page_title="Neon Spring Pendulum — 3 Modes", layout="wide")

st.markdown("""
<style>
    /* Deep Cyberpunk Black-Blue Background */
    .stApp {
        background-color: #05050a;
        color: #00f3ff;
    }
    /* Neon Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0c16 !important;
        border-right: 1px solid #00f3ff;
        box-shadow: 0 0 15px rgba(0, 243, 255, 0.2);
    }
    h1, h2, h3, label, .stMarkdown {
        color: #00f3ff !important;
        text-shadow: 0 0 8px rgba(0, 243, 255, 0.5);
    }
    /* Metric Cards Neon Glow */
    div[data-testid="stMetric"] {
        background-color: #0d0f1d;
        border: 1px solid #ff007f;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 0 12px rgba(255, 0, 127, 0.35);
    }
    div[data-testid="stMetricValue"] {
        color: #39ff14 !important;
        text-shadow: 0 0 8px rgba(57, 255, 20, 0.6);
    }
</style>
""", unsafe_allow_html=True)

st.title("⚡ NEON SPRING PENDULUM — ALL 3 MODES OF OSCILLATION (60s SIMULATION)")

# =========================================================
# 2. SIDEBAR PARAMETERS
# =========================================================
st.sidebar.header("🎛️ System Parameters")
m = st.sidebar.slider("Mass m (kg)", 0.2, 5.0, 1.0, 0.1)
l = st.sidebar.slider("Equilibrium Length l (m)", 0.5, 2.5, 1.2, 0.1)
g = st.sidebar.slider("Gravity g (m/s²)", 1.0, 20.0, 9.81, 0.1)

resonance_lock = st.sidebar.checkbox("Lock 2:1 Autoparametric Resonance (ω_z = 2ω_x)", value=True)
if resonance_lock:
    k = 4.0 * m * g / l
    st.sidebar.info(f"Locked Stiffness k = {k:.2f} N/m")
else:
    k = st.sidebar.slider("Spring Stiffness k (N/m)", 5.0, 150.0, float(4.0 * m * g / l), 0.5)

st.sidebar.header("🚀 Initial Amplitudes")
amp_z = st.sidebar.slider("Vertical Bounce Amplitude z₀ (m)", 0.05, 0.45, 0.28, 0.01)
amp_x = st.sidebar.slider("Horizontal Swing Amplitude x₀ (m)", 0.05, 0.45, 0.28, 0.01)

# Derived quantities
l0 = l - (m * g) / k
omega_z = np.sqrt(k / m)
omega_x = np.sqrt(g / l)

# Display Neon Metrics Bar
c1, c2, c3, c4 = st.columns(4)
c1.metric("Vertical Mode Freq (ω_z)", f"{omega_z:.2f} rad/s")
c2.metric("Horizontal Mode Freq (ω_x)", f"{omega_x:.2f} rad/s")
c3.metric("Frequency Ratio (ω_z / ω_x)", f"{omega_z / omega_x:.2f}")
c4.metric("Duration", "60.0 Seconds (1 Min)")

# =========================================================
# 3. SOLVE 60-SECOND ODE FOR ALL 3 MODES
# =========================================================
def spring_pendulum_ode(t, state):
    x, vx, z, vz = state
    L = np.hypot(x, l - z)
    factor = (k / m) * (1.0 - l0 / L)
    return [vx, -factor * x, vz, -g + factor * (l - z)]

t_max = 60.0      # 1 full minute duration
fps = 12          # 12 FPS * 60s = 720 frames for responsive web playback
n_frames = int(t_max * fps)
t_eval = np.linspace(0, t_max, n_frames)

inits = [
    [0.0, 0.0, amp_z, 0.0],       # Mode 1: Pure Vertical (z)
    [amp_x, 0.0, 0.0, 0.0],       # Mode 2: Pure Horizontal (x)
    [0.03, 0.0, amp_z, 0.0]       # Mode 3: Coupled 2:1 Resonance
]

sols = []
for init in inits:
    sol = solve_ivp(spring_pendulum_ode, [0, t_max], init, t_eval=t_eval, rtol=1e-7, atol=1e-7)
    sols.append({"x": sol.y[0], "z": sol.y[2], "t": sol.t})

def get_spring_coil(x_b, z_b, n_coils=9, width=0.09):
    dx, dz = x_b, z_b - l
    dist = np.hypot(dx, dz)
    ux, uz = dx / dist, dz / dist
    nx, nz = -uz, ux
    s = np.linspace(0, max(0.1, dist - 0.12), 80)
    offset = np.zeros_like(s)
    offset[6:-6] = width * np.sin(np.linspace(0, 2 * np.pi * n_coils, len(s) - 12))
    return s * ux + offset * nx, l + s * uz + offset * nz

# =========================================================
# 4. INTERACTIVE 60-SECOND NEON TIME-SERIES GRAPHS (ALL 3 MODES)
# =========================================================
st.subheader("📈 Full 1-Minute (60s) Displacement Graphs — All 3 Modes")

fig_graphs = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "Mode 1: Vertical (z) Spring Mode",
        "Mode 2: Horizontal (x) Swing Mode",
        "Mode 3: Coupled 2:1 Energy Exchange"
    )
)

for idx in range(3):
    col = idx + 1
    # Horizontal x(t) in Neon Cyan
    fig_graphs.add_trace(
        go.Scatter(x=sols[idx]["t"], y=sols[idx]["x"], mode="lines",
                   name="x(t) Horizontal", line=dict(color="#00f3ff", width=1.8),
                   showlegend=(idx == 0)),
        row=1, col=col
    )
    # Vertical z(t) in Neon Magenta
    fig_graphs.add_trace(
        go.Scatter(x=sols[idx]["t"], y=sols[idx]["z"], mode="lines",
                   name="z(t) Vertical", line=dict(color="#ff007f", width=1.8),
                   showlegend=(idx == 0)),
        row=1, col=col
    )
    fig_graphs.update_xaxes(title_text="Time (s)", range=[0, 60], gridcolor="#1a1d35", row=1, col=col)
    fig_graphs.update_yaxes(title_text="Displacement (m)", range=[-0.55, 0.55], gridcolor="#1a1d35", row=1, col=col)

fig_graphs.update_layout(
    template="plotly_dark",
    paper_bgcolor="#05050a",
    plot_bgcolor="#090b14",
    font=dict(color="#00f3ff"),
    height=340,
    margin=dict(l=30, r=30, t=50, b=30)
)
st.plotly_chart(fig_graphs, use_container_width=True)

# =========================================================
# 5. NEON SIDE-BY-SIDE 1-MINUTE ANIMATION OF ALL 3 MODES
# =========================================================
st.subheader("🎬 1-Minute Neon Motion Simulation (Click ▶ Play Below)")

fig_anim = make_subplots(
    rows=1, cols=3,
    subplot_titles=(
        "Mode 1: Pure Vertical Bounce (z)",
        "Mode 2: Pure Pendulum Swing (x)",
        "Mode 3: Coupled Resonance (x ↔ z)"
    )
)

neon_bob_colors = ["#ff007f", "#00f3ff", "#39ff14"]  # Magenta, Cyan, Lime

# Initial Frame Traces (3 traces per mode = 9 traces total)
for i in range(3):
    col = i + 1
    x0_val, z0_val = sols[i]["x"][0], sols[i]["z"][0]
    sx0, sz0 = get_spring_coil(x0_val, z0_val)
    
    # 1. Neon Glowing Trail
    fig_anim.add_trace(
        go.Scatter(x=[x0_val], y=[z0_val], mode="lines",
                   line=dict(color=neon_bob_colors[i], width=2.5), showlegend=False),
        row=1, col=col
    )
    # 2. Neon Electric Spring Coil
    fig_anim.add_trace(
        go.Scatter(x=sx0, y=sz0, mode="lines",
                   line=dict(color="#b829ff", width=3.2), showlegend=False),
        row=1, col=col
    )
    # 3. Glowing Mass Bob 'm'
    fig_anim.add_trace(
        go.Scatter(x=[x0_val], y=[z0_val], mode="markers+text", text=["m"],
                   textfont=dict(color="#05050a", size=13),
                   marker=dict(size=30, color=neon_bob_colors[i],
                               line=dict(color="#ffffff", width=2)),
                   showlegend=False),
        row=1, col=col
    )
    # Fixed Neon Top Support Bar at z = l
    fig_anim.add_shape(
        type="line", x0=-0.45, y0=l, x1=0.45, y1=l,
        line=dict(color="#00f3ff", width=6), row=1, col=col
    )
    fig_anim.update_xaxes(range=[-0.8, 0.8], title_text="x (m)", gridcolor="#1a1d35", row=1, col=col)
    fig_anim.update_yaxes(range=[-0.6, l + 0.25], title_text="z (m)", gridcolor="#1a1d35",
                          scaleanchor=f"x{col if col > 1 else ''}", scaleratio=1, row=1, col=col)

# Build 60-Second Animation Frames
frames = []
trail_len = 45
for k_frame in range(n_frames):
    frame_data = []
    start_idx = max(0, k_frame - trail_len)
    for i in range(3):
        x_now = sols[i]["x"][k_frame]
        z_now = sols[i]["z"][k_frame]
        sx, sz = get_spring_coil(x_now, z_now)
        
        frame_data.extend([
            go.Scatter(x=sols[i]["x"][start_idx:k_frame+1], y=sols[i]["z"][start_idx:k_frame+1]),
            go.Scatter(x=sx, y=sz),
            go.Scatter(x=[x_now], y=[z_now])
        ])
    frames.append(go.Frame(data=frame_data, name=f"{t_eval[k_frame]:.1f}s"))

fig_anim.frames = frames

fig_anim.update_layout(
    template="plotly_dark",
    paper_bgcolor="#05050a",
    plot_bgcolor="#090b14",
    font=dict(color="#00f3ff"),
    height=560,
    updatemenus=[dict(
        type="buttons",
        showactive=False,
        x=0.05, y=1.15,
        buttons=[
            dict(label="▶ Play 1-Min Neon Simulation", method="animate",
                 args=[None, dict(frame=dict(duration=int(1000/fps), redraw=True),
                                  fromcurrent=True, transition=dict(duration=0))]),
            dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
        ]
    )]
)

st.plotly_chart(fig_anim, use_container_width=True)
    ),
    frames=frames
)
st.plotly_chart(fig, use_container_width=True)
