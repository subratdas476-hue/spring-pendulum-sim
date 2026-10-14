import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

st.set_page_config(page_title="Spring Pendulum Simulation", layout="wide")
st.title("Interactive Spring Pendulum — Modes of Oscillation")

# Sidebar Controls
st.sidebar.header("Physical Parameters")
m = st.sidebar.slider("Mass m (kg)", 0.5, 5.0, 1.0, 0.1)
l = st.sidebar.slider("Equilibrium Length l (m)", 0.5, 2.5, 1.2, 0.1)
g = 9.81

mode_choice = st.sidebar.selectbox(
    "Select Oscillation Mode",
    [
        "Mode 1: Vertical (z) Spring Mode",
        "Mode 2: Horizontal (x) Swing Mode",
        "Mode 3: Coupled 2:1 Resonance Mode"
    ]
)

# 2:1 Resonance stiffness k = 4*m*g/l
k = 4.0 * m * g / l
l0 = l - (m * g) / k

if "Mode 1" in mode_choice:
    init_state = [0.0, 0.0, 0.28, 0.0]
elif "Mode 2" in mode_choice:
    init_state = [0.28, 0.0, 0.0, 0.0]
else:
    init_state = [0.03, 0.0, 0.28, 0.0]

def ode(t, state):
    x, vx, z, vz = state
    L = np.hypot(x, l - z)
    factor = (k / m) * (1.0 - l0 / L)
    return [vx, -factor * x, vz, -g + factor * (l - z)]

t_eval = np.linspace(0, 12, 240)
sol = solve_ivp(ode, [0, 12], init_state, t_eval=t_eval, rtol=1e-8)
x_arr, z_arr = sol.y[0], sol.y[2]

def spring_coords(x_b, z_b, n_coils=9, width=0.08):
    dx, dz = x_b, z_b - l
    dist = np.hypot(dx, dz)
    ux, uz = dx / dist, dz / dist
    nx, nz = -uz, ux
    s = np.linspace(0, dist - 0.12, 120)
    offset = np.zeros_like(s)
    offset[10:-10] = width * np.sin(np.linspace(0, 2 * np.pi * n_coils, 100))
    return s * ux + offset * nx, l + s * uz + offset * nz

# Build Plotly Animated Figure
frames = []
for idx in range(len(t_eval)):
    sx, sz = spring_coords(x_arr[idx], z_arr[idx])
    frames.append(go.Frame(data=[
        go.Scatter(x=x_arr[:idx+1], y=z_arr[:idx+1]), # Trail
        go.Scatter(x=sx, y=sz),                       # Coiled Spring
        go.Scatter(x=[x_arr[idx]], y=[z_arr[idx]])    # Red Mass m
    ]))

sx0, sz0 = spring_coords(x_arr[0], z_arr[0])
fig = go.Figure(
    data=[
        go.Scatter(x=[x_arr[0]], y=[z_arr[0]], mode="lines", line=dict(color="royalblue", width=2), name="Trajectory"),
        go.Scatter(x=sx0, y=sz0, mode="lines", line=dict(color="#333333", width=4), name="Spring"),
        go.Scatter(x=[x_arr[0]], y=[z_arr[0]], mode="markers+text", text=["m"],
                   marker=dict(size=36, color="#d90404", line=dict(width=2, color="black")), name="Mass m")
    ],
    layout=go.Layout(
        xaxis=dict(range=[-0.8, 0.8], title="x (m)", constrain="domain"),
        yaxis=dict(range=[-0.6, l + 0.3], title="z (m)", scaleanchor="x", scaleratio=1),
        height=650,
        updatemenus=[dict(type="buttons", showactive=False, buttons=[
            dict(label="▶ Play Simulation", method="animate",
                 args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True)])
        ])]
    ),
    frames=frames
)
st.plotly_chart(fig, use_container_width=True)
