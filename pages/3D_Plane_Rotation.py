import streamlit as st
import streamlit.components.v1 as components
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import hilbert
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# 1. NEON THEME PAGE CONFIGURATION
st.set_page_config(page_title="3D Spring Pendulum - Plane Rotation", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #05050a; color: #00f3ff; }
section[data-testid="stSidebar"] { background-color: #0b0c16 !important; border-right: 1px solid #00f3ff; }
h1, h2, h3, label, .stMarkdown { color: #00f3ff !important; text-shadow: 0 0 8px rgba(0, 243, 255, 0.5); }
div[data-testid="stMetric"] { background-color: #0d0f1d; border: 1px solid #ff007f; border-radius: 10px; padding: 10px; box-shadow: 0 0 12px rgba(255, 0, 127, 0.35); }
div[data-testid="stMetricValue"] { color: #39ff14 !important; text-shadow: 0 0 8px rgba(57, 255, 20, 0.6); }
</style>
""", unsafe_allow_html=True)

st.title("🌀 3D NEON SPRING PENDULUM — BEAT ENERGY EXCHANGE & PLANE ROTATION (60s)")

# 2. SIDEBAR PARAMETERS
st.sidebar.header("🎛️ 3D System Parameters")
m = st.sidebar.slider("Mass m (kg)", 0.5, 5.0, 1.5, 0.1)
l = st.sidebar.slider("Equilibrium Length l (m)", 0.3, 2.5, 1.5, 0.1)
g = st.sidebar.slider("Gravity g (m/s²)", 1.0, 20.0, 9.81, 0.1)

resonance_lock = st.sidebar.checkbox("Lock 1:1:2 Resonance (ω_z = 2ω_x = 2ω_y)", value=True)
if resonance_lock:
    k = 4.0 * m * g / l
    st.sidebar.info(f"Locked 2:1 Stiffness k = {k:.2f} N/m")
else:
    k = st.sidebar.slider("Spring Stiffness k (N/m)", 5.0, 200.0, float(4.0 * m * g / l), 0.5)

st.sidebar.header("🌀 3D Plane-Shift Controls")
z0 = st.sidebar.slider("Initial Vertical Bounce z₀ (m)", 0.08, 0.40, 0.22, 0.01)
x0 = st.sidebar.slider("Initial X Seed x₀ (m)", 0.01, 0.15, 0.04, 0.01)
vy0 = st.sidebar.slider("Transverse Y Velocity Seed vy₀ (m/s)", 0.01, 0.30, 0.08, 0.01)
torsion_coup = st.sidebar.slider("Spring Coil Precession / Torsion Factor", 0.00, 0.50, 0.18, 0.02)

l0 = l - (m * g) / k
omega_z = np.sqrt(k / m)
omega_xy = np.sqrt(g / l)
t_beat = (4.0 * np.pi * l) / (3.0 * omega_xy * z0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vertical Spring Freq (ω_z)", f"{omega_z:.2f} rad/s")
c2.metric("Horizontal Freq (ω_x = ω_y)", f"{omega_xy:.2f} rad/s")
c3.metric("Beat Period (T_beat)", f"~{t_beat:.1f} s")
c4.metric("3D Resonance Ratio", f"1 : 1 : {omega_z/omega_xy:.2f}")

# 3. CACHED 3D (x, y, z) ODE SOLVER
@st.cache_data(show_spinner=False)
def solve_3d_pendulum(m_val, l_val, g_val, k_val, l0_val, z_init, x_init, vy_init, torsion, w_xy, w_z):
    t_max = 60.0
    fps = 25
    t_eval = np.linspace(0, t_max, int(t_max * fps))

    def ode_3d(t, state):
        x, vx, y, vy, z, vz = state
        # 3D instantaneous spring length from ceiling (0, 0, l) to (x, y, z)
        L = np.sqrt(x**2 + y**2 + (l_val - z)**2)
        factor = (k_val / m_val) * (1.0 - l0_val / L)

        # Coriolis-like helical spring torsion / precession coupling during vertical bounce
        ax = -factor * x + torsion * (z / l_val) * w_xy * vy
        ay = -factor * y - torsion * (z / l_val) * w_xy * vx
        az = -g_val + factor * (l_val - z)
        return [vx, ax, vy, ay, vz, az]

    init_state = [x_init, 0.0, 0.01, vy_init, z_init, 0.0]
    sol = solve_ivp(ode_3d, [0, t_max], init_state, t_eval=t_eval, method="RK45", rtol=1e-6, atol=1e-6)
    x_a, vx_a, y_a, vy_a, z_a, vz_a = sol.y

    # Radial horizontal amplitude r_xy = sqrt(x^2 + y^2) and plane angle phi = atan2(y, x)
    r_xy = np.hypot(x_a, y_a)
    phi_deg = np.degrees(np.arctan2(y_a, x_a))

    # Energies in X, Y, and Z modes
    ex = 0.5 * m_val * vx_a**2 + 0.5 * m_val * (w_xy**2) * x_a**2
    ey = 0.5 * m_val * vy_a**2 + 0.5 * m_val * (w_xy**2) * y_a**2
    ez = 0.5 * m_val * vz_a**2 + 0.5 * m_val * (w_z**2) * z_a**2

    return {
        "t": np.round(sol.t, 2).tolist(),
        "x": np.round(x_a, 4).tolist(),
        "y": np.round(y_a, 4).tolist(),
        "z": np.round(z_a, 4).tolist(),
        "r_xy": np.round(r_xy, 4).tolist(),
        "phi": np.round(phi_deg, 1).tolist(),
        "ex": np.round(ex, 4).tolist(),
        "ey": np.round(ey, 4).tolist(),
        "ez": np.round(ez, 4).tolist()
    }

sim = solve_3d_pendulum(m, l, g, k, l0, z0, x0, vy0, torsion_coup, omega_xy, omega_z)

# 4. 60-SECOND NEON GRAPHS: (1) X, Y, Z BEATS & (2) ENERGY EXCHANGE BETWEEN X, Y, Z
st.subheader("📈 1-Minute 3D Beat Waveforms & Plane-Switching Energy Transfer")

fig_graphs = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=(
        "3D Coordinates: Vertical z(t) vs Horizontal Planes x(t) & y(t)",
        "Mode Energy Transfer: Vertical E_z ↔ Plane X (E_x) ↔ Plane Y (E_y)"
    )
)

step = 2
t_sub = sim["t"][::step]

# Left Plot: Displacements x(t), y(t), z(t)
fig_graphs.add_trace(
    go.Scattergl(x=t_sub, y=sim["z"][::step], mode="lines", name="Vertical z(t)",
                 line=dict(color="#ff007f", width=1.5)),
    row=1, col=1
)
fig_graphs.add_trace(
    go.Scattergl(x=t_sub, y=sim["x"][::step], mode="lines", name="Horizontal x(t) [East-West]",
                 line=dict(color="#00f3ff", width=1.6)),
    row=1, col=1
)
fig_graphs.add_trace(
    go.Scattergl(x=t_sub, y=sim["y"][::step], mode="lines", name="Horizontal y(t) [North-South]",
                 line=dict(color="#39ff14", width=1.6)),
    row=1, col=1
)

# Right Plot: Energies E_z, E_x, E_y showing plane switching on each beat!
fig_graphs.add_trace(
    go.Scattergl(x=t_sub, y=sim["ez"][::step], mode="lines", name="Vertical Spring Energy E_z",
                 line=dict(color="#ff007f", width=2.3)),
    row=1, col=2
)
fig_graphs.add_trace(
    go.Scattergl(x=t_sub, y=sim["ex"][::step], mode="lines", name="X-Plane Swing Energy E_x",
                 line=dict(color="#00f3ff", width=2.3)),
    row=1, col=2
)
fig_graphs.add_trace(
    go.Scattergl(x=t_sub, y=sim["ey"][::step], mode="lines", name="Y-Plane Swing Energy E_y",
                 line=dict(color="#39ff14", width=2.3)),
    row=1, col=2
)

fig_graphs.update_xaxes(title_text="Time (0 to 60s)", range=[0, 60], gridcolor="#1a1d35", row=1, col=1)
fig_graphs.update_xaxes(title_text="Time (0 to 60s)", range=[0, 60], gridcolor="#1a1d35", row=1, col=2)
fig_graphs.update_yaxes(title_text="Displacement (m)", gridcolor="#1a1d35", row=1, col=1)
fig_graphs.update_yaxes(title_text="Energy (Joules)", gridcolor="#1a1d35", row=1, col=2)

fig_graphs.update_layout(
    template="plotly_dark",
    paper_bgcolor="#05050a",
    plot_bgcolor="#090b14",
    font=dict(color="#00f3ff"),
    height=320,
    margin=dict(l=30, r=30, t=40, b=30)
)
st.plotly_chart(fig_graphs, use_container_width=True)

# 5. HARDWARE-ACCELERATED 3-PANEL NEON ANIMATION (3D VIEW + TOP-DOWN ROSETTE + SIDE VIEW)
st.subheader("🎬 1-Minute Real-Time 3D Simulation — Watch the Plane of Motion Rotate After Every Beat!")

sim_json = json.dumps({"sim": sim, "l": l})

canvas_html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  body {{ margin: 0; background: #05050a; color: #00f3ff; font-family: 'Segoe UI', sans-serif; text-align: center; }}
  .controls {{
    display: flex; justify-content: center; align-items: center; gap: 15px;
    margin-bottom: 10px; padding: 8px; background: #0b0d19;
    border: 1px solid #00f3ff; border-radius: 8px; box-shadow: 0 0 12px rgba(0, 243, 255, 0.25);
  }}
  button {{
    background: #05050a; color: #39ff14; border: 1px solid #39ff14;
    padding: 6px 14px; font-size: 13px; font-weight: bold; border-radius: 6px; cursor: pointer;
  }}
  button:hover {{ background: #39ff14; color: #05050a; }}
  input[type=range] {{ width: 300px; accent-color: #ff007f; cursor: pointer; }}
  .timer {{ font-size: 15px; font-weight: bold; color: #ff007f; min-width: 130px; }}
  .canvas-row {{ display: flex; justify-content: space-around; gap: 10px; }}
  .panel {{
    background: #090b14; border: 1px solid #1a2138; border-radius: 10px;
    padding: 8px; width: 32%; box-shadow: 0 0 15px rgba(0, 243, 255, 0.12);
  }}
  .panel-title {{ font-size: 13.5px; font-weight: bold; margin-bottom: 6px; }}
  canvas {{ width: 100%; height: 390px; background: #070810; border-radius: 6px; }}
</style>
</head>
<body>
  <div class="controls">
    <button id="playBtn" onclick="togglePlay()">⏸ Pause</button>
    <button onclick="resetSim()">⏮ Reset (0.0s)</button>
    <button onclick="clearRosette()">🧹 Clear Rosette Trail</button>
    <input type="range" id="timeSlider" min="0" max="1499" value="0" oninput="scrub(this.value)">
    <div class="timer" id="timeDisplay">Time: 0.0s / 60.0s</div>
  </div>
  <div class="canvas-row">
    <div class="panel">
      <div class="panel-title" style="color:#00f3ff; text-shadow:0 0 8px #00f3ff;">1. 3D Perspective View (x, y, z)</div>
      <canvas id="c3d" width="360" height="390"></canvas>
    </div>
    <div class="panel">
      <div class="panel-title" style="color:#39ff14; text-shadow:0 0 8px #39ff14;">2. Top-Down View (x-y Plane Rotation!)</div>
      <canvas id="ctop" width="360" height="390"></canvas>
    </div>
    <div class="panel">
      <div class="panel-title" style="color:#ff007f; text-shadow:0 0 8px #ff007f;">3. Side View (x-z Bounce ↔ Swing)</div>
      <canvas id="cside" width="360" height="390"></canvas>
    </div>
  </div>

<script>
const payload = {sim_json};
const sim = payload.sim;
const l_eq = payload.l;
const nFrames = sim.t.length;

const c3d = document.getElementById("c3d").getContext("2d");
const ctop = document.getElementById("ctop").getContext("2d");
const cside = document.getElementById("cside").getContext("2d");

let frame = 0;
let playing = true;
let trailStart = 0;
const slider = document.getElementById("timeSlider");
const timeDisplay = document.getElementById("timeDisplay");
const playBtn = document.getElementById("playBtn");

function togglePlay() {{
  playing = !playing;
  playBtn.innerText = playing ? "⏸ Pause" : "▶ Play";
}}
function resetSim() {{ frame = 0; trailStart = 0; slider.value = 0; drawFrame(0); }}
function clearRosette() {{ trailStart = frame; drawFrame(frame); }}
function scrub(val) {{ frame = parseInt(val); if (frame < trailStart) trailStart = 0; drawFrame(frame); }}

// Isometric 3D projection helper
function proj3D(x, y, z, cx, cy, scale) {{
  const angle = 0.55;
  const px = cx + (x * Math.cos(angle) - y * Math.sin(angle)) * scale;
  const py = cy - z * scale + (x * Math.sin(angle) + y * Math.cos(angle)) * 0.38 * scale;
  return [px, py];
}}

function drawSpring2D(ctx, x0, y0, x1, y1, coils=10, width=10) {{
  const dx = x1 - x0, dy = y1 - y0;
  const dist = Math.hypot(dx, dy);
  const ux = dx / dist, uy = dy / dist;
  const nx = -uy, ny = ux;
  ctx.save();
  ctx.beginPath();
  ctx.strokeStyle = "#b829ff";
  ctx.lineWidth = 2.8;
  ctx.shadowColor = "#b829ff";
  ctx.shadowBlur = 10;
  const steps = 85;
  for (let i = 0; i <= steps; i++) {{
    const f = i / steps;
    const s = f * Math.max(10, dist - 14);
    let offset = 0;
    if (f > 0.08 && f < 0.92) {{
      offset = width * Math.sin(((f - 0.08) / 0.84) * coils * 2 * Math.PI);
    }}
    const px = x0 + s * ux + offset * nx;
    const py = y0 + s * uy + offset * ny;
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }}
  ctx.stroke();
  ctx.restore();
}}

function drawFrame(idx) {{
  timeDisplay.innerText = "Time: " + sim.t[idx].toFixed(1) + "s / 60.0s";
  const scale = 150 / Math.max(0.8, l_eq);
  const cx = 180;
  const ceilY = 45;
  const eqY = ceilY + l_eq * scale;

  // ==========================================
  // PANEL 1: 3D PERSPECTIVE VIEW (x, y, z)
  // ==========================================
  c3d.clearRect(0, 0, 360, 390);
  // Draw 3D Floor Grid at equilibrium z = 0
  c3d.strokeStyle = "#161d38";
  c3d.lineWidth = 1;
  for (let g = -0.6; g <= 0.61; g += 0.3) {{
    let p1 = proj3D(g, -0.6, 0, cx, eqY, scale);
    let p2 = proj3D(g, 0.6, 0, cx, eqY, scale);
    c3d.beginPath(); c3d.moveTo(p1[0], p1[1]); c3d.lineTo(p2[0], p2[1]); c3d.stroke();
    let p3 = proj3D(-0.6, g, 0, cx, eqY, scale);
    let p4 = proj3D(0.6, g, 0, cx, eqY, scale);
    c3d.beginPath(); c3d.moveTo(p3[0], p3[1]); c3d.lineTo(p4[0], p4[1]); c3d.stroke();
  }}

  const ceil3D = proj3D(0, 0, l_eq, cx, eqY, scale);
  const bob3D = proj3D(sim.x[idx], sim.y[idx], sim.z[idx], cx, eqY, scale);
  const shadow3D = proj3D(sim.x[idx], sim.y[idx], -0.35, cx, eqY, scale);

  // 3D Trail
  c3d.save();
  c3d.beginPath();
  c3d.strokeStyle = "#00f3ff";
  c3d.lineWidth = 2;
  c3d.shadowColor = "#00f3ff";
  c3d.shadowBlur = 8;
  const s3d = Math.max(0, idx - 85);
  for (let k = s3d; k <= idx; k++) {{
    let pt = proj3D(sim.x[k], sim.y[k], sim.z[k], cx, eqY, scale);
    if (k === s3d) c3d.moveTo(pt[0], pt[1]);
    else c3d.lineTo(pt[0], pt[1]);
  }}
  c3d.stroke();
  c3d.restore();

  // Top Ceiling Disc + Coiled Spring + 3D Bob
  c3d.fillStyle = "#00f3ff";
  c3d.fillRect(ceil3D[0] - 45, ceil3D[1] - 8, 90, 8);
  drawSpring2D(c3d, ceil3D[0], ceil3D[1], bob3D[0], bob3D[1], 10, 9);

  c3d.save();
  c3d.beginPath();
  c3d.arc(bob3D[0], bob3D[1], 16, 0, 2 * Math.PI);
  c3d.fillStyle = "#ff007f";
  c3d.shadowColor = "#ff007f";
  c3d.shadowBlur = 16;
  c3d.fill();
  c3d.strokeStyle = "#ffffff";
  c3d.lineWidth = 2;
  c3d.stroke();
  c3d.restore();

  // ==========================================
  // PANEL 2: TOP-DOWN HORIZONTAL VIEW (x, y) ROSETTE
  // ==========================================
  ctop.clearRect(0, 0, 360, 390);
  const topCy = 185;
  const topScale = scale * 1.45;

  // Polar Angle Circles & Crosshairs
  ctop.strokeStyle = "#161d38";
  ctop.lineWidth = 1;
  [50, 100, 145].forEach(r => {{
    ctop.beginPath(); ctop.arc(cx, topCy, r, 0, 2 * Math.PI); ctop.stroke();
  }});
  ctop.beginPath();
  ctop.moveTo(20, topCy); ctop.lineTo(340, topCy);
  ctop.moveTo(cx, 25); ctop.lineTo(cx, 345);
  ctop.stroke();
  ctop.fillStyle = "#00f3ff";
  ctop.font = "11px sans-serif";
  ctop.fillText("+X (East)", 305, topCy - 6);
  ctop.fillText("+Y (North)", cx + 28, 38);

  // Instantaneous Swing Plane Axis Line (Dashed Yellow)
  const angleRad = Math.atan2(sim.y[idx], sim.x[idx]);
  ctop.save();
  ctop.strokeStyle = "rgba(255, 234, 0, 0.45)";
  ctop.setLineDash([5, 5]);
  ctop.lineWidth = 1.8;
  ctop.beginPath();
  ctop.moveTo(cx - 140 * Math.cos(angleRad), topCy + 140 * Math.sin(angleRad));
  ctop.lineTo(cx + 140 * Math.cos(angleRad), topCy - 140 * Math.sin(angleRad));
  ctop.stroke();
  ctop.restore();

  // Long Precessing Rosette Trail in (x, y) Plane
  const rStart = Math.max(trailStart, idx - 550);
  ctop.save();
  ctop.beginPath();
  ctop.strokeStyle = "#39ff14";
  ctop.lineWidth = 2.0;
  ctop.shadowColor = "#39ff14";
  ctop.shadowBlur = 8;
  for (let k = rStart; k <= idx; k++) {{
    const tx = cx + sim.x[k] * topScale;
    const ty = topCy - sim.y[k] * topScale;
    if (k === rStart) ctop.moveTo(tx, ty);
    else ctop.lineTo(tx, ty);
  }}
  ctop.stroke();
  ctop.restore();

  // Current (x, y) Bob Position
  const topBx = cx + sim.x[idx] * topScale;
  const topBy = topCy - sim.y[idx] * topScale;
  ctop.save();
  ctop.beginPath();
  ctop.arc(topBx, topBy, 12, 0, 2 * Math.PI);
  ctop.fillStyle = "#39ff14";
  ctop.shadowColor = "#39ff14";
  ctop.shadowBlur = 16;
  ctop.fill();
  ctop.strokeStyle = "#ffffff";
  ctop.stroke();
  ctop.restore();

  ctop.fillStyle = "#ffea00";
  ctop.font = "bold 12px sans-serif";
  ctop.fillText("Watch the green swing plane rotate each beat!", cx - 125, 372);

  // ==========================================
  // PANEL 3: SIDE VIEW (x-z) + 3-MODE ENERGY METERS
  // ==========================================
  cside.clearRect(0, 0, 360, 390);
  cside.strokeStyle = "#161d38";
  cside.beginPath();
  cside.moveTo(0, eqY); cside.lineTo(360, eqY);
  cside.moveTo(cx, 0); cside.lineTo(cx, 330);
  cside.stroke();

  cside.fillStyle = "#121629";
  cside.strokeStyle = "#00f3ff";
  cside.lineWidth = 2;
  cside.fillRect(cx - 70, ceilY - 12, 140, 12);
  cside.strokeRect(cx - 70, ceilY - 12, 140, 12);

  const sx = cx + sim.x[idx] * scale;
  const sy = eqY - sim.z[idx] * scale;
  drawSpring2D(cside, cx, ceilY, sx, sy, 10, 10);

  cside.save();
  cside.beginPath();
  cside.arc(sx, sy, 16, 0, 2 * Math.PI);
  cside.fillStyle = "#00f3ff";
  cside.shadowColor = "#00f3ff";
  cside.shadowBlur = 16;
  cside.fill();
  cside.strokeStyle = "#ffffff";
  cside.stroke();
  cside.restore();

  // Live 3-Mode Energy Bars (E_z, E_x, E_y)
  const ex = sim.ex[idx], ey = sim.ey[idx], ez = sim.ez[idx];
  const etot = Math.max(0.001, ex + ey + ez);

  cside.font = "bold 11px sans-serif";
  cside.fillStyle = "#ff007f";
  cside.fillText("Bounce E_z", 15, 342);
  cside.fillRect(88, 333, (ez / etot) * 240, 9);

  cside.fillStyle = "#00f3ff";
  cside.fillText("X-Swing E_x", 15, 360);
  cside.fillRect(88, 351, (ex / etot) * 240, 9);

  cside.fillStyle = "#39ff14";
  cside.fillText("Y-Swing E_y", 15, 378);
  cside.fillRect(88, 369, (ey / etot) * 240, 9);
}}

let lastTime = 0;
function loop(timestamp) {{
  if (playing && timestamp - lastTime > 38) {{
    frame = (frame + 1) % nFrames;
    if (frame === 0) trailStart = 0;
    slider.value = frame;
    drawFrame(frame);
    lastTime = timestamp;
  }}
  requestAnimationFrame(loop);
}}
drawFrame(0);
requestAnimationFrame(loop);
</script>
</body>
</html>
"""

components.html(canvas_html, height=490)
