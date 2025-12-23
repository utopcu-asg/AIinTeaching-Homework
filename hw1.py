"""Streamlit port of the updated 20251116 homework interactive demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

# --- Shared Configuration for Styling & Math ---
HTML_HEADER = r"""
<head>
  <script>
    window.MathJax = {
      tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
      }
    };
  </script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  <style>
    body { font-family: sans-serif; margin: 0; padding: 0; color: #111827; }
    .demo-card { 
        border: 1px solid #e5e7eb; 
        padding: 20px; 
        border-radius: 8px; 
        background: white;
    }
    .exercise h2 { margin-top: 0; font-size: 1.5rem; color: #1f2937; }
    .exercise p { line-height: 1.6; margin-bottom: 1em; }
    .equations { text-align: center; margin: 15px 0; overflow-x: auto; }
  </style>
</head>
"""

PLOTLY_CONFIG = {"modeBarButtonsToRemove": ["toImage"], "displaylogo": False}


def get_plot_style() -> dict[str, str]:
    """Return CSS-driven colors that adapt automatically to dark mode."""
    return {
        "paper_bgcolor": "var(--plot-paper-bgcolor)",
        "plot_bgcolor": "var(--plot-surface-bgcolor)",
        "font_color": "var(--plot-font-color)",
        "tick_color": "var(--plot-tick-color)",
        "grid_color": "var(--plot-grid-color)",
        "primary_line": "var(--plot-primary-line)",
        "state_line": "var(--plot-state-line)",
        "input_line": "var(--plot-input-line)",
    }


@dataclass(frozen=True)
class SimulationConstants:
    """Container for the plant, controller, and integration settings."""

    A: float
    B1: float
    B2: float
    C: float
    kp: float
    t0: float
    tf: float
    dt: float


@dataclass(frozen=True)
class SteadyStateSettings:
    """Closed-form steady-state response settings for the Question 4 system."""

    A: float
    B: float
    steady_start: float
    window: float
    dt: float


Q1_CONSTANTS = SimulationConstants(A=-1.2, B1=1.0, B2=0.35, C=1.0, kp=1.8, t0=0.0, tf=8.0, dt=0.01)
Q2_CONSTANTS = SimulationConstants(A=-0.3, B1=1.0, B2=0.35, C=0.2, kp=1.5, t0=0.0, tf=15.0, dt=0.01)
Q3_CONSTANTS = SimulationConstants(A=1.0, B1=5.0, B2=0.2, C=1.0, kp=1.8, t0=0.0, tf=8.0, dt=0.01)
STEADY_STATE = SteadyStateSettings(A=-1.2, B=1.0, steady_start=12.0, window=6.0, dt=0.005)


def compute_closed_loop_response(
    constants: SimulationConstants,
    *,
    magnitude: float,
    omega: float,
    disturbance: float,
    x0: float,
    kp_override: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-Euler integration matching the original HTML app."""
    kp_value = constants.kp if kp_override is None else kp_override
    time = np.arange(constants.t0, constants.tf + constants.dt, constants.dt)
    x_traj = np.empty_like(time)
    y_traj = np.empty_like(time)

    x_state = x0
    x_traj[0] = x_state
    y_traj[0] = constants.C * x_state

    for idx in range(len(time) - 1):
        t = time[idx]
        y_value = constants.C * x_state
        reference = magnitude * np.cos(omega * t)
        error = reference - y_value
        forcing = constants.B1 * kp_value * error + constants.B2 * disturbance
        dx = constants.A * x_state + forcing
        x_state = x_state + dx * constants.dt
        x_traj[idx + 1] = x_state
        y_traj[idx + 1] = constants.C * x_state

    return time, y_traj


def compute_steady_state_response(
    settings: SteadyStateSettings,
    *,
    amplitude: float,
    omega: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-form steady-state response of a forced first-order system."""
    steps = int(settings.window / settings.dt) + 1
    time = settings.steady_start + np.arange(steps) * settings.dt
    input_signal = amplitude * np.cos(omega * time)

    denominator = settings.A * settings.A + omega * omega
    alpha = (-settings.A * settings.B * amplitude / denominator) if denominator > 0 else 0.0
    beta = (settings.B * amplitude * omega / denominator) if denominator > 0 else 0.0

    if np.isclose(omega, 0.0, atol=1e-6):
        if np.isclose(settings.A, 0.0, atol=1e-6):
            state_signal = np.full_like(time, np.nan)
        else:
            state_signal = np.full_like(time, -settings.B * amplitude / settings.A)
    elif np.isclose(denominator, 0.0, atol=1e-12):
        state_signal = np.full_like(time, np.nan)
    else:
        state_signal = alpha * np.cos(omega * time) + beta * np.sin(omega * time)

    return time, state_signal, input_signal


def render_closed_loop_plot(time: np.ndarray, y: np.ndarray, *, height: int = 420) -> go.Figure:
    """Create a Plotly figure mirroring the styling of the HTML original."""
    style = get_plot_style()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=y,
            mode="lines",
            line=dict(width=3, color=style["primary_line"], shape="spline"),
            hovertemplate="<b>t:</b> %{x:.2f} s<br><b>y(t):</b> %{y:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        margin=dict(l=60, r=18, t=28, b=50),
        paper_bgcolor=style["paper_bgcolor"],
        plot_bgcolor=style["plot_bgcolor"],
        font=dict(family="Inter, sans-serif", color=style["font_color"]),
        xaxis=dict(
            title="Time (s)",
            gridcolor=style["grid_color"],
            zeroline=False,
            title_font=dict(color=style["font_color"]),
            tickfont=dict(color=style["tick_color"]),
        ),
        yaxis=dict(
            title="Output y(t)",
            gridcolor=style["grid_color"],
            zeroline=False,
            title_font=dict(color=style["font_color"]),
            tickfont=dict(color=style["tick_color"]),
        ),
        height=height,
        showlegend=False,
    )

    return fig


def render_state_input_plot(time: np.ndarray, x: np.ndarray, u: np.ndarray) -> go.Figure:
    """Stacked state/input view mirroring the Q4 HTML plot."""
    style = get_plot_style()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.08)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=x,
            mode="lines",
            line=dict(width=3, color=style["state_line"]),
            hovertemplate="<b>t:</b> %{x:.2f} s<extra></extra>",
            name="x(t)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=u,
            mode="lines",
            line=dict(width=2, color=style["input_line"], dash="dot"),
            hovertemplate="<b>t:</b> %{x:.2f} s<extra></extra>",
            name="u(t)",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        height=520,
        margin=dict(l=60, r=18, t=28, b=50),
        paper_bgcolor=style["paper_bgcolor"],
        plot_bgcolor=style["plot_bgcolor"],
        font=dict(family="Inter, sans-serif", color=style["font_color"]),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(
        title_text="Time (s)",
        gridcolor=style["grid_color"],
        zeroline=False,
        title_font=dict(color=style["font_color"]),
        tickfont=dict(color=style["tick_color"]),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="Time (s)",
        gridcolor=style["grid_color"],
        zeroline=False,
        title_font=dict(color=style["font_color"]),
        tickfont=dict(color=style["tick_color"]),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        title_text="x(t)",
        gridcolor=style["grid_color"],
        zeroline=False,
        showticklabels=False,
        title_font=dict(color=style["font_color"]),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="u(t)",
        gridcolor=style["grid_color"],
        zeroline=False,
        showticklabels=False,
        title_font=dict(color=style["font_color"]),
        row=2,
        col=1,
    )

    return fig


def require_password() -> bool:
    """Prompt for the password stored in Streamlit secrets before showing the app."""
    secret_password = st.secrets.get("password")
    if not secret_password:
        st.error("Password not configured. Add `password` to your `.streamlit/secrets.toml`.")
        return False

    if st.session_state.get("password_correct"):
        return True

    def _password_entered() -> None:
        if st.session_state["app_password"] == secret_password:
            st.session_state["password_correct"] = True
            st.session_state.pop("app_password", None)
            st.session_state.pop("password_error", None)
        else:
            st.session_state["password_correct"] = False
            st.session_state["password_error"] = True

    st.text_input(
        "Enter password to access the homework demo",
        type="password",
        key="app_password",
        on_change=_password_entered,
    )
    if st.session_state.get("password_error"):
        st.error("Incorrect password. Please try again.")
    else:
        st.info("This demo is password protected.")
    return False


def inject_custom_css() -> None:
    """Inject light-touch CSS for Homework 2."""
    st.markdown(
        """
        <style>
            :root {
                color-scheme: light;
                --plot-paper-bgcolor: rgba(0, 0, 0, 0);
                --plot-surface-bgcolor: rgba(248, 250, 252, 0.92);
                --plot-font-color: #111827;
                --plot-tick-color: #1f2937;
                --plot-grid-color: rgba(99, 102, 241, 0.15);
                --plot-primary-line: #3730a3;
                --plot-state-line: #f97316;
                --plot-input-line: #1d4ed8;
                --plot-vector-line: #e11d48;
            }
            body {
                background: #f5f7fb;
                color: #1f2937;
            }
            h1 {font-size: 2rem !important;}
            h2 {font-size: 1.5rem !important;}
            h3 {font-size: 1.2rem !important;}
            .stApp {
                background: #f5f7fb;
                color: #1f2937;
                font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                max-width: 1500px;
                margin: 0 auto;
            }
            .exercise-card {
                background: linear-gradient(135deg, rgba(16, 185, 129, 0.14), rgba(45, 212, 191, 0.14));
                border: 1px solid rgba(16, 185, 129, 0.25);
                border-radius: 16px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                color: #065f46;
            }
            .demo-card {
                background: #ffffff;
                border-radius: 16px;
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.1);
                padding: 2rem;
                margin-bottom: 2rem;
            }
            .equations-card {
                background: linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(14, 165, 233, 0.08));
                border: 1px solid rgba(79, 70, 229, 0.15);
                border-radius: 12px;
                padding: 1rem;
                margin: 1rem 0;
                text-align: center;
            }
            .vector-display {
                text-align: center;
                padding: 12px 16px;
                background: linear-gradient(135deg, rgba(225, 29, 72, 0.08), rgba(244, 63, 94, 0.08));
                border: 1px solid rgba(225, 29, 72, 0.2);
                border-radius: 10px;
                margin-bottom: 1.25rem;
                font-family: 'Inter', sans-serif;
                font-size: 1rem;
                color: #be123c;
            }
            @media (prefers-color-scheme: dark) {
                :root {
                    color-scheme: dark;
                    --plot-paper-bgcolor: rgba(0, 0, 0, 0);
                    --plot-surface-bgcolor: rgba(15, 23, 42, 0.92);
                    --plot-font-color: #e2e8f0;
                    --plot-tick-color: #cbd5f5;
                    --plot-grid-color: rgba(148, 163, 184, 0.25);
                    --plot-primary-line: #a5b4fc;
                    --plot-state-line: #fb923c;
                    --plot-input-line: #38bdf8;
                    --plot-vector-line: #fb7185;
                }
                body,
                .stApp {
                    background: #020617;
                    color: #f8fafc;
                }
                .exercise-card {
                    background: linear-gradient(135deg, rgba(16, 185, 129, 0.3), rgba(45, 212, 191, 0.22));
                    border: 1px solid rgba(16, 185, 129, 0.5);
                    color: #d1fae5;
                }
                .demo-card {
                    background: #0f172a;
                    box-shadow: 0 18px 40px rgba(2, 6, 23, 0.85);
                }
                .equations-card {
                    background: linear-gradient(135deg, rgba(79, 70, 229, 0.15), rgba(14, 165, 233, 0.15));
                    border: 1px solid rgba(79, 70, 229, 0.3);
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def slider_with_number(
    label: str,
    *,
    min_value: float,
    max_value: float,
    step: float,
    default: float,
    format_str: str = "%.2f",
    key: str,
) -> float:
    """Slider + number input combo block with synchronized values."""
    slider_key = f"{key}_slider"
    number_key = f"{key}_number"

    # Initialize session state if not already set
    if slider_key not in st.session_state:
        st.session_state[slider_key] = float(default)
    if number_key not in st.session_state:
        st.session_state[number_key] = float(default)

    def _sync_from_slider() -> None:
        st.session_state[number_key] = st.session_state[slider_key]

    def _sync_from_number() -> None:
        st.session_state[slider_key] = st.session_state[number_key]

    slider_col, number_col = st.columns([4, 1], gap="small")
    with slider_col:
        st.slider(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            key=slider_key,
            on_change=_sync_from_slider,
        )
    with number_col:
        st.number_input(
            label,
            min_value=min_value,
            max_value=max_value,
            step=step,
            format=format_str,
            key=number_key,
            on_change=_sync_from_number,
            label_visibility="hidden",
        )

    return float(st.session_state[number_key])


def render_simulation_constants(constants: SimulationConstants, *, label: str) -> None:
    """Display the current constants in a compact LaTeX block."""
    st.markdown(
        rf"""
        **Simulation constants ({label})**

        $$
        \begin{{aligned}}
            &A = {constants.A} \qquad
            B_1 = {constants.B1} \qquad
            B_2 = {constants.B2} \qquad
            C = {constants.C} \\[0.5em]
            &k_p = {constants.kp} \qquad
            t_0 = {constants.t0} \qquad
            t_f = {constants.tf} \qquad
            dt = {constants.dt}
        \end{{aligned}}
        $$
        """,
        unsafe_allow_html=True,
    )


def render_problem_1():
    st.header("Question 1: Identifying the system parameters from the forced response of a first-order system (40 pts)")
    st.image(
        "media/feedback.svg",
        width=800,
    )
    st.markdown(
        r"""
        Consider the feedback interconnection above where $A$, $B_{1}$, $B_{2}$, and $C$ are scalars.
        The initial condition of the system satisfies $x(0) = x_{0}$.
        """
    )

    # 1. Define the HTML Body for the static content
    body = r"""
    <body>
      <div class="demo-card">
        <div class="exercise">
          <h3>Part 1 (10 pts)</h3>
          <p>
            By hand, derive the steady-state gain from the step reference input \(r(t) = \bar{r}\) to the output \(y\) in terms of:
          </p>
          <ul>
            <li>The state matrix (scalar) \(A\)</li>
            <li>The input matrix (scalar) \(B_{1}\) for the input \(u\)</li>
            <li>The proportional gain (scalar) \(k_{p}\) for the input \(u\)</li>
            <li>The input matrix (scalar) \(B_{2}\) for the input \(d\)</li>
            <li>The output matrix (scalar) \(C\)</li>
            <li>The input magnitude \(\bar{r}\) for the input \(r\)</li>
          </ul>
          <p>Explain your reasoning and include any drawings that support your answers.</p>

          <hr style="margin: 20px 0; opacity: 0.3;">

          <h3>Part 2 (10 pts)</h3>
          <p>
            By hand, derive the steady-state gain from the step disturbance input \(d(t) = \bar{d}\) to the output \(y\) in terms of:
          </p>
          <ul>
            <li>The state matrix (scalar) \(A\)</li>
            <li>The input matrix (scalar) \(B_{1}\) for the input \(u\)</li>
            <li>The proportional gain (scalar) \(k_{p}\) for the input \(u\)</li>
            <li>The input matrix (scalar) \(B_{2}\) for the input \(d\)</li>
            <li>The output matrix (scalar) \(C\)</li>
            <li>The input magnitude \(\bar{d}\) for the input \(d\)</li>
          </ul>
          <p>Explain your reasoning and include any drawings that support your answers.</p>

          <hr style="margin: 20px 0; opacity: 0.3;">

          <h3>Part 3 (20 pts)</h3>
          <p>
            The below plot shows the complete response \(y(t)\) of the system under inputs \(r\) and \(d\). Identify:
          </p>
          <ul>
            <li>The state matrix (scalar) \(A\)</li>
            <li>The input matrix (scalar) \(B_{1}\) for the input \(u\)</li>
            <li>The proportional gain (scalar) \(k_{p}\) for the input \(u\)</li>
            <li>The input matrix (scalar) \(B_{2}\) for the input \(d\)</li>
            <li>The output matrix (scalar) \(C\)</li>
          </ul>
          <p>
            by setting the values of the initial condition \(x(0)=x_{0}\), sinusoidal reference input \(r(t) = \bar{r} \cos(\omega t) \), and step disturbance input \(d(t) = \bar{d}\).
          </p>
          <p>
            If any of the parameters is not uniquely identifiable, state the most precise condition for the parameter.
          </p>
          <p>Explain your reasoning and include any plots that support your answers.</p>
        </div>
      </div>
    </body>
    """

    # 2. Render the HTML component
    # Adjust height to fit the content comfortably
    components.html(f"<!DOCTYPE html><html>{HTML_HEADER}{body}</html>", height=950, scrolling=False)

    # 3. Render Native Streamlit Interactions (Sliders & Plot)
    # These must be outside the HTML component to function correctly
    with st.container():
        st.markdown("***You can hover on the plot to get time and magnitude measurements.***")
        col1, col2 = st.columns(2)
        with col1:
            magnitude = slider_with_number(
                r"Reference input magnitude $\bar{r}$",
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                default=1.0,
                key="q1_magnitude",
            )
            disturbance = slider_with_number(
                r"Disturbance input magnitude $\bar{d}$",
                min_value=-1.5,
                max_value=1.5,
                step=0.01,
                default=0.5,
                key="q1_disturbance",
            )
        with col2:
            omega = slider_with_number(
                r"Reference input frequency $\omega$",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                default=2.0,
                key="q1_omega",
                format_str="%.1f",
            )
            x0 = slider_with_number(
                r"Initial state $x_{0}$",
                min_value=-2.0,
                max_value=2.0,
                step=0.01,
                default=-0.2,
                key="q1_x0",
            )

        time, y = compute_closed_loop_response(
            Q1_CONSTANTS,
            magnitude=magnitude,
            omega=omega,
            disturbance=disturbance,
            x0=x0,
        )
        fig = render_closed_loop_plot(time, y)
        st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOTLY_CONFIG)


def render_problem_2() -> None:
    st.header(
        "Question 2: Identifying the system parameters from the steady-state response of a first-order system (30 pts)"
    )
    st.markdown(
        r"""
        Consider the feedback interconnection above where $A$, $B_{1}$, $B_{2}$, and $C$ are scalars.

        The initial condition of the system satisfies $x(0) = x_{0}$. 
        """
    )

    body = r"""
    <body>
      <div class="demo-card">
        <div class="exercise">
        
          <h3>Part 1 (10 pts)</h3>
          <p>
            By hand, derive the steady-state gain from the sinusoidal reference input \(r(t) = \bar{r}\cos(\omega t)\) to the output \(y\) in terms of:
          </p>
          <ul>
            <li>The state matrix (scalar) \(A\)</li>
            <li>The input matrix (scalar) \(B_{1}\) for the input \(u\)</li>
            <li>The proportional gain (scalar) \(k_{p}\) for the input \(u\)</li>
            <li>The input matrix (scalar) \(B_{2}\) for the input \(d\)</li>
            <li>The output matrix (scalar) \(C\)</li>
            <li>The input frequency \(\omega\) for the input \(r\)</li>
            <li>The input magnitude \(\bar{r}\) for the input \(r\)</li>
          </ul>

          <p>Explain your reasoning and include any drawings that support your answers.</p>

          <p style="background-color: #f9f9f9; padding: 10px; border-left: 4px solid #ccc; margin-top: 10px;">
            <strong>Hint:</strong> Derive the closed-loop dynamics as a first-order system 
            \(\dot{y} = \tilde{A} y + \tilde{B}_{1} r + \tilde{B}_{2} d\) 
            where \(\tilde{A}\), \(\tilde{B}_{1}\), and \(\tilde{B}_{2}\) depend on the parameters listed above.
          </p>

          <hr style="margin: 20px 0; opacity: 0.3;">

          <h3>Part 2 (20 pts)</h3>
          <p>
            Assume \(d(t)=0\) and \(x(0)=0\). The below plot shows the steady-state response \(y(t)\) of the system under input \(r\).
          </p>
          
          <p>Identify:</p>
          <ul>
            <li>The state matrix (scalar) \(A\)</li>
            <li>The input matrix (scalar) \(B_{1}\) for the input \(u\)</li>
            <li>The proportional gain (scalar) \(k_{p}\) for the input \(u\)</li>
            <li>The output matrix (scalar) \(C\)</li>
          </ul>

          <p>
            by setting the values for \(\bar{r}\) and \(\omega\).
          </p>

          <p>
            If any of the parameters is not uniquely identifiable, state the most precise condition for the parameter.
          </p>

          <p>Explain your reasoning and include any plots that support your answers.</p>
        </div>
      </div>
    </body>
    """
    components.html(f"<!DOCTYPE html><html>{HTML_HEADER}{body}</html>", height=800, scrolling=False)

    with st.container():
        st.markdown("***You can hover on the plot to get time and magnitude measurements.***")
        col1, col2 = st.columns(2)
        with col1:
            magnitude = slider_with_number(
                r"Reference input magnitude $\bar{r}$",
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                default=1.0,
                key="q2_magnitude",
            )
        with col2:
            omega = slider_with_number(
                r"Reference input frequency $\omega$",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                default=2.0,
                key="q2_omega",
                format_str="%.1f",
            )

        time, y = compute_closed_loop_response(
            Q2_CONSTANTS,
            magnitude=magnitude,
            omega=omega,
            disturbance=0.0,
            x0=0.0,
        )
        fig = render_closed_loop_plot(time, y, height=440)
        st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOTLY_CONFIG)


def render_problem_3() -> None:
    st.header("Question 3: Stability by tuning the proportional gain (20 pts)")
    st.markdown(
        r"""
        Consider the feedback interconnection above where $A$, $B_{1}$, $B_{2}$, and $C$ are scalars.
        The initial condition of the system satisfies $x(0) = x_{0}$.
        """
    )

    body = r"""
    <body>
      <div class="demo-card">
        <div class="exercise">
        
          <h3>Part 1 (10 pts)</h3>
          <p>
            By hand, derive a condition on \(k_{p}\) that ensures that the closed-loop system is stable, in terms of:
          </p>
          <ul>
            <li>The state matrix (scalar) \(A\)</li>
            <li>The input matrix (scalar) \(B_{1}\) for the input \(u\)</li>
            <li>The input matrix (scalar) \(B_{2}\) for the input \(d\)</li>
            <li>The output matrix (scalar) \(C\)</li>
          </ul>

          <p>Explain your reasoning and include any drawings that support your answers.</p>
          
          <p style="background-color: #f9f9f9; padding: 10px; border-left: 4px solid #ccc; margin-top: 10px;">
            <strong>Hint:</strong> Derive the closed-loop dynamics as a first-order system 
            \(\dot{y} = \tilde{A} y + \tilde{B}_{1} r + \tilde{B}_{2} d\) 
            where \(\tilde{A}\), \(\tilde{B}_{1}\), and \(\tilde{B}_{2}\) depend on the parameters listed above.
          </p>

          <hr style="margin: 20px 0; opacity: 0.3;">

          <h3>Part 2 (10 pts)</h3>
          <p>
            The system parameters are given as 
          </p>
          <ul>
            <li>The state matrix (scalar) \(A\) is  1</li>
            <li>The input matrix (scalar) \(B_{1}\) for the input \(u\) is  5</li>
            <li>The input matrix (scalar) \(B_{2}\) for the input \(d\) is 0.2</li>
            <li>The output matrix (scalar) \(C\) is 1</li>
          </ul>

          <p>
            The below plot shows the complete response \(y(t)\) of the system under inputs \(r\) and \(d\). By tuning the parameters, find the value of \(k_{p}\) that makes the system marginally stable. 
          </p>

          <p>
            Next, find the set of \(k_{p}\) values that makes the system stable.
          </p>

          <p>
            Using the values for the system parameters, show that your answer matches the stability condition derived in Part 1.
          </p>

          <p>Explain your reasoning and include any plots that support your answers.</p>
        </div>
      </div>
    </body>
    """
    components.html(f"<!DOCTYPE html><html>{HTML_HEADER}{body}</html>", height=760, scrolling=False)

    with st.container():
        st.markdown("***You can hover on the plot to get time and magnitude measurements.***")
        col1, col2 = st.columns(2)
        with col1:
            magnitude = slider_with_number(
                r"Reference input magnitude $\bar{r}$",
                min_value=0.0,
                max_value=2.0,
                step=0.01,
                default=1.0,
                key="q3_magnitude",
            )
            disturbance = slider_with_number(
                r"Disturbance input magnitude $\bar{d}$",
                min_value=-1.5,
                max_value=1.5,
                step=0.01,
                default=0.5,
                key="q3_disturbance",
            )
        with col2:
            omega = slider_with_number(
                r"Reference input frequency $\omega$",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                default=2.0,
                key="q3_omega",
                format_str="%.1f",
            )
            x0 = slider_with_number(
                r"Initial state $x_{0}$",
                min_value=-2.0,
                max_value=2.0,
                step=0.01,
                default=-0.2,
                key="q3_x0",
            )
        kp = slider_with_number(
            r"Proportional gain $k_{p}$",
            min_value=0.0,
            max_value=5.0,
            step=0.01,
            default=Q3_CONSTANTS.kp,
            key="q3_kp",
        )

        time, y = compute_closed_loop_response(
            Q3_CONSTANTS,
            magnitude=magnitude,
            omega=omega,
            disturbance=disturbance,
            x0=x0,
            kp_override=kp,
        )
        fig = render_closed_loop_plot(time, y)
        st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOTLY_CONFIG)
        st.markdown(f"Selected proportional gain: **kₚ = {kp:.2f}**")


def render_problem_4() -> None:
    st.header("Question 4: Steady-state of a first-order system under sinusoidal inputs (20 pts)")

    body = r"""
    <body>
      <div class="demo-card">
        <div class="exercise">
        
          <h3>Part 1 (10 pts)</h3>
          <p>
            Note that the amplitude values ($y$-axes) are not on the same scale both plots and the values are not given for the ($y$-axes).
          </p>
          <p>
            Explain your reasoning and include any plots that support your answers.
          </p>
        </div>
        <ul>
            <li>
                <strong>Fact 1:</strong> The period of a signal in seconds is \(\frac{2\pi}{\omega}\) where \(\omega\) is the frequency in radians.
            </li>
            <li>
                <strong>Fact 2:</strong> Time lag \(T^{\text{lag}}\) between the input and output is equal to minus phase shift (lag) divided by the input frequency:
                \(T^{\text{lag}} = \frac{-\angle G(j\omega)}{\omega}\).
            </li>
        </ul>
        
      </div>
    </body>
    """
    components.html(f"<!DOCTYPE html><html>{HTML_HEADER}{body}</html>", height=260, scrolling=False)

    with st.container():
        st.markdown("***You can hover on the plot to get time measurements.***")
        omega = st.slider("Input frequency ω (rad/s)", min_value=2.0, max_value=5.0, value=2.0, step=0.01)
        time, x_signal, u_signal = compute_steady_state_response(
            STEADY_STATE,
            amplitude=1.0,
            omega=omega,
        )
        fig = render_state_input_plot(time, x_signal, u_signal)
        st.plotly_chart(fig, use_container_width=True, theme=None, config=PLOTLY_CONFIG)


def run_hw1() -> None:
    """Main entry point for the Homework 1 page."""
    inject_custom_css()
    st.title("Homework 1: Feedback Control Systems")
    st.write("Total points: 110 (= 40 + 30 + 20 + 20).")
    st.write("Submit your work to Canvas as a single pdf containing solutions to all problems.")
    st.write(
        "Solutions must be typed, NOT handwritten. You can use any text processors such as Latex, MS Word, Markdown."
    )
    st.divider()
    render_problem_1()
    st.divider()
    render_problem_2()
    st.divider()
    render_problem_3()
    st.divider()
    render_problem_4()


def main() -> None:
    st.set_page_config(page_title="ASE 370C Homework 1", page_icon="📈", layout="wide")
    if not require_password():
        st.stop()
    run_hw1()


if __name__ == "__main__":
    main()
