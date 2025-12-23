"""Streamlit port of the Linear Systems Homework 2 interactive demo."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import json
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
        "vector_line": "var(--plot-vector-line)",
    }


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


def simulate_linear_system(
    A: np.ndarray, x0: np.ndarray, t_max: float, dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate dx/dt = Ax using Forward Euler."""
    time = np.arange(0, t_max + dt, dt)
    x_traj = np.zeros((len(time), len(x0)))
    x = x0.copy()
    for i, t in enumerate(time):
        x_traj[i] = x
        dx = A @ x
        x = x + dx * dt
    return time, x_traj


def render_time_series_plot(time: np.ndarray, x_traj: np.ndarray) -> go.Figure:
    """Classic x1(t), x2(t) time series plot."""
    style = get_plot_style()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=x_traj[:, 0],
            name="x₁(t)",
            mode="lines",
            line=dict(width=3, color=style["state_line"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=time,
            y=x_traj[:, 1],
            name="x₂(t)",
            mode="lines",
            line=dict(width=3, color=style["input_line"]),
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
        ),
        yaxis=dict(
            title="State",
            gridcolor=style["grid_color"],
            zeroline=False,
        ),
        legend=dict(x=0.8, y=0.95),
        hovermode="x unified",
        height=400,
    )
    return fig


def render_phase_portrait(
    A: np.ndarray,
    trajectories: list[tuple[np.ndarray, np.ndarray]],
    x_range: tuple[float, float] = (-2, 2),
    y_range: tuple[float, float] = (-2, 2),
) -> go.Figure:
    """Phase portrait with trajectories and (conceptually) interactive vector field."""
    style = get_plot_style()
    fig = go.Figure()

    colors = ["#1d4ed8", "#f97316", "#10b981", "#8b5cf6"]
    for i, (x1, x2) in enumerate(trajectories):
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=x1,
                y=x2,
                mode="lines",
                line=dict(width=2, color=color),
                showlegend=False,
            )
        )
        # Add arrowhead
        if len(x1) > 10:
            idx = len(x1) // 2
            dx = x1[idx + 1] - x1[idx - 1]
            dy = x2[idx + 1] - x2[idx - 1]
            angle = np.arctan2(dy, dx) * 180 / np.pi
            fig.add_trace(
                go.Scatter(
                    x=[x1[idx]],
                    y=[x2[idx]],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=color,
                        symbol="triangle-up",
                        angle=90 - angle,
                    ),
                    showlegend=False,
                )
            )

    fig.update_layout(
        margin=dict(l=60, r=18, t=28, b=50),
        paper_bgcolor=style["paper_bgcolor"],
        plot_bgcolor=style["plot_bgcolor"],
        font=dict(family="Inter, sans-serif", color=style["font_color"]),
        xaxis=dict(
            title="x₁",
            gridcolor=style["grid_color"],
            zeroline=True,
            zerolinecolor="#888",
            range=list(x_range),
            fixedrange=True,
        ),
        yaxis=dict(
            title="x₂",
            gridcolor=style["grid_color"],
            zeroline=True,
            zerolinecolor="#888",
            range=list(y_range),
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True,
        ),
        height=500,
        hovermode="closest",
    )

    return fig


def render_problem_1():
    body = r"""
    <body>
      <div class="demo-card">
        <div class="exercise">
          <h2>Question 1</h2>
          <p>Consider the following matrices:</p>
          <div class="equations">
            \[
            A = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}, \qquad B = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}
            \]
          </div>
          <p><strong>a)</strong> Find the matrix exponential \(e^{At}\).</p>
          <p><strong>b)</strong> Find the matrix exponential \(e^{Bt}\).</p>
          <p><strong>c)</strong> Find the matrix exponential \(e^{Ct}\) where \(C = B - A\). What do you observe?</p>
        </div>
      </div>
    </body>
    """
    # Combine header + body
    components.html(f"<!DOCTYPE html><html>{HTML_HEADER}{body}</html>", height=350)


def render_problem_2():
    body = r"""
    <body>
      <div class="demo-card">
        <div class="exercise">
          <h2>Question 2</h2>
          <p>Consider the linear system \(\dot{x} = Ax\) where</p>
          <div class="equations">
            \[
            A = \begin{bmatrix} -2 & 1 & 0 \\ 0 & -1 & 2 \\ 0 & -1 & -1 \end{bmatrix}
            \]
          </div>
          <p>Determine whether the system is stable or unstable by computing the eigenvalues of \(A\). Show your work.</p>
        </div>
      </div>
    </body>
    """
    components.html(f"<!DOCTYPE html><html>{HTML_HEADER}{body}</html>", height=300)


def render_problem_3():
    body = r"""
    <body>
      <div class="demo-card">
        <div class="exercise">
          <h2>Question 3</h2>
          <p>An \(n\)-dimensional linear system \(\dot{x} = Ax\) has all eigenvalues equal to \(0\). Is the system stable or unstable? Justify your response with analytical mathematical calculations.</p>
        </div>
      </div>
    </body>
    """
    components.html(f"<!DOCTYPE html><html>{HTML_HEADER}{body}</html>", height=200)


def render_problem_4():
    # 1. Calculate Data in Python
    # Part A: A = [0 1; 1 0], x0 = [1, -1]
    A_a = np.array([[0.0, 1.0], [1.0, 0.0]])
    x0 = np.array([1.0, -1.0])
    t_a, traj_a = simulate_linear_system(A_a, x0, t_max=5.0, dt=0.01)
    
    # Part B: A = [-1 -5; 5 -1], x0 = [1, -1]
    A_b = np.array([[-1.0, -5.0], [5.0, -1.0]])
    t_b, traj_b = simulate_linear_system(A_b, x0, t_max=5.0, dt=0.01)

    # Prepare data payloads
    data_a = {"t": t_a.tolist(), "x1": traj_a[:, 0].tolist(), "x2": traj_a[:, 1].tolist()}
    data_b = {"t": t_b.tolist(), "x1": traj_b[:, 0].tolist(), "x2": traj_b[:, 1].tolist()}
    
    json_a = json.dumps(data_a)
    json_b = json.dumps(data_b)

    # 2. Define Body Content (using f-string for data injection)
    body = f"""
    <body>
      <style>
        .plot-container {{
            width: 100%;
            height: 350px;
            margin-bottom: 15px;
            border: 1px solid #f3f4f6;
            border-radius: 6px;
        }}
        hr {{ border: 0; border-top: 1px solid #e5e7eb; margin: 30px 0; }}
      </style>

      <div class="demo-card">
        <div class="exercise">
          <h2>Question 4</h2>
          <p>Consider a 2-dimensional linear system $\dot{{x}} = Ax$. The system is simulated from a particular initial condition, and the resulting state trajectory is shown below.</p>
        </div>

        <p><strong>a)</strong></p>
        <div id="plot4a" class="plot-container"></div>
        <div class="exercise">
          <p>Based on this response, is the system stable or unstable? Justify your answer. (Hint: what are the mathematical conditions for stability?)</p>
        </div>

        <hr>

        <p><strong>b)</strong></p>
        <div id="plot4b" class="plot-container"></div>
        <div class="exercise">
          <p>Based on this response, is the system stable or unstable? Justify your answer, and explain why your conclusion is the same or different from part (a).</p>
        </div>
      </div>

      <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
      <script>
        const commonLayout = {{
            margin: {{ l: 60, r: 20, t: 30, b: 50 }},
            font: {{ family: 'Inter, sans-serif', color: '#111827' }},
            xaxis: {{ title: 'Time (s)', gridcolor: 'rgba(99,102,241,0.15)', zeroline: false }},
            yaxis: {{ title: 'State', gridcolor: 'rgba(99,102,241,0.15)', zeroline: false }},
            legend: {{ x: 0.85, y: 1.0 }},
            hovermode: 'x unified',
            hoverlabel: {{ bgcolor: 'rgba(255,255,255,0.95)', font: {{ size: 12 }} }}
        }};

        // Render Plot A
        const dataA = {json_a};
        const tracesA = [
            {{ x: dataA.t, y: dataA.x1, name: 'x₁(t)', type: 'scatter', mode: 'lines', line: {{ width: 3, color: '#1d4ed8' }} }},
            {{ x: dataA.t, y: dataA.x2, name: 'x₂(t)', type: 'scatter', mode: 'lines', line: {{ width: 3, color: '#f97316' }} }}
        ];
        Plotly.newPlot('plot4a', tracesA, commonLayout, {{displayModeBar: false, responsive: true}});

        // Render Plot B
        const dataB = {json_b};
        const tracesB = [
            {{ x: dataB.t, y: dataB.x1, name: 'x₁(t)', type: 'scatter', mode: 'lines', line: {{ width: 3, color: '#1d4ed8' }} }},
            {{ x: dataB.t, y: dataB.x2, name: 'x₂(t)', type: 'scatter', mode: 'lines', line: {{ width: 3, color: '#f97316' }} }}
        ];
        Plotly.newPlot('plot4b', tracesB, commonLayout, {{displayModeBar: false, responsive: true}});
      </script>
    </body>
    """

    # Combine with shared header
    components.html(f"<!DOCTYPE html><html>{HTML_HEADER}{body}</html>", height=1100, scrolling=False)


def render_problem_5():
    # 1. Calculate Data in Python
    A = np.array([[2.0, 4.0], [-1.0, 2.0]])
    ics = [[0.1, 0.0], [-0.1, 0.0], [0.0, 0.1], [0.0, -0.1]]
    
    traces_data = []
    for ic in ics:
        _, traj = simulate_linear_system(A, np.array(ic), t_max=2.0, dt=0.005)
        mask = (np.abs(traj[:, 0]) <= 2.1) & (np.abs(traj[:, 1]) <= 2.1)
        traces_data.append({
            "x": traj[mask, 0].tolist(),
            "y": traj[mask, 1].tolist(),
            "mode": "lines",
            "line": {"color": "#3b82f6", "width": 2},
            "showlegend": False,
            "hoverinfo": "skip"
        })

    traces_json = json.dumps(traces_data)

    # 2. Define Body Content
    body = f"""
    <body>
      <style>
        #vectorDisplay {{
            text-align: center; 
            padding: 12px 16px; 
            background: linear-gradient(135deg, rgba(225, 29, 72, 0.08), rgba(244, 63, 94, 0.08)); 
            border: 1px solid rgba(225, 29, 72, 0.2); 
            border-radius: 10px; 
            margin-bottom: 20px; 
            font-family: 'Inter', sans-serif; 
            font-size: 1rem; 
            color: #be123c;
            min-height: 24px;
        }}
      </style>

      <div class="demo-card">
        <div class="exercise">
          <h2>Question 5</h2>
          <p>Consider a 2-dimensional linear system $\dot{{x}} = Ax$. The phase portrait of the system is shown below.</p>
        </div>

        <div id="plot5" style="width: 100%; height: 400px;"></div>

        <div id="vectorDisplay">
          <em>Hover over the plot to see the vector field</em>
        </div>

        <div class="exercise">
          <p><strong>a)</strong> Is the system stable or unstable? Justify your answer.</p>
          <p><strong>b)</strong> Identify the matrix $A$.</p>
        </div>
      </div>

      <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
      <script>
        const traces = {traces_json};

        const layout = {{
            margin: {{t: 20, l: 40, r: 40, b: 40}},
            xaxis: {{range: [-2, 2], title: 'x1'}},
            yaxis: {{range: [-2, 2], title: 'x2', scaleanchor: "x", scaleratio: 1}},
            showlegend: false,
            hovermode: false
        }};

        Plotly.newPlot('plot5', traces, layout, {{displayModeBar: false, responsive: true}});
        
        // Interactive Vector Listener
        const plotDiv = document.getElementById('plot5');
        const vectorDisplay = document.getElementById('vectorDisplay');
        
        plotDiv.addEventListener('mousemove', function(evt) {{
            const bb = plotDiv.getBoundingClientRect();
            const xaxis = plotDiv._fullLayout.xaxis;
            const yaxis = plotDiv._fullLayout.yaxis;
            
            const mouseX = evt.clientX - bb.left;
            const mouseY = evt.clientY - bb.top;
            
            const x1 = xaxis.p2d(mouseX - plotDiv._fullLayout.margin.l);
            const x2 = yaxis.p2d(mouseY - plotDiv._fullLayout.margin.t);
            
            if (x1 >= -2 && x1 <= 2 && x2 >= -2 && x2 <= 2) {{
                // A = [2 4; -1 2]
                const ax = 2*x1 + 4*x2;
                const ay = -x1 + 2*x2;
                
                vectorDisplay.innerHTML = `<strong>x</strong> = (${{x1.toFixed(2)}}, ${{x2.toFixed(2)}})`;
                
                const arrowScale = 0.15;
                const tipX = x1 + ax * arrowScale;
                const tipY = x2 + ay * arrowScale;
                
                const newAnnotations = [
                    {{
                        x: tipX, y: tipY, ax: x1, ay: x2, xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                        showarrow: true, arrowhead: 2, arrowsize: 1.5, arrowwidth: 2.5, arrowcolor: '#e11d48'
                    }},
                    {{
                        x: tipX, y: tipY, xref: 'x', yref: 'y',
                        text: `(${{ax.toFixed(2)}}, ${{ay.toFixed(2)}})`,
                        showarrow: true, arrowhead: 0, ax: 0, ay: -20,
                        font: {{ size: 12, color: '#e11d48', family: 'Inter, sans-serif' }},
                        bgcolor: 'rgba(255,255,255,0.8)', borderpad: 2
                    }}
                ];
                
                Plotly.relayout('plot5', {{ annotations: newAnnotations }});
            }}
        }});
        
        plotDiv.addEventListener('mouseleave', function() {{
            vectorDisplay.innerHTML = '<em>Hover over the plot to see the vector field</em>';
            Plotly.relayout('plot5', {{ annotations: [] }});
        }});
      </script>
    </body>
    """

    components.html(f"<!DOCTYPE html><html>{HTML_HEADER}{body}</html>", height=750, scrolling=False)


def render_problem_6():
    # 1. Define Body Content
    # We use double braces {{ }} for JS/CSS because this is an f-string.
    body = f"""
    <body>
      <style>
        #vectorDisplay6 {{
            text-align: center; 
            padding: 12px 16px; 
            background: linear-gradient(135deg, rgba(225, 29, 72, 0.08), rgba(244, 63, 94, 0.08)); 
            border: 1px solid rgba(225, 29, 72, 0.2); 
            border-radius: 10px; 
            margin-bottom: 20px; 
            font-family: 'Inter', sans-serif; 
            font-size: 1rem; 
            color: #be123c;
            min-height: 24px;
        }}
      </style>

      <div class="demo-card">
        <div class="exercise">
          <h2>Question 6</h2>
          <p>Consider a 2-dimensional linear system $\dot{{x}} = SQ^{{-1}}AQS^{{-1}}x$, where $Q$ and $S$ are invertible matrices. The vector field of the system is shown below.</p>
        </div>

        <div id="plot6" style="width: 100%; height: 400px;"></div>

        <div id="vectorDisplay6">
          <em>Hover over the plot to see the vector field</em>
        </div>

        <div class="exercise">
          <p><strong>a)</strong> Is $\displaystyle\lim_{{t \\to \infty}} e^{{At}}$ finite? Is $\displaystyle\lim_{{t \\to \infty}} \det(e^{{At}})$ finite?</p>
          <p><strong>b)</strong> Suppose that $S = Q$. Show that $\displaystyle\lim_{{t \\to \infty}} e_2^\\top e^{{At}} e_2 = 0$, where $e_2 = \\begin{{bmatrix}} 0 \\\\ 1 \\end{{bmatrix}}$.</p>
        </div>
      </div>

      <script src="https://cdn.plot.ly/plotly-2.24.1.min.js"></script>
      
      <script>
        // Question 6 - interactive vector field only (no trajectories)
        
        const traces = [{{
          x: [0], y: [0],
          type: 'scatter',
          mode: 'markers',
          marker: {{ size: 8, color: '#888' }},
          showlegend: false
        }}];
        
        const layout = {{
          margin: {{ l: 40, r: 40, t: 20, b: 40 }},
          xaxis: {{ title: 'x1', range: [-2, 2] }},
          yaxis: {{ title: 'x2', range: [-2, 2], scaleanchor: 'x', scaleratio: 1 }},
          hovermode: false
        }};
        
        Plotly.newPlot('plot6', traces, layout, {{ displayModeBar: false, responsive: true }});
        
        // Interactive Vector Listener
        const plotDiv6 = document.getElementById('plot6');
        const vectorDisplay6 = document.getElementById('vectorDisplay6');
        
        plotDiv6.addEventListener('mousemove', function(evt) {{
          const bb = plotDiv6.getBoundingClientRect();
          const xaxis = plotDiv6._fullLayout.xaxis;
          const yaxis = plotDiv6._fullLayout.yaxis;
          
          const mouseX = evt.clientX - bb.left;
          const mouseY = evt.clientY - bb.top;
          
          const x1 = xaxis.p2d(mouseX - plotDiv6._fullLayout.margin.l);
          const x2 = yaxis.p2d(mouseY - plotDiv6._fullLayout.margin.t);
          
          if (x1 >= -2 && x1 <= 2 && x2 >= -2 && x2 <= 2) {{
            // A = [1 0; 0 -1] (Saddle)
            const ax = x1;
            const ay = -x2;
            
            vectorDisplay6.innerHTML = `<strong>x</strong> = (${{x1.toFixed(2)}}, ${{x2.toFixed(2)}})`;
            
            const arrowScale = 0.5;
            const tipX = x1 + ax * arrowScale;
            const tipY = x2 + ay * arrowScale;
            
            const newAnnotations = [
              {{
                x: tipX, y: tipY, ax: x1, ay: x2, xref: 'x', yref: 'y', axref: 'x', ayref: 'y',
                showarrow: true, arrowhead: 2, arrowsize: 1.5, arrowwidth: 2.5, arrowcolor: '#e11d48'
              }},
              {{
                x: tipX + 0.15, y: tipY + 0.15, xref: 'x', yref: 'y',
                text: `(${{ax.toFixed(2)}}, ${{ay.toFixed(2)}})`,
                showarrow: false,
                font: {{ size: 12, color: '#e11d48', family: 'Inter, sans-serif' }},
                bgcolor: 'rgba(255,255,255,0.85)', borderpad: 3
              }}
            ];
            
            Plotly.relayout('plot6', {{ annotations: newAnnotations }});
          }}
        }});
        
        plotDiv6.addEventListener('mouseleave', function() {{
          vectorDisplay6.innerHTML = '<em>Hover over the plot to see the vector field</em>';
          Plotly.relayout('plot6', {{ annotations: [] }});
        }});
      </script>
    </body>
    """

    components.html(f"<!DOCTYPE html><html>{HTML_HEADER}{body}</html>", height=750, scrolling=False)
    
def run_hw2() -> None:
    """Main entry point for the Homework 2 page."""
    inject_custom_css()
    st.title("Homework 2: Linear Systems")
    st.divider()

    render_problem_1()
    st.divider()
    render_problem_2()
    st.divider()
    render_problem_3()
    st.divider()
    render_problem_4()
    st.divider()
    render_problem_5()
    st.divider()
    render_problem_6()


def main() -> None:
    st.set_page_config(page_title="ASE 370C Homework 2", page_icon="📈", layout="wide")
    if not require_password():
        st.stop()
    run_hw2()


if __name__ == "__main__":
    main()
