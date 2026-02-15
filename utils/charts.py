import plotly.graph_objects as go
from utils.style import get_theme_config
from utils.data import get_value_col


def style_fig(fig, y_axis_title=None):
    """Apply standard styling to a Plotly figure."""
    t = get_theme_config()

    fig.update_layout(
        template=t["PLOT_TEMPLATE"],
        font_family="Inter",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=9, color=t["TEXT_SUB"]),
        ),
    )
    fig.update_xaxes(
        showgrid=False,
        linecolor=t["BORDER_COLOR"],
        tickfont=dict(color=t["TEXT_SUB"]),
    )

    y_title = y_axis_title or ""
    fig.update_yaxes(
        showgrid=True,
        gridcolor=t["GRID_COLOR"],
        linecolor=t["BORDER_COLOR"],
        tickfont=dict(color=t["TEXT_SUB"]),
        title_text=y_title,
    )
    return fig


def plot_scarcity_line(df, name, color, dash, broadcast_x=None):
    """Return a Scatter trace for a scarcity-price series (or None)."""
    if df is None or df.empty:
        return None

    vcol = get_value_col(df)
    y = df[vcol]
    x = df["Date"]

    if broadcast_x is not None and len(df) == 1:
        val = y.iloc[0]
        y = [val] * len(broadcast_x)
        x = broadcast_x

    return go.Scatter(
        x=x, y=y,
        name=name, mode="lines",
        line=dict(color=color, dash=dash, width=2),
    )
