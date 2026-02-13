import plotly.graph_objects as go
from utils.style import get_theme_config
from utils.data import get_chart_val_col

def style_fig(fig, y_axis_title=None):
    theme = get_theme_config()
    
    fig.update_layout(
        template=theme['PLOT_TEMPLATE'],
        font_family="Inter",
        title_text="", # FORCE EMPTY TITLE
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color=theme['TEXT_SUB'])
        )
    )
    fig.update_xaxes(showgrid=False, linecolor=theme['BORDER_COLOR'], tickfont=dict(color=theme['TEXT_SUB']))
    
    # Check if a title was provided
    if y_axis_title:
        fig.update_layout(yaxis_title=y_axis_title)
        fig.update_yaxes(showgrid=True, gridcolor=theme['GRID_COLOR'], linecolor=theme['BORDER_COLOR'], tickfont=dict(color=theme['TEXT_SUB']), title_text=y_axis_title)
    else:
        fig.update_layout(yaxis_title="")
        fig.update_yaxes(showgrid=True, gridcolor=theme['GRID_COLOR'], linecolor=theme['BORDER_COLOR'], tickfont=dict(color=theme['TEXT_SUB']), title_text="")
        
    return fig

def plot_scarcity_line(df_scarcity, name, color, dash_style, broadcast_x=None):
     if df_scarcity is not None and not df_scarcity.empty:
        val_col = get_chart_val_col(df_scarcity)
        
        y_vals = df_scarcity[val_col]
        x_vals = df_scarcity['Date']
        
        # Broadcasting logic: If 1D view (1 data point) vs Hourly X-axis
        if broadcast_x is not None and len(df_scarcity) == 1:
            val = y_vals.iloc[0]
            y_vals = [val] * len(broadcast_x)
            x_vals = broadcast_x
            
        return go.Scatter(
             x=x_vals, y=y_vals,
             name=name, mode='lines', line=dict(color=color, dash=dash_style, width=2)
        )
     return None
