import plotly.graph_objects as go
import numpy as np

# Create iteration range
iterations = np.linspace(0, 60000, 1000)


# Define weight curves
def rgb_recon_weight(iters):
    """RGB Recon Loss: constant at 1.0 until 20K, then decreases to 0.3 by 60K"""
    weights = np.ones_like(iters)
    mask = iters > 20000
    weights[mask] = 1.0 - (iters[mask] - 20000) / (60000 - 20000) * 0.7
    return weights


def sds_weight(iters):
    """SDS Loss: ramps to 1.0 by iter 100, stays 1.0 until 50K, then decreases"""
    weights = np.zeros_like(iters)
    mask1 = iters <= 100
    weights[mask1] = iters[mask1] / 100
    mask2 = (iters > 100) & (iters <= 50000)
    weights[mask2] = 1.0
    mask3 = iters > 50000
    weights[mask3] = 1.0 - (iters[mask3] - 50000) / (60000 - 50000) * 0.4
    return weights


def contact_temporal_weight(iters):
    """Contact/Temporal Loss: 0 until 20K, ramps 20K-20.5K, stays 1.0 from 25K-50K, decays to 0.5 by 60K"""
    weights = np.zeros_like(iters)
    mask1 = (iters > 20000) & (iters <= 20500)
    weights[mask1] = (iters[mask1] - 20000) / 500
    mask2 = (iters > 20500) & (iters <= 50000)
    weights[mask2] = 1.0
    mask3 = iters > 50000
    weights[mask3] = 1.0 - (iters[mask3] - 50000) / (60000 - 50000) * 0.5
    return weights


def total_loss_trajectory(iters):
    """Total loss trajectory with three distinct regions"""
    loss = np.zeros_like(iters)
    # Region 1: Steep initial decrease 0-5K
    mask1 = iters <= 5000
    loss[mask1] = 1.0 - (iters[mask1] / 5000) * 0.6  # Drop from 1.0 to 0.4
    # Continue decreasing 5K-20K at slower rate
    mask2 = (iters > 5000) & (iters <= 20000)
    loss[mask2] = 0.4 - ((iters[mask2] - 5000) / 15000) * 0.15  # Drop from 0.4 to 0.25
    # Slight increase at 20K due to contact/temporal activation
    mask3 = (iters > 20000) & (iters <= 22000)
    loss[mask3] = 0.25 + ((iters[mask3] - 20000) / 2000) * 0.08  # Increase to 0.33
    # Resume decreasing through decay phase
    mask4 = iters > 22000
    loss[mask4] = 0.33 - ((iters[mask4] - 22000) / 38000) * 0.18  # Drop to 0.15
    return loss


# Calculate weights and loss
rgb_weights = rgb_recon_weight(iterations)
sds_weights = sds_weight(iterations)
contact_weights = contact_temporal_weight(iterations)
temporal_weights = contact_temporal_weight(iterations)
total_loss = total_loss_trajectory(iterations)

# ========================================
# FIGURE 1: Timeline Diagram
# ========================================
# ===== ROW 1: Timeline Diagram =====
fig1 = go.Figure()
# Phase 1 block (0-20K) - Phase 3 SDS only
fig1.add_trace(go.Scatter(
    x=[0, 20000, 20000, 0, 0],
    y=[0.3, 0.3, 0.9, 0.9, 0.3],
    fill='toself',
    fillcolor='rgba(31,184,205,0.25)',
    line=dict(color='#1FB8CD', width=3),
    name='',
    showlegend=False,
    hoverinfo='skip'
))

# Phase 2 block (20K-60K) - All phases
fig1.add_trace(go.Scatter(
    x=[20000, 60000, 60000, 20000, 20000],
    y=[0.3, 0.3, 0.9, 0.9, 0.3],
    fill='toself',
    fillcolor='rgba(46,139,87,0.25)',
    line=dict(color='#2E8B57', width=3),
    name='',
    showlegend=False,
    hoverinfo='skip'
))

# Add detailed annotations for timeline blocks
fig1.add_annotation(
    x=10000, y=0.75,
    text="<b>EPOCH 0-9</b>",
    showarrow=False,
    font=dict(size=11, color="#1FB8CD", family="Arial Black"),
    
)

fig1.add_annotation(
    x=10000, y=0.60,
    text="0-20K iterations",
    showarrow=False,
    font=dict(size=9, color="#1FB8CD"),
    
)

fig1.add_annotation(
    x=10000, y=0.45,
    text="Phase 3: SDS Loss Only",
    showarrow=False,
    font=dict(size=9, color="#1FB8CD"),
    
)

fig1.add_annotation(
    x=40000, y=0.75,
    text="<b>EPOCH 10-29</b>",
    showarrow=False,
    font=dict(size=11, color="#2E8B57", family="Arial Black"),
    
)

fig1.add_annotation(
    x=40000, y=0.60,
    text="20K-60K iterations",
    showarrow=False,
    font=dict(size=9, color="#2E8B57"),
    
)

fig1.add_annotation(
    x=40000, y=0.45,
    text="Phase 3: SDS + Phase 4: Contact + Phase 5: Temporal",
    showarrow=False,
    font=dict(size=9, color="#2E8B57"),
    
)

# Add vertical line at 20K boundary
fig1.add_shape(
    type="line",
    x0=20000, x1=20000,
    y0=0.3, y1=0.9,
    line=dict(color="black", width=3),
    
)
fig1.update_xaxes(
    tickmode='array',
    tickvals=[0, 10000, 20000, 30000, 40000, 50000, 60000],
    ticktext=['0k', '10k', '20k', '30k', '40k', '50k', '60k'],
    title_text='Iterations (k)',
    showgrid=True
)
fig1.update_yaxes(title_text='', range=[0.2, 1.0], showticklabels=False, showgrid=False)

fig1.update_layout(
    title="Training Phase Timeline",
    height=400,  # Control individual height
    hovermode='x unified'
)

# ========================================
# FIGURE 2: Loss Weight Scheduling
# ========================================
fig2 = go.Figure()
# ===== ROW 2: Weight Scheduling Curves =====
fig2.add_trace(go.Scatter(
    x=iterations, y=rgb_weights,
    name='RGB Recon Loss',
    line=dict(color='#9B59B6', width=3),
    mode='lines',
    legendgroup='weights'
))

fig2.add_trace(go.Scatter(
    x=iterations, y=sds_weights,
    name='SDS Loss (Phase 3)',
    line=dict(color='#1FB8CD', width=3),
    mode='lines',
    legendgroup='weights'
))

fig2.add_trace(go.Scatter(
    x=iterations, y=contact_weights,
    name='Contact Loss (Phase 4)',
    line=dict(color='#DB4545', width=3),
    mode='lines',
    legendgroup='weights'
))

fig2.add_trace(go.Scatter(
    x=iterations, y=temporal_weights,
    name='Temporal Loss (Phase 5)',
    line=dict(color='#2E8B57', width=3),
    mode='lines',
    legendgroup='weights'
))

# Add shaded ramp-up region (20K-20.5K) - GRAY
fig2.add_vrect(
    x0=20000, x1=20500,
    fillcolor="gray", opacity=0.25,
    layer="below", line_width=0,
    
)

fig2.add_annotation(
    # x=20250, y=1.12,
    x=15000, y=0.9,
    text="500-iter ramp-up",
    showarrow=False,
    font=dict(size=10, color="gray"),
    
)

# Add shaded decay region (50K-60K) - LIGHTER GRAY
fig2.add_vrect(
    x0=50000, x1=60000,
    fillcolor="lightgray", opacity=0.2,
    layer="below", line_width=0,
    
)

# Add vertical dashed lines at ALL critical points including 20.5K
critical_points = [20000, 20500, 50000, 60000]
point_labels = ['Phase\nTransition\n20K', 'Ramp\nComplete\n20.5K', 'Decay\nStart\n50K', 'End\n60K']

# Give each label its own placement to avoid overlap
label_y = {20000: 1.22, 20500: 1.10, 50000: 1.22, 60000: 1.22}
label_xshift = {20000: -18, 20500: +18, 50000: 0, 60000: 0}

for i, point in enumerate(critical_points):
    fig2.add_vline(
        x=point,
        line_dash="dash",
        line_color="rgba(100,100,100,0.5)",
        line_width=1.5
    )

    fig2.add_annotation(
        x=point, y=label_y[point],
        xshift=label_xshift[point],
        text=point_labels[i],
        showarrow=False,
        font=dict(size=10, color="gray")
    )


# Add annotations for weight curves - key insights
fig2.add_annotation(
    x=20250, y=0.45,
    text="Ramp-up<br>prevents shock",
    showarrow=True,
    ax=50, ay=-35,
    font=dict(size=9, color="#DB4545"),
    arrowcolor="#DB4545",
    bgcolor="rgba(255,255,255,0.8)",
    
)

fig2.add_annotation(
    x=35000, y=0.98,
    text="Balanced objectives",
    showarrow=True,
    ax=0, ay=25,
    font=dict(size=9, color="#2E8B57"),
    arrowcolor="#2E8B57",
    bgcolor="rgba(255,255,255,0.8)",
    
)

fig2.add_annotation(
    x=55000, y=0.60,
    text="Gradual decay<br>prevents oscillation",
    showarrow=True,
    ax=-50, ay=-25,
    font=dict(size=9, color="#9B59B6"),
    arrowcolor="#9B59B6",
    bgcolor="rgba(255,255,255,0.8)",
    
)
fig2.update_xaxes(
    tickmode='array',
    tickvals=[0, 10000, 20000, 30000, 40000, 50000, 60000],
    ticktext=['0k', '10k', '20k', '30k', '40k', '50k', '60k'],
    title_text='Iterations (k)',
    showgrid=True
)
fig2.update_yaxes(title_text='Loss Weight', range=[-0.05, 1.30], showgrid=True)

fig2.update_layout(
    title="Loss Weight Scheduling",
    height=500,
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    hovermode='x unified'
)

# ========================================
# FIGURE 3: Total Loss Trajectory
# ========================================
fig3 = go.Figure()
# ===== ROW 3: Loss Trajectory =====
fig3.add_trace(go.Scatter(
    x=iterations, y=total_loss,
    name='Total Loss Trajectory',
    line=dict(color='#5D878F', width=4),
    mode='lines',
    showlegend=False
))

# Add vertical lines at phase boundaries
fig3.add_vline(
    x=20000,
    line_dash="solid",
    line_color="rgba(0,0,0,0.6)",
    line_width=2,
    
)

fig3.add_vline(
    x=50000,
    line_dash="solid",
    line_color="rgba(0,0,0,0.6)",
    line_width=2,
    
)

# Add shaded regions for loss trajectory - three distinct regions
fig3.add_vrect(
    x0=0, x1=20000,
    fillcolor="rgba(31,184,205,0.12)",
    layer="below", line_width=0,
    
)

fig3.add_vrect(
    x0=20000, x1=50000,
    fillcolor="rgba(46,139,87,0.12)",
    layer="below", line_width=0,
    
)

fig3.add_vrect(
    x0=50000, x1=60000,
    fillcolor="rgba(155,89,182,0.12)",
    layer="below", line_width=0,
    
)

# Add region labels with clear boundaries
fig3.add_annotation(
    x=10000, y=0.92,
    text="<b>Region 1: Geometric Foundation (SDS only)</b>",
    showarrow=False,
    font=dict(size=10, color="#1FB8CD"),
    bgcolor="rgba(255,255,255,0.7)",
    borderpad=3,
    
)

fig3.add_annotation(
    x=35000, y=0.92,
    text="<b>Region 2: Refinement (All phases active)</b>",
    showarrow=False,
    font=dict(size=10, color="#2E8B57"),
    bgcolor="rgba(255,255,255,0.7)",
    borderpad=3,
    
)

fig3.add_annotation(
    x=55000, y=0.92,
    text="<b>Region 3: Fine-tuning<br>(Balanced decay)</b>",
    showarrow=False,
    font=dict(size=10, color="#9B59B6"),
    bgcolor="rgba(255,255,255,0.7)",
    borderpad=3,
    
)

# Add "initial shock" annotation at the spike
fig3.add_annotation(
    x=21000, y=0.33,
    text="Initial shock from<br>new objectives",
    showarrow=True,
    ax=60, ay=-50,
    font=dict(size=9, color="#DB4545"),
    arrowcolor="#DB4545",
    bgcolor="rgba(255,255,255,0.9)",
    bordercolor="#DB4545",
    borderwidth=1,
    
)

# Update axes
fig3.update_xaxes(
    tickmode='array',
    tickvals=[0, 10000, 20000, 30000, 40000, 50000, 60000],
    ticktext=['0k', '10k', '20k', '30k', '40k', '50k', '60k'],
    title_text='Iterations (k)',  # Proper title here
    showgrid=True
)
fig3.update_yaxes(title_text='Norm. Loss', range=[0, 1.0], showgrid=True)

fig3.update_layout(
    title="Total Loss Trajectory",
    height=500,
    hovermode='x unified'
)

# Save the chart
fig1.write_image('./diagram/timeline_diagram.png')
fig2.write_image('./diagram/weight_scheduling.png')
fig3.write_image('./diagram/loss_trajectory.png')
# fig.write_image('./diagram/training_visualization.svg', format='svg')
