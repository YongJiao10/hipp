#!/usr/bin/env python3
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.patches import Arc, PathPatch
from matplotlib.path import Path
import seaborn as sns
from .network_colors import load_network_hex_colors

def FC_vec2mat(x: np.ndarray, n: int = 100, upper: bool = False, include_diag: bool = False, dtype=np.float32) -> np.ndarray:
    """
    Convert flat correlation vector to a symmetric matrix.
    Supports 1D, 2D (p x vectors), and 3D (p x q x vectors) inputs.
    """
    # If the input is already a square matrix, return it directly
    if x.ndim == 2 and x.shape[0] == x.shape[1] == n:
        return x.astype(dtype)
    if x.ndim == 3 and x.shape[1] == x.shape[2] == n:
        return x.astype(dtype)

    k = 0 if include_diag else (1 if upper else -1)

    if upper:
        indices = np.triu_indices(n, k=k)
    else:
        indices = np.tril_indices(n, k=k)
    
    if x.ndim == 1:
        fc = np.zeros((n, n))
        fc[indices] = x
        fc.T[indices] = x
        if not include_diag:
            np.fill_diagonal(fc, 1.0)
    elif x.ndim == 2:
        p, _ = x.shape
        fc = np.zeros((p, n, n))
        fc[:, indices[0], indices[1]] = x
        fc[:, indices[1], indices[0]] = x
        if not include_diag:
            for i in range(p):
                np.fill_diagonal(fc[i], 1.0)
    elif x.ndim == 3:
        p, q, _ = x.shape
        fc = np.zeros((p, q, n, n))
        fc[:, :, indices[0], indices[1]] = x
        fc[:, :, indices[1], indices[0]] = x
        if not include_diag:
            for i in range(p):
                for j in range(q):
                    np.fill_diagonal(fc[i, j], 1.0)
    else:
        fc = x
    
    return fc.astype(dtype)

def FC_mat2vec(fc: np.ndarray, upper: bool = False, include_diag: bool = False, dtype=np.float32) -> np.ndarray:
    """
    Convert symmetric matrix back to flat correlation vector.
    """
    n = fc.shape[-1]
    k = 0 if include_diag else (1 if upper else -1)
    
    if upper:
        indices = np.triu_indices(n, k=k)
    else:
        indices = np.tril_indices(n, k=k)
    
    if fc.ndim == 2:
        x = fc[indices]
    elif fc.ndim == 3:
        x = fc[:, indices[0], indices[1]]
    elif fc.ndim == 4:
        x = fc[:, :, indices[0], indices[1]]
    else:
        raise ValueError("Input array must have 2, 3, or 4 dimensions")
    
    return x.astype(dtype)

def rearrange_FC_by_df(mat: np.ndarray, df: pd.DataFrame, net_type: str = '8net') -> tuple[np.ndarray, list[int], list[str]]:
    """
    Rearrange correlation matrix rows and columns based on Kong2022 network order.
    Groups left and right parts of each network together.
    Unifies the ROI ordering between 8net and 17net by always sorting at the 17net level.
    """
    df = df.copy()
    df['orig_idx'] = range(len(df))
    
    # Always sort by 17net network order to keep ROI order identical between 8net and 17net
    family17_order = [
        "VisualA", "VisualB", "VisualC", "SomMotA", "SomMotB",
        "DorsAttnA", "DorsAttnB", "SalVenAttnA", "SalVenAttnB",
        "ContA", "ContB", "ContC", "Aud", "Language",
        "DefaultA", "DefaultB", "DefaultC"
    ]
    
    left_indices = []
    right_indices = []
    
    for net in family17_order:
        lh_df = df[(df['family17'] == net) & (df['hemisphere'].isin(['L', 'LH']))]
        lh_df = lh_df.sort_values(by='parcel_id')
        
        rh_df = df[(df['family17'] == net) & (df['hemisphere'].isin(['R', 'RH']))]
        rh_df = rh_df.sort_values(by='parcel_id')
        
        if len(lh_df) > 0 or len(rh_df) > 0:
            left_indices.append(lh_df['orig_idx'].tolist())
            right_indices.append(rh_df['orig_idx'].tolist())
            
    new_order = []
    for left, right in zip(left_indices, right_indices):
        new_order.extend(left)
        new_order.extend(right)
        
    df_sorted = df.iloc[new_order].reset_index(drop=True)
    
    if mat.ndim == 3:
        new_mat = mat[:, new_order, :][:, :, new_order]
    else:
        new_mat = mat[np.ix_(new_order, new_order)]
        
    # Calculate boundaries (pos) and network labels based on the target net_type
    if net_type == '8net':
        net_col = 'canonical8'
    else:
        net_col = 'family17'
        
    pos = []
    current_net = None
    active_networks = []
    for idx, row in df_sorted.iterrows():
        net_name = row[net_col]
        if current_net is None:
            current_net = net_name
            active_networks.append(net_name)
        elif net_name != current_net:
            pos.append(idx)
            current_net = net_name
            active_networks.append(net_name)
            
    return new_mat, pos, active_networks

def rearrange_FC(mat: np.ndarray, df: pd.DataFrame | None = None, net_type: str = '8net') -> tuple[np.ndarray, list[int], list[str]]:
    """
    Rearrange a functional connectivity (FC) matrix so that
    the left and right parts of each network are grouped together.
    """
    if df is None:
        if mat.shape[-1] == 100:
            # Original 100-ROI Yeo 7 rearrangement
            pos = [9, 15, 23, 30, 33, 37, 50, 58, 66, 73, 78, 80, 89]
            left_indices = []
            right_indices = []
            for i in range(7):
                if i == 0:
                    left_indices.append(list(range(0, pos[i])))
                else:
                    left_indices.append(list(range(pos[i-1], pos[i])))
            for i in range(7, 14):
                if i == 13:
                    right_indices.append(list(range(pos[i-1], 100)))
                else:
                    right_indices.append(list(range(pos[i-1], pos[i])))
            new_order = []
            new_pos = []
            current_index = 0
            for left, right in zip(left_indices, right_indices):
                new_order.extend(left)
                new_order.extend(right)
                current_index += len(left) + len(right)
                new_pos.append(current_index)
            new_pos.pop()
            if mat.ndim == 3:
                new_mat = mat[:, new_order, :][:, :, new_order]
            else:
                new_mat = mat[np.ix_(new_order, new_order)]
            net_names = ["Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Limbic", "Control", "Default"]
            return new_mat, new_pos, net_names
        else:
            raise ValueError("df DataFrame is required for matrices of size other than 100.")
            
    return rearrange_FC_by_df(mat, df, net_type)

def draw_fc_heatmap_on_ax(ax, cax, mat, pos, net_names, net_colors, mask, normalize=False, pos_value=False,
                          dark_mode=True, scale=0.6, cb_title="Fisher's $z$"):
    """
    Core drawing logic for the functional connectivity heatmap on a specified axis.
    """
    gridcolor = 'white' if dark_mode else 'black'

    if dark_mode:
        colors = [
            '#00ffff', '#0081ff','#0000ff', '#4b0082', 
            '#000000', '#7f0000', '#FF0000', '#FF8C00', '#FFFF00'
        ]
        cmap = ListedColormap(colors)
    else:
        cmap = 'bwr'

    vmin, vmax = np.nanmin(mat[~mask]), np.nanmax(mat[~mask])
    if normalize:
        vmin, vmax = (0, 1) if pos_value else (-1, 1)

    if vmin == vmax:
        if pos_value:
            vmin = 0
            vmax = 1
        else:
            vmin = -1
            vmax = 1

    if not dark_mode:
        ax.plot([0, len(mat)], [0, len(mat)], color=gridcolor, linewidth=scale)
        ax.axvline(x=0, color=gridcolor, linewidth=scale)
        ax.axhline(y=len(mat), color=gridcolor, linewidth=scale+1)
    else:
        ax.plot([0, len(mat)], [0, len(mat)], color=gridcolor, linewidth=scale*0.5)
        ax.axvline(x=0, color=gridcolor, linewidth=scale*0.5)
        ax.axhline(y=len(mat), color=gridcolor, linewidth=scale*0.5)
    
    for p in pos:
        ax.plot([0, p], [p, p], color=gridcolor, linewidth=scale)
        ax.plot([p, p], [p, len(mat)], color=gridcolor, linewidth=scale)
    
    if pos_value:
        norm = Normalize(vmin=max(0, vmin), vmax=vmax)
    else:
        v_limit = max(abs(vmin), abs(vmax))
        if v_limit <= 1e-5:
            v_limit = 1.0
        norm = TwoSlopeNorm(vmin=-v_limit, vcenter=0, vmax=v_limit)

    sns.heatmap(mat, mask=mask, fmt=".2f", cmap=cmap, norm=norm,
                cbar=True, square=True, linewidths=0, linecolor=gridcolor,
                cbar_kws={'shrink': 0.8, 'pad': 0.05}, ax=ax, cbar_ax=cax)

    cbar = ax.collections[0].colorbar
    if cbar is not None:
        cbar.ax.tick_params(labelsize=14*scale, colors='white' if dark_mode else 'black')
        if cb_title:
            cbar.ax.set_title(cb_title, fontsize=16*scale, pad=10, color='white' if dark_mode else 'black')

    ax.set_xticks([])
    ax.set_yticks([])

    # Network labels positioning
    tick_pos = pos + [len(mat)]
    fontsize = 20 * scale
    middle_pos = [(1 + tick_pos[k] + (tick_pos[k-1] if k > 0 else 0)) / 2 for k in range(len(tick_pos))]

    abbrev_map = {
        "Default": "DMN",
        "Visual": "VN",
        "Somatomotor": "SMN",
        "DorsalAttention": "DAN",
        "VentralAttention": "VAN",
        "Control": "CON",
        "Auditory": "AUD",
        "Language": "LAN",
        "Limbic": "LIM",
        "DefaultA": "DMN-A", "DefaultB": "DMN-B", "DefaultC": "DMN-C",
        "ContA": "CON-A", "ContB": "CON-B", "ContC": "CON-C",
        "SalVenAttnA": "VAN-A", "SalVenAttnB": "VAN-B",
        "DorsAttnA": "DAN-A", "DorsAttnB": "DAN-B",
        "SomMotA": "SMN-A", "SomMotB": "SMN-B",
        "VisualA": "VN-A", "VisualB": "VN-B", "VisualC": "VN-C",
        "Aud": "AUD", "Language": "LAN"
    }

    for k, p in enumerate(middle_pos):
        color = net_colors[k % len(net_colors)]
        net_label = net_names[k]
        net_label_display = abbrev_map.get(net_label, net_label)
        
        # Bottom labels (rotated 45 degrees, matching user's original layout)
        y_offset = len(mat) + 1
        ax.text(p, y_offset, net_label_display, 
               ha='center', va='top', color=color, 
               fontsize=fontsize, rotation=45, weight='bold')
        
        # Left labels (horizontal)
        ax.text(-1, p, net_label_display, 
               ha='right', va='center', color=color, 
               fontsize=fontsize, weight='bold')

def draw_chord_on_ax(ax, cax, conn_matrix, max_line_width=12, min_line_width=0.5, 
                    colormap='coolwarm', nodewidth=3, colorbar_pos='bottom',
                    max_strength=None, shadow=False, node_names=None, colors=None,
                    tick_color='white'):
    """
    Core drawing logic for the chord diagram on a specified axis.
    """
    n_nodes = conn_matrix.shape[0]
    
    abbrev_map = {
        "Default": "DMN",
        "Visual": "VN",
        "Somatomotor": "SMN",
        "DorsalAttention": "DAN",
        "VentralAttention": "VAN",
        "Control": "CON",
        "Auditory": "AUD",
        "Language": "LAN",
        "Limbic": "LIM",
        "DefaultA": "DMN-A", "DefaultB": "DMN-B", "DefaultC": "DMN-C",
        "ContA": "CON-A", "ContB": "CON-B", "ContC": "CON-C",
        "SalVenAttnA": "VAN-A", "SalVenAttnB": "VAN-B",
        "DorsAttnA": "DAN-A", "DorsAttnB": "DAN-B",
        "SomMotA": "SMN-A", "SomMotB": "SMN-B",
        "VisualA": "VN-A", "VisualB": "VN-B", "VisualC": "VN-C",
        "Aud": "AUD", "Language": "LAN"
    }
    
    if node_names is None:
        node_names = [str(i) for i in range(n_nodes)]
    else:
        node_names = [abbrev_map.get(name, name) for name in node_names]
        
    if colors is None:
        hex_colors = load_network_hex_colors()
        abbrev_to_full = {
            "DMN": "Default", "VN": "Visual", "SMN": "Somatomotor",
            "DAN": "DorsalAttention", "VAN": "VentralAttention",
            "CON": "Control", "AUD": "Auditory", "LAN": "Language",
            "DMN-A": "DefaultA", "DMN-B": "DefaultB", "DMN-C": "DefaultC",
            "CON-A": "ContA", "CON-B": "ContB", "CON-C": "ContC",
            "VAN-A": "SalVenAttnA", "VAN-B": "SalVenAttnB",
            "DAN-A": "DorsAttnA", "DAN-B": "DorsAttnB",
            "SMN-A": "SomMotA", "SMN-B": "SomMotB",
            "VN-A": "VisualA", "VN-B": "VisualB", "VN-C": "VisualC"
        }
        colors = []
        for name in node_names:
            full_name = abbrev_to_full.get(name, name)
            colors.append(hex_colors.get(full_name, "#808080"))
            
    ax.set_aspect('equal')
    
    angles = np.linspace(0, 2*np.pi, n_nodes, endpoint=False) + np.pi/2
    pos = {node: (np.cos(angle), np.sin(angle)) 
           for node, angle in zip(node_names, angles)}
    
    if isinstance(colormap, (list, tuple)):
        cmap_full = LinearSegmentedColormap.from_list('custom_listed_cmap', list(colormap))
    else:
        cmap_full = plt.colormaps[colormap]

    data_min = np.nanmin(conn_matrix)
    data_max = np.nanmax(conn_matrix)
    max_abs_value = max(abs(data_min), abs(data_max))
    if max_abs_value <= 1e-5:
        max_abs_value = 1.0

    if data_min < 0 and data_max > 0:
        cmap = cmap_full
        norm = plt.Normalize(-max_abs_value, max_abs_value)
    elif data_min >= 0:
        cmap = LinearSegmentedColormap.from_list("trunc_pos", cmap_full(np.linspace(0.5, 1, 256)))
        norm = plt.Normalize(0, max_abs_value)
    else:
        cmap = LinearSegmentedColormap.from_list("trunc_neg", cmap_full(np.linspace(0, 0.5, 256)))
        norm = plt.Normalize(-max_abs_value, 0)
    
    max_strength = np.nanmax(np.abs(conn_matrix)) if max_strength is None else max_strength
    if max_strength is None or max_strength <= 1e-5:
        max_strength = 1.0
        
    def get_line_width(strength):
        return min_line_width + (max_line_width - min_line_width) * abs(strength) / max_strength
    
    arc_width = 360 / n_nodes - (5 if n_nodes <= 8 else 4)
    radius = 1.0
    
    # Scale fontsize based on number of nodes (17 networks vs 8 networks)
    fontsize = 28 if n_nodes <= 8 else 13
    
    for i, (node, (x, y)) in enumerate(pos.items()):
        node_angle = np.degrees(angles[i])
        start_angle = node_angle - arc_width/2
        end_angle = node_angle + arc_width/2
        
        self_strength = conn_matrix[i, i]
        outer_x, outer_y = np.cos(angles[i]), np.sin(angles[i])
        if self_strength != 0 and np.isfinite(self_strength):
            ax.add_patch(Arc((outer_x, outer_y), width=0.4, height=0.4,
                            angle=node_angle, theta1=-90, theta2=90,
                            color=cmap(norm(self_strength)),
                            linewidth=get_line_width(self_strength)))
        if shadow:
            ax.add_patch(Arc((0.015, -0.015),
                            width=2*radius, height=2*radius,
                            angle=0, theta1=start_angle, theta2=end_angle,
                            color='gray', alpha=0.3, linewidth=nodewidth))
        
        arc = Arc((0, 0), width=2*radius, height=2*radius,
                 angle=0, theta1=start_angle, theta2=end_angle,
                 color=colors[i], linewidth=nodewidth)
        ax.add_patch(arc)
        
        text_radius = 0.88
        text_x = text_radius * np.cos(angles[i])
        text_y = text_radius * np.sin(angles[i])
        text_angle = node_angle - 90 + (180 if text_y < 0 else 0)
        ax.text(text_x, text_y, node, color=colors[i],
                rotation=text_angle, ha='center', va='center',
                rotation_mode='anchor', fontsize=fontsize, fontweight='bold')
    
    link_radius = 0.8
    for i in range(n_nodes):
        for j in range(i+1, n_nodes):
            val = conn_matrix[i, j]
            if val != 0 and np.isfinite(val):
                start_pos = (link_radius*np.cos(angles[i]), link_radius*np.sin(angles[i]))
                end_pos = (link_radius*np.cos(angles[j]), link_radius*np.sin(angles[j]))
                
                mid_x, mid_y = (start_pos[0] + end_pos[0])/2, (start_pos[1] + end_pos[1])/2
                control_point = (mid_x * 0.3, mid_y * 0.3)
                
                path = Path([start_pos, control_point, end_pos],
                          [Path.MOVETO, Path.CURVE3, Path.CURVE3])
                ax.add_patch(PathPatch(path, facecolor='none',
                                     edgecolor=cmap(norm(val)),
                                     linewidth=get_line_width(val)))
    
    scale = 1.25
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    ax.axis('off')
    
    # Add colorbar if cax is provided
    if cax is not None:
        orientation = 'vertical' if colorbar_pos in ['right', 'left'] else 'horizontal'
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

        if data_min < 0 and data_max > 0:
            ticks = [-max_abs_value, 0, max_abs_value]
            tick_labels = [f'{-max_abs_value:.1f}', '0', f'{max_abs_value:.1f}']
        elif data_min >= 0:
            ticks = [0, max_abs_value]
            tick_labels = ['0', f'{max_abs_value:.1f}']
        else:
            ticks = [-max_abs_value, 0]
            tick_labels = [f'{-max_abs_value:.1f}', '0']

        cbar = plt.colorbar(sm, cax=cax, orientation=orientation, ticks=ticks)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(labelsize=14 if n_nodes <= 8 else 10, length=0, colors=tick_color)

        if orientation == 'horizontal':
            cbar.ax.set_xticklabels(tick_labels, fontsize=14 if n_nodes <= 8 else 10)
        else:
            cbar.ax.set_yticklabels(tick_labels, fontsize=14 if n_nodes <= 8 else 10)

def plot_fc_heatmap(fc_values, parcels_df=None, net_type='8net', pos_value=False, normalize=False, dark_mode=True,
                    threshold=None, cmap=None, title='', scale=.6, cb_title="Fisher's $z$",
                    horizontal_layout=False, save_path=None):
    """
    Plot functional connectivity heatmap with flexible layout orientation.
    """
    # Handle input type wrapping
    if isinstance(fc_values, np.ndarray) and fc_values.ndim == 2:
        fc_values_list = [[fc_values]]
    elif isinstance(fc_values, list) and isinstance(fc_values[0], np.ndarray) and fc_values[0].ndim == 2:
        fc_values_list = [fc_values]
    else:
        fc_values_list = fc_values

    processed_mats_list = []
    pos_list = []
    net_names_list = []
    
    for fc_list in fc_values_list:
        processed_mats = []
        pos_sublist = []
        net_names_sublist = []
        for fc in fc_list:
            if threshold is not None:
                fc = apply_threshold(fc, threshold)
            n_rois = parcels_df.shape[0] if parcels_df is not None else 100
            mat = FC_vec2mat(fc, n=n_rois)
            mat, pos, net_names = rearrange_FC(mat, df=parcels_df, net_type=net_type)
            if normalize:
                mat = normalize_matrix(mat, pos_value)
            processed_mats.append(mat)
            pos_sublist.append(pos)
            net_names_sublist.append(net_names)
        processed_mats_list.append(processed_mats)
        pos_list.append(pos_sublist)
        net_names_list.append(net_names_sublist)

    masks = [np.triu(np.ones_like(mat, dtype=bool), 1) for mat in processed_mats_list[0]]

    n_datasets = len(processed_mats_list)
    n_timepoints = len(processed_mats_list[0])

    if horizontal_layout:
        fig, axes = plt.subplots(n_datasets, n_timepoints,
                               figsize=(9 * n_timepoints * scale, 7.5 * n_datasets * scale))
    else:
        fig, axes = plt.subplots(n_timepoints, n_datasets,
                               figsize=(8.5 * n_datasets * scale, 7.5 * n_timepoints * scale))

    if n_datasets == 1 and n_timepoints == 1:
        axes = np.array([[axes]])
    elif n_datasets == 1 or n_timepoints == 1:
        axes = np.array(axes).reshape((n_datasets, n_timepoints) if horizontal_layout else (n_timepoints, n_datasets))

    # Apply dark mode background if requested
    if dark_mode:
        fig.patch.set_facecolor('#111111')
    else:
        fig.patch.set_facecolor('white')

    # Load standard colors for labels
    hex_colors = load_network_hex_colors()
    color_map_names = {
        "DefaultA": "DefaultA", "DefaultB": "DefaultB", "DefaultC": "DefaultC",
        "ContA": "ContA", "ContB": "ContB", "ContC": "ContC",
        "SalVenAttnA": "SalVenAttnA", "SalVenAttnB": "SalVenAttnB",
        "DorsAttnA": "DorsAttnA", "DorsAttnB": "DorsAttnB",
        "SomMotA": "SomMotA", "SomMotB": "SomMotB",
        "VisualA": "VisualA", "VisualB": "VisualB", "VisualC": "VisualC",
        "Aud": "Aud", "Language": "Language",
        "Default": "Default", "Visual": "Visual", "Somatomotor": "Somatomotor",
        "DorsalAttention": "DorsalAttention", "VentralAttention": "VentralAttention",
        "Control": "Control", "Auditory": "Auditory"
    }

    for i in range(n_timepoints):
        for j in range(n_datasets):
            if horizontal_layout:
                ax = axes[j, i]
            else:
                ax = axes[i, j]

            ax.set_facecolor('none')

            mat = processed_mats_list[j][i]
            pos = pos_list[j][i]
            net_names = net_names_list[j][i]
            net_colors = [hex_colors.get(color_map_names.get(name, name), "#808080") for name in net_names]

            draw_fc_heatmap_on_ax(
                ax=ax,
                cax=None,  # let Seaborn create colorbar on axes
                mat=mat,
                pos=pos,
                net_names=net_names,
                net_colors=net_colors,
                mask=masks[i],
                normalize=normalize,
                pos_value=pos_value,
                dark_mode=dark_mode,
                scale=scale,
                cb_title=cb_title
            )

    if title:
        fig.suptitle(title, fontsize=24*scale, y=0.98, color='white' if dark_mode else 'black')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.patch.set_alpha(0)
        plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

def plot_chord(conn_matrix: np.ndarray, max_line_width: float = 12, min_line_width: float = 0.5, 
               colormap: str | list | tuple = 'coolwarm', nodewidth: float = 3,
               colorbar_pos: str = 'bottom', colorbar_width: float = 0.018, colorbar_length: float = 0.3, colorbar_pad: float = 0.45, 
               max_strength: float | None = None, shadow: bool = False, node_names: list[str] | None = None, 
               colors: list[str] | None = None, save_path: str | None = None):
    """
    Plot functional connectivity chord diagram.
    """
    fig = plt.figure(figsize=(8, 7))
    start, end = 0, 0.9
    center = (end - start) / 2
    ax = fig.add_axes([start, start, end, end])
    
    # Position colorbar inside the figure
    cax_pos = {'right': [center + colorbar_pad, center - colorbar_length/2, colorbar_width, colorbar_length],
               'bottom': [center - colorbar_length/2, center - colorbar_pad, colorbar_length, colorbar_width]}
    cax = fig.add_axes(cax_pos[colorbar_pos])
    
    draw_chord_on_ax(
        ax=ax,
        cax=cax,
        conn_matrix=conn_matrix,
        max_line_width=max_line_width,
        min_line_width=min_line_width,
        colormap=colormap,
        nodewidth=nodewidth,
        colorbar_pos=colorbar_pos,
        max_strength=max_strength,
        shadow=shadow,
        node_names=node_names,
        colors=colors,
        tick_color='black'
    )
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.patch.set_alpha(0)
        plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

def compute_network_fc(fc_matrix: np.ndarray, df: pd.DataFrame, net_type: str = '8net') -> tuple[np.ndarray, list[str]]:
    """
    Compute network-level functional connectivity by averaging ROI-to-ROI connection strengths.
    
    Args:
        fc_matrix: 2D array of shape (400, 400) representing ROI-to-ROI Fisher-z FC.
        df: DataFrame containing parcel definitions.
        net_type: '8net' or '17net'
        
    Returns:
        net_fc: 2D array of shape (n_networks, n_networks)
        net_names: list of network names in the display order
    """
    if net_type == '8net':
        net_col = 'canonical8'
        net_order = [
            "Visual", "Somatomotor", "DorsalAttention", "VentralAttention",
            "Control", "Auditory", "Language", "Default"
        ]
    else:
        net_col = 'family17'
        net_order = [
            "VisualA", "VisualB", "VisualC", "SomMotA", "SomMotB",
            "DorsAttnA", "DorsAttnB", "SalVenAttnA", "SalVenAttnB",
            "ContA", "ContB", "ContC", "Aud", "Language",
            "DefaultA", "DefaultB", "DefaultC"
        ]
        
    n_nets = len(net_order)
    net_fc = np.zeros((n_nets, n_nets))
    
    # Get indices for each network
    net_indices = {}
    for net in net_order:
        idx = df[df[net_col] == net].index.tolist()
        net_indices[net] = idx
        
    for i, net_i in enumerate(net_order):
        for j, net_j in enumerate(net_order):
            idx_i = net_indices[net_i]
            idx_j = net_indices[net_j]
            if len(idx_i) > 0 and len(idx_j) > 0:
                # Use nanmean to ignore diagonal elements (which are set to NaN)
                net_fc[i, j] = np.nanmean(fc_matrix[np.ix_(idx_i, idx_j)])
            else:
                net_fc[i, j] = 0.0
                
    return net_fc, net_order

def plot_combined_fc(fc_matrix: np.ndarray, parcels_df: pd.DataFrame, net_type: str = '8net', 
                     dark_mode: bool = True, title: str = '', scale: float = 0.7, 
                     cb_title: str = "Fisher's $z$", save_path: str | None = None):
    """
    Create a combined figure placing the Chord diagram in the blank upper-right triangle of the Heatmap.
    Renders the Chord diagram at its native size in-memory first to preserve perfect font size and
    line proportions, then overlays it as a scaled image.
    """
    import io
    
    # 1. Render the Chord Diagram normally to a BytesIO buffer at its original perfect proportions
    fig_chord = plt.figure(figsize=(8, 7))
    start, end = 0, 0.9
    center = (end - start) / 2
    ax_c = fig_chord.add_axes([start, start, end, end])
    
    colorbar_pos = 'bottom'
    colorbar_pad = 0.46
    colorbar_length = 0.3
    colorbar_width = 0.018
    cax_pos = {'right': [center + colorbar_pad, center - colorbar_length/2, colorbar_width, colorbar_length],
               'bottom': [center - colorbar_length/2, center - colorbar_pad, colorbar_length, colorbar_width]}
    cax_c = fig_chord.add_axes(cax_pos[colorbar_pos])
    
    # Compute network FC
    net_fc, net_names_order = compute_network_fc(fc_matrix, parcels_df, net_type=net_type)
    
    # Use larger font size for chord labels as it will be scaled down as a whole image
    draw_chord_on_ax(
        ax=ax_c,
        cax=cax_c,
        conn_matrix=net_fc,
        node_names=net_names_order,
        colormap='coolwarm',
        colorbar_pos='bottom',
        shadow=False,
        tick_color='white' if dark_mode else 'black'
    )
    
    # Save fig_chord to buffer with transparent background
    buf = io.BytesIO()
    fig_chord.patch.set_alpha(0)
    fig_chord.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor=fig_chord.get_facecolor(), edgecolor='none')
    plt.close(fig_chord)
    buf.seek(0)
    
    # Load the rendered image back
    chord_img = plt.imread(buf, format='png')
    
    # 2. Draw Heatmap exactly as in plot_fc_heatmap
    n_datasets = 1
    n_timepoints = 1
    
    fig, axes = plt.subplots(n_timepoints, n_datasets,
                           figsize=(8.5 * n_datasets * scale, 7.5 * n_timepoints * scale))
    
    if n_datasets == 1 and n_timepoints == 1:
        ax = axes
    else:
        ax = axes[0, 0]
        
    if dark_mode:
        fig.patch.set_facecolor('#111111')
    else:
        fig.patch.set_facecolor('white')
    ax.set_facecolor('none')
        
    n_rois = parcels_df.shape[0]
    mat = FC_vec2mat(fc_matrix, n=n_rois)
    mat_rearranged, pos, net_names_rearranged = rearrange_FC(mat, df=parcels_df, net_type=net_type)
    
    # Define mask (upper triangle masked out)
    mask = np.triu(np.ones_like(mat_rearranged, dtype=bool), 1)
    
    # Get network colors
    hex_colors = load_network_hex_colors()
    color_map_names = {
        "DefaultA": "DefaultA", "DefaultB": "DefaultB", "DefaultC": "DefaultC",
        "ContA": "ContA", "ContB": "ContB", "ContC": "ContC",
        "SalVenAttnA": "SalVenAttnA", "SalVenAttnB": "SalVenAttnB",
        "DorsAttnA": "DorsAttnA", "DorsAttnB": "DorsAttnB",
        "SomMotA": "SomMotA", "SomMotB": "SomMotB",
        "VisualA": "VisualA", "VisualB": "VisualB", "VisualC": "VisualC",
        "Aud": "Aud", "Language": "Language",
        "Default": "Default", "Visual": "Visual", "Somatomotor": "Somatomotor",
        "DorsalAttention": "DorsalAttention", "VentralAttention": "VentralAttention",
        "Control": "Control", "Auditory": "Auditory"
    }
    net_colors = [hex_colors.get(color_map_names.get(name, name), "#808080") for name in net_names_rearranged]
    
    draw_fc_heatmap_on_ax(
        ax=ax,
        cax=None,  # let Seaborn create colorbar on axes
        mat=mat_rearranged,
        pos=pos,
        net_names=net_names_rearranged,
        net_colors=net_colors,
        mask=mask,
        normalize=False,
        pos_value=False,
        dark_mode=dark_mode,
        scale=scale,
        cb_title=cb_title
    )
    
    if title:
        fig.suptitle(title, fontsize=24*scale, y=0.98, color='white' if dark_mode else 'black')
        
    # 3. Get the actual position of the square heatmap axis to place the Chord diagram
    fig.canvas.draw()  # Force layout calculation
    bbox = ax.get_position(original=False)  # Get active position after aspect ratio
    
    # Define ax_chord inset relative to the heatmap bbox (so it stays perfectly in the upper-right corner)
    # Using mathematically optimal coordinates to maximize chord size without crossing the diagonal:
    left = bbox.x0 + bbox.width * 0.425
    bottom = bbox.y0 + bbox.height * 0.415
    width = bbox.width * 0.59
    height = bbox.height * 0.59
    
    ax_chord = fig.add_axes([left, bottom, width, height])
    ax_chord.imshow(chord_img)
    ax_chord.axis('off')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.patch.set_alpha(0)
        plt.savefig(f'{save_path}.png', bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
