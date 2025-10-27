def plot_gaussian_voronoi_watershed(original_image, cls_bg, markers,
                                    labels=None, class_names=None,
                                    output_path=None): 
    """Plot figures for debug..."""
    """Plot figures for debug with 2x3 layout showing different processing stages.
    
    Layout:
        Row 1: Original Image | Pure Voronoi | Pure Watershed
        Row 2: Original+Voronoi Boundaries | Watershed Mask+Original | (Legend)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import time
    from scipy import ndimage
    from matplotlib.colors import ListedColormap
    import os
    from io import BytesIO
    from PIL import Image
    
    # Check if the system supports font rendering
    try:
        # Try to set font
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        # Test rendering with a simple figure
        fig = plt.figure()
        plt.text(0.5, 0.5, 'test')
        buf = BytesIO()
        fig.savefig(buf)
        plt.close(fig)
    except:
        pass
    
    # Create 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=300)
    
    # Convert tensor to numpy array
    if original_image.dim() == 3:
        orig_img = original_image.permute(1, 2, 0).detach().cpu().numpy()
    else:
        orig_img = original_image.detach().cpu().numpy()
    
    # Normalize image to [0,1]
    orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min())
    
    cls_bg_np = cls_bg.detach().cpu().numpy()
    markers_np = markers.detach().cpu().numpy()
    
    # Prepare color map
    colors = plt.cm.tab20(np.linspace(0, 1, 20)) if labels is not None else None
    
    # Title mapping
    titles = {
        'original': 'Original Image',
        'voronoi': 'Pure Voronoi',
        'watershed': 'Pure Watershed',
        'original_edges': 'Original+Red Edges',
        'watershed_mask': 'Watershed Mask+Original'
    }
    
    # === Row 1 Col 1: Original Image ===
    ax = axes[0, 0]
    ax.imshow(orig_img, cmap='gray' if len(orig_img.shape) == 2 else None)
    ax.set_title(titles['original'], fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # === Row 1 Col 2: Pure Voronoi ===
    ax = axes[0, 1]
    # Create colored display for Voronoi
    voronoi_colored = np.zeros((*cls_bg_np.shape, 3))
    unique_regions = np.unique(cls_bg_np)
    
    for i, region_id in enumerate(unique_regions):
        if region_id == 16:  # Background regions in white
            voronoi_colored[cls_bg_np == region_id] = [1, 1, 1]
        elif region_id == -1:  # Internal regions in black
            voronoi_colored[cls_bg_np == region_id] = [0, 0, 0]
    ax.imshow(voronoi_colored)
    for region_id in unique_regions:
        if (region_id >= 0):  # Ignore background and boundaries
            region_mask = cls_bg_np == region_id
            if region_mask.any():
                # Calculate region center
                y, x = np.where(region_mask)
                center_x = int(np.mean(x))
                center_y = int(np.mean(y))
                ax.text(center_x, center_y, str(region_id), color='black', fontsize=8, ha='center', va='center')
    ax.set_title(titles['voronoi'], fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # === Row 1 Col 3: Pure Watershed ===
    ax = axes[0, 2]
    # Create colored display for watershed results
    watershed_colored = np.zeros((*markers_np.shape, 3))
    unique_markers = np.unique(markers_np)
    
    for i, marker_id in enumerate(unique_markers):
        if marker_id <= 0:  # Background and boundaries in black
            watershed_colored[markers_np == marker_id] = [0, 0, 0]
        else:
            color_idx = (marker_id - 1) % len(colors) if colors is not None else (marker_id - 1) % 10
            marker_color = colors[color_idx][:3] if colors is not None else plt.cm.tab10(color_idx)[:3]
            watershed_colored[markers_np == marker_id] = marker_color
    
    ax.imshow(watershed_colored)
    
    for marker_id in unique_markers:
        if marker_id > 0:
            region_mask = markers_np == marker_id
            if region_mask.any():
                y, x = np.where(region_mask)
                center_x = int(np.mean(x))
                center_y = int(np.mean(y))
                ax.text(center_x, center_y, str(marker_id), color='red', 
                       fontsize=16, ha='center', va='center')
    
    ax.set_title(titles['watershed'], fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # === Row 2 Col 1: Original Image with Voronoi Boundaries ===
    ax = axes[1, 0]
    ax.imshow(orig_img, cmap='gray' if len(orig_img.shape) == 2 else None)

    # Find edges of black regions
    black_edges = ndimage.binary_dilation(cls_bg_np == -1) & ~(cls_bg_np == -1)

    # Create overlay image with red edges
    overlay = np.zeros_like(orig_img)
    if len(orig_img.shape) == 3:  # Color image
        overlay = orig_img.copy()
        overlay[black_edges, 0] = 1  # Red channel
        overlay[black_edges, 1:] = 0  # Green and blue channels
    else:  # Grayscale image
        overlay = np.stack([orig_img] * 3, axis=-1)
        overlay[black_edges, 0] = 1  # Red channel
        overlay[black_edges, 1:] = 0  # Green and blue channels

    ax.imshow(overlay)
    ax.set_title(titles['original_edges'], fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # === Row 2 Col 2: Watershed Mask over Original Image ===
    ax = axes[1, 1]
    ax.imshow(orig_img, cmap='gray' if len(orig_img.shape) == 2 else None)
    
    # Create transparent overlay for watershed mask
    if labels is not None and colors is not None:
        labels_np = labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)
        
        # Create and overlay transparent mask for each class
        for idx, label in enumerate(unique_labels):
            # Find instance indices for this class
            class_instances = np.where(labels_np == label)[0]
            
            # Create total mask for this class
            class_mask = np.zeros_like(markers_np, dtype=bool)
            for instance_idx in class_instances:
                # Instance labels in markers start from 1
                instance_mask = (markers_np == instance_idx + 1)
                class_mask |= instance_mask
            
            if class_mask.any():
                # Create colored transparent mask
                color_rgba = colors[idx % len(colors)]
                
                # Create RGBA image for transparent overlay
                overlay = np.zeros((*class_mask.shape, 4))
                overlay[class_mask] = [*color_rgba[:3], 0.6]  # 60% transparency
                
                # Display overlay
                ax.imshow(overlay, alpha=0.8)
    
    ax.set_title(titles['watershed_mask'], fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # === Row 2 Col 3: For Legend ===
    ax = axes[1, 2]
    ax.axis('off')  # Hide axes
    
    # Add legend in this blank area
    if labels is not None and colors is not None:
        labels_np = labels.detach().cpu().numpy()
        unique_labels = np.unique(labels_np)
        legend_patches = []
        
        for idx, label in enumerate(unique_labels):
            if class_names is not None and label < len(class_names):
                class_name = class_names[label]
            else:
                class_name = f'Class {label}'
            
            color_rgba = colors[idx % len(colors)]
            legend_patches.append(
                mpatches.Patch(color=color_rgba[:3], 
                             label=class_name, alpha=0.7)
            )
        
        # Place legend in center of Row 2 Col 3 subplot
        if legend_patches:
            ax.legend(handles=legend_patches, 
                     loc='center',
                     fontsize=10, 
                     fancybox=True, 
                     shadow=True)
    
    # Adjust layout
    plt.tight_layout(pad=2.0)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img = Image.open(buf)
        img.save(output_path, format='PNG', optimize=True, compress_level=9)
        buf.close()
    else:
        os.makedirs('debug', exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        output_path_default = f'debug/{timestamp}-Gaussian-Voronoi-2x3.png'
        plt.savefig(output_path_default, bbox_inches='tight', facecolor='white')
        
    plt.close()


def visualize_loss_calculation(image, mask_tensor, mu_j, V_j, xy_centered, xy_rotated, max_x, max_y, L_j, L_target, instance_loss, j, class_id):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.patches import Ellipse
    import time
    from matplotlib.colors import LinearSegmentedColormap
    
    os.makedirs('vis_loss', exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), dpi=150)
    
    class_names = {
        0: 'ship',
        1: 'aircraft',
        2: 'car',
        3: 'tank',
        4: 'bridge',
        5: 'harbor'
    }
    class_name = class_names.get(class_id, f'Class {class_id}')
    
    fig.suptitle(f'Loss Calculation Process for {class_name} (Instance {j+1})', fontsize=16, fontweight='bold')
    

    img_np = image.permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    ax = axes[0, 0]
    ax.imshow(img_np)
    ax.set_title("1. Original Image", fontsize=12)
    
    mu_np = mu_j.detach().cpu().numpy()
    ax.plot(mu_np[0], mu_np[1], 'ro', markersize=8)
    ax.text(mu_np[0]+5, mu_np[1]+5, f"Center ({mu_np[0]:.1f}, {mu_np[1]:.1f})", color='white',  
            bbox=dict(facecolor='red', alpha=0.7))
    ax.axis('off')
    
    ax = axes[0, 1]
    mask_np = mask_tensor.detach().cpu().numpy()
    
    ax.imshow(img_np)
    mask_overlay = np.zeros((*mask_np.shape, 4))
    mask_overlay[mask_np] = [1, 0, 0, 0.5]
    ax.imshow(mask_overlay)
    ax.set_title("2. Extracted Mask", fontsize=12)
    ax.axis('off')
    
    ax = axes[0, 2]
    xy_centered_np = xy_centered.detach().cpu().numpy()
    
    dist_from_center = np.sqrt(xy_centered_np[:, 0]**2 + xy_centered_np[:, 1]**2)
    normalized_dist = dist_from_center / max(1e-7, dist_from_center.max())
    
    scatter = ax.scatter(xy_centered_np[:, 0], xy_centered_np[:, 1],  
                         s=2, c=normalized_dist, cmap='viridis', alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax.set_title("3. Centered Coordinates", fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    ax = axes[1, 0]
    xy_rotated_np = xy_rotated.detach().cpu().numpy()
    
    ax.scatter(xy_rotated_np[:, 0], xy_rotated_np[:, 1],  
               s=2, c=normalized_dist, cmap='viridis', alpha=0.7)
    ax.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)
    
    max_x_np = max_x.detach().cpu().numpy()
    max_y_np = max_y.detach().cpu().numpy()
    ax.plot([-max_x_np, max_x_np], [0, 0], 'r-', linewidth=2)
    ax.plot([0, 0], [-max_y_np, max_y_np], 'r-', linewidth=2)
    
    ax.text(max_x_np, 2, f"max_x = {max_x_np:.2f}", color='red', ha='right')
    ax.text(2, max_y_np, f"max_y = {max_y_np:.2f}", color='red', va='top')
    
    ellipse = Ellipse((0, 0), 2*max_x_np, 2*max_y_np,  
                      facecolor='none', edgecolor='orange', linewidth=2, alpha=0.7)
    ax.add_patch(ellipse)
    
    ax.set_title("4. Rotated Coordinates", fontsize=12)
    ax.set_xlabel("Principal Axis 1")
    ax.set_ylabel("Principal Axis 2")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    
    L_j_np = L_j.detach().cpu().numpy()
    L_target_np = L_target.detach().cpu().numpy()
    
    table_data = [
        ["Predicted", f"{L_j_np[0]:.2f}", f"{L_j_np[1]:.2f}"],
        ["Target", f"{L_target_np[0]:.2f}", f"{L_target_np[1]:.2f}"],
        ["Ratio", f"{L_target_np[0]/max(1e-7, L_j_np[0]):.2f}", f"{L_target_np[1]/max(1e-7, L_j_np[1]):.2f}"]
    ]
    
    table = ax.table(cellText=table_data,  
                     colLabels=["Eigenvalues", "λ1 (along V1)", "λ2 (along V2)"],
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    ax.text(0.5, 0.1, f"GWD Loss: {instance_loss.item():.6f}",  
            ha='center', va='center', fontsize=12,  
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    ax.set_title("5. Loss Calculation", fontsize=12)
    ax.axis('off')
    
    ax = axes[1, 2]
    
    V_j_np = V_j.detach().cpu().numpy()
    
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='-', alpha=0.5)
    
    max_eigen_sqrt = max(np.sqrt(L_j_np[1]), np.sqrt(L_target_np.max())) if L_target_np.size > 0 else np.sqrt(L_j_np[1])
    arrow_scale = max(max_eigen_sqrt, 1.0)

    ax.arrow(0, 0, arrow_scale * V_j_np[0, 0], arrow_scale * V_j_np[1, 0],  
             head_width=0.05*arrow_scale, head_length=0.1*arrow_scale, fc='cyan', ec='cyan', label='V1 (corresponds to λ1)')
    ax.arrow(0, 0, arrow_scale * V_j_np[0, 1], arrow_scale * V_j_np[1, 1],  
             head_width=0.05*arrow_scale, head_length=0.1*arrow_scale, fc='magenta', ec='magenta', label='V2 (corresponds to λ2)')

    width_pred = 2 * np.sqrt(L_j_np[1])
    height_pred = 2 * np.sqrt(L_j_np[0])
    angle_pred = np.rad2deg(np.arctan2(V_j_np[1, 1], V_j_np[0, 1]))
    eigen_ellipse = Ellipse((0, 0), width=width_pred, height=height_pred, angle=angle_pred,
                            facecolor='none', edgecolor='green', linewidth=2,
                            linestyle='--', label='Predicted')
    ax.add_patch(eigen_ellipse)

    if L_target_np[0] >= L_target_np[1]:
        width_target = 2 * np.sqrt(L_target_np[0])
        height_target = 2 * np.sqrt(L_target_np[1])
        angle_target = np.rad2deg(np.arctan2(V_j_np[1, 0], V_j_np[0, 0]))
    else:
        width_target = 2 * np.sqrt(L_target_np[1])
        height_target = 2 * np.sqrt(L_target_np[0])
        angle_target = np.rad2deg(np.arctan2(V_j_np[1, 1], V_j_np[0, 1]))

    target_ellipse = Ellipse((0, 0), width=width_target, height=height_target, angle=angle_target,
                             facecolor='none', edgecolor='orange', linewidth=2,
                             label='Target')
    ax.add_patch(target_ellipse)
    
    ax.set_title("6. Eigenvectors & Ellipses", fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    max_dim = 1.1 * max(width_pred, height_pred, width_target, height_target, 2 * arrow_scale)
    ax.set_xlim(-max_dim / 2, max_dim / 2)
    ax.set_ylim(-max_dim / 2, max_dim / 2)
    
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    plt.savefig(f'vis_loss/loss_calc_{timestamp}_{class_name}_inst{j+1}.png',  
                bbox_inches='tight', facecolor='white')
    plt.close()


# Add debug visualization helper function
def save_debug_visualization(image, masks, scores, shape_metrics, metrics_values, best_mask_idx, class_id, metric_name):
    """Save debug visualization image showing all masks and their scores
    
    Args:
        image: Original image
        masks: All masks
        scores: SAM confidence scores
        shape_metrics: Shape metrics
        metrics_values: Combined scores
        best_mask_idx: Index of the best mask selected
        class_id: Instance class ID
        metric_name: Name of the metric used
    """
    import matplotlib.pyplot as plt
    import os
    import time
    import numpy as np
    
    # Create save directory
    os.makedirs('vis', exist_ok=True)
    
    # Create figure
    n_masks = len(masks)
    n_cols = min(3, n_masks)  # Maximum 3 subplots per row
    n_rows = (n_masks + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows), dpi=150,
                            gridspec_kw={'bottom': 0.5})
    if n_rows * n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Get class names
    class_names = {
        0: 'ship',
        1: 'aircraft',
        2: 'car',
        3: 'tank',
        4: 'bridge',
        5: 'harbor'
    }
    class_name = class_names.get(class_id, f'Class {class_id}')
    
    # Set title
    fig.suptitle(f'{class_name} - Mask Evaluation using {metric_name}', fontsize=16, fontweight='bold')
    
    # Convert image to numpy array for visualization
    img_np = image.permute(1, 2, 0).detach().cpu().numpy()
    
    # Normalize image to [0,1]
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    # Plot each mask
    for i, (mask, ax) in enumerate(zip(masks, axes)):
        # Display original image
        ax.imshow(img_np)
        
        # Create mask overlay
        mask_colored = np.zeros((*mask.shape, 4))  # RGBA
        mask_colored[mask] = [1, 0, 0, 0.3]  # Semi-transparent red
        from scipy import ndimage
        mask_dilated = ndimage.binary_dilation(mask, iterations=1)
        mask_edge = mask_dilated & ~mask
        
        mask_colored[mask_edge] = [0, 0, 1, 1]  # Solid blue for edge

        # Overlay mask
        ax.imshow(mask_colored)
        
        title_color = 'red' if i == best_mask_idx else 'black'
        ax.set_title(f"Mask {i+1}", color=title_color, fontweight='bold' if i == best_mask_idx else 'normal')
        ax.axis('off')
        
        metric_text = ""
        for metric_key, metric_value in shape_metrics[i].items():
            metric_text += f"{metric_key}: {metric_value:.4f}\n"
        
        text_info = f"SAM Score: {scores[i]:.4f}\n{metric_text}Combined Score: {metrics_values[i]:.4f}"
        
        bbox = ax.get_position()
        fig.text(bbox.x0 + bbox.width/2, bbox.y0 - 0.03, 
                text_info, 
                ha='center', va='top', 
                fontsize=9,
                linespacing=1.5,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Hide empty subplots
    for i in range(len(masks), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout(rect=[0, 0.25, 1, 0.95])
    
    # Save image
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = f'vis/{timestamp}-{class_name}-masks.png'
    plt.savefig(output_path, bbox_inches='tight', facecolor='white')
    plt.close()