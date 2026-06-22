#Subbotina Stephenson and Jackson 2026
#functions/classes not from HCIPy used in this work are included here to de-clutter
import scipy.sparse as sp
import numpy as np

def build_Laplacian_operator(pupil_grid, aperture):
    """
    Builds a sparse finite-difference Laplacian matrix with explicit edge substitution.
    
    Interior pixels: Standard 5-point stencil
      ∇²φ ≈ (-4φ_center + φ_left + φ_right + φ_down + φ_up) / dx² = curvature_center
    
    Boundary pixels: One-sided stencil using available interior neighbors
      Each valid neighbor contributes: (φ_neighbor - φ_center) / dx² to ∇²φ
      Average over available directions: ∇²φ ≈ Σ(φ_neighbor - φ_center) / (dx² * N) = curvature_boundary
      
      This EXPLICITLY SUBSTITUTES the measured curvature at boundary pixels.
      When solving L·φ = curvature, the RHS directly contains measured edge values.
    """
    dims = pupil_grid.dims
    if len(dims) == 2:
        Ny, Nx = dims
    else:
        raise ValueError("pupil_grid must be 2D")
    
    num_pixels = pupil_grid.size
    inside = (aperture > 0.5).ravel()
    
    L = sp.lil_matrix((num_pixels, num_pixels))
    
    x_coords = np.unique(pupil_grid.x)
    if len(x_coords) > 1:
        dx = x_coords[1] - x_coords[0]
    else:
        dx = 1.0
    
    for idx in range(num_pixels):
        if not inside[idx]:
            # Exterior pixels: dummy constraint
            L[idx, idx] = 1.0
            continue
            
        y_idx = idx // Nx
        x_idx = idx % Nx
        
        left  = idx - 1  if x_idx > 0      else None
        right = idx + 1  if x_idx < Nx - 1 else None
        down  = idx - Nx if y_idx > 0      else None
        up    = idx + Nx if y_idx < Ny - 1 else None
        
        neighbors = [left, right, down, up]
        is_boundary = any(n is None or not inside[n] for n in neighbors)
        
        if not is_boundary:
            # INTERIOR: Standard 5-point stencil
            # Enforces: ∇²φ = curvature (via L·φ = curvature_RHS)
            L[idx, idx]   = -4.0 / dx**2
            L[idx, left]  =  1.0 / dx**2
            L[idx, right] =  1.0 / dx**2
            L[idx, down]  =  1.0 / dx**2
            L[idx, up]    =  1.0 / dx**2
        else:
            # BOUNDARY: One-sided stencil with explicit edge measurement substitution
            # Collect available interior neighbors for this boundary pixel
            valid_neighbors = []
            if left is not None and inside[left]:
                valid_neighbors.append(('left', left))
            if right is not None and inside[right]:
                valid_neighbors.append(('right', right))
            if down is not None and inside[down]:
                valid_neighbors.append(('down', down))
            if up is not None and inside[up]:
                valid_neighbors.append(('up', up))
            
            if len(valid_neighbors) >= 1:
                # One-sided Laplacian stencil using available directions
                # Formula: ∇²φ ≈ [Σ(φ_neighbor - φ_center)] / dx²
                # = [Σφ_neighbor - N·φ_center] / dx²
                
                n_valid = len(valid_neighbors)
                L[idx, idx] = -float(n_valid) / dx**2  # Center: -N/dx²
                
                for direction, neighbor_idx in valid_neighbors:
                    # Each neighbor: 1/dx²
                    L[idx, neighbor_idx] = 1.0 / dx**2
                
                # The RHS (curvature_signal[idx]) is the EXPLICITLY SUBSTITUTED 
                # measured curvature at this boundary pixel
            else:
                # Isolated boundary: identity constraint
                L[idx, idx] = 1.0
                
    return L.tocsr()
