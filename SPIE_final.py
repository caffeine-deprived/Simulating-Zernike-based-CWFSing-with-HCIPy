#Subbotina Stephenson and Jackson 2026
import matplotlib.pyplot as plt
from hcipy import *
import numpy as np
from scipy.optimize import curve_fit
import scipy.sparse.linalg as spla
from hcipy import Field
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from functions import *

#####################
##############
#-------
# PHYSICAL PARAMETERS
wavelength =  8*(10**-7)          # 800 nanometers is based off of REVOLT desired wavelength for wfsing
diameter = 8*(10**-2)             # entrance pupil diameter
num_pix = 352              # CCD array resolution
propagation_distances = np.array([1,2,3,4, 5, 10, 20, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 2000000]) #can be changed out, but intially is set broad. 
# Reccomend changing distances to shorter + more detailed increments for high spatial complexities as their peak sensing distance occurs clsoe to the focal plane.
# For example, in this work's SPIE poster we started with the propagation distances provided above but then added in more specific ones according to where the optimimum gain == 1 distance was for various modes to make the SPIE plot appear smoother.
regularization_param = 1e-12  # Tikhonov regularization parameter (tune for better conditioning when using regularized solver - note: regularization is not used in the main loop by default, but can be enabled by uncommenting the regularized solver section)
#####################
##############
#-------
# CREATING ZERNIKE BASIS
zernike_modes_to_plot = list(range(100))
oversizing_factor = 30/15 #add oversizing to prevent edge issues/wrapping 
pupil_grid = make_pupil_grid(num_pix, diameter * oversizing_factor)
aperture = make_circular_aperture(diameter)(pupil_grid) 
L_matrix = build_Laplacian_operator(pupil_grid, aperture) # Generate the sparse matrix system based on your physical grid layout
zernike_basis = make_zernike_basis(100, diameter, pupil_grid, ansi=True, starting_mode=0) #note the ansi setting and starting mode; important for labelling later
x_coords = np.unique(pupil_grid.x) #grid spacing calculation
all_mode_results = {}  # Dictionary to store results for each mode
if len(x_coords) > 1:
    dx = x_coords[1] - x_coords[0]
else:
    dx = 1.0
#####################
##############
#-------
# CALCULATIONS and FIRST PLOT: Reconstructed Zernike Coeff vs. Propagated Distance
print(f"\n{'='*60}")
print("Plotting Reconstructed Mode Coefficients vs Propagation Distance")
print(f"{'='*60}")
fig, ax = plt.subplots(figsize=(16, 8))
line_styles, markers = ['-', '--', '-.', ':'], ['o', 's', '^', 'D', 'v']
colors = plt.cm.tab10(np.linspace(0, 1, len(zernike_modes_to_plot))) #upper and lower bounds for a cmap (aesthetic plots)
dictionary_of_mode_and_peak_z_and_peak_coeff = dict()
for idx, mode in enumerate(zernike_modes_to_plot):
    if mode == 0: #piston
        n = 0
    else:
        n = int(np.ceil((-3 + np.sqrt(9 + 8 * mode)) / 2))
    m = 2 * mode - n * (n + 2)
    print(f"\n\n{'#'*70}")
    print(f"# TESTING MODE {mode}")
    print(f"{'#'*70}\n")
    #make the zernike phase abberation using the basis made earlier
    phase_aberration = zernike_basis[mode] * 1.000 #as opposed to 0.3 or some partial stroke
    input_field = Field(aperture * np.exp(1j * phase_aberration), pupil_grid)
    input_field_wavefront = Wavefront(input_field, wavelength)
    results = []

    skip_mode = False #this is for figuring out what subsections of the first 100 modes to plot (see later in this loop)
    
    for dist_idx, prop_dist in enumerate(propagation_distances):
        print(f"\n{'='*60}")
        print(f"Mode {mode} - Testing propagation distance: {prop_dist} m")
        print(f"{'='*60}")
        propagator = FresnelPropagator(pupil_grid, prop_dist)
        propagator_minus = FresnelPropagator(pupil_grid, -prop_dist)
        # propagations (+/-L)
        field_plus = propagator.forward(input_field_wavefront)
        intensity_plus = field_plus.intensity
        field_minus = propagator_minus.forward(input_field_wavefront)
        intensity_minus = field_minus.intensity
        # Avoid division by zero by adding a small epsilon denominator threshold
        total_intensity = intensity_plus + intensity_minus
        safe_denominator = np.where(total_intensity == 0, 1e-10, total_intensity)
        curvature_signal = (intensity_plus - intensity_minus) / safe_denominator # Calculate curvature signal: (I_+ - I_-) / (I_+ + I_-)
        curvature_signal *= aperture # mask to clean up edge noise outside the pupil
        curvature_ravel = curvature_signal.ravel()  #Reconstruct phase from curvature signal

        # Compute curvature RMS (inside aperture only) - for frequency response analysis
        curvature_rms = np.sqrt(np.mean(curvature_signal[aperture > 0.5]**2))

        # SCALING: The wavenumber k = 2π/λ couples curvature to the phase Laplacian
        # For CWFS: ∇²φ = -(k/z) * curvature_signal (negative sign for Fresnel)
        k = 2 * np.pi / wavelength  # wavenumber
        curvature_scale = -k / prop_dist  # Scale by -k/z (note negative sign)

        # Direct solve without regularization
        # Scale the RHS first, then solve: L @ phase = curvature_scaled
        L_solver = L_matrix.tocsr()
        curvature_scaled = curvature_ravel * curvature_scale

        #print(f"Matrix condition number estimate (Original L): {sp.linalg.norm(L_solver) * sp.linalg.norm(L_solver.T):.2e}")
        #print(f"Curvature scaled - min: {np.min(curvature_scaled[aperture > 0.5]):.6e}, max: {np.max(curvature_scaled[aperture > 0.5]):.6e}, std: {np.std(curvature_scaled[aperture > 0.5]):.6e}")
        # Direct sparse solve (UNREGULARIZED)
        phase_reconstructed_ravel = spla.spsolve(L_solver, curvature_scaled)
        phase_reconstructed = Field(phase_reconstructed_ravel, pupil_grid)
    
        # Remove piston (constant phase offset) from reconstructed phase
        inside_aperture = aperture > 0.5
        mean_phase_recon = np.mean(phase_reconstructed[inside_aperture])
        phase_reconstructed = Field(phase_reconstructed.ravel() - mean_phase_recon, pupil_grid)

        # Compute reconstruction error
        reconstruction_error = phase_aberration - phase_reconstructed
        rms_error = np.sqrt(np.mean(reconstruction_error[aperture > 0.5]**2))
        
        # Project input and reconstructed phase onto Zernike modes
        input_coeffs = []
        recon_coeffs = []
        
        for i, zern_mode in enumerate(zernike_basis):
            # Project onto each Zernike mode (normalize by aperture area)
            input_coeff = np.sum(phase_aberration[aperture > 0.5] * zern_mode[aperture > 0.5]) / np.sum(aperture > 0.5)
            recon_coeff = np.sum(phase_reconstructed[aperture > 0.5] * zern_mode[aperture > 0.5]) / np.sum(aperture > 0.5)
            
            input_coeffs.append(input_coeff)
            recon_coeffs.append(recon_coeff)
        
        if dist_idx == 0:
            initial_coeff = recon_coeffs[mode]
            
            #BELOW are the sorting logics for the various plots produced in Subbotina Stephenson and Jackson 2026.
            #FIRST: Studying radial edge orders only (excluding piston)
            if m+n != 0 or m == 0: 
                print(f"--> SKIPPING MODE {mode}: NOT A RADIAL EDGE ORDER")
                skip_mode = True
                break
            #SECOND: Studying radial edge orders and spherical orders only
            """if m+n != 0 and m != 0: 
                print(f"--> SKIPPING MODE {mode}: NOT A RADIAL EDGE ORDER OR SPHERICAL ORDER")
                skip_mode = True
                break"""
            #THIRD: Studying all modes with a >1 gain on first propagation (ie. less spatially complex modes)
            """if initial_coeff <= 1.000: 
                print(f"--> SKIPPING MODE {mode}: Initial coeff {initial_coeff:.4f} is not high enough")
                skip_mode = True
                break"""
            #FOURTH: All modes with a <1 gain on first proapgation (ie. so complex that their talbot length is close to the focal plane)
            """if initial_coeff >= 1.000: 
                print(f"--> SKIPPING MODE {mode}: Initial coeff {initial_coeff:.4f} is too high")
                skip_mode = True
                break"""

        results.append({
            'distance': prop_dist,
            'curvature': curvature_signal,
            'curvature_rms': curvature_rms,
            'phase_input': phase_aberration,
            'phase_reconstructed': phase_reconstructed,
            'error': reconstruction_error,
            'rms_error': rms_error,
            'input_coeffs': input_coeffs,
            'recon_coeffs': recon_coeffs
        })

        print(f"Reconstruction RMS Error: {rms_error:.6f} rad")
        #print(f"Input phase range: [{np.min(phase_aberration[aperture > 0.5]):.4f}, {np.max(phase_aberration[aperture > 0.5]):.4f}] rad")
        #print(f"Reconstructed phase range: [{np.min(phase_reconstructed[aperture > 0.5]):.4f}, {np.max(phase_reconstructed[aperture > 0.5]):.4f}] rad")
        
        error = input_coeffs[mode] - recon_coeffs[mode]
        #printing results for specific mode
        print(f"\nZernike Mode {mode} Coefficients (Distance: {prop_dist} m):")
        print(f"Input Coeff: {input_coeffs[mode]:>14.6e}")
        print(f"Recon Coeff: {recon_coeffs[mode]:>14.6e}")
        print(f"Error:       {error:>14.6e}")
    if skip_mode:
        continue

    all_mode_results[mode] = results
    distances = [r['distance'] for r in results]
    recon_coeffs_mode = [r['recon_coeffs'][mode] for r in results]
    
    #finding optimal distance, aka where gain is closest (or just is) one
    recon_arr = np.array(recon_coeffs_mode)
    crossover_idx = np.argmin(np.abs(recon_arr - 1.0)) #use argmin instead of an exact 'gain==1' logic because there are modes which never reach gain==1
    
    optimal_z = distances[crossover_idx]
    print(f'optimal z for mode {mode} is {optimal_z} meters')
    crossover_coeff = recon_coeffs_mode[crossover_idx] 
    dictionary_of_mode_and_peak_z_and_peak_coeff[mode] = optimal_z, float(crossover_coeff)
    #below are stylistic if/else: plotting different colors/labels depending on mode
    if m+n == 0:
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        
        ax.plot(distances, recon_coeffs_mode, linestyle=line_style, marker=marker, 
                linewidth=2.5, markersize=6, color=color, label=rf"$Z_{{{n}}}^{{{m}}}$")
        #ax.plot(optimal_z, peak_coeff, 'o', markersize=12, color=color, markeredgewidth=2,  markerfacecolor='none', alpha=0.7)
    else:
        print('grey')
        line_style = line_styles[idx % len(line_styles)]
        marker = markers[idx % len(markers)]
        color = colors[idx % len(colors)]
        
        ax.plot(distances, recon_coeffs_mode, linestyle=line_style, marker=marker, 
                linewidth=2.5, markersize=6, color='grey', alpha=0.2)
        #ax.plot(optimal_z, peak_coeff, 'o', markersize=12, color='grey', markeredgewidth=2, markerfacecolor='none', alpha=0.7)


plt.axhline(y=1, color='red', linestyle='-', linewidth=2, label='optimal sensing distance')
ax.set_xlabel('Propagation Distance (m)', fontsize=25)
ax.set_xscale('log') #remember to take x axis out of log space if you are plotting a short distance regime!

ax.set_ylabel('Reconstructed Zernike Coefficient (rad)', fontsize=25)

ax.tick_params(axis='both', which='major', labelsize=22)
plt.suptitle('Reconstructed Mode Coefficients vs Propagation Distance', fontsize=27, fontweight='bold') #fontsize is optimized for SPIE poster
#plt.title('coupled asymmetric modes are included in grey', fontsize=23)
ax.grid(True, alpha=0.43, which='both')
ax.legend(fontsize=15, loc='upper right', ncol=1)
ax.axhline(y=0.0, color='black', linestyle='-', linewidth=0.5, alpha=1.000)

plt.tight_layout()
plt.show()
plt.savefig("reconstructed_coeff_vs_distance.png", dpi=300)
plt.close()

#####################
##############
#-------
# SECOND PLOT: Selected Zernike modes

fig = plt.figure(figsize=(16, 14))

# --- layout parameters (from chat with LLM) ---
ax_size = 0.135      # Slightly smaller to leave room for the vertical gap
dx = 0.145           # Horizontal step scaled to match
dy = 0.195           # The crucial fix: increased significantly to make physical room for 26pt font
y_top = 0.88         # Top margin
x_right_edge = 0.92  # The fixed X coordinate for the rightmost edge of the shelf
# Calculate the true coordinate extent of the oversized grid
grid_extent = (-oversizing_factor * diameter/2, oversizing_factor * diameter/2, 
               -oversizing_factor * diameter/2, oversizing_factor * diameter/2)
# ---end chat with LLM ---


for i in range(100):
    phase_aberration = zernike_basis[i]
    masked_phase = phase_aberration.copy()
    masked_phase[aperture < 0.5] = np.nan
    # -- Calculate labels (from LLM + wikipedia) --
    if i == 0:
        n = 0
    else:
        n = int(np.ceil((-3 + np.sqrt(9 + 8 * i)) / 2)) 
    m = 2 * i - n * (n + 2)
    x_center = x_right_edge - ((n - m) / 2) * dx
    y_center = y_top - n * dy
    ax = fig.add_axes([x_center - ax_size/2, y_center - ax_size/2, ax_size, ax_size])
    # -- end LLM chat --

    if i in dictionary_of_mode_and_peak_z_and_peak_coeff: #only plot those of particular interest in color
        im = ax.imshow(masked_phase.shaped, cmap='RdBu', origin='lower', extent=grid_extent)
    else: #otherwise, to a greyish existence it goes
        im = ax.imshow(masked_phase.shaped, cmap='Greys', origin='lower', alpha=0.8, extent=grid_extent)

    #crop anything outside pupil diameter so there is no buffer space 
    ax.set_xlim(-diameter/2, diameter/2)
    ax.set_ylim(-diameter/2, diameter/2)
    ax.set_title(rf"$Z_{{{n}}}^{{{m}}}$", fontsize=26, pad=6)
    ax.axis('off')
plt.savefig("zernike_modes_transparent.png", transparent=True, bbox_inches="tight", dpi=300)
plt.show()

#####################
##############
#-------
# THIRD PLOT: Optimal Distance v. Mode number (Zernike mode displayed instead of traditional markers)
modes = list(dictionary_of_mode_and_peak_z_and_peak_coeff.keys())
distances = [dictionary_of_mode_and_peak_z_and_peak_coeff[m][0] for m in modes]

min_mode, max_mode = min(modes) if modes else 0, max(modes) if modes else 100


fig, ax = plt.subplots(figsize=(22, 12))
for mode in modes:
    x_val = dictionary_of_mode_and_peak_z_and_peak_coeff[mode][0]
    #ANSI n/m calculation for labels
    if mode == 0:
        n = 0
    else:
        n = int(np.ceil((-3 + np.sqrt(9 + 8 * mode)) / 2))
    m = 2 * mode - n * (n + 2)
    
    #making Zernike mode marker
    phase_aberration = zernike_basis[mode]
    masked_phase = phase_aberration.copy()
    masked_phase[aperture < 0.5] = np.nan
    img_array = masked_phase.shaped
    #custom labels located close in proximity to the actual zernike mod being labelled
    imagebox = OffsetImage(img_array, zoom=0.3, cmap='RdBu', norm=plt.Normalize(vmin=-1, vmax=1))
    ab = AnnotationBbox(imagebox, (x_val, mode), frameon=False)
    ax.add_artist(ab)
    ax.annotate(rf"$Z_{{{n}}}^{{{m}}}$", xy=(x_val, mode), xytext=(30, 0), textcoords="offset points", fontsize=22, va='center', bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8, lw=0.5))

ax.set_ylim(min_mode - 2, max_mode + 2) #stylistic (for breathing room)
ax.set_xlim(0.8, max(distances) * 3)  #stylistic (so that labels don't overlap)
ax.set_xscale('log') 
ax.set_xlabel('Logarithmic Propagation Distance (m)', fontsize=25)
ax.set_ylabel('mode', fontsize=25)
plt.suptitle('Optimal Propagation Distance for each Zernike Mode', fontsize=27, fontweight='bold', y=0.94)
plt.title('Optimal distance occurs where gain ==1', fontsize=25, y=0.94)
ax.tick_params(axis='both', which='major', labelsize=22)
ax.grid(True, alpha=0.3, which='both')

#lastly, adding a nice exponential line of best fit
def fit_func(x, a, b):
    return a * x**b
x,y = np.array(distances), np.array(modes)
popt, pcov = curve_fit(fit_func, x, y)
x_fit = np.logspace(np.log10(min(x)), np.log10(max(x)), 500)
y_fit = fit_func(x_fit, *popt)
ax.plot(x_fit, y_fit, color='grey', linestyle='--', linewidth=7, alpha=0.5, zorder=0) #label=rf'Fit: $y = {popt[0]:.0f}x^{{{popt[1]:.2f}}}$')
#ax.legend(fontsize=16, loc='upper right')
plt.tight_layout()
plt.savefig("zernike_scatter_images.png", dpi=300)
plt.show()
plt.close()



