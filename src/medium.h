#pragma once

#include "phase_function.h"
#include "spectrum.h"
#include "volume.h"
#include <variant>

struct Scene;

struct MediumBase {
    PhaseFunction phase_function;
};

struct HomogeneousMedium : public MediumBase {
    Spectrum sigma_a, sigma_s;
};

struct HeterogeneousMedium : public MediumBase {
    VolumeSpectrum albedo, density;
};

// #Edited (Phase 1 & Phase 2)
// Struct: GlowDischargeMedium
// Description: Volumetric medium representing glow discharge plasma.
//              Stores all physics parameters for the ODE solver and emission computation.
struct GlowDischargeMedium : public MediumBase {
    // Physics coefficients (from Equations 14 in the paper)
    Real alpha;          // #Edited (Phase 1) - Ionization coefficient (controls brightness growth)
    Real beta;           // #Edited (Phase 1) - Recombination coefficient (drag term)
    Real eta;            // #Edited (Phase 2) - Attachment coefficient (e⁻ → negative ion conversion)
    Real rho;            // #Edited (Phase 2) - Ion scale coefficient (ions move rho times slower, rho > 1)
    Real sigma_0;        // Collision cross section (0.42e-18)

    // Drift field parameters
    Vector3 drift_center;    // Center of the glow region
    Real drift_magnitude;    // Magnitude of drift velocity ||μ||
    Vector3 drift_direction; // Direction (for linear drift)

    // Boundary
    Real boundary_radius;    // Spherical boundary radius ∂Ω

    // Emission color (simplified - no spectral)
    Spectrum emission_color; // RGB color, e.g., (0.8, 0.3, 0.8) for purple/pink

    // Solver parameters
    int num_integration_steps; // Number of ODE integration steps (default: 64)
};

using Medium = std::variant<HomogeneousMedium, HeterogeneousMedium, GlowDischargeMedium>;

/// the maximum of sigma_t = sigma_s + sigma_a over the whole space
Spectrum get_majorant(const Medium &medium, const Ray &ray);
Spectrum get_sigma_s(const Medium &medium, const Vector3 &p);
Spectrum get_sigma_a(const Medium &medium, const Vector3 &p);

inline PhaseFunction get_phase_function(const Medium &medium) {
    return std::visit([&](const auto &m) { return m.phase_function; }, medium);
}

// #Edited (Phase 1 & Phase 2) - Glow discharge helper functions
Spectrum get_glow_emission(const GlowDischargeMedium &medium, const Vector3 &p);  // Computes emission L_e using ODE solver
Vector3 get_drift_velocity(const GlowDischargeMedium &medium, const Vector3 &p);  // Returns drift field μ(p)
bool is_inside_glow_boundary(const GlowDischargeMedium &medium, const Vector3 &p); // Checks if p ∈ Ω
Real compute_divergence(const GlowDischargeMedium &medium, const Vector3 &p);     // #Edited (Phase 2) - Computes ∇·μ(p)
