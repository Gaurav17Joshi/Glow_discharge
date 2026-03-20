#include "medium.h"
#include "glow_discharge_solver.h"

struct get_majorant_op {
    Spectrum operator()(const HomogeneousMedium &m);
    Spectrum operator()(const HeterogeneousMedium &m);
    Spectrum operator()(const GlowDischargeMedium &m);

    const Ray &ray;
};

struct get_sigma_s_op {
    Spectrum operator()(const HomogeneousMedium &m);
    Spectrum operator()(const HeterogeneousMedium &m);
    Spectrum operator()(const GlowDischargeMedium &m);

    const Vector3 &p;
};

struct get_sigma_a_op {
    Spectrum operator()(const HomogeneousMedium &m);
    Spectrum operator()(const HeterogeneousMedium &m);
    Spectrum operator()(const GlowDischargeMedium &m);

    const Vector3 &p;
};

#include "media/homogeneous.inl"
#include "media/heterogeneous.inl"
#include "media/glow_discharge.inl"

Spectrum get_majorant(const Medium &medium, const Ray &ray) {
    return std::visit(get_majorant_op{ray}, medium);
}

Spectrum get_sigma_s(const Medium &medium, const Vector3 &p) {
    return std::visit(get_sigma_s_op{p}, medium);
}

Spectrum get_sigma_a(const Medium &medium, const Vector3 &p) {
    return std::visit(get_sigma_a_op{p}, medium);
}

// Glow discharge helper functions implementation

// #Edited (Phase 1)
// Function: get_drift_velocity
// Description: Computes the drift velocity field μ(x) at a given point.
//              Implements a simple radial drift field pointing outward from the glow center.
// Input:
//   - medium: GlowDischargeMedium struct containing drift field parameters
//   - p: 3D point where drift velocity is evaluated
// Output: Vector3 representing drift velocity μ(p) = drift_magnitude * normalize(p - center)
Vector3 get_drift_velocity(const GlowDischargeMedium &medium, const Vector3 &p) {
    // Simple radial drift field (points outward from center)
    Vector3 dir = p - medium.drift_center;
    Real dist = length(dir);

    if (dist < Real(1e-6)) {
        return Vector3{Real(0), Real(0), Real(0)};
    }

    // Radial drift with constant magnitude
    return medium.drift_magnitude * normalize(dir);
}

// #Edited (Phase 1)
// Function: is_inside_glow_boundary
// Description: Checks if a point is inside the glow discharge boundary region.
// Input:
//   - medium: GlowDischargeMedium struct containing boundary parameters
//   - p: 3D point to test
// Output: bool - true if ||p - center|| <= boundary_radius, false otherwise
bool is_inside_glow_boundary(const GlowDischargeMedium &medium, const Vector3 &p) {
    Real dist = length(p - medium.drift_center);
    return dist <= medium.boundary_radius;
}

// #Edited (Phase 2)
// Function: compute_divergence
// Description: Computes the divergence of the drift velocity field ∇·μ(x).
//              For a radial drift field μ(x) = c * normalize(x - center),
//              the divergence is ∇·μ = 2c/r where r = ||x - center||.
//              This term appears in the coupled ODE system (Equations 14).
// Input:
//   - medium: GlowDischargeMedium struct containing drift field parameters
//   - p: 3D point where divergence is evaluated
// Output: Real - divergence value ∇·μ(p)
Real compute_divergence(const GlowDischargeMedium &medium, const Vector3 &p) {
    // For radial drift field: μ(x) = drift_magnitude · normalize(x - center)
    // ∇·μ = drift_magnitude · (2/r) where r = ||x - center||
    Real r = length(p - medium.drift_center);
    if (r < Real(1e-6)) {
        return Real(0);
    }
    return medium.drift_magnitude * (Real(2.0) / r);
}

// #Edited (Phase 1, updated Phase 2)
// Function: get_glow_emission
// Description: Computes the emission radiance at a point in the glow discharge.
//              Uses the full ODE solver to compute electron density E(x), then
//              applies Equation 4 from the paper: L_e = (σ₀/4π) · ||μ|| · E · color
// Input:
//   - medium: GlowDischargeMedium struct containing all physics parameters
//   - p: 3D point where emission is evaluated
// Output: Spectrum - RGB emission radiance at point p
Spectrum get_glow_emission(const GlowDischargeMedium &medium, const Vector3 &p) {
    // Check if we're inside the glow region
    if (!is_inside_glow_boundary(medium, p)) {
        return make_zero_spectrum();
    }

    // Solve for electron density at this point
    GlowSolver solver{medium};
    Real E = solver.solve_density(p);

    // Get drift velocity magnitude
    Vector3 mu = get_drift_velocity(medium, p);
    Real mu_magnitude = length(mu);

    // Compute emission intensity (Equation 4 from paper)
    // L_e = (sigma_0 / 4π) * ||mu|| * E
    Real intensity = (medium.sigma_0 / (Real(4.0) * Real(c_PI))) * mu_magnitude * E;

    // Multiply by emission color
    return intensity * medium.emission_color;
}
