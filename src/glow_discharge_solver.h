#pragma once

#include "medium.h"
#include "vector.h"
#include <cmath>
#include <utility>
#include <algorithm>

// #Edited (Phase 2 - Complete Rewrite)
// Struct: GlowSolver
// Description: Full ODE solver for glow discharge plasma simulation.
//              Tracks three particle species: electrons (E), positive ions (P), negative ions (N).
//              Implements Algorithm 1 from the paper using log-space reparameterization.
struct GlowSolver {
    const GlowDischargeMedium &medium;

    // #Edited (Phase 2)
    // Function: trace_backwards
    // Description: Traces backwards along the drift field from point x until hitting
    //              the glow discharge boundary ∂Ω. This is step 1 of Algorithm 1.
    // Input:
    //   - x: Starting 3D point inside the glow region
    // Output: pair<Vector3, Real>
    //   - first: Boundary point x_∂ where trajectory exits the glow region
    //   - second: Total distance τ traveled from x_∂ to x along drift field
    std::pair<Vector3, Real> trace_backwards(const Vector3 &x) {
        Vector3 pos = x;
        Real total_distance = 0.0;
        Real dt = 0.01; // Fixed time step

        for (int i = 0; i < medium.num_integration_steps; i++) {
            // Check if we're outside boundary
            if (!is_inside_glow_boundary(medium, pos)) {
                break;
            }

            // Get drift velocity at current position (points outward from center)
            Vector3 mu = get_drift_velocity(medium, pos);
            Real mu_len = length(mu);

            if (mu_len < Real(1e-6)) {
                break;
            }

            // Step backwards (opposite to drift direction)
            Vector3 step = -normalize(mu) * dt;
            pos = pos + step;
            total_distance += dt;
        }

        return {pos, total_distance};
    }

    // #Edited (Phase 2)
    // Function: integrate_densities
    // Description: Integrates the coupled ODE system (Equations 14) forward from boundary
    //              to target point. Uses log-space reparameterization (X = exp(X̃)) for
    //              numerical stability. This is step 2 of Algorithm 1.
    //              Boundary conditions: E = P = N = 1 → Ẽ = P̃ = Ñ = 0
    // Input:
    //   - x_boundary: Starting point on the boundary ∂Ω
    //   - x_target: Target point where densities are needed (currently unused, could be used for validation)
    //   - tau: Total integration time/distance
    // Output: tuple<Real, Real, Real>
    //   - Ẽ(x): Log-space electron density
    //   - P̃(x): Log-space positive ion density
    //   - Ñ(x): Log-space negative ion density
    std::tuple<Real, Real, Real> integrate_densities(const Vector3 &x_boundary,
                                                       const Vector3 &x_target,
                                                       Real tau) {
        if (tau <= Real(0)) {
            return {Real(0), Real(0), Real(0)};
        }

        Vector3 pos = x_boundary;

        // Log-space densities (X = exp(X_tilde))
        // Boundary conditions: E = P = N = 1 → E_tilde = P_tilde = N_tilde = 0
        Real E_tilde = Real(0);
        Real P_tilde = Real(0);
        Real N_tilde = Real(0);

        Real dt = tau / Real(medium.num_integration_steps);

        for (int i = 0; i < medium.num_integration_steps; i++) {
            // Get drift velocity and its magnitude
            Vector3 mu = get_drift_velocity(medium, pos);
            Real mu_len = length(mu);

            // Compute divergence of drift velocity
            Real div_mu = compute_divergence(medium, pos);

            // Equations 14 from paper (drift-aligned, log-space)
            // ∇_μ Ẽ = (α - η)||μ|| - β exp(P̃) - (∇·μ)
            Real dE = (medium.alpha - medium.eta) * mu_len
                      - medium.beta * std::exp(P_tilde)
                      - div_mu;

            // ∇_μ P̃ = (α/ρ)||μ|| exp(Ẽ - P̃) - (β/ρ)exp(Ẽ) - (β/ρ)exp(Ñ) - (∇·μ)
            Real dP = (medium.alpha / medium.rho) * mu_len * std::exp(E_tilde - P_tilde)
                      - (medium.beta / medium.rho) * std::exp(E_tilde)
                      - (medium.beta / medium.rho) * std::exp(N_tilde)
                      - div_mu;

            // ∇_μ Ñ = (η/ρ)||μ|| exp(Ẽ - Ñ) - (β/ρ)exp(P̃) - (∇·μ)
            Real dN = (medium.eta / medium.rho) * mu_len * std::exp(E_tilde - N_tilde)
                      - (medium.beta / medium.rho) * std::exp(P_tilde)
                      - div_mu;

            // Forward Euler integration step
            E_tilde += dE * dt;
            P_tilde += dP * dt;
            N_tilde += dN * dt;

            // Clamp to prevent numerical overflow
            E_tilde = std::min(E_tilde, Real(50.0));
            P_tilde = std::min(P_tilde, Real(50.0));
            N_tilde = std::min(N_tilde, Real(50.0));

            // Move position forward along drift field
            if (mu_len > Real(1e-6)) {
                pos = pos + dt * mu;
            }
        }

        return {E_tilde, P_tilde, N_tilde};
    }

    // #Edited (Phase 2)
    // Function: solve_density
    // Description: Main entry point for computing electron density at any point x.
    //              Implements complete Algorithm 1 from the paper:
    //              1) Trace backwards to find boundary point and distance
    //              2) Integrate coupled ODEs forward from boundary
    //              3) Convert from log-space to linear-space
    // Input:
    //   - x: 3D point where electron density is needed
    // Output: Real - electron number density E(x) in linear space
    Real solve_density(const Vector3 &x) {
        // 1. Trace backwards to boundary
        auto [x_boundary, tau] = trace_backwards(x);

        // 2. Integrate forward with full ODE system
        auto [E_tilde, P_tilde, N_tilde] = integrate_densities(x_boundary, x, tau);

        // 3. Convert from log-space to linear-space
        // E = exp(E_tilde)
        Real E = std::exp(E_tilde);

        return E;
    }
};
