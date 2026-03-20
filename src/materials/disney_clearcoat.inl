#include "../microfacet.h"

Spectrum eval_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return make_zero_spectrum();
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real ag = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real ag2 = ag * ag;

    Vector3 h = normalize(dir_in + dir_out);
    Real n_dot_in = dot(frame.n, dir_in);
    Real h_dot_out = dot(h, dir_out);
    Real h_dot_n = dot(h, frame.n);
    
    if (n_dot_in <= 0 || h_dot_out <= 0 || h_dot_n <= 0) return make_zero_spectrum();

    Real R0 = 0.04;
    Real Fc = R0 + (1 - R0) * pow(1 - h_dot_out, 5);

    Real denom = c_PI * log(ag2) * (1 + (ag2 - 1) * h_dot_n * h_dot_n);
    Real Dc = (ag2 - 1) / denom;

    Real Gc = smith_masking_gtr2(to_local(frame, dir_in), 0.25) * 
              smith_masking_gtr2(to_local(frame, dir_out), 0.25);

    return make_const_spectrum(Fc * Dc * Gc / (4 * n_dot_in));
}

Real pdf_sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        return 0;
    }
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real ag = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real ag2 = ag * ag;

    Vector3 h = normalize(dir_in + dir_out);
    Real h_dot_out = dot(h, dir_out);
    Real h_dot_n = dot(h, frame.n);
    
    if (h_dot_out <= 0 || h_dot_n <= 0) return 0;

    Real denom = c_PI * log(ag2) * (1 + (ag2 - 1) * h_dot_n * h_dot_n);
    Real Dc = (ag2 - 1) / denom;

    return Dc * h_dot_n / (4 * h_dot_out);
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyClearcoat &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
    Real clearcoat_gloss = eval(bsdf.clearcoat_gloss, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real ag = (1 - clearcoat_gloss) * 0.1 + clearcoat_gloss * 0.001;
    Real ag2 = ag * ag;

    // Sample GTR1
    Real cos_theta_h = sqrt((1 - pow(ag2, 1 - rnd_param_uv.x)) / (1 - ag2));
    Real sin_theta_h = sqrt(max(Real(0), 1 - cos_theta_h * cos_theta_h));
    Real phi_h = 2 * c_PI * rnd_param_uv.y;

    Vector3 h_local = Vector3(
        sin_theta_h * cos(phi_h),
        sin_theta_h * sin(phi_h),
        cos_theta_h
    );

    Vector3 h = to_world(frame, h_local);
    Vector3 dir_out = normalize(-dir_in + 2 * dot(dir_in, h) * h);
    
    if (dot(frame.n, dir_out) <= 0) return {};

    return BSDFSampleRecord{
        dir_out,
        Real(0) /* eta */, ag /* roughness */
    };
}

TextureSpectrum get_texture_op::operator()(const DisneyClearcoat &bsdf) const {
    return make_constant_spectrum_texture(make_zero_spectrum());
}
