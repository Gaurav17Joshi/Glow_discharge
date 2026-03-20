#include "../microfacet.h"

inline Real anisotropic_gtr2(Real n_dot_h, const Vector3 &h_l, Real ax, Real ay) {
    Real ax2 = ax * ax;
    Real ay2 = ay * ay;
    Real term = (h_l.x * h_l.x) / ax2 + (h_l.y * h_l.y) / ay2 + h_l.z * h_l.z;
    return 1 / (c_PI * ax * ay * term * term);
}

inline Real anisotropic_smith_masking(const Vector3 &v_local, Real ax, Real ay) {
    Real term = (v_local.x * ax) * (v_local.x * ax) + (v_local.y * ay) * (v_local.y * ay);
    Real lambda = (sqrt(1 + term / (v_local.z * v_local.z)) - 1) / 2;
    return 1 / (1 + lambda);
}

Spectrum eval_op::operator()(const DisneyMetal &bsdf) const {
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
    
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real ax = std::max(Real(0.0001), roughness * roughness / aspect);
    Real ay = std::max(Real(0.0001), roughness * roughness * aspect);

    Vector3 h = normalize(dir_in + dir_out);
    Vector3 h_l = to_local(frame, h);
    Real n_dot_in = dot(frame.n, dir_in);
    Real n_dot_out = dot(frame.n, dir_out);
    Real h_dot_out = dot(h, dir_out);
    
    if (n_dot_in <= 0 || n_dot_out <= 0) return make_zero_spectrum();

    Spectrum Fm = base_color + (1 - base_color) * pow(1 - h_dot_out, 5);
    Real Dm = anisotropic_gtr2(dot(frame.n, h), h_l, ax, ay);
    Real Gm = anisotropic_smith_masking(to_local(frame, dir_in), ax, ay) *
              anisotropic_smith_masking(to_local(frame, dir_out), ax, ay);

    return (Fm * Dm * Gm) / (4 * n_dot_in);
}

Real pdf_sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0 ||
            dot(vertex.geometric_normal, dir_out) < 0) {
        // No light below the surface
        return 0;
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real ax = std::max(Real(0.0001), roughness * roughness / aspect);
    Real ay = std::max(Real(0.0001), roughness * roughness * aspect);

    Vector3 h = normalize(dir_in + dir_out);
    Vector3 h_l = to_local(frame, h);
    Real n_dot_in = dot(frame.n, dir_in);
    Real h_dot_in = dot(h, dir_in); // Same as h_dot_out
    
    if (n_dot_in <= 0 || h_dot_in <= 0) return 0;

    Real Dm = anisotropic_gtr2(dot(frame.n, h), h_l, ax, ay);
    Real Gm_in = anisotropic_smith_masking(to_local(frame, dir_in), ax, ay);

    return (Dm * Gm_in) / (4 * n_dot_in);
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyMetal &bsdf) const {
    if (dot(vertex.geometric_normal, dir_in) < 0) {
        // No light below the surface
        return {};
    }
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) < 0) {
        frame = -frame;
    }
    
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
    roughness = std::clamp(roughness, Real(0.01), Real(1));

    Real aspect = sqrt(1 - 0.9 * anisotropic);
    Real ax = std::max(Real(0.0001), roughness * roughness / aspect);
    Real ay = std::max(Real(0.0001), roughness * roughness * aspect);

    Vector3 local_dir_in = to_local(frame, dir_in);
    
    // Stretch
    Vector3 v_h = normalize(Vector3{ax * local_dir_in.x, ay * local_dir_in.y, local_dir_in.z});
    
    Real r = sqrt(rnd_param_uv.x);
    Real phi = 2 * c_PI * rnd_param_uv.y;
    Real t1 = r * cos(phi);
    Real t2 = r * sin(phi);
    Real s = 0.5 * (1 + v_h.z);
    t2 = (1 - s) * sqrt(1 - t1 * t1) + s * t2;
    Vector3 disk_N{t1, t2, sqrt(max(Real(0), 1 - t1*t1 - t2*t2))};
    
    // Reproject
    Frame hemi_frame(v_h);
    Vector3 hemi_N = to_world(hemi_frame, disk_N);
    
    // Unstretch
    Vector3 m_l = normalize(Vector3{ax * hemi_N.x, ay * hemi_N.y, max(Real(0), hemi_N.z)});
    
    Vector3 m = to_world(frame, m_l);
    Vector3 dir_out = normalize(-dir_in + 2 * dot(dir_in, m) * m);
    
    return BSDFSampleRecord{
        dir_out,
        Real(0) /* eta */, roughness /* roughness */
    };
}

TextureSpectrum get_texture_op::operator()(const DisneyMetal &bsdf) const {
    return bsdf.base_color;
}
