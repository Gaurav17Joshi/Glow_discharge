#include "../microfacet.h"

inline Real bsdf_anisotropic_gtr2(Real n_dot_h, const Vector3 &h_l, Real ax, Real ay) {
    Real ax2 = ax * ax;
    Real ay2 = ay * ay;
    Real term = (h_l.x * h_l.x) / ax2 + (h_l.y * h_l.y) / ay2 + h_l.z * h_l.z;
    return 1 / (c_PI * ax * ay * term * term);
}

inline Real bsdf_anisotropic_smith_masking(const Vector3 &v_local, Real ax, Real ay) {
    Real term = (v_local.x * ax) * (v_local.x * ax) + (v_local.y * ay) * (v_local.y * ay);
    Real lambda = (sqrt(1 + term / (v_local.z * v_local.z)) - 1) / 2;
    return 1 / (1 + lambda);
}

Spectrum eval_op::operator()(const DisneyBSDF &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }

    // Check if we are inside the object
    if (dot(vertex.geometric_normal, dir_in) <= 0) {
        // Only glass lobe is active
        return (*this)(DisneyGlass{
            bsdf.base_color,
            bsdf.roughness,
            bsdf.anisotropic,
            bsdf.eta
        });
    }

    // Outside: Blended BSDF
    Spectrum base_color = eval(bsdf.base_color, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real roughness = eval(bsdf.roughness, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real sheen = eval(bsdf.sheen, vertex.uv, vertex.uv_screen_size, texture_pool);

    Spectrum f_diffuse = (*this)(DisneyDiffuse{
        bsdf.base_color,
        bsdf.roughness,
        bsdf.subsurface
    });

    Spectrum f_sheen = (*this)(DisneySheen{
        bsdf.base_color,
        bsdf.sheen_tint
    });

    Spectrum f_clearcoat = (*this)(DisneyClearcoat{
        bsdf.clearcoat_gloss
    });

    Spectrum f_glass = (*this)(DisneyGlass{
        bsdf.base_color,
        bsdf.roughness,
        bsdf.anisotropic,
        bsdf.eta
    });

    // Modified Metal Lobe
    Spectrum f_metal;
    {
        // Custom Metal Eval
        Real anisotropic = eval(bsdf.anisotropic, vertex.uv, vertex.uv_screen_size, texture_pool);
        Real specular = eval(bsdf.specular, vertex.uv, vertex.uv_screen_size, texture_pool);
        Real specular_tint = eval(bsdf.specular_tint, vertex.uv, vertex.uv_screen_size, texture_pool);
        
        Real r_clamped = std::clamp(roughness, Real(0.01), Real(1));
        Real aspect = sqrt(1 - 0.9 * anisotropic);
        Real ax = std::max(Real(0.0001), r_clamped * r_clamped / aspect);
        Real ay = std::max(Real(0.0001), r_clamped * r_clamped * aspect);

        Vector3 h = normalize(dir_in + dir_out);
        Vector3 h_l = to_local(frame, h);
        Real n_dot_in = dot(frame.n, dir_in);
        Real n_dot_out = dot(frame.n, dir_out);
        Real h_dot_out = dot(h, dir_out);

        if (n_dot_in > 0 && n_dot_out > 0 && h_dot_out > 0) {
            Real lum = luminance(base_color);
            Spectrum C_tint = lum > 0 ? base_color / lum : make_const_spectrum(1);
            Spectrum Ks = make_const_spectrum(1 - specular_tint) + specular_tint * C_tint;
            Real R0 = pow((bsdf.eta - 1) / (bsdf.eta + 1), 2);
            Spectrum C0 = specular * R0 * (1 - metallic) * Ks + metallic * base_color;

            Spectrum Fm = C0 + (make_const_spectrum(1) - C0) * pow(1 - h_dot_out, 5);
            Real Dm = bsdf_anisotropic_gtr2(dot(frame.n, h), h_l, ax, ay);
            Real Gm_in = bsdf_anisotropic_smith_masking(to_local(frame, dir_in), ax, ay);

            f_metal = Fm * Dm * Gm_in / (4 * n_dot_in);
        } else {
            f_metal = make_zero_spectrum();
        }
    }

    Spectrum result = make_zero_spectrum();

    // Diffuse
    result += (1 - specular_transmission) * (1 - metallic) * f_diffuse;

    // Sheen 
    result += (1 - metallic) * sheen * f_sheen;

    // Metal
    result += (1 - specular_transmission * (1 - metallic)) * f_metal;

    // Clearcoat
    result += 0.25 * clearcoat * f_clearcoat;

    // Glass
    result += (1 - metallic) * specular_transmission * f_glass;

    Real max_val = max(result);
    if (max_val > 10.0) {
        result = result * (10.0 / max_val);
    }

    return result;
}

Real pdf_sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }

    if (dot(vertex.geometric_normal, dir_in) <= 0) {
        return (*this)(DisneyGlass{
            bsdf.base_color,
            bsdf.roughness,
            bsdf.anisotropic,
            bsdf.eta
        });
    }

    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);

    Real w_diffuse = (1 - metallic) * (1 - specular_transmission);
    Real w_metal = (1 - specular_transmission * (1 - metallic));
    Real w_glass = (1 - metallic) * specular_transmission;
    Real w_clearcoat = 0.25 * clearcoat;

    Real total_weight = w_diffuse + w_metal + w_glass + w_clearcoat;
    if (total_weight == 0) return 0;

    Real pdf = 0;
    if (w_diffuse > 0) {
        pdf += w_diffuse * (*this)(DisneyDiffuse{
            bsdf.base_color, bsdf.roughness, bsdf.subsurface});
    }
    if (w_metal > 0) {
        pdf += w_metal * (*this)(DisneyMetal{
            bsdf.base_color, bsdf.roughness, bsdf.anisotropic});
    }
    if (w_glass > 0) {
        pdf += w_glass * (*this)(DisneyGlass{
            bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta});
    }
    if (w_clearcoat > 0) {
        pdf += w_clearcoat * (*this)(DisneyClearcoat{bsdf.clearcoat_gloss});
    }

    return pdf / total_weight;
}

std::optional<BSDFSampleRecord>
        sample_bsdf_op::operator()(const DisneyBSDF &bsdf) const {
    // Flip the shading frame if it is inconsistent with the geometry normal
    Frame frame = vertex.shading_frame;
    if (dot(frame.n, dir_in) * dot(vertex.geometric_normal, dir_in) < 0) {
        frame = -frame;
    }

    if (dot(vertex.geometric_normal, dir_in) <= 0) {
        return (*this)(DisneyGlass{
            bsdf.base_color,
            bsdf.roughness,
            bsdf.anisotropic,
            bsdf.eta
        });
    }

    Real metallic = eval(bsdf.metallic, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real specular_transmission = eval(bsdf.specular_transmission, vertex.uv, vertex.uv_screen_size, texture_pool);
    Real clearcoat = eval(bsdf.clearcoat, vertex.uv, vertex.uv_screen_size, texture_pool);

    Real w_diffuse = (1 - metallic) * (1 - specular_transmission);
    Real w_metal = (1 - specular_transmission * (1 - metallic));
    Real w_glass = (1 - metallic) * specular_transmission;
    Real w_clearcoat = 0.25 * clearcoat;

    Real total_weight = w_diffuse + w_metal + w_glass + w_clearcoat;
    if (total_weight == 0) return {};

    Real rnd = rnd_param_w * total_weight;

    if (rnd < w_diffuse) {
        return (*this)(DisneyDiffuse{
            bsdf.base_color, bsdf.roughness, bsdf.subsurface});
    }
    rnd -= w_diffuse;

    if (rnd < w_metal) {
        return (*this)(DisneyMetal{
            bsdf.base_color, bsdf.roughness, bsdf.anisotropic});
    }
    rnd -= w_metal;

    if (rnd < w_glass) {
        return (*this)(DisneyGlass{
            bsdf.base_color, bsdf.roughness, bsdf.anisotropic, bsdf.eta});
    }
    rnd -= w_glass;

    return (*this)(DisneyClearcoat{bsdf.clearcoat_gloss});
}

TextureSpectrum get_texture_op::operator()(const DisneyBSDF &bsdf) const {
    return bsdf.base_color;
}