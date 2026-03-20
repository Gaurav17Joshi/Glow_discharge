#pragma once

inline Spectrum integrate_glow_emission_segment(const GlowDischargeMedium &medium,
                                                const Ray &ray,
                                                Real t_max,
                                                pcg32_state &rng) {
    if (t_max <= Real(0)) {
        return make_zero_spectrum();
    }

    constexpr int num_samples = 16;
    Spectrum emission = make_zero_spectrum();
    for (int i = 0; i < num_samples; i++) {
        Real t = (Real(i) + next_pcg32_real<Real>(rng)) / Real(num_samples) * t_max;
        emission += get_glow_emission(medium, ray.org + t * ray.dir);
    }
    return emission * (t_max / Real(num_samples));
}

// The simplest volumetric renderer:
// single absorption only homogeneous volume
// only handle directly visible light sources
Spectrum vol_path_tracing_1(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);

    Ray ray = sample_primary(scene.camera, screen_pos);

    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    if (!vertex_) {
        return make_zero_spectrum();
    }

    PathVertex vertex = *vertex_;

    int medium_id = scene.camera.medium_id;

    Spectrum transmittance = make_const_spectrum(1); // Default: no attenuation

    if (medium_id >= 0) {
        const Medium &medium = scene.media[medium_id];

        Spectrum sigma_a = get_sigma_a(medium, vertex.position);

        Real t_hit = distance(ray.org, vertex.position);

        Real sigma_a_mono = sigma_a[0];

        transmittance = make_const_spectrum(exp(-sigma_a_mono * t_hit));
    }

    if (is_light(scene.shapes[vertex.shape_id])) {
        Spectrum Le = emission(vertex, -ray.dir, scene);

        return transmittance * Le;
    }

    return make_zero_spectrum();
}

// The second simplest volumetric renderer:
// single monochromatic homogeneous volume with single scattering,
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_2(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);

    Real t_hit = infinity<Real>();
    if (vertex_) {
        t_hit = distance(ray.org, vertex_->position);
    }

    int medium_id = scene.camera.medium_id;
    if (medium_id < 0) {
        if (vertex_ && is_light(scene.shapes[vertex_->shape_id])) {
            return emission(*vertex_, -ray.dir, scene);
        }
        return make_zero_spectrum();
    }

    const Medium &medium = scene.media[medium_id];

    Spectrum sigma_a = get_sigma_a(medium, ray.org);
    Spectrum sigma_s = get_sigma_s(medium, ray.org);

    Real sigma_a_mono = sigma_a[0];
    Real sigma_s_mono = sigma_s[0];
    Real sigma_t = sigma_a_mono + sigma_s_mono; // Extinction coefficient

    Real u = next_pcg32_real<Real>(rng);
    Real t_sampled = -log(Real(1) - u) / sigma_t;

    if (t_sampled < t_hit) {
        Vector3 scatter_pos = ray.org + t_sampled * ray.dir;

        Real transmittance = exp(-sigma_t * t_sampled);
        Real trans_pdf = exp(-sigma_t * t_sampled) * sigma_t;

        Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
        Real light_w = next_pcg32_real<Real>(rng);
        Real shape_w = next_pcg32_real<Real>(rng);
        int light_id = sample_light(scene, light_w);
        const Light &light = scene.lights[light_id];
        PointAndNormal point_on_light =
            sample_point_on_light(light, scatter_pos, light_uv, shape_w, scene);

        Vector3 dir_to_light = normalize(point_on_light.position - scatter_pos);
        Real dist_to_light = distance(point_on_light.position, scatter_pos);

        Ray shadow_ray{scatter_pos, dir_to_light,
                       get_shadow_epsilon(scene),
                       (Real(1) - get_shadow_epsilon(scene)) * dist_to_light};

        if (!occluded(scene, shadow_ray)) {
            PhaseFunction phase_function = get_phase_function(medium);
            Real phase_eval = eval(phase_function, -ray.dir, dir_to_light)[0];

            Real transmittance_to_light = exp(-sigma_t * dist_to_light);

            Real G = max(-dot(dir_to_light, point_on_light.normal), Real(0)) /
                     (dist_to_light * dist_to_light);

            Spectrum Le = emission(light, -dir_to_light, Real(0), point_on_light, scene);

            Real light_pdf = pdf_point_on_light(light, point_on_light, scatter_pos, scene);
            Real pdf_light_choice = light_pmf(scene, light_id);
            Real combined_light_pdf = light_pdf * pdf_light_choice;

            Spectrum L_scatter =
                make_const_spectrum((transmittance / trans_pdf) * sigma_s_mono) *
                make_const_spectrum(phase_eval * transmittance_to_light * G / combined_light_pdf) *
                Le;

            return L_scatter;
        }

        return make_zero_spectrum();

    } else {
        if (!vertex_) {
            return make_zero_spectrum();
        }
        PathVertex vertex = *vertex_;

        Real transmittance = exp(-sigma_t * t_hit);
        Real trans_pdf = exp(-sigma_t * t_hit); 

        if (is_light(scene.shapes[vertex.shape_id])) {
            Spectrum Le = emission(vertex, -ray.dir, scene);
            return make_const_spectrum(transmittance / trans_pdf) * Le;
        }

        return make_zero_spectrum();
    }
}

// The third volumetric renderer (not so simple anymore): 
// multiple monochromatic homogeneous volumes with multiple scattering
// no need to handle surface lighting, only directly visible light source
Spectrum vol_path_tracing_3(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};

    int current_medium = scene.camera.medium_id;
    Spectrum path_throughput = make_const_spectrum(1);
    Spectrum radiance = make_zero_spectrum();
    int bounces = 0;
    int max_depth = scene.options.max_depth;

    while (true) {
        bool scatter = false;
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        Real t_hit = vertex_ ? distance(ray.org, vertex_->position) : infinity<Real>();

        if (current_medium >= 0) {
            const Medium &medium = scene.media[current_medium];
            Spectrum sigma_a = get_sigma_a(medium, ray.org);
            Spectrum sigma_s = get_sigma_s(medium, ray.org);
            Real sigma_t = sigma_a[0] + sigma_s[0];

            Real u = next_pcg32_real<Real>(rng);
            Real t = -log(Real(1) - u) / sigma_t;

            if (t < t_hit) {
                scatter = true;
                ray.org = ray.org + t * ray.dir;
                
                path_throughput *= (sigma_s / sigma_t);
            } else {
            }
        }

        // --- Handle Surface Hit / Emission ---
        if (!scatter) {
            if (!vertex_) {
                break;
            }
            const PathVertex &vertex = *vertex_;

            if (is_light(scene.shapes[vertex.shape_id])) {
                radiance += path_throughput * emission(vertex, -ray.dir, scene);
            }

            if (max_depth != -1 && bounces >= max_depth - 1) {
                break;
            }

            if (vertex.material_id == -1) {
                // Update medium
                if (vertex.interior_medium_id != vertex.exterior_medium_id) {
                    if (dot(ray.dir, vertex.geometric_normal) > 0) {
                        current_medium = vertex.exterior_medium_id;
                    } else {
                        current_medium = vertex.interior_medium_id;
                    }
                }
                
                // Continue ray
                ray.org = vertex.position + ray.dir * get_intersection_epsilon(scene);
                bounces++;
                continue;
            } else {
                break;
            }
        }

        if (scatter) {
            if (max_depth != -1 && bounces >= max_depth - 1) {
                break;
            }

            const Medium &medium = scene.media[current_medium];
            PhaseFunction phase = get_phase_function(medium);
            
            Vector2 uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            std::optional<Vector3> next_dir = sample_phase_function(phase, -ray.dir, uv);

            if (next_dir) {
                Real p = eval(phase, -ray.dir, *next_dir)[0];
                Real pdf = pdf_sample_phase(phase, -ray.dir, *next_dir);
                if (pdf > 0) {
                    path_throughput *= (p / pdf);
                    ray.dir = *next_dir;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        if (scene.options.rr_depth != -1 && bounces >= scene.options.rr_depth) {
            Real rr_prob = min(max(path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
            path_throughput /= rr_prob;
        }

        bounces++;
    }

    return radiance;
}

// The fourth volumetric renderer:
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// still no surface lighting
Spectrum vol_path_tracing_4(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};

    int current_medium_id = scene.camera.medium_id;
    Spectrum radiance = make_zero_spectrum();
    Spectrum path_throughput = make_const_spectrum(1);
    int bounces = 0;
    int max_depth = scene.options.max_depth;
    int rr_depth = scene.options.rr_depth;

    // MIS caching variables
    Real dir_pdf = 0;                              
    Vector3 nee_p_cache = ray.org;                 
    Real multi_trans_pdf = 1;                      
    bool never_scatter = true;                      

    while (true) {
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);

        bool scatter = false;
        Spectrum transmittance = make_const_spectrum(1);
        Real trans_pdf = 1;

        if (current_medium_id >= 0) {
            const Medium &medium = scene.media[current_medium_id];

            Vector3 sample_pos = vertex_ ? vertex_->position : ray.org + ray.dir * Real(1e10);
            Spectrum sigma_a = get_sigma_a(medium, sample_pos);
            Spectrum sigma_s = get_sigma_s(medium, sample_pos);
            Real sigma_a_mono = sigma_a[0];
            Real sigma_s_mono = sigma_s[0];
            Real sigma_t = sigma_a_mono + sigma_s_mono;

            if (sigma_t > 0) {
                Real t_hit = vertex_ ? distance(ray.org, vertex_->position) : infinity<Real>();
                Real u = next_pcg32_real<Real>(rng);
                Real t_sampled = -log(Real(1) - u) / sigma_t;

                if (t_sampled < t_hit) {
                    scatter = true;
                    never_scatter = false;
                    ray.org = ray.org + t_sampled * ray.dir;
                    transmittance = make_const_spectrum(exp(-sigma_t * t_sampled));
                    trans_pdf = exp(-sigma_t * t_sampled) * sigma_t;

                    path_throughput *= transmittance / make_const_spectrum(trans_pdf);
                    path_throughput *= make_const_spectrum(sigma_s_mono);
                } else {
                    transmittance = make_const_spectrum(exp(-sigma_t * t_hit));
                    trans_pdf = exp(-sigma_t * t_hit);
                    path_throughput *= transmittance / make_const_spectrum(trans_pdf);
                }
            }
        }

        if (scatter) {
            const Medium &medium = scene.media[current_medium_id];
            PhaseFunction phase_function = get_phase_function(medium);
            Spectrum sigma_a = get_sigma_a(medium, ray.org);
            Spectrum sigma_s = get_sigma_s(medium, ray.org);
            Real sigma_t = sigma_a[0] + sigma_s[0];

            Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            Real light_w = next_pcg32_real<Real>(rng);
            Real shape_w = next_pcg32_real<Real>(rng);
            int light_id = sample_light(scene, light_w);
            const Light &light = scene.lights[light_id];
            PointAndNormal point_on_light =
                sample_point_on_light(light, ray.org, light_uv, shape_w, scene);

            Vector3 dir_to_light = normalize(point_on_light.position - ray.org);
            Real dist_to_light = distance(point_on_light.position, ray.org);

            Spectrum T_light = make_const_spectrum(1);
            Real p_trans_nee = 1;
            Real p_trans_dir = 1;
            Vector3 shadow_pos = ray.org;
            int shadow_medium = current_medium_id;
            int shadow_bounces = 0;
            bool blocked = false;

            while (true) {
                Ray shadow_ray{shadow_pos, dir_to_light,
                              get_shadow_epsilon(scene),
                              distance(shadow_pos, point_on_light.position) - get_shadow_epsilon(scene)};
                std::optional<PathVertex> shadow_isect = intersect(scene, shadow_ray, ray_diff);

                Real next_t = distance(shadow_pos, point_on_light.position);
                if (shadow_isect) {
                    next_t = min(next_t, distance(shadow_pos, shadow_isect->position));
                }

                // Account for transmittance through this segment
                if (shadow_medium >= 0) {
                    const Medium &shadow_med = scene.media[shadow_medium];
                    Spectrum shadow_sigma_a = get_sigma_a(shadow_med, shadow_pos);
                    Spectrum shadow_sigma_s = get_sigma_s(shadow_med, shadow_pos);
                    Real shadow_sigma_t = shadow_sigma_a[0] + shadow_sigma_s[0];
                    T_light *= make_const_spectrum(exp(-shadow_sigma_t * next_t));
                    p_trans_nee *= exp(-shadow_sigma_t * next_t);
                    p_trans_dir *= exp(-shadow_sigma_t * next_t);
                }

                if (!shadow_isect || distance(shadow_pos, shadow_isect->position) >=
                    distance(shadow_pos, point_on_light.position) - get_shadow_epsilon(scene)) {
                    break;
                }

                if (shadow_isect->material_id >= 0) {
                    blocked = true;
                    break;
                }

                shadow_bounces++;
                if (max_depth != -1 && bounces + shadow_bounces + 1 >= max_depth) {
                    blocked = true;
                    break;
                }

                if (shadow_isect->interior_medium_id != shadow_isect->exterior_medium_id) {
                    if (dot(dir_to_light, shadow_isect->geometric_normal) > 0) {
                        shadow_medium = shadow_isect->exterior_medium_id;
                    } else {
                        shadow_medium = shadow_isect->interior_medium_id;
                    }
                }

                shadow_pos = shadow_isect->position + dir_to_light * get_shadow_epsilon(scene);
            }

            if (!blocked && max(T_light) > 0) {
                Real phase_eval = eval(phase_function, -ray.dir, dir_to_light)[0];
                Real G = max(-dot(dir_to_light, point_on_light.normal), Real(0)) /
                         (dist_to_light * dist_to_light);
                Spectrum Le = emission(light, -dir_to_light, Real(0), point_on_light, scene);
                Real pdf_nee = pdf_point_on_light(light, point_on_light, ray.org, scene) *
                              light_pmf(scene, light_id);

                // MIS weight
                Real pdf_phase = pdf_sample_phase(phase_function, -ray.dir, dir_to_light) * G * p_trans_dir;
                Real w_nee = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_phase * pdf_phase);

                Spectrum nee_contrib = T_light * make_const_spectrum(phase_eval * G / pdf_nee) * Le * make_const_spectrum(w_nee);
                radiance += path_throughput * nee_contrib;
            }

            nee_p_cache = ray.org;
            multi_trans_pdf = 1;
        }

        if (!scatter) {
            if (vertex_) {
                PathVertex vertex = *vertex_;

                // Check if we hit a light
                if (is_light(scene.shapes[vertex.shape_id])) {
                    Spectrum Le = emission(vertex, -ray.dir, scene);

                    if (never_scatter) {
                        // Direct visibility, no MIS needed
                        radiance += path_throughput * Le;
                    } else {
                        // Apply MIS weight
                        Real dist = distance(nee_p_cache, vertex.position);
                        Real G = abs(dot(vertex.geometric_normal, ray.dir)) / (dist * dist);
                        Real pdf_nee = pdf_point_on_light(
                            scene.lights[get_area_light_id(scene.shapes[vertex.shape_id])],
                            PointAndNormal{vertex.position, vertex.geometric_normal},
                            nee_p_cache, scene) *
                            light_pmf(scene, get_area_light_id(scene.shapes[vertex.shape_id]));
                        Real pdf_phase = dir_pdf * multi_trans_pdf * G;
                        Real w_phase = (pdf_phase * pdf_phase) / (pdf_phase * pdf_phase + pdf_nee * pdf_nee);
                        radiance += path_throughput * Le * make_const_spectrum(w_phase);
                    }
                }

                if (max_depth != -1 && bounces >= max_depth - 1) {
                    break;
                }

                if (vertex.material_id == -1) {
                    // Accumulate transmittance PDF for MIS
                    if (current_medium_id >= 0) {
                        const Medium &medium = scene.media[current_medium_id];
                        Spectrum sigma_a = get_sigma_a(medium, vertex.position);
                        Spectrum sigma_s = get_sigma_s(medium, vertex.position);
                        Real sigma_t = sigma_a[0] + sigma_s[0];
                        multi_trans_pdf *= exp(-sigma_t * distance(ray.org, vertex.position));
                    }

                    // Update medium
                    if (vertex.interior_medium_id != vertex.exterior_medium_id) {
                        if (dot(ray.dir, vertex.geometric_normal) > 0) {
                            current_medium_id = vertex.exterior_medium_id;
                        } else {
                            current_medium_id = vertex.interior_medium_id;
                        }
                    }

                    ray.org = vertex.position + ray.dir * get_shadow_epsilon(scene);
                    bounces++;
                    continue;
                }

                // Hit opaque surface, terminate
                break;
            } else {
                break;
            }
        }

        if (scatter) {
            const Medium &medium = scene.media[current_medium_id];
            PhaseFunction phase_function = get_phase_function(medium);

            Vector2 rnd_param{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            std::optional<Vector3> dir_out = sample_phase_function(
                phase_function, -ray.dir, rnd_param);

            if (!dir_out) {
                break;
            }

            Real phase_eval = eval(phase_function, -ray.dir, *dir_out)[0];
            dir_pdf = pdf_sample_phase(phase_function, -ray.dir, *dir_out);

            path_throughput *= make_const_spectrum(phase_eval / dir_pdf);

            ray.dir = *dir_out;
        }

        if (bounces >= rr_depth) {
            Real rr_prob = min(max(path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
            path_throughput /= rr_prob;
        }

        bounces++;

        if (bounces > 1000) {
            break;
        }
    }

    return radiance;
}

// The fifth volumetric renderer:
// multiple monochromatic homogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing_5(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};

    int current_medium_id = scene.camera.medium_id;
    Spectrum radiance = make_zero_spectrum();
    Spectrum path_throughput = make_const_spectrum(1);
    int bounces = 0;
    int max_depth = scene.options.max_depth;
    int rr_depth = scene.options.rr_depth;

    Real dir_pdf = 0;
    Vector3 nee_p_cache = ray.org;
    Real multi_trans_pdf = 1;
    bool never_scatter = true;

    while (true) {
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        Real t_hit = vertex_ ? distance(ray.org, vertex_->position)
                             : (Real(2) * scene.bounds.radius);

        // Glow discharge is emissive-only. Integrate its radiance along the
        // current free-flight segment while preserving the existing surface logic.
        if (current_medium_id >= 0) {
            const Medium &medium = scene.media[current_medium_id];
            if (std::holds_alternative<GlowDischargeMedium>(medium)) {
                const auto &glow_medium = std::get<GlowDischargeMedium>(medium);
                radiance += path_throughput *
                            integrate_glow_emission_segment(glow_medium, ray, t_hit, rng);
            }
        }

        bool scatter = false;
        Spectrum transmittance = make_const_spectrum(1);
        Real trans_pdf = 1;

        if (current_medium_id >= 0) {
            const Medium &medium = scene.media[current_medium_id];
            Vector3 sample_pos = vertex_ ? vertex_->position : ray.org + ray.dir * t_hit;
            Spectrum sigma_a = get_sigma_a(medium, sample_pos);
            Spectrum sigma_s = get_sigma_s(medium, sample_pos);
            Spectrum sigma_t = sigma_a + sigma_s;  // RGB extinction
            Real sigma_t_scalar = (sigma_t[0] + sigma_t[1] + sigma_t[2]) / Real(3);

            if (sigma_t_scalar > 0) {
                Real t_hit = vertex_ ? distance(ray.org, vertex_->position) : infinity<Real>();
                Real u = next_pcg32_real<Real>(rng);
                Real t_sampled = -log(Real(1) - u) / sigma_t_scalar;

                if (t_sampled < t_hit) {
                    scatter = true;
                    never_scatter = false;
                    ray.org = ray.org + t_sampled * ray.dir;
                    transmittance = Spectrum(exp(-sigma_t[0] * t_sampled),
                                            exp(-sigma_t[1] * t_sampled),
                                            exp(-sigma_t[2] * t_sampled));
                    trans_pdf = exp(-sigma_t_scalar * t_sampled) * sigma_t_scalar;
                    path_throughput *= transmittance / make_const_spectrum(trans_pdf);
                    path_throughput *= sigma_s;  // RGB scattering coefficient
                } else {
                    transmittance = Spectrum(exp(-sigma_t[0] * t_hit),
                                            exp(-sigma_t[1] * t_hit),
                                            exp(-sigma_t[2] * t_hit));
                    trans_pdf = exp(-sigma_t_scalar * t_hit);
                    path_throughput *= transmittance / make_const_spectrum(trans_pdf);
                }
            }
        }

        if (scatter) {
            const Medium &medium = scene.media[current_medium_id];
            PhaseFunction phase_function = get_phase_function(medium);

            Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            Real light_w = next_pcg32_real<Real>(rng);
            Real shape_w = next_pcg32_real<Real>(rng);
            int light_id = sample_light(scene, light_w);
            const Light &light = scene.lights[light_id];
            PointAndNormal point_on_light =
                sample_point_on_light(light, ray.org, light_uv, shape_w, scene);

            Vector3 dir_to_light = normalize(point_on_light.position - ray.org);
            Real dist_to_light = distance(point_on_light.position, ray.org);

            Spectrum T_light = make_const_spectrum(1);
            Real p_trans_nee = 1;
            Real p_trans_dir = 1;
            Vector3 shadow_pos = ray.org;
            int shadow_medium = current_medium_id;
            int shadow_bounces = 0;
            bool blocked = false;

            while (true) {
                Ray shadow_ray{shadow_pos, dir_to_light, get_shadow_epsilon(scene),
                              distance(shadow_pos, point_on_light.position) - get_shadow_epsilon(scene)};
                std::optional<PathVertex> shadow_isect = intersect(scene, shadow_ray, ray_diff);

                Real next_t = distance(shadow_pos, point_on_light.position);
                if (shadow_isect) {
                    next_t = min(next_t, distance(shadow_pos, shadow_isect->position));
                }

                if (shadow_medium >= 0) {
                    const Medium &shadow_med = scene.media[shadow_medium];
                    Spectrum shadow_sigma_a = get_sigma_a(shadow_med, shadow_pos);
                    Spectrum shadow_sigma_s = get_sigma_s(shadow_med, shadow_pos);
                    Spectrum shadow_sigma_t = shadow_sigma_a + shadow_sigma_s;  // RGB
                    Real shadow_sigma_t_scalar = (shadow_sigma_t[0] + shadow_sigma_t[1] + shadow_sigma_t[2]) / Real(3);
                    // Chromatic transmittance (per-channel)
                    T_light *= Spectrum(exp(-shadow_sigma_t[0] * next_t),
                                       exp(-shadow_sigma_t[1] * next_t),
                                       exp(-shadow_sigma_t[2] * next_t));
                    // PDF uses scalar
                    p_trans_nee *= exp(-shadow_sigma_t_scalar * next_t);
                    p_trans_dir *= exp(-shadow_sigma_t_scalar * next_t);
                }

                if (!shadow_isect || distance(shadow_pos, shadow_isect->position) >=
                    distance(shadow_pos, point_on_light.position) - get_shadow_epsilon(scene)) {
                    break;
                }

                if (shadow_isect->material_id >= 0) {
                    blocked = true;
                    break;
                }

                shadow_bounces++;
                if (max_depth != -1 && bounces + shadow_bounces + 1 >= max_depth) {
                    blocked = true;
                    break;
                }

                if (shadow_isect->interior_medium_id != shadow_isect->exterior_medium_id) {
                    if (dot(dir_to_light, shadow_isect->geometric_normal) > 0) {
                        shadow_medium = shadow_isect->exterior_medium_id;
                    } else {
                        shadow_medium = shadow_isect->interior_medium_id;
                    }
                }

                shadow_pos = shadow_isect->position + dir_to_light * get_shadow_epsilon(scene);
            }

            if (!blocked && max(T_light) > 0) {
                Real phase_eval = eval(phase_function, -ray.dir, dir_to_light)[0];
                Real G = max(-dot(dir_to_light, point_on_light.normal), Real(0)) /
                         (dist_to_light * dist_to_light);
                Spectrum Le = emission(light, -dir_to_light, Real(0), point_on_light, scene);
                Real pdf_nee = pdf_point_on_light(light, point_on_light, ray.org, scene) *
                              light_pmf(scene, light_id);
                Real pdf_phase = pdf_sample_phase(phase_function, -ray.dir, dir_to_light) * G * p_trans_dir;
                Real w_nee = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_phase * pdf_phase);
                Spectrum nee_contrib = T_light * make_const_spectrum(phase_eval * G / pdf_nee) * Le * make_const_spectrum(w_nee);
                radiance += path_throughput * nee_contrib;
            }

            nee_p_cache = ray.org;
            multi_trans_pdf = 1;
        }

        if (!scatter) {
            if (vertex_) {
                PathVertex vertex = *vertex_;

                // Add emission if we hit a light
                if (is_light(scene.shapes[vertex.shape_id])) {
                    Spectrum Le = emission(vertex, -ray.dir, scene);
                    if (never_scatter) {
                        radiance += path_throughput * Le;
                    } else {
                        Real dist = distance(nee_p_cache, vertex.position);
                        Real G = abs(dot(vertex.geometric_normal, ray.dir)) / (dist * dist);
                        Real pdf_nee = pdf_point_on_light(
                            scene.lights[get_area_light_id(scene.shapes[vertex.shape_id])],
                            PointAndNormal{vertex.position, vertex.geometric_normal},
                            nee_p_cache, scene) *
                            light_pmf(scene, get_area_light_id(scene.shapes[vertex.shape_id]));
                        Real pdf_phase = dir_pdf * multi_trans_pdf * G;
                        Real w_phase = (pdf_phase * pdf_phase) / (pdf_phase * pdf_phase + pdf_nee * pdf_nee);
                        radiance += path_throughput * Le * make_const_spectrum(w_phase);
                    }
                }

                if (max_depth != -1 && bounces >= max_depth - 1) {
                    break;
                }

                if (vertex.material_id == -1) {
                    if (current_medium_id >= 0) {
                        const Medium &medium = scene.media[current_medium_id];
                        Spectrum sigma_a = get_sigma_a(medium, vertex.position);
                        Spectrum sigma_s = get_sigma_s(medium, vertex.position);
                        Spectrum sigma_t = sigma_a + sigma_s;
                        Real sigma_t_scalar = (sigma_t[0] + sigma_t[1] + sigma_t[2]) / Real(3);
                        multi_trans_pdf *= exp(-sigma_t_scalar * distance(ray.org, vertex.position));
                    }

                    if (vertex.interior_medium_id != vertex.exterior_medium_id) {
                        if (dot(ray.dir, vertex.geometric_normal) > 0) {
                            current_medium_id = vertex.exterior_medium_id;
                        } else {
                            current_medium_id = vertex.interior_medium_id;
                        }
                    }

                    ray.org = vertex.position + ray.dir * get_shadow_epsilon(scene);
                    bounces++;
                    continue;
                }

                if (vertex.material_id >= 0) {
                    const Material &mat = scene.materials[vertex.material_id];
                    never_scatter = false;

                    // Surface Next Event Estimation
                    Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
                    Real light_w = next_pcg32_real<Real>(rng);
                    Real shape_w = next_pcg32_real<Real>(rng);
                    int light_id = sample_light(scene, light_w);
                    const Light &light = scene.lights[light_id];
                    PointAndNormal point_on_light =
                        sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);

                    Vector3 dir_to_light = normalize(point_on_light.position - vertex.position);
                    Real dist_to_light = distance(point_on_light.position, vertex.position);

                    // Trace shadow ray
                    Spectrum T_light = make_const_spectrum(1);
                    Vector3 shadow_pos = vertex.position;
                    int shadow_medium = current_medium_id;
                    bool blocked = false;

                    while (true) {
                        Ray shadow_ray{shadow_pos, dir_to_light, get_shadow_epsilon(scene),
                                      distance(shadow_pos, point_on_light.position) - get_shadow_epsilon(scene)};
                        std::optional<PathVertex> shadow_isect = intersect(scene, shadow_ray, ray_diff);

                        Real next_t = distance(shadow_pos, point_on_light.position);
                        if (shadow_isect) {
                            next_t = min(next_t, distance(shadow_pos, shadow_isect->position));
                        }

                        if (shadow_medium >= 0) {
                            const Medium &shadow_med = scene.media[shadow_medium];
                            Spectrum shadow_sigma_a = get_sigma_a(shadow_med, shadow_pos);
                            Spectrum shadow_sigma_s = get_sigma_s(shadow_med, shadow_pos);
                            Spectrum shadow_sigma_t = shadow_sigma_a + shadow_sigma_s;  // RGB
                            // Chromatic transmittance (per-channel)
                            T_light *= Spectrum(exp(-shadow_sigma_t[0] * next_t),
                                               exp(-shadow_sigma_t[1] * next_t),
                                               exp(-shadow_sigma_t[2] * next_t));
                        }

                        if (!shadow_isect || distance(shadow_pos, shadow_isect->position) >=
                            distance(shadow_pos, point_on_light.position) - get_shadow_epsilon(scene)) {
                            break;
                        }

                        if (shadow_isect->material_id >= 0) {
                            blocked = true;
                            break;
                        }

                        if (shadow_isect->interior_medium_id != shadow_isect->exterior_medium_id) {
                            if (dot(dir_to_light, shadow_isect->geometric_normal) > 0) {
                                shadow_medium = shadow_isect->exterior_medium_id;
                            } else {
                                shadow_medium = shadow_isect->interior_medium_id;
                            }
                        }

                        shadow_pos = shadow_isect->position + dir_to_light * get_shadow_epsilon(scene);
                    }

                    if (!blocked && max(T_light) > 0) {
                        Spectrum bsdf_eval = eval(mat, -ray.dir, dir_to_light, vertex, scene.texture_pool);
                        Real G = max(-dot(dir_to_light, point_on_light.normal), Real(0)) /
                                 (dist_to_light * dist_to_light);
                        Spectrum Le = emission(light, -dir_to_light, Real(0), point_on_light, scene);
                        Real pdf_nee = pdf_point_on_light(light, point_on_light, vertex.position, scene) *
                                      light_pmf(scene, light_id);
                        Spectrum nee_contrib = T_light * bsdf_eval * make_const_spectrum(G / pdf_nee) * Le;
                        radiance += path_throughput * nee_contrib;
                    }

                    Vector2 bsdf_rnd_param_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
                    Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
                    std::optional<BSDFSampleRecord> bsdf_sample =
                        sample_bsdf(mat, -ray.dir, vertex, scene.texture_pool,
                                   bsdf_rnd_param_uv, bsdf_rnd_param_w);

                    if (!bsdf_sample) {
                        break;
                    }

                    Spectrum bsdf_eval = eval(mat, -ray.dir, bsdf_sample->dir_out, vertex, scene.texture_pool);
                    Real bsdf_pdf = pdf_sample_bsdf(mat, -ray.dir, bsdf_sample->dir_out, vertex, scene.texture_pool);

                    if (bsdf_pdf <= 0) {
                        break;
                    }

                    path_throughput *= bsdf_eval / make_const_spectrum(bsdf_pdf);
                    ray.dir = bsdf_sample->dir_out;
                    ray.org = vertex.position + ray.dir * get_shadow_epsilon(scene);

                    // Update medium (if transmitting through surface)
                    if (vertex.interior_medium_id != vertex.exterior_medium_id) {
                        if (dot(ray.dir, vertex.geometric_normal) > 0) {
                            current_medium_id = vertex.exterior_medium_id;
                        } else {
                            current_medium_id = vertex.interior_medium_id;
                        }
                    }

                    dir_pdf = bsdf_pdf;
                    nee_p_cache = vertex.position;
                    multi_trans_pdf = 1;
                    bounces++;
                    continue;
                }

                break;
            } else {
                break;
            }
        }

        if (scatter) {
            const Medium &medium = scene.media[current_medium_id];
            PhaseFunction phase_function = get_phase_function(medium);

            Vector2 rnd_param{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            std::optional<Vector3> dir_out = sample_phase_function(phase_function, -ray.dir, rnd_param);

            if (!dir_out) {
                break;
            }

            Real phase_eval = eval(phase_function, -ray.dir, *dir_out)[0];
            dir_pdf = pdf_sample_phase(phase_function, -ray.dir, *dir_out);
            path_throughput *= make_const_spectrum(phase_eval / dir_pdf);
            ray.dir = *dir_out;
        }

        if (bounces >= rr_depth) {
            Real rr_prob = min(max(path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
            path_throughput /= rr_prob;
        }

        bounces++;

        if (bounces > 1000) {
            break;
        }
    }

    return radiance;
}

// #Edited (Phase 2 - Full Version)
// Function: vol_path_tracing_glow_full
// Description: Full volumetric path tracer with glow discharge support AND proper BSDF handling.
//              Combines glow emission integration with complete surface scattering, bounces, and lighting.
//              This is vol_path_tracing_5 with glow discharge already integrated at lines 574-580.
// Input:
//   - scene: Scene containing shapes, media, materials, and lights
//   - x, y: Pixel coordinates
//   - rng: Random number generator state
// Output: Spectrum - RGB radiance for this pixel
Spectrum vol_path_tracing_glow_full(const Scene &scene,
                            int x, int y, /* pixel coordinates */
                            pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};

    int current_medium_id = scene.camera.medium_id;
    Spectrum radiance = make_zero_spectrum();
    Spectrum path_throughput = make_const_spectrum(1);
    int bounces = 0;
    int max_depth = scene.options.max_depth;
    int rr_depth = scene.options.rr_depth;

    Real dir_pdf = 0;
    Vector3 nee_p_cache = ray.org;
    Real multi_trans_pdf = 1;
    bool never_scatter = true;

    while (true) {
        std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
        Real t_hit = vertex_ ? distance(ray.org, vertex_->position)
                             : (Real(2) * scene.bounds.radius);

        // Glow discharge is emissive-only. Integrate its radiance along the
        // current free-flight segment while preserving the existing surface logic.
        if (current_medium_id >= 0) {
            const Medium &medium = scene.media[current_medium_id];
            if (std::holds_alternative<GlowDischargeMedium>(medium)) {
                const auto &glow_medium = std::get<GlowDischargeMedium>(medium);
                radiance += path_throughput *
                            integrate_glow_emission_segment(glow_medium, ray, t_hit, rng);
            }
        }

        bool scatter = false;
        Spectrum transmittance = make_const_spectrum(1);
        Real trans_pdf = 1;

        if (current_medium_id >= 0) {
            const Medium &medium = scene.media[current_medium_id];
            Vector3 sample_pos = vertex_ ? vertex_->position : ray.org + ray.dir * t_hit;
            Spectrum sigma_a = get_sigma_a(medium, sample_pos);
            Spectrum sigma_s = get_sigma_s(medium, sample_pos);
            Spectrum sigma_t = sigma_a + sigma_s;  // RGB extinction
            Real sigma_t_scalar = (sigma_t[0] + sigma_t[1] + sigma_t[2]) / Real(3);

            if (sigma_t_scalar > 0) {
                Real t_hit = vertex_ ? distance(ray.org, vertex_->position) : infinity<Real>();
                Real u = next_pcg32_real<Real>(rng);
                Real t_sampled = -log(Real(1) - u) / sigma_t_scalar;

                if (t_sampled < t_hit) {
                    scatter = true;
                    never_scatter = false;
                    ray.org = ray.org + t_sampled * ray.dir;
                    transmittance = Spectrum(exp(-sigma_t[0] * t_sampled),
                                            exp(-sigma_t[1] * t_sampled),
                                            exp(-sigma_t[2] * t_sampled));
                    trans_pdf = exp(-sigma_t_scalar * t_sampled) * sigma_t_scalar;
                    path_throughput *= transmittance / make_const_spectrum(trans_pdf);
                    path_throughput *= sigma_s;  // RGB scattering coefficient
                } else {
                    transmittance = Spectrum(exp(-sigma_t[0] * t_hit),
                                            exp(-sigma_t[1] * t_hit),
                                            exp(-sigma_t[2] * t_hit));
                    trans_pdf = exp(-sigma_t_scalar * t_hit);
                    path_throughput *= transmittance / make_const_spectrum(trans_pdf);
                }
            }
        }

        if (scatter) {
            const Medium &medium = scene.media[current_medium_id];
            PhaseFunction phase_function = get_phase_function(medium);

            Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            Real light_w = next_pcg32_real<Real>(rng);
            Real shape_w = next_pcg32_real<Real>(rng);
            int light_id = sample_light(scene, light_w);
            const Light &light = scene.lights[light_id];
            PointAndNormal point_on_light =
                sample_point_on_light(light, ray.org, light_uv, shape_w, scene);

            Vector3 dir_to_light = normalize(point_on_light.position - ray.org);
            Real dist_to_light = distance(point_on_light.position, ray.org);

            Spectrum T_light = make_const_spectrum(1);
            Real p_trans_nee = 1;
            Real p_trans_dir = 1;
            Vector3 shadow_pos = ray.org;
            int shadow_medium = current_medium_id;
            int shadow_bounces = 0;
            bool blocked = false;

            while (true) {
                Ray shadow_ray{shadow_pos, dir_to_light, get_shadow_epsilon(scene),
                              distance(shadow_pos, point_on_light.position) - get_shadow_epsilon(scene)};
                std::optional<PathVertex> shadow_isect = intersect(scene, shadow_ray, ray_diff);

                Real next_t = distance(shadow_pos, point_on_light.position);
                if (shadow_isect) {
                    next_t = min(next_t, distance(shadow_pos, shadow_isect->position));
                }

                if (shadow_medium >= 0) {
                    const Medium &shadow_med = scene.media[shadow_medium];
                    Spectrum shadow_sigma_a = get_sigma_a(shadow_med, shadow_pos);
                    Spectrum shadow_sigma_s = get_sigma_s(shadow_med, shadow_pos);
                    Spectrum shadow_sigma_t = shadow_sigma_a + shadow_sigma_s;  // RGB
                    Real shadow_sigma_t_scalar = (shadow_sigma_t[0] + shadow_sigma_t[1] + shadow_sigma_t[2]) / Real(3);
                    // Chromatic transmittance (per-channel)
                    T_light *= Spectrum(exp(-shadow_sigma_t[0] * next_t),
                                       exp(-shadow_sigma_t[1] * next_t),
                                       exp(-shadow_sigma_t[2] * next_t));
                    // PDF uses scalar
                    p_trans_nee *= exp(-shadow_sigma_t_scalar * next_t);
                    p_trans_dir *= exp(-shadow_sigma_t_scalar * next_t);
                }

                if (!shadow_isect || distance(shadow_pos, shadow_isect->position) >=
                    distance(shadow_pos, point_on_light.position) - get_shadow_epsilon(scene)) {
                    break;
                }

                if (shadow_isect->material_id >= 0) {
                    blocked = true;
                    break;
                }

                shadow_bounces++;
                if (max_depth != -1 && bounces + shadow_bounces + 1 >= max_depth) {
                    blocked = true;
                    break;
                }

                if (shadow_isect->interior_medium_id != shadow_isect->exterior_medium_id) {
                    if (dot(dir_to_light, shadow_isect->geometric_normal) > 0) {
                        shadow_medium = shadow_isect->exterior_medium_id;
                    } else {
                        shadow_medium = shadow_isect->interior_medium_id;
                    }
                }

                shadow_pos = shadow_isect->position + dir_to_light * get_shadow_epsilon(scene);
            }

            if (!blocked && max(T_light) > 0) {
                Real phase_eval = eval(phase_function, -ray.dir, dir_to_light)[0];
                Real G = max(-dot(dir_to_light, point_on_light.normal), Real(0)) /
                         (dist_to_light * dist_to_light);
                Spectrum Le = emission(light, -dir_to_light, Real(0), point_on_light, scene);
                Real pdf_nee = pdf_point_on_light(light, point_on_light, ray.org, scene) *
                              light_pmf(scene, light_id);
                Real pdf_phase = pdf_sample_phase(phase_function, -ray.dir, dir_to_light) * G * p_trans_dir;
                Real w_nee = (pdf_nee * pdf_nee) / (pdf_nee * pdf_nee + pdf_phase * pdf_phase);
                Spectrum nee_contrib = T_light * make_const_spectrum(phase_eval * G / pdf_nee) * Le * make_const_spectrum(w_nee);
                radiance += path_throughput * nee_contrib;
            }

            nee_p_cache = ray.org;
            multi_trans_pdf = 1;
        }

        if (!scatter) {
            if (vertex_) {
                PathVertex vertex = *vertex_;

                // Add emission if we hit a light
                if (is_light(scene.shapes[vertex.shape_id])) {
                    Spectrum Le = emission(vertex, -ray.dir, scene);
                    if (never_scatter) {
                        radiance += path_throughput * Le;
                    } else {
                        Real dist = distance(nee_p_cache, vertex.position);
                        Real G = abs(dot(vertex.geometric_normal, ray.dir)) / (dist * dist);
                        Real pdf_nee = pdf_point_on_light(
                            scene.lights[get_area_light_id(scene.shapes[vertex.shape_id])],
                            PointAndNormal{vertex.position, vertex.geometric_normal},
                            nee_p_cache, scene) *
                            light_pmf(scene, get_area_light_id(scene.shapes[vertex.shape_id]));
                        Real pdf_phase = dir_pdf * multi_trans_pdf * G;
                        Real w_phase = (pdf_phase * pdf_phase) / (pdf_phase * pdf_phase + pdf_nee * pdf_nee);
                        radiance += path_throughput * Le * make_const_spectrum(w_phase);
                    }
                }

                if (max_depth != -1 && bounces >= max_depth - 1) {
                    break;
                }

                if (vertex.material_id == -1) {
                    if (current_medium_id >= 0) {
                        const Medium &medium = scene.media[current_medium_id];
                        Spectrum sigma_a = get_sigma_a(medium, vertex.position);
                        Spectrum sigma_s = get_sigma_s(medium, vertex.position);
                        Spectrum sigma_t = sigma_a + sigma_s;
                        Real sigma_t_scalar = (sigma_t[0] + sigma_t[1] + sigma_t[2]) / Real(3);
                        multi_trans_pdf *= exp(-sigma_t_scalar * distance(ray.org, vertex.position));
                    }

                    if (vertex.interior_medium_id != vertex.exterior_medium_id) {
                        if (dot(ray.dir, vertex.geometric_normal) > 0) {
                            current_medium_id = vertex.exterior_medium_id;
                        } else {
                            current_medium_id = vertex.interior_medium_id;
                        }
                    }

                    ray.org = vertex.position + ray.dir * get_shadow_epsilon(scene);
                    bounces++;
                    continue;
                }

                if (vertex.material_id >= 0) {
                    const Material &mat = scene.materials[vertex.material_id];
                    never_scatter = false;

                    // Surface Next Event Estimation
                    Vector2 light_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
                    Real light_w = next_pcg32_real<Real>(rng);
                    Real shape_w = next_pcg32_real<Real>(rng);
                    int light_id = sample_light(scene, light_w);
                    const Light &light = scene.lights[light_id];
                    PointAndNormal point_on_light =
                        sample_point_on_light(light, vertex.position, light_uv, shape_w, scene);

                    Vector3 dir_to_light = normalize(point_on_light.position - vertex.position);
                    Real dist_to_light = distance(point_on_light.position, vertex.position);

                    // Trace shadow ray
                    Spectrum T_light = make_const_spectrum(1);
                    Vector3 shadow_pos = vertex.position;
                    int shadow_medium = current_medium_id;
                    bool blocked = false;

                    while (true) {
                        Ray shadow_ray{shadow_pos, dir_to_light, get_shadow_epsilon(scene),
                                      distance(shadow_pos, point_on_light.position) - get_shadow_epsilon(scene)};
                        std::optional<PathVertex> shadow_isect = intersect(scene, shadow_ray, ray_diff);

                        Real next_t = distance(shadow_pos, point_on_light.position);
                        if (shadow_isect) {
                            next_t = min(next_t, distance(shadow_pos, shadow_isect->position));
                        }

                        if (shadow_medium >= 0) {
                            const Medium &shadow_med = scene.media[shadow_medium];
                            Spectrum shadow_sigma_a = get_sigma_a(shadow_med, shadow_pos);
                            Spectrum shadow_sigma_s = get_sigma_s(shadow_med, shadow_pos);
                            Spectrum shadow_sigma_t = shadow_sigma_a + shadow_sigma_s;  // RGB
                            // Chromatic transmittance (per-channel)
                            T_light *= Spectrum(exp(-shadow_sigma_t[0] * next_t),
                                               exp(-shadow_sigma_t[1] * next_t),
                                               exp(-shadow_sigma_t[2] * next_t));
                        }

                        if (!shadow_isect || distance(shadow_pos, shadow_isect->position) >=
                            distance(shadow_pos, point_on_light.position) - get_shadow_epsilon(scene)) {
                            break;
                        }

                        if (shadow_isect->material_id >= 0) {
                            blocked = true;
                            break;
                        }

                        if (shadow_isect->interior_medium_id != shadow_isect->exterior_medium_id) {
                            if (dot(dir_to_light, shadow_isect->geometric_normal) > 0) {
                                shadow_medium = shadow_isect->exterior_medium_id;
                            } else {
                                shadow_medium = shadow_isect->interior_medium_id;
                            }
                        }

                        shadow_pos = shadow_isect->position + dir_to_light * get_shadow_epsilon(scene);
                    }

                    if (!blocked && max(T_light) > 0) {
                        Spectrum bsdf_eval = eval(mat, -ray.dir, dir_to_light, vertex, scene.texture_pool);
                        Real G = max(-dot(dir_to_light, point_on_light.normal), Real(0)) /
                                 (dist_to_light * dist_to_light);
                        Spectrum Le = emission(light, -dir_to_light, Real(0), point_on_light, scene);
                        Real pdf_nee = pdf_point_on_light(light, point_on_light, vertex.position, scene) *
                                      light_pmf(scene, light_id);
                        Spectrum nee_contrib = T_light * bsdf_eval * make_const_spectrum(G / pdf_nee) * Le;
                        radiance += path_throughput * nee_contrib;
                    }

                    Vector2 bsdf_rnd_param_uv{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
                    Real bsdf_rnd_param_w = next_pcg32_real<Real>(rng);
                    std::optional<BSDFSampleRecord> bsdf_sample =
                        sample_bsdf(mat, -ray.dir, vertex, scene.texture_pool,
                                   bsdf_rnd_param_uv, bsdf_rnd_param_w);

                    if (!bsdf_sample) {
                        break;
                    }

                    Spectrum bsdf_eval = eval(mat, -ray.dir, bsdf_sample->dir_out, vertex, scene.texture_pool);
                    Real bsdf_pdf = pdf_sample_bsdf(mat, -ray.dir, bsdf_sample->dir_out, vertex, scene.texture_pool);

                    if (bsdf_pdf <= 0) {
                        break;
                    }

                    path_throughput *= bsdf_eval / make_const_spectrum(bsdf_pdf);
                    ray.dir = bsdf_sample->dir_out;
                    ray.org = vertex.position + ray.dir * get_shadow_epsilon(scene);

                    // Update medium (if transmitting through surface)
                    if (vertex.interior_medium_id != vertex.exterior_medium_id) {
                        if (dot(ray.dir, vertex.geometric_normal) > 0) {
                            current_medium_id = vertex.exterior_medium_id;
                        } else {
                            current_medium_id = vertex.interior_medium_id;
                        }
                    }

                    dir_pdf = bsdf_pdf;
                    nee_p_cache = vertex.position;
                    multi_trans_pdf = 1;
                    bounces++;
                    continue;
                }

                break;
            } else {
                break;
            }
        }

        if (scatter) {
            const Medium &medium = scene.media[current_medium_id];
            PhaseFunction phase_function = get_phase_function(medium);

            Vector2 rnd_param{next_pcg32_real<Real>(rng), next_pcg32_real<Real>(rng)};
            std::optional<Vector3> dir_out = sample_phase_function(phase_function, -ray.dir, rnd_param);

            if (!dir_out) {
                break;
            }

            Real phase_eval = eval(phase_function, -ray.dir, *dir_out)[0];
            dir_pdf = pdf_sample_phase(phase_function, -ray.dir, *dir_out);
            path_throughput *= make_const_spectrum(phase_eval / dir_pdf);
            ray.dir = *dir_out;
        }

        if (bounces >= rr_depth) {
            Real rr_prob = min(max(path_throughput), Real(0.95));
            if (next_pcg32_real<Real>(rng) > rr_prob) {
                break;
            }
            path_throughput /= rr_prob;
        }

        bounces++;

        if (bounces > 1000) {
            break;
        }
    }

    return radiance;
}

// Glow discharge volumetric renderer (milestone)
// Simplified renderer for pure emissive glow discharge volumes
// #Edited (Phase 1)
// Function: vol_path_tracing_glow
// Description: Specialized volumetric path tracer for glow discharge rendering.
//              Renders emissive volumes by integrating emission along camera rays.
//              Limitation: Only handles direct emission, no scattering or surface lighting.
// Input:
//   - scene: Scene containing shapes, media, and lights
//   - x, y: Pixel coordinates
//   - rng: Random number generator state
// Output: Spectrum - RGB radiance for this pixel
Spectrum vol_path_tracing_glow(const Scene &scene,
                               int x, int y,
                               pcg32_state &rng) {
    int w = scene.camera.width, h = scene.camera.height;
    Vector2 screen_pos((x + next_pcg32_real<Real>(rng)) / w,
                       (y + next_pcg32_real<Real>(rng)) / h);
    Ray ray = sample_primary(scene.camera, screen_pos);
    RayDifferential ray_diff = RayDifferential{Real(0), Real(0)};

    std::optional<PathVertex> vertex_ = intersect(scene, ray, ray_diff);
    Spectrum radiance = make_zero_spectrum();

    // Check if ray hit a shape with interior medium
    if (vertex_) {
        const Shape &shape = scene.shapes[vertex_->shape_id];
        int interior_medium_id = std::visit([](const auto &s) { return s.interior_medium_id; }, shape);

        if (interior_medium_id >= 0) {
            const Medium &medium = scene.media[interior_medium_id];

            // Check if it's a glow discharge medium
            if (std::holds_alternative<GlowDischargeMedium>(medium)) {
                const auto &glow_medium = std::get<GlowDischargeMedium>(medium);

                // Sample along the ray from camera to surface
                Real t_max = distance(ray.org, vertex_->position);

                // Sample multiple points along the ray and accumulate emission
                int num_samples = 32; // Number of samples along the ray
                for (int i = 0; i < num_samples; i++) {
                    // Stratified sampling along the ray
                    Real t = (Real(i) + next_pcg32_real<Real>(rng)) / Real(num_samples) * t_max;
                    Vector3 sample_point = ray.org + t * ray.dir;

                    // Evaluate emission at this point
                    Spectrum emission = get_glow_emission(glow_medium, sample_point);

                    // Accumulate emission (Monte Carlo estimate of integral)
                    // L_o = ∫ L_e(x) dt ≈ (1/N) * Σ L_e(x_i) * t_max
                    radiance += emission * t_max / Real(num_samples);
                }
            }
        }

        // Also include surface emission if we hit a light
        if (is_light(scene.shapes[vertex_->shape_id])) {
            radiance += emission(*vertex_, -ray.dir, scene);
        }
    }

    return radiance;
}

// The final volumetric renderer:
// multiple chromatic heterogeneous volumes with multiple scattering
// with MIS between next event estimation and phase function sampling
// with surface lighting
Spectrum vol_path_tracing(const Scene &scene,
                          int x, int y, /* pixel coordinates */
                          pcg32_state &rng) {
    // Homework 2: implememt this!
    return make_zero_spectrum();
}
