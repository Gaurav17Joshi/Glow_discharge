Spectrum get_majorant_op::operator()(const GlowDischargeMedium &m) {
    // Glow discharge is pure emission: sigma_t = sigma_a + sigma_s = 0
    return make_zero_spectrum();
}

Spectrum get_sigma_s_op::operator()(const GlowDischargeMedium &m) {
    // No scattering in glow discharge
    return make_zero_spectrum();
}

Spectrum get_sigma_a_op::operator()(const GlowDischargeMedium &m) {
    // No absorption in glow discharge
    return make_zero_spectrum();
}
