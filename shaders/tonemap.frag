#version 450
layout(binding = 0) uniform sampler2D hdr_image;
layout(location = 0) in  vec2 uv;
layout(location = 0) out vec4 out_color;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main() {
    vec3 hdr = texture(hdr_image, vec2(uv.x, 1.0 - uv.y)).rgb;
    vec3 ldr = aces(hdr);
    out_color = vec4(pow(ldr, vec3(1.0 / 2.2)), 1.0);
}
