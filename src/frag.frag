#version 330 core

in vec3 position;
in vec3 normal;

out vec4 fragColor;

uniform struct {
    vec4 ambient;
    vec4 diffuse;
    vec4 specular;
    vec4 emission;
    float shininess;
} phong;

const int max_lights = 10;

uniform struct {
    int len;
    vec4 positions[max_lights];
    vec4 colors[max_lights];
    mat4 lightspaces[max_lights];
    bool has_shadow[max_lights];
    sampler2DArray shadows;
} lights;

uniform bool enable_lighting;

vec3 homosub(vec4 p, vec4 q) {
    return p.xyz * q.w - q.xyz * p.w;
}

vec3 dehomo(vec4 p) {
    return p.xyz / p.w;
}

float shadow_texture(vec2 uv, int idx) {
    return texture(
        lights.shadows,
        vec3(uv, float(idx))
    ).x;
}

void main (void) {
    if (!enable_lighting) {
        fragColor = vec4(0.5f * normalize(normal) + 0.5f, 1.0f);
        return;
    }

    fragColor = phong.emission;

    int si = 0;
    for (int i = 0; i < lights.len; i++) {
        float shadow = 1.0;

        vec3 lightspace = dehomo(
            lights.lightspaces[i] *
            vec4(position, 1.0)
        ) * 0.5 + vec3(0.5);

        if (lights.has_shadow[i] &&
            shadow_texture(lightspace.xy, si++) < lightspace.z - 0.005f) {

            shadow = 0.0;
        }

        vec3 n = normal;
        vec3 l = normalize(homosub(
            lights.positions[i],
            vec4(position, 1.0f)
        ));
        vec3 h = normalize(vec3(0.0, 0.0, 1.0) + l);

        vec4 reflection = phong.ambient +
            shadow * phong.diffuse * max(dot(n, l), 0.0) +
            shadow * phong.specular * pow(max(dot(n, h), 0.0), phong.shininess);

        fragColor += reflection * lights.colors[i];
    }
}

