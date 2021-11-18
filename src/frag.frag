#version 330 core

in vec3 position;
in vec3 normal;

out vec4 fragColor;

uniform mat4 modelview;

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
    sampler2D shadows[max_lights];
} lights;

uniform bool enable_lighting;

vec3 homosub(vec4 p, vec4 q) {
    return p.xyz * q.w - q.xyz * p.w;
}

#define CASE(N) case N: \
    return texture(lights.shadows[N], uv).x;

// TODO: handle this, probably with texture2DArray
float shadow_texture(vec2 uv, int idx) {
    switch (idx) {
        CASE(0)
        CASE(1)
        CASE(2)
        CASE(3)
        CASE(4)
        CASE(5)
        CASE(6)
        CASE(7)
        CASE(8)
        CASE(9)
        default:
            return 1.0;
    }
}

void main (void) {
    if (!enable_lighting) {
        fragColor = vec4(0.5f * normalize(normal) + 0.5f, 1.0f);
        return;
    }

    fragColor = phong.emission;


    for (int i = 0; i < lights.len; i++) {
        float shadow = 1.0;
        vec4 lightspace = lights.lightspaces[i] * vec4(position, 1.0);
        // sampler2DArray depth = lights.shadows[i];
        if (shadow_texture(lightspace.xy, i) < lightspace.z) {
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
