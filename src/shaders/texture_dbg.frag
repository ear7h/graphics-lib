#version 330 core

in vec2 uv;

out vec4 fragColor;

uniform sampler2DArray tex;
uniform uint idx;
uniform float near;
uniform float far;

float linearize(float z) {
	z = z * 2.0 - 1.0;
	return (2.0 * near) / (far + near - z * (far - near));
}

void main() {
    float depth = texture(tex, vec3(uv, float(idx))).x;

    fragColor = vec4(vec3(linearize(depth)), 1.0f);
}
