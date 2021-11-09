#version 330 core

in vec2 uv;
out vec4 fragColor;

uniform sampler2D tex;
uniform int idx;

void main() {
    // fragColor = vec4(uv, 1.0, 1.0);
	float f = 0.0f;

    // for (int i = 0; i < 10; i++) {
		//f = max(f, texture(tex, vec3(uv, float(i))).z);
		f = texture(tex, uv).z;
	// }

    fragColor = vec4(f, f, f, 1.0);
}
