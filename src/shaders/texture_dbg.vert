# version 330 core

layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 vertex_normal;

out vec2 uv;

void main() {
    gl_Position = vec4(vertex_position, 1.0);
    uv = 1.5 * vertex_position.xy + vec2(0.5f);
}
