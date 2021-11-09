# version 330 core

layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 vertex_normal;

uniform mat4 lightspace;

void main() {
    gl_Position = lightspace * vec4(vertex_position, 1.0f);
}
