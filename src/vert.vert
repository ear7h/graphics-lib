# version 330 core

layout (location = 0) in vec3 vertex_position;
layout (location = 1) in vec3 vertex_normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

out vec3 position;
out vec3 normal;

void main() {
    gl_Position = projection * view * model * vec4(vertex_position, 1.0f);

    vec4 p = model * vec4(vertex_position, 1.0f);
    position = p.xyz / p.w;
    normal = normalize(
        (
            transpose(inverse(model)) *
            vec4(vertex_normal, 0.0)
        ).xyz
    );
}
