#version 330 core

in vec4 position; // raw position in the model coord
in vec3 normal;   // raw normal in the model coord

uniform mat4 modelview; // from model coord to eye coord
uniform mat4 view;      // from world coord to eye coord

// Material parameters
uniform vec4 ambient;
uniform vec4 diffuse;
uniform vec4 specular;
uniform vec4 emision;
uniform float shininess;

// Light source parameters
const int maximal_allowed_lights = 10;
uniform bool enablelighting;
uniform int nlights;
uniform vec4 lightpositions[ maximal_allowed_lights ];
uniform vec4 lightcolors[ maximal_allowed_lights ];

// Output the frag color
out vec4 fragColor;

vec3 normaldiff(vec4 q, vec4 p) {
    return normalize(p.w * q.xyz - q.w * p.xyz);
}

void main (void){
    if (!enablelighting){
        // Default normal coloring (you don't need to modify anything here)
        vec3 N = normalize(normal);
        fragColor = vec4(0.5f*N + 0.5f , 1.0f);
    } else {

        // HW3: You will compute the lighting here.
        fragColor = emision;

        for (int i = 0; i < nlights; i++) {
            vec3 n = normalize((transpose(inverse(modelview)) * vec4(normal, 1.0)).xyz);
            vec3 l = normaldiff(modelview * lightpositions[i], modelview * position);
            vec3 h = normalize(vec3(0.0, 0.0, 1.0) + l);

            vec4 curColor = ambient
                + diffuse * max(dot(n, l), 0.0)
                + specular * pow(max(dot(n, h), 0.0), shininess);

            fragColor += curColor * lightcolors[i];
        }
    }
}
