#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
} ubo;

// The per-vertex data
// NOTE: names must match the `Vertex` struct in Rust
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

// The per-instance data
layout(location = 3) in mat4 model_matrix;

layout(location = 0) out vec3 fragColor;

out gl_PerVertex {
    vec4 gl_Position;
};

void main() {
    gl_Position = ubo.proj * ubo.view * model_matrix * vec4(position, 1.0);
    fragColor = (normal);
}
