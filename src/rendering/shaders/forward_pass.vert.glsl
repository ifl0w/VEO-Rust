#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_EXT_nonuniform_qualifier : enable

layout(constant_id = 0) const bool use_instancing = false;

// Uniforms
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec4 position;
} camera_ubo;

layout(binding = 1) buffer InstanceDataBuffer {
    mat4 model_matrix[];
} instance_ssbo;

layout(push_constant) uniform PushConsts {
    mat4 model_matrix;
    int instance_data_offset;
} pushConsts;

// The per-vertex data
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragPosition;

void main() {
    mat4 model_matrix;
    if (use_instancing) {
        model_matrix = pushConsts.model_matrix * instance_ssbo.model_matrix[pushConsts.instance_data_offset + gl_InstanceIndex];
    } else {
        model_matrix = pushConsts.model_matrix;
    }

    vec4 worldCoords = model_matrix * vec4(position, 1.0);
    gl_Position = camera_ubo.proj * camera_ubo.view * worldCoords;

    fragColor = normal;
    fragNormal = normal;
    fragPosition = worldCoords.xyz;
}
