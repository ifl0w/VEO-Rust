#version 450
#extension GL_ARB_separate_shader_objects : enable

// Uniforms
layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec4 position;
} camera_ubo;

layout(push_constant) uniform PushConsts {
    mat4 model_matrix;
} pushConsts;

//layout(location = 1) uniform ModelMatrix {
//    mat4 model_matrix;
//};

// The per-vertex data
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

// The per-instance data
//layout(location = 3) in mat4 instance_model_matrix;

/*
layout(location = 0) out FragmentData {
    vec3 color;
    vec3 normal;
    vec3 position;
} frag;*/

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec3 fragPosition;


//layout(location = 0) out fragment_data frag;

void main() {
    vec4 worldCoords = pushConsts.model_matrix * vec4(position, 1.0);
    //    vec4 worldCoords = vec4(position + vec3(0,0,-10), 1.0);
    //    gl_Position = vec4(position.xy, 0.0, 1.0); //camera_ubo.proj * camera_ubo.view * worldCoords;
    gl_Position = camera_ubo.proj * camera_ubo.view * worldCoords;

    /*
    frag.color = normal;
    frag.normal = normal;
    frag.position = gl_Position.xyz;
    */
    fragColor = normal;
    fragNormal = normal;
    fragPosition = worldCoords.xyz;
}
