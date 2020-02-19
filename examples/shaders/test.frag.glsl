#version 450
#extension GL_ARB_separate_shader_objects : enable

#define PI 3.1415926535897932384626433832795

layout(binding = 0) uniform UniformBufferObject {
    mat4 view;
    mat4 proj;
    vec4 position;
} camera_ubo;

/*
layout(location = 0) in fragment_data {
    vec3 color;
    vec3 normal;
    vec3 position;
} frag;
*/
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec3 fragPosition;

layout(location = 0) out vec4 outColor;

struct Light {
    vec3 color;
    int type; // 0 = directional, 1 = point, 2 = spotlight
    vec3 direction;
};

struct Material {
    vec3 albedo;
    float metallic;
    float roughness;
};

vec3 fresnelSchlick(float cosTheta, vec3 f0)
{
    return f0 + (1.0 - f0) * pow(max(1.0 - cosTheta, 0), 5.0);
}

float chiGGX(float v)
{
    return v > 0 ? 1 : 0;
}

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    roughness = clamp(roughness, 0.025, 1.0);
    float a = roughness*roughness;
    float a2 = a*a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH*NdotH;

    float num = chiGGX(NdotH) * a2;
    float denom = (NdotH2 * (a2 - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float DistributionBeckmann(vec3 N, vec3 H, float roughness)
{
    float m2 = max(roughness * roughness, 0.0001);
    float NdotH = max(dot(N, H), 0.0001);
    float cos2Alpha = NdotH*NdotH;
    float tan2Alpha = (1 - cos2Alpha) / cos2Alpha;

    float num = exp(-tan2Alpha/m2);
    float denom = PI * m2 * cos2Alpha * cos2Alpha;

    return num/denom;
}

float geomAttinuationWikipedia(vec3 H, vec3 N, vec3 V, vec3 L) {
    float VdotH = max(dot(V,H), 0.0001);
    float HdotN = max(dot(H,N), 0);
    float VdotN = max(dot(V,N), 0);
    float LdotN = max(dot(L,N), 0);
    float t1 = 2*(HdotN*VdotN) / VdotH;
    float t2 = 2*(HdotN*LdotN) / VdotH;
    return min(1, min(t1,t2));
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float num   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return num / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx2  = GeometrySchlickGGX(NdotV, roughness);
    float ggx1  = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

vec3 evaluateLight(Light light, Material mat, vec3 position, vec3 normal) {
    vec3 toLight = normalize(-light.direction.xyz);
    vec3 toCam = normalize(camera_ubo.position.xyz - position);

    float attenuation = 1;
    float intensityFactor = 1;

    vec3 halfVec = normalize(toCam + toLight);

    vec3 f0 = vec3(0.04);
    f0 = mix(f0, mat.albedo, mat.metallic);
    vec3 f = fresnelSchlick(max(dot(halfVec, toCam), 0.0), f0);

    float NDF = DistributionGGX(normal, halfVec, mat.roughness);
    //	float NDF = DistributionBeckmann(norm, halfVec, roughness);
    float G = GeometrySmith(normal, toCam, toLight, mat.roughness);
    //	float G = geomAttinuationWikipedia(halfVec, norm, toCam, toLight);

    vec3 numerator = NDF * G * f;
    float denominator = 4.0 * max(dot(toCam, normal), 0.0) * max(dot(normal, toLight), 0.0);
    vec3 specular = numerator / max(denominator, 0.0001);

    vec3 kD = vec3(1.0) - specular;
    kD *= 1.0 - mat.metallic;

    vec3 radiance = (light.color.xyz * 1 / attenuation) * intensityFactor;

    float NdotL = max(dot(normal, toLight), 0.0);

    vec3 color = (kD * mat.albedo / PI + specular) * radiance * NdotL;

    return color;
}

void main() {
    Light sun;
    sun.color = vec3(1,1,0.8);
    sun.type = 0;
    sun.direction = vec3(-1, -1, -1);

    Material defaultMat;
    defaultMat.albedo = vec3(0.7, 0.7, 0.7);
    defaultMat.metallic = 0;
    defaultMat.roughness = 0.4;

    vec3 shadedColor = evaluateLight(sun, defaultMat, fragPosition, normalize(fragNormal));
    outColor = vec4(shadedColor, 1.0);
    //outColor = vec4(fragColor, 1.0);
}
