// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x43fac (spirv, Fragment).
// Programs: hydra_reproject_match
// Role: Compare reprojected frames and produce matching evidence.
// Semantic names are reconstruction aids, not recovered author identifiers.
// Generic type/slot names deliberately retain uncertainty. See manifest.json for renames.
// Embedded source templates still require their original runtime defines.
#version 450
#if defined(GL_AMD_gpu_shader_half_float)
#extension GL_AMD_gpu_shader_half_float : require
#elif defined(GL_EXT_shader_explicit_arithmetic_types_float16)
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#else
#error No extension available for FP16.
#endif
#extension GL_EXT_shader_16bit_storage : require

layout(constant_id = 2) const float specialization2 = 0.02500000037252902984619140625;
layout(constant_id = 3) const float specialization3 = 0.5;
const float16_t float16_tValue154 = float16_t(specialization3);

layout(push_constant, std430) uniform symbolX46X48 {
    float memberFloat0;
    float memberFloat1;
}
pushConstants;

layout(set = 0, binding = 2) uniform texture2D textureSet0Binding2;
layout(set = 0, binding = 1) uniform sampler samplerSet0Binding1;
layout(set = 0, binding = 3) uniform texture2D textureSet0Binding3;
layout(set = 1, binding = 0) uniform texture2D textureSet1Binding0;
layout(set = 0, binding = 0) uniform sampler samplerSet0Binding0;
layout(set = 1, binding = 1) uniform texture2D textureSet1Binding1;

layout(location = 0) in vec2 inputLocation0;
layout(location = 0) out float outputLocation0;

void main() {
    vec4 sampledValue55 = texture(
        sampler2D(textureSet1Binding0, samplerSet0Binding0),
        (inputLocation0 + texture(sampler2D(textureSet0Binding2, samplerSet0Binding1), inputLocation0).xy) *
            pushConstants.memberFloat1);
    float floatValue58 = sampledValue55.x;
    vec4 sampledValue69 = texture(
        sampler2D(textureSet1Binding1, samplerSet0Binding0),
        (inputLocation0 + texture(sampler2D(textureSet0Binding3, samplerSet0Binding1), inputLocation0).xy) *
            pushConstants.memberFloat0);
    float floatValue70 = sampledValue69.x;
    float16_t float16_tValue78 = float16_t(floatValue58 + dFdx(floatValue58));
    float16_t float16_tValue84 = float16_t(floatValue58 + dFdy(floatValue58));
    float16_t float16_tValue90 = float16_t(floatValue70 + dFdx(floatValue70));
    float16_t float16_tValue96 = float16_t(floatValue70 + dFdy(floatValue70));
    float16_t float16_tValue99 = float16_t(floatValue58);
    float16_t float16_tValue102 = float16_t(floatValue70);
    float floatValue130 = 1.0 / specialization2;
    outputLocation0 =
        float((min(max(float16_tValue99 - max(float16_tValue102, max(float16_tValue90, float16_tValue96)),
                       min(float16_tValue102, min(float16_tValue90, float16_tValue96)) - float16_tValue99),
                   max(float16_tValue102 - max(float16_tValue99, max(float16_tValue78, float16_tValue84)),
                       min(float16_tValue99, min(float16_tValue78, float16_tValue84)) - float16_tValue102)) *
               float16_t(floatValue130)) *
              float16_tValue154);
}
