// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x4c5e0 (spirv, Fragment).
// Programs: hydra_match_txt(false)_dmatch(none)
// Role: Evaluate motion-compensated luminance/depth agreement and frame-usage masks.
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
#extension GL_EXT_samplerless_texture_functions : require

layout(constant_id = 0) const float specialization0 = 0.125;
const float16_t float16_tValue36 = float16_t(specialization0);
layout(constant_id = 2) const float specialization2 = 40.0;
const float16_t float16_tValue186 = float16_t(specialization2);
layout(constant_id = 5) const float specialization5 = 0.5;

layout(push_constant, std430) uniform symbolX54X56 {
    float memberFloat0;
    float memberFloat1;
    mat4 memberMat42;
    mat4 memberMat43;
}
pushConstants;

layout(set = 0, binding = 2) uniform texture2D textureSet0Binding2;
layout(set = 1, binding = 0) uniform texture2D textureSet1Binding0;
layout(set = 0, binding = 0) uniform sampler samplerSet0Binding0;
layout(set = 1, binding = 1) uniform texture2D textureSet1Binding1;
layout(set = 0, binding = 3, rgba8) uniform restrict writeonly image2D storageImageSet0Binding3;
layout(set = 0, binding = 4, rgba8) uniform restrict writeonly image2D storageImageSet0Binding4;

layout(location = 0) in vec2 inputLocation0;
layout(location = 0) out float outputLocation0;

void main() {
    f16vec2 f16vec2Value24 = f16vec2(textureSize(textureSet1Binding0, 0));
    f16vec2 f16vec2Value32 = f16vec2(inputLocation0) * f16vec2Value24;
    vec4 sampledValue61 =
        texture(sampler2D(textureSet0Binding2, samplerSet0Binding0),
                vec2((f16vec2Value32 * float16_tValue36) / f16vec2(textureSize(textureSet0Binding2, 0))) *
                    pushConstants.memberFloat0);
    f16vec3 f16vec3Value64 = f16vec3(sampledValue61.xyz);
    f16vec2 f16vec2Value68 = f16vec3Value64.xy;
    f16vec2 f16vec2Value69 = f16vec2Value32 + f16vec2Value68;
    f16vec2 f16vec2Value74 = f16vec2Value32 - f16vec2Value68;
    f16vec2 f16vec2Value79 = f16vec2(float16_t(1.0)) / f16vec2Value24;
    vec4 sampledValue99 = texture(sampler2D(textureSet1Binding0, samplerSet0Binding0),
                                  vec2(f16vec2Value79 * f16vec2Value74) * pushConstants.memberFloat1);
    float floatValue102 = sampledValue99.x;
    vec4 sampledValue113 = texture(sampler2D(textureSet1Binding1, samplerSet0Binding0),
                                   vec2(f16vec2Value79 * f16vec2Value69) * pushConstants.memberFloat0);
    float floatValue114 = sampledValue113.x;
    float16_t float16_tValue121 = float16_t(floatValue102 + dFdx(floatValue102));
    float16_t float16_tValue127 = float16_t(floatValue102 + dFdy(floatValue102));
    float16_t float16_tValue133 = float16_t(floatValue114 + dFdx(floatValue114));
    float16_t float16_tValue139 = float16_t(floatValue114 + dFdy(floatValue114));
    float16_t float16_tValue142 = float16_t(floatValue102);
    float16_t float16_tValue145 = float16_t(floatValue114);
    f16vec4 f16vec4Value183 =
        f16vec4(float16_tValue142, min(float16_tValue145, min(float16_tValue133, float16_tValue139)),
                float16_tValue145, min(float16_tValue142, min(float16_tValue121, float16_tValue127))) -
        f16vec4(max(float16_tValue145, max(float16_tValue133, float16_tValue139)), float16_tValue142,
                max(float16_tValue142, max(float16_tValue121, float16_tValue127)), float16_tValue145);
    float16_t float16_tValue246 = (f16vec3Value64.z < float16_t(1.0))
                                      ? float16_t(1.0)
                                      : (float16_tValue186 * min(max(f16vec4Value183.x, f16vec4Value183.y),
                                                                 max(f16vec4Value183.z, f16vec4Value183.w)));
    if (float16_tValue246 < float16_t(1.0)) {
        imageStore(storageImageSet0Binding3, ivec2(f16vec2Value74), vec4(0.0));
        imageStore(storageImageSet0Binding4, ivec2(f16vec2Value69), vec4(0.0));
    }
    outputLocation0 = float(float16_tValue246) * specialization5;
}
