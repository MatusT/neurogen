// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x4f054 (spirv, Fragment).
// Programs: hydra_resolve_io_reproject
// Role: Warp and combine frame colors using optical flow and/or reprojection evidence.
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

layout(constant_id = 4) const bool specialization4 = false;
layout(constant_id = 1) const float specialization1 = 0.125;
layout(constant_id = 0) const float specialization0 = 2.0;
layout(constant_id = 2) const float specialization2 = 0.949999988079071044921875;
layout(constant_id = 3) const float specialization3 = 2.0;

layout(push_constant, std430) uniform symbolX83X85 { float memberFloat0; }
pushConstants;

layout(set = 1, binding = 0) uniform texture2D textureSet1Binding0;
layout(set = 0, binding = 0) uniform sampler samplerSet0Binding0;
layout(set = 1, binding = 1) uniform texture2D textureSet1Binding1;
layout(set = 0, binding = 4) uniform texture2D textureSet0Binding4;
layout(set = 0, binding = 3) uniform texture2D textureSet0Binding3;
layout(set = 0, binding = 2) uniform sampler samplerSet0Binding2;
layout(set = 0, binding = 5) uniform texture2D textureSet0Binding5;
layout(set = 0, binding = 6) uniform texture2D textureSet0Binding6;
layout(set = 0, binding = 8) uniform texture2D textureSet0Binding8;
layout(set = 0, binding = 1) uniform sampler samplerSet0Binding1;
layout(set = 0, binding = 9) uniform texture2D textureSet0Binding9;
layout(set = 0, binding = 7) uniform texture2D textureSet0Binding7;

layout(location = 0) in vec2 inputLocation0;
layout(location = 0) out vec4 outputLocation0;

void main() {
    vec2 vec2Value268;
    if (specialization4) {
        vec2Value268 = vec2(inputLocation0.x, 1.0 - inputLocation0.y);
    } else {
        vec2Value268 = inputLocation0;
    }
    vec2 vec2Value76 = vec2(f16vec2(textureSize(textureSet0Binding4, 0)));
    vec2 sampledValue100 =
        textureLod(sampler2D(textureSet0Binding3, samplerSet0Binding2),
                   (((inputLocation0 * vec2Value76) / vec2(f16vec2(textureSize(textureSet0Binding3, 0)))) *
                    specialization1) *
                       pushConstants.memberFloat0,
                   0.0)
            .xy /
        vec2Value76;
    f16vec2 f16vec2Value153 =
        f16vec2(mix(vec2(1.0),
                    vec2(float(float16_t(textureLod(sampler2D(textureSet0Binding5, samplerSet0Binding0),
                                                    inputLocation0, specialization0)
                                             .x)),
                         float(float16_t(textureLod(sampler2D(textureSet0Binding6, samplerSet0Binding0),
                                                    inputLocation0, specialization0)
                                             .x))),
                    vec2(specialization2)) *
                0.5);
    float16_t float16_tValue159 = f16vec2Value153.x + f16vec2Value153.y;
    float16_t float16_tValue174 = float16_t(
        clamp(1.0 - (specialization3 * textureLod(sampler2D(textureSet0Binding4, samplerSet0Binding0),
                                                  inputLocation0, specialization0)
                                           .x),
              0.0, 1.0 - float(float16_tValue159)));
    outputLocation0 = vec4(
        vec3(mix(
            fma(f16vec3(
                    textureLod(sampler2D(textureSet1Binding0, samplerSet0Binding0), vec2Value268, 0.0).xyz),
                f16vec2Value153.xxx,
                fma(f16vec3(textureLod(sampler2D(textureSet1Binding1, samplerSet0Binding0), vec2Value268, 0.0)
                                .xyz),
                    f16vec2Value153.yyy,
                    mix(f16vec3(textureLod(sampler2D(textureSet1Binding0, samplerSet0Binding0),
                                           vec2Value268 - sampledValue100, 0.0)
                                    .xyz),
                        f16vec3(textureLod(sampler2D(textureSet1Binding1, samplerSet0Binding0),
                                           vec2Value268 + sampledValue100, 0.0)
                                    .xyz),
                        f16vec3(float16_t(0.5))) *
                        float16_tValue174)) *
                (float16_t(1.0) / (float16_tValue159 + float16_tValue174)),
            mix(f16vec3(
                    textureLod(sampler2D(textureSet1Binding0, samplerSet0Binding0),
                               vec2Value268 + textureLod(sampler2D(textureSet0Binding8, samplerSet0Binding1),
                                                         inputLocation0, 0.0)
                                                  .xy,
                               0.0)
                        .xyz),
                f16vec3(
                    textureLod(sampler2D(textureSet1Binding1, samplerSet0Binding0),
                               vec2Value268 + textureLod(sampler2D(textureSet0Binding9, samplerSet0Binding1),
                                                         inputLocation0, 0.0)
                                                  .xy,
                               0.0)
                        .xyz),
                f16vec3(float16_t(0.5))),
            f16vec3(float16_t(clamp(
                1.0 -
                    (specialization3 *
                     textureLod(sampler2D(textureSet0Binding7, samplerSet0Binding2), inputLocation0, 0.0).x),
                0.0, 1.0))))),
        1.0);
}
