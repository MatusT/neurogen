// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x44850 (spirv, Fragment).
// Programs: hydra_reproject_blur
// Role: Filter reprojection matching evidence.
// Semantic names are reconstruction aids, not recovered author identifiers.
// Generic type/slot names deliberately retain uncertainty. See manifest.json for renames.
// Embedded source templates still require their original runtime defines.
#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(push_constant, std430) uniform symbolX16X18 {
    uint memberUint0;
    int memberInt1;
}
pushConstants;

layout(set = 1, binding = 0) uniform texture2D textureSet1Binding0;
layout(set = 0, binding = 0) uniform sampler samplerSet0Binding0;

layout(location = 0) out float outputLocation0;
layout(location = 0) in vec2 inputLocation0;

void main() {
    vec2 vec2Value42 = mix(vec2(0.0, 1.2000000476837158203125), vec2(1.2000000476837158203125, 0.0),
                           bvec2(pushConstants.memberUint0 != 0u)) /
                       vec2(textureSize(textureSet1Binding0, pushConstants.memberInt1));
    float floatValue47 = float(pushConstants.memberInt1);
    outputLocation0 =
        0.4000000059604644775390625 *
        textureLod(sampler2D(textureSet1Binding0, samplerSet0Binding0), inputLocation0, floatValue47).x;
    outputLocation0 +=
        (0.300000011920928955078125 * textureLod(sampler2D(textureSet1Binding0, samplerSet0Binding0),
                                                 inputLocation0 + vec2Value42, floatValue47)
                                          .x);
    outputLocation0 +=
        (0.300000011920928955078125 * textureLod(sampler2D(textureSet1Binding0, samplerSet0Binding0),
                                                 inputLocation0 - vec2Value42, floatValue47)
                                          .x);
}
