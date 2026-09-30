// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x3e674 (spirv, Fragment).
// Programs: depth_mipmap_conversion
// Role: Prepare or reduce depth for motion/reprojection comparisons.
// Semantic names are reconstruction aids, not recovered author identifiers.
// Generic type/slot names deliberately retain uncertainty. See manifest.json for renames.
// Embedded source templates still require their original runtime defines.
#version 450

layout(constant_id = 0) const uint specialization0 = 0u;
const bool boolValue25 = (specialization0 == 1u);
const float floatValue28 = boolValue25 ? 1000000015047466219876688855040.0 : 0.0;
const bool boolValue118 = (specialization0 == 0u);

layout(set = 1, binding = 0) uniform utexture2D textureSet1Binding0;
layout(set = 0, binding = 0) uniform sampler samplerSet0Binding0;

layout(location = 0) in vec2 inputLocation0;
layout(location = 0) out float outputLocation0;

void main() {
    vec4 vec4Value135 =
        vec4(textureGather(usampler2D(textureSet1Binding0, samplerSet0Binding0), inputLocation0) &
             uvec4(16777215u)) *
        vec4(5.9604651880817982601001858711243e-08);
    float floatValue202;
    float floatValue203;
    switch (specialization0) {
    case 1u: {
        floatValue203 =
            min(floatValue28, min(min(vec4Value135.x, vec4Value135.y), min(vec4Value135.z, vec4Value135.w)));
        floatValue202 = 0.0;
        break;
    }
    case 2u: {
        floatValue203 =
            max(floatValue28, max(max(vec4Value135.x, vec4Value135.y), max(vec4Value135.z, vec4Value135.w)));
        floatValue202 = 0.0;
        break;
    }
    default: {
        floatValue203 =
            floatValue28 + (((vec4Value135.x + vec4Value135.y) + vec4Value135.z) + vec4Value135.w);
        floatValue202 = 4.0;
        break;
    }
    }
    float floatValue204;
    if (boolValue118) {
        floatValue204 = floatValue203 / floatValue202;
    } else {
        floatValue204 = floatValue203;
    }
    outputLocation0 = floatValue204;
}
