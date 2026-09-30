// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x3e000 (spirv, Fragment).
// Programs: depth_mipmap_internal
// Role: Prepare or reduce depth for motion/reprojection comparisons.
// Semantic names are reconstruction aids, not recovered author identifiers.
// Generic type/slot names deliberately retain uncertainty. See manifest.json for renames.
// Embedded source templates still require their original runtime defines.
#version 450

layout(constant_id = 0) const uint specialization0 = 0u;
const bool boolValue22 = (specialization0 == 1u);
const float floatValue25 = boolValue22 ? 1000000015047466219876688855040.0 : 0.0;
const bool boolValue108 = (specialization0 == 0u);

layout(set = 1, binding = 0) uniform texture2D textureSet1Binding0;
layout(set = 0, binding = 0) uniform sampler samplerSet0Binding0;

layout(location = 0) in vec2 inputLocation0;
layout(location = 0) out float outputLocation0;

void main() {
    vec4 gatheredTexels105 =
        textureGather(sampler2D(textureSet1Binding0, samplerSet0Binding0), inputLocation0);
    float floatValue187;
    float floatValue188;
    switch (specialization0) {
    case 1u: {
        floatValue188 = min(floatValue25, min(min(gatheredTexels105.x, gatheredTexels105.y),
                                              min(gatheredTexels105.z, gatheredTexels105.w)));
        floatValue187 = 0.0;
        break;
    }
    case 2u: {
        floatValue188 = max(floatValue25, max(max(gatheredTexels105.x, gatheredTexels105.y),
                                              max(gatheredTexels105.z, gatheredTexels105.w)));
        floatValue187 = 0.0;
        break;
    }
    default: {
        floatValue188 = floatValue25 + (((gatheredTexels105.x + gatheredTexels105.y) + gatheredTexels105.z) +
                                        gatheredTexels105.w);
        floatValue187 = 4.0;
        break;
    }
    }
    float floatValue189;
    if (boolValue108) {
        floatValue189 = floatValue188 / floatValue187;
    } else {
        floatValue189 = floatValue188;
    }
    outputLocation0 = floatValue189;
}
