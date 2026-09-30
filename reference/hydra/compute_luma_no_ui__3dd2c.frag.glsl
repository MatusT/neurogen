// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x3dd2c (spirv, Fragment).
// Programs: compute_luma_no_ui
// Role: Convert color to luminance; optional UI/extended-data specialization.
// Semantic names are reconstruction aids, not recovered author identifiers.
// Generic type/slot names deliberately retain uncertainty. See manifest.json for renames.
// Embedded source templates still require their original runtime defines.
#version 450

layout(set = 1, binding = 0) uniform texture2D textureSet1Binding0;
layout(set = 0, binding = 0) uniform sampler samplerSet0Binding0;

layout(location = 0) out float outputLocation0;
layout(location = 0) in vec2 inputLocation0;

void main() {
    outputLocation0 =
        dot(vec3(0.2989999949932098388671875, 0.58700001239776611328125, 0.114000000059604644775390625),
            texture(sampler2D(textureSet1Binding0, samplerSet0Binding0), inputLocation0).xyz);
}
