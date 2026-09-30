// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x3d7ac (spirv, Fragment).
// Programs: split_zs_blit
// Role: Depth/stencil transfer utility.
// Semantic names are reconstruction aids, not recovered author identifiers.
// Generic type/slot names deliberately retain uncertainty. See manifest.json for renames.
// Embedded source templates still require their original runtime defines.
#version 450

layout(constant_id = 0) const bool specialization0 = false;

layout(set = 1, binding = 0) uniform texture2D textureSet1Binding0;
layout(set = 0, binding = 0) uniform sampler samplerSet0Binding0;
layout(set = 1, binding = 1) uniform utexture2D textureSet1Binding1;
layout(set = 0, binding = 1) uniform utexture1D textureSet0Binding1;

layout(location = 0) noperspective in vec2 inputLocation0;
layout(location = 0) out uint outputLocation0;

void main() {
    vec2 vec2Value12 = inputLocation0;
    vec2 vec2Value81;
    if (specialization0) {
        vec2 vec2Value80 = vec2Value12;
        vec2Value80.y = 1.0 - vec2Value12.y;
        vec2Value81 = vec2Value80;
    } else {
        vec2Value81 = vec2Value12;
    }
    outputLocation0 =
        uint(texture(sampler2D(textureSet1Binding0, samplerSet0Binding0), vec2Value81).x * 16777215.0) |
        (texelFetch(usampler1D(textureSet0Binding1, samplerSet0Binding0),
                    int(texture(usampler2D(textureSet1Binding1, samplerSet0Binding0), vec2Value81).x), 0)
             .x
         << 24u);
}
