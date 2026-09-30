// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x3d2bc (spirv, Vertex).
// Programs: hydra_resolve_io_reproject, hydra_resolve_io, hydra_resolve_reproject
// Role: Warp and combine frame colors using optical flow and/or reprojection evidence.
// Semantic names are reconstruction aids, not recovered author identifiers.
// Generic type/slot names deliberately retain uncertainty. See manifest.json for renames.
// Embedded source templates still require their original runtime defines.
#version 450

const vec2 vec2Value18[3] = vec2[](vec2(-1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
layout(constant_id = 5) const bool specialization5 = false;

layout(location = 0) noperspective out vec2 outputLocation0;

void main() {
    gl_Position = vec4(vec2Value18[gl_VertexIndex], 0.0, 1.0);
    outputLocation0 = fma(vec2Value18[gl_VertexIndex], vec2(0.5), vec2(0.5));
    if (specialization5) {
        outputLocation0 = vec2(outputLocation0.x, 1.0 - outputLocation0.y);
    }
}
