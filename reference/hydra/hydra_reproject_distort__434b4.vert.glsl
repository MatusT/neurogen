// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x434b4 (spirv, Vertex).
// Programs: hydra_reproject_distort
// Role: Rasterize the reprojection mesh / its fragment payload.
// Semantic names are reconstruction aids, not recovered author identifiers.
// Generic type/slot names deliberately retain uncertainty. See manifest.json for renames.
// Embedded source templates still require their original runtime defines.
#version 450

layout(constant_id = 1) const int specialization1 = 99;
layout(constant_id = 4) const float specialization4 = 0.010204100050032138824462890625;
layout(constant_id = 5) const float specialization5 = 0.02222220040857791900634765625;
const vec2 vec2Value26 = vec2(specialization4, specialization5);
layout(constant_id = 0) const bool specialization0 = false;
layout(constant_id = 2) const int specialization2 = 98;
layout(constant_id = 3) const int specialization3 = 45;
const ivec2 ivec2Value126 = ivec2(specialization2, specialization3);

layout(set = 1, binding = 0, std140) uniform uniformsSet1Binding0 {
    mat4 memberUint0;
    mat4 memberMat41;
}
symbolX82;

layout(push_constant, std430) uniform symbolX67X69 { uint memberUint0; }
pushConstants;

layout(set = 2, binding = 0) uniform texture2D textureSet2Binding0;
layout(set = 0, binding = 0) uniform sampler samplerSet0Binding0;

layout(location = 0) out vec2 outputLocation0;

void main() {
    ivec2 ivec2Value17 = ivec2(gl_VertexIndex % specialization1, gl_VertexIndex / specialization1);
    vec2 vec2Value27 = vec2(ivec2Value17) * vec2Value26;
    vec4 sampledValue43 = textureLod(sampler2D(textureSet2Binding0, samplerSet0Binding0), vec2Value27, 0.0);
    float floatValue46 = sampledValue43.x;
    float floatValue166;
    if (specialization0) {
        floatValue166 = fma(floatValue46, 2.0, -1.0);
    } else {
        floatValue166 = floatValue46;
    }
    mat4 mat4Value167;
    if (pushConstants.memberUint0 != 0u) {
        mat4Value167 = symbolX82.memberMat41;
    } else {
        mat4Value167 = symbolX82.memberUint0;
    }
    vec4 vec4Value92 = mat4Value167 * vec4(fma(vec2Value27, vec2(2.0), vec2(-1.0)), floatValue166, 1.0);
    vec2 vec2Value115 = clamp(vec2Value27 - fma(vec4Value92.xy, vec2(0.5 / vec4Value92.w), vec2(0.5)),
                              vec2(-0.25), vec2(0.25)) *
                        0.5;
    bvec2 bvec2Value121 = greaterThan(ivec2Value17, ivec2(0));
    bvec2 bvec2Value127 = lessThan(ivec2Value17, ivec2Value126);
    outputLocation0 = vec2Value115;
    gl_Position = vec4(fma(mix(vec2Value27, vec2Value27 - vec2Value115,
                               bvec2(bvec2Value121.x && bvec2Value127.x, bvec2Value121.y && bvec2Value127.y)),
                           vec2(2.0), vec2(-1.0)),
                       0.0, 1.0);
}
