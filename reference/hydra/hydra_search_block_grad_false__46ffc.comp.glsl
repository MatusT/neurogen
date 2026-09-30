// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x46ffc (spirv, GLCompute).
// Programs: hydra_search_block_grad(false)
// Role: Block motion search using previous/current luminance and an optional coarser motion field.
// Semantic names are reconstruction aids, not recovered author identifiers.
// Generic type/slot names deliberately retain uncertainty. See manifest.json for renames.
// Embedded source templates still require their original runtime defines.
#version 450
#extension GL_EXT_samplerless_texture_functions : require
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

const vec2 vec2Value56[9] =
    vec2[](vec2(0.25), vec2(1.25, 0.25), vec2(-0.75, 0.25), vec2(0.25, 1.25), vec2(0.25, -0.75), vec2(1.25),
           vec2(-0.75), vec2(-0.75, 1.25), vec2(1.25, -0.75));
layout(constant_id = 2) const uint specialization2 = 9u;
layout(constant_id = 3) const uint specialization3 = 8u;
layout(constant_id = 4) const uint specialization4 = 8u;
const uint uintValue131 = (specialization4 - specialization3);
const uint uintValue132 = (uintValue131 / 2u);
const uint uintValue176 = (specialization4 / 2u);
const uint uintValue185 = (specialization4 / 2u);
layout(constant_id = 5) const uint specialization5 = 256u;
const uint uintValue277 = (specialization5 >> 1u);
layout(constant_id = 0) const uint specialization0 = 5u;
layout(constant_id = 1) const uint specialization1 = 5u;
const uint uintValue300 = (specialization0 * specialization1);
const uint uintValue301 = (uintValue300 * specialization2);
const uint uintValue369 = (specialization0 * specialization1);

layout(push_constant, std430) uniform symbolX20X22 {
    uint memberUint0;
    uint memberUint1;
    float memberFloat2;
    float memberFloat3;
}
pushConstants;

layout(set = 1, binding = 2) uniform texture2D textureSet1Binding2;
layout(set = 0, binding = 0) uniform sampler samplerSet0Binding0;
layout(set = 1, binding = 0) uniform texture2D textureSet1Binding0;
layout(set = 1, binding = 1) uniform texture2D textureSet1Binding1;
layout(set = 1, binding = 3, rgba16f) uniform restrict writeonly image2D storageImageSet1Binding3;

shared vec3 sharedVec3115[specialization2];
shared ivec2 sharedIvec2264[specialization5];

void main() {
    do {
        vec2 vec2Value18 = vec2(textureSize(textureSet1Binding2, 0));
        vec2 localInvocation424;
        if (pushConstants.memberUint1 != 0u) {
            localInvocation424 =
                fma(vec2(gl_WorkGroupID.xy), vec2(0.5), vec2Value56[gl_LocalInvocationID.z]) /
                (vec2Value18 * pushConstants.memberFloat2);
        } else {
            localInvocation424 =
                clamp(fma(vec2(gl_WorkGroupID.xy), vec2(0.5), vec2Value56[gl_LocalInvocationID.z]), vec2(0.0),
                      vec2Value18 * pushConstants.memberFloat2) /
                vec2Value18;
        }
        vec3 vec3Value111 = round(
            vec3(2.0, 2.0, 1.0) *
            textureLod(sampler2D(textureSet1Binding2, samplerSet0Binding0), localInvocation424, 0.0).xyz);
        sharedVec3115[gl_LocalInvocationID.z] = vec3Value111;
        uvec2 uvec2Value134 = (gl_WorkGroupID.xy * uvec2(specialization3)) - uvec2(uintValue132);
        vec2 vec2Value141 = vec2(1.0) / vec2(textureSize(textureSet1Binding0, 0));
        vec2 localInvocation148 = fma(vec2(gl_LocalInvocationID.xy), vec2(0.5), vec2(-0.5));
        vec2 vec2Value155 = vec3Value111.xy;
        vec2 vec2Value160 = vec2Value141 * fma(vec2(-2.0), localInvocation148, vec2(1.0) - vec2Value155);
        vec2 vec2Value167 = vec2Value141 * (vec2Value155 + vec2(1.0));
        uint uintValue425;
        float floatValue426;
        floatValue426 = 0.0;
        uintValue425 = 0u;
        float floatValue434;
        for (; uintValue425 < uintValue176; floatValue426 = floatValue434, uintValue425++) {
            floatValue434 = floatValue426;
            for (uint loopIndex432 = 0u; loopIndex432 < uintValue185;) {
                vec2 vec2Value195 = vec2(uvec2Value134 + (uvec2(2u) * uvec2(uintValue425, loopIndex432)));
                vec4 vec4Value233 = abs(textureGather(sampler2D(textureSet1Binding0, samplerSet0Binding0),
                                                      clamp(fma(vec2Value195, vec2Value141, vec2Value160),
                                                            vec2(0.0), vec2(pushConstants.memberFloat3))) -
                                        textureGather(sampler2D(textureSet1Binding1, samplerSet0Binding0),
                                                      clamp(fma(vec2Value195, vec2Value141, vec2Value167),
                                                            vec2(0.0), vec2(pushConstants.memberFloat2))));
                vec2 vec2Value239 = vec4Value233.xy + vec4Value233.zw;
                floatValue434 += (vec2Value239.x + vec2Value239.y);
                loopIndex432++;
                continue;
            }
        }
        sharedIvec2264[gl_LocalInvocationIndex] = ivec2(
            int(3072.0 * (floatValue426 + (0.00999999977648258209228515625 * length(localInvocation148)))),
            int(gl_LocalInvocationIndex));
        barrier();
        if (gl_LocalInvocationIndex < uintValue277) {
            uint uintValue287 = gl_LocalInvocationIndex + uintValue277;
            ivec2 ivec2Value439 =
                mix(sharedIvec2264[uintValue287], ivec2(107374182), bvec2(uintValue287 >= uintValue301));
            sharedIvec2264[gl_LocalInvocationIndex] =
                mix(ivec2Value439, sharedIvec2264[gl_LocalInvocationIndex],
                    bvec2(sharedIvec2264[gl_LocalInvocationIndex].x < ivec2Value439.x));
        }
        uint uintValue321 = uintValue277 >> 1u;
        barrier();
        uint uintValue356;
        for (uint loopIndex430 = uintValue321; loopIndex430 > 0u; loopIndex430 = uintValue356) {
            if (gl_LocalInvocationIndex < loopIndex430) {
                uint uintValue341 = gl_LocalInvocationIndex + loopIndex430;
                sharedIvec2264[gl_LocalInvocationIndex] =
                    mix(sharedIvec2264[uintValue341], sharedIvec2264[gl_LocalInvocationIndex],
                        bvec2(sharedIvec2264[gl_LocalInvocationIndex].x < sharedIvec2264[uintValue341].x));
            }
            uintValue356 = loopIndex430 >> 1u;
            barrier();
        }
        if (gl_LocalInvocationIndex != 0u) {
            break;
        }
        uint uintValue366 = uint(sharedIvec2264[0].y);
        uint uintValue373 = uintValue366 % uintValue369;
        imageStore(storageImageSet1Binding3, ivec2(gl_WorkGroupID.xy),
                   vec4(sharedVec3115[uintValue366 / uintValue369] +
                            vec3(fma(vec2(float(uintValue373 % gl_WorkGroupSize.y),
                                          float(uintValue373 / gl_WorkGroupSize.y)),
                                     vec2(0.5), vec2(-0.5)),
                                 0.0),
                        0.0));
        break;
    } while (false);
}
