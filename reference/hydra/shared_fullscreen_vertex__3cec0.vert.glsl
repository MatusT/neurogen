// Readable review copy; constants, expressions and control flow are preserved.
// Origin: libFrameGenerationSdkCore.so @ 0x3cec0 (spirv, Vertex).
// Programs: split_zs_blit, compute_luma_no_ui, depth_mipmap_internal, depth_mipmap_conversion,
// hydra_match_txt(false)_dmatch(none), hydra_match_txt(false)_dmatch(pos), hydra_match_txt(false)_dmatch(z),
// hydra_reproject_match, hydra_reproject_blur, aiflow_resolve, inter_fuse_geometry_generation,
// inter_fuse_geometry_match, inter_fuse_geometry_iter, inter_fuse_geometry_iter_invert,
// inter_fuse_geometry_expand, engine_mv_match, ui_resolve_generate_mv, ui_resolve, inter_fuse_resolve_hybrid,
// inter_fuse_resolve_of, inter_fuse_resolve_hiof, inter_fuse_resolve_geometry, rotate_90 Role: Shared vertex
// stage used by the listed programs. Semantic names are reconstruction aids, not recovered author
// identifiers. Generic type/slot names deliberately retain uncertainty. See manifest.json for renames.
// Embedded source templates still require their original runtime defines.
#version 450

const vec2 vec2Value18[3] = vec2[](vec2(-1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));

layout(location = 0) noperspective out vec2 outputLocation0;

void main() {
    gl_Position = vec4(vec2Value18[gl_VertexIndex], 0.0, 1.0);
    outputLocation0 = fma(vec2Value18[gl_VertexIndex], vec2(0.5), vec2(0.5));
}
