// SDKCore hydra_resolve_io_reproject, 0x4f054.
// Resolve unwarped fallback, optical-flow midpoint, and camera-reprojected midpoint.
// Both motion paths target t=0.5. Matching errors control their contribution;
// unused source pixels favor an unwarped fallback where optical matching fails.
struct ResolveParams {
    currentUvScale: f32,
}

@group(0) @binding(0) var linearSampler: sampler;
@group(0) @binding(1) var reprojectionSampler: sampler;
@group(0) @binding(2) var pointSampler: sampler;
@group(0) @binding(3) var halfMotion: texture_2d<f32>;
@group(0) @binding(4) var opticalError: texture_2d<f32>;
@group(0) @binding(5) var previousUsage: texture_2d<f32>;
@group(0) @binding(6) var currentUsage: texture_2d<f32>;
@group(0) @binding(7) var reprojectionError: texture_2d<f32>;
@group(0) @binding(8) var previousOffset: texture_2d<f32>;
@group(0) @binding(9) var currentOffset: texture_2d<f32>;
@group(1) @binding(0) var previousColor: texture_2d<f32>;
@group(1) @binding(1) var currentColor: texture_2d<f32>;
@group(3) @binding(0) var<uniform> params: ResolveParams;

@fragment
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    // One flow texel represents an 8x8 luminance block. Convert its half-motion
    // from luminance pixels to normalized UVs for the full-resolution colors.
    let lumaSize = vec2<f32>(textureDimensions(opticalError));
    let flowUv = uv * lumaSize / vec2<f32>(textureDimensions(halfMotion)) * 0.125 * params.currentUvScale;
    let flowUvOffset = textureSampleLevel(halfMotion, pointSampler, flowUv, 0.0).xy / lumaSize;

    // Despite the binding names, these masks store 1 for UNUSED source pixels.
    // Mip 2 smooths local coverage and error before they affect blend weights.
    // Each fallback weight is at least 0.025, keeping the normalization nonzero.
    let unusedCoverage = vec2(
        textureSampleLevel(previousUsage, linearSampler, uv, 2.0).r,
        textureSampleLevel(currentUsage, linearSampler, uv, 2.0).r,
    );
    let fallbackWeights = mix(vec2(1.0), unusedCoverage, vec2(0.95)) * 0.5;
    let fallbackSum = fallbackWeights.x + fallbackWeights.y;

    // Error 0 gives full confidence; 0.5 or more rejects optical motion. Reserve
    // the fallback share first, then normalize the remaining weighted colors.
    let opticalWeight = clamp(
        1.0 - 2.0 * textureSampleLevel(opticalError, linearSampler, uv, 2.0).r,
        0.0,
        1.0 - fallbackSum,
    );
    let unwarpedPrevious = textureSampleLevel(previousColor, linearSampler, uv, 0.0).rgb;
    let unwarpedCurrent = textureSampleLevel(currentColor, linearSampler, uv, 0.0).rgb;
    let opticalMidpoint = mix(
        textureSampleLevel(previousColor, linearSampler, uv - flowUvOffset, 0.0).rgb,
        textureSampleLevel(currentColor, linearSampler, uv + flowUvOffset, 0.0).rgb,
        vec3(0.5),
    );
    let opticalWithFallback = (
        unwarpedPrevious * fallbackWeights.x +
        unwarpedCurrent * fallbackWeights.y +
        opticalMidpoint * opticalWeight
    ) / (fallbackSum + opticalWeight);

    // Camera reprojection supplies an independent midpoint estimate. Its mesh
    // already stores midpoint-to-source offsets, so both are added to the UV.
    let previousUv = uv + textureSampleLevel(previousOffset, reprojectionSampler, uv, 0.0).xy;
    let currentUv = uv + textureSampleLevel(currentOffset, reprojectionSampler, uv, 0.0).xy;
    let reprojectedMidpoint = mix(
        textureSampleLevel(previousColor, linearSampler, previousUv, 0.0).rgb,
        textureSampleLevel(currentColor, linearSampler, currentUv, 0.0).rgb,
        vec3(0.5),
    );
    let reprojectionWeight = clamp(
        1.0 - 2.0 * textureSampleLevel(reprojectionError, pointSampler, uv, 0.0).r,
        0.0,
        1.0,
    );

    // Prefer camera reprojection where its filtered error is low. Output is opaque.
    return vec4(mix(opticalWithFallback, reprojectedMidpoint, vec3(reprojectionWeight)), 1.0);
}
