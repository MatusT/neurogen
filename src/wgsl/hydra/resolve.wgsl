// SDKCore hydra_resolve_io_reproject, 0x4f054.
// Resolve unwarped fallback, optical-flow midpoint, and camera-reprojected midpoint.
struct ResolveParams { currentUvScale: f32 }
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
@fragment fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    let lumaSize = vec2<f32>(textureDimensions(opticalError));
    let flowUv = uv * lumaSize / vec2<f32>(textureDimensions(halfMotion)) * 0.125 * params.currentUvScale;
    let flowUvOffset = textureSampleLevel(halfMotion, pointSampler, flowUv, 0.0).xy / lumaSize;
    let usage = vec2(textureSampleLevel(previousUsage, linearSampler, uv, 2.0).r,
                     textureSampleLevel(currentUsage, linearSampler, uv, 2.0).r);
    let fallbackWeights = mix(vec2(1.0), usage, vec2(0.95)) * 0.5;
    let fallbackSum = fallbackWeights.x + fallbackWeights.y;
    let opticalWeight = clamp(1.0 - 2.0 * textureSampleLevel(opticalError, linearSampler, uv, 2.0).r, 0.0, 1.0 - fallbackSum);
    let previous = textureSampleLevel(previousColor, linearSampler, uv, 0.0).rgb;
    let current = textureSampleLevel(currentColor, linearSampler, uv, 0.0).rgb;
    let opticalMidpoint = mix(textureSampleLevel(previousColor, linearSampler, uv - flowUvOffset, 0.0).rgb,
                              textureSampleLevel(currentColor, linearSampler, uv + flowUvOffset, 0.0).rgb, vec3(0.5));
    let opticalWithFallback = (previous * fallbackWeights.x + current * fallbackWeights.y + opticalMidpoint * opticalWeight) /
                              (fallbackSum + opticalWeight);
    let previousUv = uv + textureSampleLevel(previousOffset, reprojectionSampler, uv, 0.0).xy;
    let currentUv = uv + textureSampleLevel(currentOffset, reprojectionSampler, uv, 0.0).xy;
    let reprojectedMidpoint = mix(textureSampleLevel(previousColor, linearSampler, previousUv, 0.0).rgb,
                                  textureSampleLevel(currentColor, linearSampler, currentUv, 0.0).rgb, vec3(0.5));
    let reprojectionWeight = clamp(1.0 - 2.0 * textureSampleLevel(reprojectionError, pointSampler, uv, 0.0).r, 0.0, 1.0);
    return vec4(mix(opticalWithFallback, reprojectedMidpoint, vec3(reprojectionWeight)), 1.0);
}
