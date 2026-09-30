"""Translate recovered BASIC interpolation shaders. Requires naga and glslangValidator.

The application consumes hand-cleaned WGSL. This produces comparison translations
under reference/hydra/translated and never overwrites the application shaders.
See reference/hydra/README.md for adaptations and provenance.
"""
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent.parent
FILES = {
    'fullscreen': 'shared_fullscreen_vertex__3cec0.vert',
    'luma': 'compute_luma_no_ui__3dd2c.frag',
    'depth_reduce': 'depth_mipmap_internal__3e000.frag',
    'search': 'hydra_search_block_grad_false__46ffc.comp',
    'match': 'hydra_match_txt_false_dmatch_none__4c5e0.frag',
    'reproject_vertex': 'hydra_reproject_distort__434b4.vert',
    'reproject_fragment': 'hydra_reproject_distort__4337c.frag',
    'reproject_match': 'hydra_reproject_match__43fac.frag',
    'reproject_blur': 'hydra_reproject_blur__44850.frag',
    'resolve': 'hydra_resolve_io_reproject__4f054.frag',
}

def adapt(name, text):
    text = '\n'.join(l for l in text.splitlines() if not l.startswith('#'))
    text = '#version 450\n#extension GL_EXT_samplerless_texture_functions : require\n' + text
    text = re.sub(r'layout\(constant_id = \d+\) ', '', text)
    text = text.replace('layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2)',
                        'layout(local_size_x = 5, local_size_y = 5, local_size_z = 9)')
    text = text.replace('layout(push_constant, std430)', 'layout(set = 3, binding = 0, std140)')
    text = re.sub(r'f16vec([234])', r'vec\1', text).replace('float16_t', 'float')
    text = text.replace('int16_t', 'int')
    text = re.sub(r'\b(\d+)s\b', r'\1', text)
    text = text.replace('noperspective ', '')
    if name == 'reproject_vertex':
        # Runtime geometry dimensions, independent of the shader binary's defaults.
        text = re.sub(r'const (?:int|float) specialization([12345]) = [^;]+;',
                      lambda m: '#define specialization'+m[1]+' '+{
                          '1':'int(mesh.cells.x + 1u)', '2':'int(mesh.cells.x)',
                          '3':'int(mesh.cells.y)', '4':'(1.0 / float(mesh.cells.x))',
                          '5':'(1.0 / float(mesh.cells.y))'}[m[1]], text)
        text = re.sub(r'const (vec2|ivec2) (\w+) = ([^;]+);', lambda m: '#define '+m[2]+' ('+m[3]+')', text)
        text = text.replace('#version 450', '#version 450\nlayout(set=3,binding=1,std140) uniform Mesh { uvec4 cells; } mesh;')
        text = text.replace('textureLod(sampler2D(textureSet2Binding0, samplerSet0Binding0), vec2Value27, 0.0)',
            'texelFetch(textureSet2Binding0, clamp(ivec2(vec2Value27 * vec2(textureSize(textureSet2Binding0, 0))), ivec2(0), textureSize(textureSet2Binding0, 0)-1), 0)')
    if name in ('fullscreen', 'reproject_vertex'):
        # Flip clip Y only: texel/UV coordinates remain top-left throughout.
        pos = text.rfind('}')
        text = text[:pos] + 'gl_Position.y = -gl_Position.y;\n' + text[pos:]
    if name == 'search':
        # The native code writes the same candidate from all 25 XY lanes.
        # Have one lane do it, then synchronize before readers access it.
        text = text.replace('sharedVec3115[gl_LocalInvocationID.z] = vec3Value111;',
            'if (gl_LocalInvocationID.x == 0u && gl_LocalInvocationID.y == 0u) { sharedVec3115[gl_LocalInvocationID.z] = vec3Value111; }\nbarrier();')
    return text

def main():
    out = ROOT/'reference/hydra/translated'
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='neurogen-hydra-') as tmp:
        for name, origin in FILES.items():
            text = adapt(name, (ROOT/'reference/hydra'/(origin+'.glsl')).read_text())
            source = Path(tmp)/(name+'.glsl')
            source.write_text(text)
            stage = {'comp':'compute', 'vert':'vert', 'frag':'frag'}[origin.split('.')[-1]]
            target = out/(name+'.wgsl')
            spirv = Path(tmp)/(name+'.spv')
            subprocess.run(['glslangValidator','-V','-S','comp' if stage=='compute' else stage,str(source),'-o',str(spirv)],check=True,stdout=subprocess.DEVNULL)
            result = subprocess.run(['naga','--keep-coordinate-space',str(spirv),str(target)],capture_output=True,text=True)
            if result.returncode:
                print(name, result.stdout, result.stderr)
                raise SystemExit(result.returncode)
            target.write_text('// Recovered SDKCore '+origin+'. Port: scripts/port-hydra.py.\n'+target.read_text())
            print(name)

if __name__ == '__main__':
    main()
