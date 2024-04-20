function makeShaderCode(outputFormat: string, filterOp: string = SPD_FILTER_AVERAGE, numMips: number): string {
    const mipsBindings = Array(numMips).fill(0)
        .map((_, i) => `@group(0) @binding(${i + 1}) var dst_mip_${i + 1}: texture_storage_2d_array<${outputFormat}, write>;`)
        .join('\n');

    const mipsAccessorBody = Array(numMips).fill(0)
        .map((_, i) => {
            return `${i === 0 ? '' : ' else '}if mip == ${i + 1} {
                textureStore(dst_mip_${i + 1}, uv, slice, value);
            }`;
        })
        .join('');
    const mipsAccessor = `fn store_dst_mip(value: vec4<f32>, uv: vec2<u32>, slice: u32, mip: u32) {\n${mipsAccessorBody}\n}`

    return /* wgsl */`
    // This file is part of the FidelityFX SDK.
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files(the “Software”), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell 
// copies of the Software, and to permit persons to whom the Software is 
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


// Helpers ------------------------------------------------------------------------------------------------------------

/**
 * A helper function performing a remap 64x1 to 8x8 remapping which is necessary for 2D wave reductions.
 * 
 * The 64-wide lane indices to 8x8 remapping is performed as follows:
 *     00 01 08 09 10 11 18 19
 *      02 03 0a 0b 12 13 1a 1b
 *      04 05 0c 0d 14 15 1c 1d
 *      06 07 0e 0f 16 17 1e 1f
 *      20 21 28 29 30 31 38 39
 *      22 23 2a 2b 32 33 3a 3b
 *      24 25 2c 2d 34 35 3c 3d
 *      26 27 2e 2f 36 37 3e 3f
 * 
 * @param a: The input 1D coordinate to remap.
 *
 * @returns The remapped 2D coordinates.
 */
fn remap_for_wave_reduction(a: u32) -> vec2<u32> {
    return vec2<u32>(
        insertBits(extractBits(a, 2u, 3u), a, 0u, 1u),
        insertBits(extractBits(a, 3u, 3u), extractBits(a, 1u, 2u), 0u, 2u)
    );
}

fn map_to_xy(local_invocation_index: u32) -> vec2<u32> {
    let sub_xy: vec2<u32> = remap_for_wave_reduction(local_invocation_index % 64);
    return vec2<u32>(
        sub_xy.x + 8 * ((local_invocation_index >> 6) % 2),
        sub_xy.y + 8 * ((local_invocation_index >> 7))
    );
}

/*
 * Compute a linear value from a SRGB value.
 * 
 * @param value: The value to convert to linear from SRGB.
 * 
 *  @returns A value in SRGB space.
 */
fn srgb_to_linear(value: f32) -> f32 {
    let j = vec3<f32>(0.0031308 * 12.92, 12.92, 1.0 / 2.4);
    let k = vec2<f32>(1.055, -0.055);
    return clamp(j.x, value * j.y, pow(value, j.z) * k.x + k.y);
}

// Resources & Accessors -----------------------------------------------------------------------------------------------
struct DownsamplePassMeta {
    work_group_offset: vec2<u32>,
    num_work_groups: u32,
    mips: u32,
}

// In the original version dst_mip_i is an image2Darray [SPD_MAX_MIP_LEVELS+1], i.e., 12+1, but WGSL doesn't support arrays of textures yet
// Also these are read_write because for mips 7-13, the workgroup reads from mip level 6 - since most formats don't support read_write access in WGSL yet, this is split in two passes and we can just use write access
@group(0) @binding(0) var src_mip_0: texture_2d_array<f32>;
${mipsBindings}

@group(1) @binding(0) var<uniform> downsample_pass_meta : DownsamplePassMeta;

fn get_mips() -> u32 {
    return downsample_pass_meta.mips;
}

fn get_num_work_groups() -> u32 {
    return downsample_pass_meta.num_work_groups;
}

fn get_work_group_offset() -> vec2<u32> {
    return downsample_pass_meta.work_group_offset;
}

fn load_src_image(uv: vec2<u32>, slice: u32) -> vec4<f32> {
    return textureLoad(src_mip_0, uv, slice, 0);
}

${mipsAccessor}

// Workgroup -----------------------------------------------------------------------------------------------------------

// WGSL doesn't support array<array<f32, 16>, 16> yet?
var<workgroup> spd_intermediate_r: array<f32, 256>;
var<workgroup> spd_intermediate_g: array<f32, 256>;
var<workgroup> spd_intermediate_b: array<f32, 256>;
var<workgroup> spd_intermediate_a: array<f32, 256>;

fn to_intermediate_index(x: u32, y: u32) -> u32 {
    return y * 16 + x;
}
fn get_intermediate_r(x: u32, y: u32) -> f32 {
    return spd_intermediate_r[to_intermediate_index(x, y)];
}
fn get_intermediate_g(x: u32, y: u32) -> f32 {
    return spd_intermediate_g[to_intermediate_index(x, y)];
}
fn get_intermediate_b(x: u32, y: u32) -> f32 {
    return spd_intermediate_b[to_intermediate_index(x, y)];
}
fn get_intermediate_a(x: u32, y: u32) -> f32 {
    return spd_intermediate_a[to_intermediate_index(x, y)];
}
fn set_intermediate_r(x: u32, y: u32, v: f32) {
    spd_intermediate_r[to_intermediate_index(x, y)] = v;
}
fn set_intermediate_g(x: u32, y: u32, v: f32) {
    spd_intermediate_g[to_intermediate_index(x, y)] = v;
}
fn set_intermediate_b(x: u32, y: u32, v: f32) {
    spd_intermediate_b[to_intermediate_index(x, y)] = v;
}
fn set_intermediate_a(x: u32, y: u32, v: f32) {
    spd_intermediate_a[to_intermediate_index(x, y)] = v;
}

// Cotnrol flow --------------------------------------------------------------------------------------------------------

fn spd_barrier() {
    // in glsl this does: groupMemoryBarrier(); barrier();
    storageBarrier();
    // textureBarrier should not be needed as long as we split downsamping into two passes
    //textureBarrier(); // a storage texture is in handle space? // requires readonly_and_readwrite_storage_textures extension
    workgroupBarrier();
}

${filterOp}

fn spd_store(pix: vec2<u32>, out_value: vec4<f32>, mip: u32, slice: u32) {
    store_dst_mip(out_value, pix, slice, mip + 1);
}

fn spd_load_intermediate(x: u32, y: u32) -> vec4<f32> {
    return vec4<f32>(get_intermediate_r(x, y), get_intermediate_g(x, y), get_intermediate_b(x, y), get_intermediate_a(x, y));
}

fn spd_store_intermediate(x: u32, y: u32, value: vec4<f32>) {
    set_intermediate_r(x, y, value.x);
    set_intermediate_g(x, y, value.y);
    set_intermediate_b(x, y, value.z);
    set_intermediate_a(x, y, value.w);
}

fn spd_reduce_intermediate(i0: vec2<u32>, i1: vec2<u32>, i2: vec2<u32>, i3: vec2<u32>) -> vec4<f32> {
    let v0 = spd_load_intermediate(i0.x, i0.y);
    let v1 = spd_load_intermediate(i1.x, i1.y);
    let v2 = spd_load_intermediate(i2.x, i2.y);
    let v3 = spd_load_intermediate(i3.x, i3.y);
    return spd_reduce_4(v0, v1, v2, v3);
}

fn spd_reduce_load_4(base: vec2<u32>, slice: u32) -> vec4<f32> {
    let v0 = load_src_image(base + vec2<u32>(0, 0), slice);
    let v1 = load_src_image(base + vec2<u32>(0, 1), slice);
    let v2 = load_src_image(base + vec2<u32>(1, 0), slice);
    let v3 = load_src_image(base + vec2<u32>(1, 1), slice);
    return spd_reduce_4(v0, v1, v2, v3);
}

// Main logic ---------------------------------------------------------------------------------------------------------

fn spd_downsample_mips_0_1(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, mip: u32, slice: u32) {
    var v: array<vec4<f32>, 4>;

    let workgroup64 = workgroup_id.xy * 64;
    let workgroup32 = workgroup_id.xy * 32;
    let workgroup16 = workgroup_id.xy * 16;

    var tex = workgroup64 + vec2<u32>(x * 2, y * 2);
    var pix = workgroup32 + vec2<u32>(x, y);
    v[0] = spd_reduce_load_4(tex, slice);
    spd_store(pix, v[0], 0, slice);

    tex = workgroup64 + vec2<u32>(x * 2 + 32, y * 2);
    pix = workgroup32 + vec2<u32>(x + 16, y);
    v[1] = spd_reduce_load_4(tex, slice);
    spd_store(pix, v[1], 0, slice);

    tex = workgroup64 + vec2<u32>(x * 2, y * 2 + 32);
    pix = workgroup32 + vec2<u32>(x, y + 16);
    v[2] = spd_reduce_load_4(tex, slice);
    spd_store(pix, v[2], 0, slice);

    tex = workgroup64 + vec2<u32>(x * 2 + 32, y * 2 + 32);
    pix = workgroup32 + vec2<u32>(x + 16, y + 16);
    v[3] = spd_reduce_load_4(tex, slice);
    spd_store(pix, v[3], 0, slice);

    if mip <= 1 {
        return;
    }

    for (var i = 0u; i < 4u; i++) {
        spd_store_intermediate(x, y, v[i]);
        spd_barrier();
        if local_invocation_index < 64 {
            v[i] = spd_reduce_intermediate(
                vec2<u32>(x * 2 + 0, y * 2 + 0),
                vec2<u32>(x * 2 + 1, y * 2 + 0),
                vec2<u32>(x * 2 + 0, y * 2 + 1),
                vec2<u32>(x * 2 + 1, y * 2 + 1)
            );
            spd_store(workgroup16 + vec2<u32>(x + (i % 2) * 8, y + (i / 2) * 8), v[i], 1, slice);
        }
        spd_barrier();
    }

    if local_invocation_index < 64 {
        spd_store_intermediate(x + 0, y + 0, v[0]);
        spd_store_intermediate(x + 8, y + 0, v[1]);
        spd_store_intermediate(x + 0, y + 8, v[2]);
        spd_store_intermediate(x + 8, y + 8, v[3]);
    }
}

fn spd_downsample_mip_2(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, mip: u32, slice: u32) {
    if local_invocation_index < 64u {
        let v = spd_reduce_intermediate(
            vec2<u32>(x * 2 + 0, y * 2 + 0),
            vec2<u32>(x * 2 + 1, y * 2 + 0),
            vec2<u32>(x * 2 + 0, y * 2 + 1),
            vec2<u32>(x * 2 + 1, y * 2 + 1)
        );
        spd_store(workgroup_id.xy * 8 + vec2<u32>(x, y), v, mip, slice);
        // store to LDS, try to reduce bank conflicts
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0 x
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        // ...
        // x 0 x 0 x 0 x 0 x 0 x 0 x 0 x 0
        spd_store_intermediate(x * 2 + y % 2, y * 2, v);
    }
}

fn spd_downsample_mip_3(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, mip: u32, slice: u32) {
    if local_invocation_index < 16u {
        // x 0 x 0
        // 0 0 0 0
        // 0 x 0 x
        // 0 0 0 0
        let v = spd_reduce_intermediate(
            vec2<u32>(x * 4 + 0 + 0, y * 4 + 0),
            vec2<u32>(x * 4 + 2 + 0, y * 4 + 0),
            vec2<u32>(x * 4 + 0 + 1, y * 4 + 2),
            vec2<u32>(x * 4 + 2 + 1, y * 4 + 2)
        );
        spd_store(workgroup_id.xy * 4 + vec2<u32>(x, y), v, mip, slice);
        // store to LDS
        // x 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
        // 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0 0
        // ...
        // 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x 0
        // ...
        // 0 0 0 x 0 0 0 x 0 0 0 x 0 0 0 x
        // ...
        spd_store_intermediate(x * 4 + y, y * 4, v);
    }
}

fn spd_downsample_mip_4(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, mip: u32, slice: u32) {
    if local_invocation_index < 4u {
        // x 0 0 0 x 0 0 0
        // ...
        // 0 x 0 0 0 x 0 0
        let v = spd_reduce_intermediate(
            vec2<u32>(x * 8 + 0 + 0 + y * 2, y * 8 + 0),
            vec2<u32>(x * 8 + 4 + 0 + y * 2, y * 8 + 0),
            vec2<u32>(x * 8 + 0 + 1 + y * 2, y * 8 + 4),
            vec2<u32>(x * 8 + 4 + 1 + y * 2, y * 8 + 4)
        );
        spd_store(workgroup_id.xy * 2 + vec2<u32>(x, y), v, mip, slice);
        // store to LDS
        // x x x x 0 ...
        // 0 ...
        spd_store_intermediate(x + y * 2, 0, v);
    }
}

fn spd_downsample_mip_5(workgroup_id: vec2<u32>, local_invocation_index: u32, mip: u32, slice: u32) {
    if local_invocation_index < 1u {
        // x x x x 0 ...
        // 0 ...
        let v = spd_reduce_intermediate(vec2<u32>(0, 0), vec2<u32>(1, 0), vec2<u32>(2, 0), vec2<u32>(3, 0));
        spd_store(workgroup_id.xy, v, mip, slice);
    }
}

fn spd_downsample_next_four(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, base_mip: u32, mips: u32, slice: u32) {
    if mips <= base_mip {
        return;
    }
    spd_barrier();
    spd_downsample_mip_2(x, y, workgroup_id, local_invocation_index, base_mip, slice);

    if mips <= base_mip + 1 {
        return;
    }
    spd_barrier();
    spd_downsample_mip_3(x, y, workgroup_id, local_invocation_index, base_mip + 1, slice);

    if mips <= base_mip + 2 {
        return;
    }
    spd_barrier();
    spd_downsample_mip_4(x, y, workgroup_id, local_invocation_index, base_mip + 2, slice);

    if mips <= base_mip + 3 {
        return;
    }
    spd_barrier();
    spd_downsample_mip_5(workgroup_id, local_invocation_index, base_mip + 3, slice);
}

fn spd_downsample_mips_6_7(x: u32, y: u32, mips: u32, slice: u32) {
    var tex = vec2<u32>(x * 4 + 0, y * 4 + 0);
    var pix = vec2<u32>(x * 2 + 0, y * 2 + 0);
    let v0 = spd_reduce_load_4(tex, slice);
    spd_store(pix, v0, 0, slice);

    tex = vec2<u32>(x * 4 + 2, y * 4 + 0);
    pix = vec2<u32>(x * 2 + 1, y * 2 + 0);
    let v1 = spd_reduce_load_4(tex, slice);
    spd_store(pix, v1, 0, slice);

    tex = vec2<u32>(x * 4 + 0, y * 4 + 2);
    pix = vec2<u32>(x * 2 + 0, y * 2 + 1);
    let v2 = spd_reduce_load_4(tex, slice);
    spd_store(pix, v2, 0, slice);

    tex = vec2<u32>(x * 4 + 2, y * 4 + 2);
    pix = vec2<u32>(x * 2 + 1, y * 2 + 1);
    let v3 = spd_reduce_load_4(tex, slice);
    spd_store(pix, v3, 0, slice);

    if mips <= 7 {
        return;
    }
    // no barrier needed, working on values only from the same thread

    let v = spd_reduce_4(v0, v1, v2, v3);
    spd_store(vec2<u32>(x, y), v, 1, slice);
    spd_store_intermediate(x, y, v);
}

/// Downsamples a 64x64 tile based on the work group id.
/// If after downsampling it's the last active thread group, computes the remaining MIP levels.
///
/// @param [in] workGroupID             index of the work group / thread group
/// @param [in] localInvocationIndex    index of the thread within the thread group in 1D
/// @param [in] mips                    the number of total MIP levels to compute for the input texture
/// @param [in] numWorkGroups           the total number of dispatched work groups / thread groups for this slice
/// @param [in] slice                   the slice of the input texture
fn spd_downsample(workgroup_id: vec2<u32>, local_invocation_index: u32, mips: u32, num_work_groups: u32, slice: u32) {
    let xy = map_to_xy(local_invocation_index);
    spd_downsample_mips_0_1(xy.x, xy.y, workgroup_id, local_invocation_index, mips, slice);
    spd_downsample_next_four(xy.x, xy.y, workgroup_id, local_invocation_index, 2, mips, slice);
}

// Entry points -------------------------------------------------------------------------------------------------------

@compute
@workgroup_size(256, 1, 1)
fn downsample(@builtin(local_invocation_index) local_invocation_index: u32, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    spd_downsample(
        workgroup_id.xy + get_work_group_offset(),
        local_invocation_index,
        get_mips(),
        get_num_work_groups(),
        workgroup_id.z
    );
}
    `;
}

const SPD_FILTER_AVERAGE: string = /* wgsl */`
fn spd_reduce_4(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32> {
    return (v0 + v1 + v2 + v3) * 0.25;
}
`;

const SPD_FILTER_MIN = /* wgsl */`
fn spd_reduce_4(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32> {
    return min(min(v0, v1), min(v2, v3));
}
`;

const SPD_FILTER_MAX = /* wgsl */`
fn spd_reduce_4(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32> {
    return max(max(v0, v1), max(v2, v3));
}
`;

const SPD_FILTER_MINMAX = /* wgsl */`
fn spd_reduce_4(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32> {
    let max4 = max(max(v0.xy, v1.xy), max(v2.xy, v3.xy));
    return vec4<f32>(min(min(v0.x, v1.x), min(v2.x, v3.x)), max(max4.x, max4.y), 0, 0);
}
`;

const SUPPORTED_FORMATS: Set<string> = new Set([
    'rgba8unorm',
    'rgba8snorm',
    'rgba8uint',
    'rgba8sint',
    'bgra8unorm', // if bgra8unorm-storage is enabled
    'rgba16uint',
    'rgba16sint',
    'rgba16float',
    'r32uint',
    'r32sint',
    'r32float',
    'rg32uint',
    'rg32sint',
    'rg32float',
    'rgba32uint',
    'rgba32sint',
    'rgba32float',
]);

/**
 * The names of all predefined filters of {@link WebGPUSinglePassDownsampler}.
 * Custom ones can be registered with an instance of {@link WebGPUSinglePassDownsampler} using @see WebGPUSinglePassDownsampler.registerFilter
 */
export enum SPDFilters {
    /**
     * Takes the channel-wise averages of 4 pixels
     */
    Average = 'average',

    /**
     * Takes the channel-wise minimum of 4 pixels.
     */
    Min = 'min',
    
    /**
     * Takes the channel-wise maximum of 4 pixels.
     */
    Max = 'max',

    /**
     * Takes the minimum of the red channel and the maximum of the red and green channel and stores the result in the red and green channel respectively.
     * This really only makes sense for single-channel input textures (where only the red channel holds any data), e.g., for generating a min-max pyramid of a depth buffer.
     */
    MinMax = 'minmax',
}

class SPDPassInner {
    constructor(private pipeline: GPUComputePipeline, private bindGroups: Array<GPUBindGroup>, private dispatchDimensions: [number, number, number]) {}
    encode(computePass: GPUComputePassEncoder) {
        computePass.setPipeline(this.pipeline);
        this.bindGroups.forEach((bindGroup, index) => {
            computePass.setBindGroup(index, bindGroup);
        });
        computePass.dispatchWorkgroups(...this.dispatchDimensions);
    }
}

export class SPDPass {
    /**
     * The texture the mipmaps will be written to by this {@link SPDPass}, once {@link SPDPass.encode} is called.
     */
    readonly target: GPUTexture

    /** @ignore */
    constructor(private passes: Array<SPDPassInner>, target: GPUTexture) {
        this.target = target;
    }
    /**
     * Encodes the configured mipmap generation pass with the given {@link GPUComputePassEncoder}.
     * @param computePassEncoder the {@link GPUComputePassEncoder} to encode this mipmap generation pass with.
     * @returns the {@link computePassEncoder}
     */
    encode(computePassEncoder: GPUComputePassEncoder): GPUComputePassEncoder {
        this.passes.forEach(p => p.encode(computePassEncoder));
        computePassEncoder.setBindGroup(0, null);
        computePassEncoder.setBindGroup(1, null);
        return computePassEncoder;
    }
}

export interface SPDPassConfig {
    /**
     * The name of the filter to use for downsampling the given texture.
     * Should be one of the filters registered with @type {WebGPUSinglePassDownsampler}.
     * Defaults to {@link SPDFilters.Average}
     */
    filter?: string,

    /**
     * The target texture the generated mipmaps are written to.
     * Its usage must include {@link GPUTextureUsage.STORAGE_BINDING}.
     * Its format must support {@link GPUStorageTextureAccess:"write-only"}.
     * Its size must be big enough to store the first mip level generated for the input texture.
     * It must support generating a {@link GPUTextureView} with {@link GPUTextureViewDimension:"2d-array"}.
     * Defaults to the given input texture.
     */
    target?: GPUTexture,

    /**
     * The upper left corner of the image region mipmaps should be generated for.
     * Defaults to [0,0].
     */
    offset?: [number, number],

    /**
     * The size of the image reagion mipmaps should be generated for.
     * Default to [texture.width - 1 - offset[0], texture.height - 1 - offset[1]]
     */
    size?: [number, number],

    /**
     * The number of mipmaps to generate.
     * Defaults to target.mipLevelCount
     */
    numMips?: number,
}

interface GPUDownsamplingMeta {
    workgroupOffset: [number, number],
    numWorkGroups: number,
    numMips: number,
}

class SPDPipeline {
    constructor(readonly mipsLayout: GPUBindGroupLayout, readonly pipelines: GPUComputePipeline) {}
}

export interface SPDPrepareFormatDescriptor {
    /**
     * The texture format to prepare downsampling pipelines for.
     */
    format: GPUTextureFormat,

    /**
     * The names of downsampling filters that to prepare downsampling pipelines for the given {@link format} for.
     * Defaults to {@link SPDFilters.Average}.
     */
    filters?: Set<string>,
}

export interface SPDPrepareDeviceDescriptor {
    /**
     * The device to prepare downsampling pipelines for.
     */
    device: GPUDevice,

    /**
     * The formats to prepare downsampling pipelines for.
     */
    formats?: Array<SPDPrepareFormatDescriptor>,
}

class DevicePipelines {
    private device: WeakRef<GPUDevice>;
    private maxMipsPerPass: number;
    private internalResourcesBindGroupLayout: GPUBindGroupLayout;
    private pipelines: Map<GPUTextureFormat, Map<string, Map<number, SPDPipeline>>>;

    constructor(device: GPUDevice) {
        this.device = new WeakRef(device);
        this.maxMipsPerPass = Math.min(device.limits.maxStorageTexturesPerShaderStage, 6);
        this.pipelines = new Map();
        this.internalResourcesBindGroupLayout = device.createBindGroupLayout({
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.COMPUTE,
                buffer: {
                    type: 'uniform',
                    hasDynamicOffset: false,
                    minBindingSize: 16,
                },
            }],
        });
    }

    preparePipelines(pipelineConfigs?: Array<SPDPrepareFormatDescriptor>) {
        pipelineConfigs?.map(c => {
            Array.from(c.filters ?? [SPD_FILTER_AVERAGE]).map(f => {
                for (let i = 0; i < this.maxMipsPerPass; ++i) {
                    this.getOrCreatePipeline(c.format, f, i + 1);
                }
            })
            
        });
    }

    private createPipeline(targetFormat: GPUTextureFormat, filterCode: string, numMips: number): SPDPipeline | undefined {
        const device = this.device.deref();
        if (!device) {
            return undefined;
        }
        const mipsBindGroupLayout = device.createBindGroupLayout({
            entries: Array(Math.min(numMips, this.maxMipsPerPass) + 1).fill(0).map((_, i) => {
                const entry: GPUBindGroupLayoutEntry = {
                    binding: i,
                    visibility: GPUShaderStage.COMPUTE,
                };
                if (i === 0) {
                    entry.texture = {
                        sampleType: 'unfilterable-float',
                        viewDimension: '2d-array',
                        multisampled: false,
                    };
                } else {
                    entry.storageTexture = {
                        access: 'write-only',
                        format: targetFormat,
                        viewDimension: '2d-array',
                    };
                }
                return entry;
            })
        });

        const module = device.createShaderModule({
            code: makeShaderCode(targetFormat, filterCode, Math.min(numMips, this.maxMipsPerPass)),
        });

        return new SPDPipeline(
            mipsBindGroupLayout,
            device.createComputePipeline({
                compute: {
                    module,
                    entryPoint: 'downsample',
                },
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [
                        mipsBindGroupLayout,
                        this.internalResourcesBindGroupLayout,
                    ],
                }),
            }),
        );
    }

    private getOrCreatePipeline(targetFormat: GPUTextureFormat, filterCode: string, numMipsToCreate: number): SPDPipeline | undefined {
        if (!this.pipelines.has(targetFormat)) {
            this.pipelines.set(targetFormat, new Map());
        }
        if (!this.pipelines.get(targetFormat)?.has(filterCode)) {
            this.pipelines.get(targetFormat)?.set(filterCode, new Map());
        }
        if (!this.pipelines.get(targetFormat)?.get(filterCode)?.has(numMipsToCreate)) {
            const pipelines = this.createPipeline(targetFormat, filterCode, numMipsToCreate);
            if (pipelines) {
                this.pipelines.get(targetFormat)?.get(filterCode)?.set(numMipsToCreate, pipelines);
            }
        }
        return this.pipelines.get(targetFormat)?.get(filterCode)?.get(numMipsToCreate);
    }

    private createMetaBindGroup(device: GPUDevice, meta: GPUDownsamplingMeta): GPUBindGroup {
        const metaBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(metaBuffer, 0, new Uint32Array([
            ...meta.workgroupOffset,
            meta.numWorkGroups,
            meta.numMips,
        ]));
        return device.createBindGroup({
            layout: this.internalResourcesBindGroupLayout,
            entries: [{
                binding: 0,
                resource: {
                    buffer: metaBuffer,
                },
            }]
        });
    }

    preparePass(texture: GPUTexture, target: GPUTexture, filterCode: string, offset: [number, number], size: [number, number], numMipsTotal: number): SPDPass | undefined {
        const device = this.device.deref();
        if (!device) {
            return undefined;
        }

        const passes = [];
        for (let baseMip = 0; baseMip < numMipsTotal - 1; baseMip += this.maxMipsPerPass) {
            const numMipsThisPass = Math.min(numMipsTotal - 1 - baseMip, this.maxMipsPerPass);

            const baseMipOffset = offset.map(o => Math.floor(o / Math.pow(2, baseMip)));
            const baseMipSize = size.map(s => Math.max(Math.floor(s / Math.pow(2, baseMip)), 1));
            const workgroupOffset = baseMipOffset.map(o => o / 64) as [number, number];
            const dispatchDimensions = baseMipOffset.map((o, i) => ((o + baseMipSize[i] - 1) / 64) + 1 - workgroupOffset[i]) as [number, number];
            const numWorkGroups = dispatchDimensions.reduce((product, v) => v * product, 1);

            const metaBindGroup = this.createMetaBindGroup(device, {
                workgroupOffset,
                numWorkGroups,
                numMips: numMipsThisPass
            });

            // todo: handle missing pipeline
            const pipeline = this.getOrCreatePipeline(target.format, filterCode, numMipsThisPass)!;

            const mipViews = Array(numMipsThisPass + 1).fill(0).map((_, i) => {
                if (baseMip === 0 && i === 0) {
                    return texture.createView({
                        dimension: '2d-array',
                        baseMipLevel: 0,
                        mipLevelCount: 1,
                        baseArrayLayer: 0,
                        arrayLayerCount: texture.depthOrArrayLayers,
                    });
                } else {
                    const mip = baseMip + i;
                    return target.createView({
                        dimension: '2d-array',
                        baseMipLevel: texture === target ? mip : mip - 1,
                        mipLevelCount: 1,
                        baseArrayLayer: 0,
                        arrayLayerCount: target.depthOrArrayLayers, 
                    });
                }
            });

            const mipsBindGroup = device.createBindGroup({
                layout: pipeline.mipsLayout,
                entries: mipViews.map((v, i) => {
                    return {
                        binding: i,
                        resource: v,
                    };
                }),
            });
            passes.push(new SPDPassInner(pipeline.pipelines, [mipsBindGroup, metaBindGroup], [...dispatchDimensions, Math.min(texture.depthOrArrayLayers, target.depthOrArrayLayers)]));
        }
        return new SPDPass(passes, target);
    }
}

/**
 * Returns the maximum number of mip levels for a given n-dimensional size.
 * @param size the size to compute the maximum number of mip levels for
 * @returns the maximum number of mip levels for the given size
 */
export function maxMipLevelCount(...size: number[]): number {
    return 1 + Math.log2(Math.max(0, ...size));
}

/**
 * A helper class for downsampling a 2D {@link GPUTexture} (array) using as few passes as possible on a {@link GPUDevice} depending on its {@link GPUSupportedLimits}.
 * Up to 6 mip levels can be generated within a single pass, if {@link GPUSupportedLimits.maxStorageTexturesPerShaderStage} supports it.
 */
export class WebGPUSinglePassDownsampler {
    private filters: Map<string, string>;
    private devicePipelines: WeakMap<GPUDevice, DevicePipelines>;

    /**
     * Sets the preferred device limits for {@link WebGPUSinglePassDownsampler} in a given record of limits.
     * Existing preferred device limits are either increased or left untouched.
     * If {@link limits} is undefined, creates a new record of preferred device limits for {@link WebGPUSinglePassDownsampler}.
     * The result can be used to set {@link GPUDeviceDescriptor.requiredLimits} when requesting a device.
     * @param limits a record of device limits set to update with the preferred limits for {@link WebGPUSinglePassDownsampler}
     * @returns the updated or created set of device limits with all preferred limits for {@link WebGPUSinglePassDownsampler} set
     */
    static setPreferredLimits(limits?: Record<string, number | GPUSize64>): Record<string, number | GPUSize64> {
        if (!limits) {
            limits = {};
        }
        limits.maxStorageTexturesPerShaderStage = Math.max(limits.maxStorageTexturesPerShaderStage ?? 6);
        return limits;
    }

    /**
     * Creates a new {@link WebGPUSinglePassDownsampler}.
     * On its own, {@link WebGPUSinglePassDownsampler} does not allocate any GPU resources.
     * Optionally, prepare GPU resources for a given {@link SPDPrepareDeviceDescriptor}.
     * @param prepareDescriptor an optional descriptor for preparing GPU resources
     * @see WebGPUSinglePassDownsampler.prepareDeviceResources
     */
    constructor(prepareDescriptor?: SPDPrepareDeviceDescriptor) {
        this.filters = new Map([
            [SPDFilters.Average, SPD_FILTER_AVERAGE],
            [SPDFilters.Min, SPD_FILTER_MIN],
            [SPDFilters.Max, SPD_FILTER_MAX],
            [SPDFilters.MinMax, SPD_FILTER_MINMAX],
        ]);
        this.devicePipelines = new Map();

        if (prepareDescriptor) {
            this.prepareDeviceResources(prepareDescriptor);
        }
    }

    /**
     * Prepares GPU resources required by {@link WebGPUSinglePassDownsampler} to downsample textures for a given {@link SPDPrepareDeviceDescriptor}.
     * @param prepareDescriptor a descriptor for preparing GPU resources
     */
    prepareDeviceResources(prepareDescriptor: SPDPrepareDeviceDescriptor) {
        this.getOrCreateDevicePipelines(prepareDescriptor.device)?.preparePipelines(prepareDescriptor?.formats?.map(format => {
            return {
                ...format,
                filters: new Set(Array.from(format.filters ?? []).map(filter => this.filters.get(filter) ?? SPD_FILTER_AVERAGE)),
            };
        }));
    }

    private getOrCreateDevicePipelines(device: GPUDevice): DevicePipelines | undefined {
        if (!this.devicePipelines.has(device)) {
            this.devicePipelines.set(device, new DevicePipelines(device));
        }
        return this.devicePipelines.get(device);
    }

    /**
     * Deregisters all resources stored for a given device.
     * @param device the device resources should be deregistered for
     */
    deregisterDevice(device: GPUDevice) {
        this.devicePipelines.delete(device);
    }

    /**
     * Registers a new downsampling filter operation that can be injected into the downsampling shader for new pipelines.
     * 
     * The given WGSL code must (at least) specify a function to reduce four values into one with the following name and signature:
     * 
     *   spd_reduce_4(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32>
     * 
     * @param name the unique name of the filter operation
     * @param wgsl the WGSL code to inject into the downsampling shader as the filter operation
     */
    registerFilter(name: string, wgsl: string) {
        if (this.filters.has(name)) {
            console.warn(`[GPUSinglePassDownsampler::registerFilter]: overriding existing filter '${name}'. Previously generated pipelines are not affected.`);
        }
        this.filters.set(name, wgsl);
    }

    /**
     * Prepares a pass to downsample a texture.
     * The produces {@link SPDPass} can be used multiple time to repeatedly downsampling a texture, e.g., for downsampling the depth buffer each frame. 
     * For one-time use, @see WebGPUSinglePassDownsampler.generateMipmaps can be used instead.
     * @param device the device the {@link SPDPass} should be prepared for
     * @param texture the texture that is to be processed by the {@link SPDPass}. Must support generating a {@link GPUTextureView} with {@link GPUTextureViewDimension:"2d-array"}.
     * @param config the config for the {@link SPDPass}
     * @returns the prepared {@link SPDPass} or undefined if preparation failed or if no mipmaps would be generated.
     * @throws if the {@link GPUTextureFormat} of {@link SPDPassConfig.target} is not supported (does not support {@link GPUStorageTextureAccess:"write-only"} on the given {@link device}).
     * @throws if the size of {@link SPDPassConfig.target} is too small to store the first mip level generated for {@link texture}
     * @throws if {@link texture} or {@link SPDPassConfig.target} is not a 2d texture.
     */
    preparePass(device: GPUDevice, texture: GPUTexture, config?: SPDPassConfig): SPDPass | undefined {
        const target = config?.target ?? texture;
        const filter = config?.filter ?? SPDFilters.Average;
        const offset = (config?.offset ?? [0, 0]).map((o, d) => Math.max(0, Math.min(o, (d === 0 ? texture.width : texture.height) - 1))) as [number, number];
        const size = (config?.size ?? [texture.width, texture.height]).map((s, d) => Math.max(0, Math.min(s, (d === 0 ? texture.width : texture.height) - offset[d]))) as [number, number];
        const numMips = Math.min(Math.max(config?.numMips ?? target.mipLevelCount, 0), 1 + Math.log2(Math.max(...size)));

        if (numMips < 2) {
            console.warn(`[GPUSinglePassDownsampler::prepare]: no mips to create (numMips = ${numMips})`);
            return undefined;
        }
        if (!SUPPORTED_FORMATS.has(target.format)) {
            throw new Error(`[GPUSinglePassDownsampler::prepare]: format ${target.format} not supported`);
        }
        if (target.format === 'bgra8unorm' && !device.features.has('bgra8unorm-storage')) {
            throw new Error(`[GPUSinglePassDownsampler::prepare]: format ${target.format} not supported without feature 'bgra8unorm-storage' enabled`);
        }
        if (target.width < Math.max(1, Math.floor(size[0] / 2)) || target.height < Math.max(1, Math.floor(size[1] / 2))) {
            throw new Error(`[GPUSinglePassDownsampler::prepare]: target too small (${[target.width, target.height]}) for input size ${size}`);
        }
        if (target.dimension !== '2d' || texture.dimension !== '2d') {
            throw new Error('[GPUSinglePassDownsampler::prepare]: texture or target is not a 2d texture');
        }
        if (!this.filters.has(filter)) {
            console.warn(`[GPUSinglePassDownsampler::prepare]: unknown filter ${filter}, falling back to average`);
        }
        if (filter === SPD_FILTER_MINMAX && target.format.includes('r32')) {
            console.warn(`[GPUSinglePassDownsampler::prepare]: filter ${filter} makes no sense for one-component target format ${target.format}`);
        }
        const filterCode = this.filters.get(filter) ?? SPD_FILTER_AVERAGE;

        return this.getOrCreateDevicePipelines(device)?.preparePass(texture, target, filterCode, offset, size, numMips);
    }

    /**
     * Generates mipmaps for the given texture.
     * For textures that will be downsampled more than once, consider generating a {@link SPDPass} using @see WebGPUSinglePassDownsampler.preparePass and calling its {@link SPDPass.encode} method.
     * This way, allocated GPU resources for downsampling the texture can be reused.
     * @param device the device to use for downsampling the texture
     * @param texture the texture to generate mipmaps for. Must support generating a {@link GPUTextureView} with {@link GPUTextureViewDimension:"2d-array"}.
     * @param config the config for mipmap generation
     * @returns true if mipmaps were generated, false otherwise
     * @throws if @see WebGPUSinglePassDownsampler.preparePass threw an error.
     */
    generateMipmaps(device: GPUDevice, texture: GPUTexture, config?: SPDPassConfig): boolean {
        const pass = this.preparePass(device, texture, config);
        if (!pass) {
            return false;
        } else {
            const commandEncoder = device.createCommandEncoder();
            pass?.encode(commandEncoder.beginComputePass()).end();
            device.queue.submit([commandEncoder.finish()]);
            return true;
        }
    }
}

