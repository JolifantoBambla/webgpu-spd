function makeShaderCode(outputFormat: string, filterOp: string = SPD_FILTER_AVERAGE, numMips: number, scalarType: SPDScalarType): string {
    const texelType = scalarType === SPDScalarType.I32 ? 'i32' : (scalarType === SPDScalarType.U32 ? 'u32' : 'f32');
    const useF16 = scalarType === SPDScalarType.F16;

    const filterCode = filterOp === SPD_FILTER_AVERAGE && !['f32', 'f16'].includes(texelType) ? filterOp.replace('* SPDScalar(0.25)', '/ 4') : filterOp;

    const mipsBindings = Array(numMips).fill(0)
        .map((_, i) => `@group(0) @binding(${i + 1}) var dst_mip_${i + 1}: texture_storage_2d_array<${outputFormat}, write>;`)
        .join('\n');

    // todo: get rid of this branching as soon as WGSL supports arrays of texture_storage_2d_array
    const mipsAccessorBody = Array(numMips).fill(0)
        .map((_, i) => {
            if (i == 5 && numMips > 6) {
                return ` else if mip == 6 {
                    textureStore(dst_mip_6, uv, slice, ${useF16 ? `vec4<${texelType}>(value)` : 'value'});
                    mip_dst_6_buffer[slice][uv.y][uv.x] = value;
                }`
            }
            return `${i === 0 ? '' : ' else '}if mip == ${i + 1} {
                textureStore(dst_mip_${i + 1}, uv, slice, ${useF16 ? `vec4<${texelType}>(value)` : 'value'});
            }`;
        })
        .join('');
    
    const mipsAccessor = `fn store_dst_mip(value: vec4<SPDScalar>, uv: vec2<u32>, slice: u32, mip: u32) {\n${mipsAccessorBody}\n}`
    const midMipAccessor =`return mip_dst_6_buffer[slice][uv.y][uv.x];`;

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


// Definitions --------------------------------------------------------------------------------------------------------

${useF16 ? 'enable f16;' : ''}
alias SPDScalar = ${scalarType};

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
/*
fn srgb_to_linear(value: SPDScalar) -> SPDScalar {
    let j = vec3<SPDScalar>(0.0031308 * 12.92, 12.92, 1.0 / 2.4);
    let k = vec2<SPDScalar>(1.055, -0.055);
    return clamp(j.x, value * j.y, pow(value, j.z) * k.x + k.y);
}
*/

// Resources & Accessors -----------------------------------------------------------------------------------------------
struct DownsamplePassMeta {
    work_group_offset: vec2<u32>,
    num_work_groups: u32,
    mips: u32,
}

// In the original version dst_mip_i is an image2Darray [SPD_MAX_MIP_LEVELS+1], i.e., 12+1, but WGSL doesn't support arrays of textures yet
// Also these are read_write because for mips 7-13, the workgroup reads from mip level 6 - since most formats don't support read_write access in WGSL yet, we use a single read_write buffer in such cases instead
@group(0) @binding(0) var src_mip_0: texture_2d_array<${texelType}>;
${mipsBindings}

@group(1) @binding(0) var<uniform> downsample_pass_meta : DownsamplePassMeta;
@group(1) @binding(1) var<storage, read_write> spd_global_counter: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> mip_dst_6_buffer: array<array<array<vec4<f32>, 64>, 64>>;

fn get_mips() -> u32 {
    return downsample_pass_meta.mips;
}

fn get_num_work_groups() -> u32 {
    return downsample_pass_meta.num_work_groups;
}

fn get_work_group_offset() -> vec2<u32> {
    return downsample_pass_meta.work_group_offset;
}

fn load_src_image(uv: vec2<u32>, slice: u32) -> vec4<SPDScalar> {
    return vec4<SPDScalar>(textureLoad(src_mip_0, uv, slice, 0));
}

fn load_mid_mip_image(uv: vec2<u32>, slice: u32) -> vec4<SPDScalar> {
    ${numMips > 6 ? midMipAccessor : 'return vec4<SPDScalar>();'}
}

${mipsAccessor}

// Workgroup -----------------------------------------------------------------------------------------------------------

${useF16 ? `
var<workgroup> spd_intermediate_rg: array<array<vec2<SPDScalar>, 16>, 16>;
var<workgroup> spd_intermediate_bg: array<array<vec2<SPDScalar>, 16>, 16>;
`: `
var<workgroup> spd_intermediate_r: array<array<SPDScalar, 16>, 16>;
var<workgroup> spd_intermediate_g: array<array<SPDScalar, 16>, 16>;
var<workgroup> spd_intermediate_b: array<array<SPDScalar, 16>, 16>;
var<workgroup> spd_intermediate_a: array<array<SPDScalar, 16>, 16>;
`}
var<workgroup> spd_counter: atomic<u32>;

fn spd_increase_atomic_counter(slice: u32) {
    atomicStore(&spd_counter, atomicAdd(&spd_global_counter[slice], 1));
}

fn spd_get_atomic_counter() -> u32 {
    return workgroupUniformLoad(&spd_counter);
}

fn spd_reset_atomic_counter(slice: u32) {
    atomicStore(&spd_global_counter[slice], 0);
}

// Cotnrol flow --------------------------------------------------------------------------------------------------------

fn spd_barrier() {
    // in glsl this does: groupMemoryBarrier(); barrier();
    workgroupBarrier();
}

// Only last active workgroup should proceed
fn spd_exit_workgroup(num_work_groups: u32, local_invocation_index: u32, slice: u32) -> bool {
    // global atomic counter
    if (local_invocation_index == 0) {
        spd_increase_atomic_counter(slice);
    }
    storageBarrier();
    return spd_get_atomic_counter() != (num_work_groups - 1);
}

// Pixel access --------------------------------------------------------------------------------------------------------

${filterCode}

fn spd_store(pix: vec2<u32>, out_value: vec4<SPDScalar>, mip: u32, slice: u32) {
    store_dst_mip(out_value, pix, slice, mip + 1);
}

fn spd_load_intermediate(x: u32, y: u32) -> vec4<SPDScalar> {
    return vec4<SPDScalar>(${useF16 ? `
        spd_intermediate_rg[x][y],
        spd_intermediate_ba[x][y],` : `
        spd_intermediate_r[x][y],
        spd_intermediate_g[x][y],
        spd_intermediate_b[x][y],
        spd_intermediate_a[x][y],`
    });
}

fn spd_store_intermediate(x: u32, y: u32, value: vec4<SPDScalar>) {
${useF16 ? `
        spd_intermediate_rg[x][y] = value.rg;
        spd_intermediate_ba[x][y] = value.ba;` : `
        spd_intermediate_r[x][y] = value.r;
        spd_intermediate_g[x][y] = value.g;
        spd_intermediate_b[x][y] = value.b;
        spd_intermediate_a[x][y] = value.a;`}
}

fn spd_reduce_intermediate(i0: vec2<u32>, i1: vec2<u32>, i2: vec2<u32>, i3: vec2<u32>) -> vec4<SPDScalar> {
    let v0 = spd_load_intermediate(i0.x, i0.y);
    let v1 = spd_load_intermediate(i1.x, i1.y);
    let v2 = spd_load_intermediate(i2.x, i2.y);
    let v3 = spd_load_intermediate(i3.x, i3.y);
    return spd_reduce_4(v0, v1, v2, v3);
}

fn spd_reduce_load_4(base: vec2<u32>, slice: u32) -> vec4<SPDScalar> {
    let v0 = load_src_image(base + vec2<u32>(0, 0), slice);
    let v1 = load_src_image(base + vec2<u32>(0, 1), slice);
    let v2 = load_src_image(base + vec2<u32>(1, 0), slice);
    let v3 = load_src_image(base + vec2<u32>(1, 1), slice);
    return spd_reduce_4(v0, v1, v2, v3);
}

fn spd_reduce_load_mid_mip_4(base: vec2<u32>, slice: u32) -> vec4<SPDScalar> {
    let v0 = load_mid_mip_image(base + vec2<u32>(0, 0), slice);
    let v1 = load_mid_mip_image(base + vec2<u32>(0, 1), slice);
    let v2 = load_mid_mip_image(base + vec2<u32>(1, 0), slice);
    let v3 = load_mid_mip_image(base + vec2<u32>(1, 1), slice);
    return spd_reduce_4(v0, v1, v2, v3);
}

// Main logic ---------------------------------------------------------------------------------------------------------

fn spd_downsample_mips_0_1(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, mip: u32, slice: u32) {
    var v: array<vec4<SPDScalar>, 4>;

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

fn spd_downsample_last_four(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, base_mip: u32, mips: u32, slice: u32) {
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
    let v0 = spd_reduce_load_mid_mip_4(tex, slice);
    spd_store(pix, v0, 6, slice);

    tex = vec2<u32>(x * 4 + 2, y * 4 + 0);
    pix = vec2<u32>(x * 2 + 1, y * 2 + 0);
    let v1 = spd_reduce_load_mid_mip_4(tex, slice);
    spd_store(pix, v1, 6, slice);

    tex = vec2<u32>(x * 4 + 0, y * 4 + 2);
    pix = vec2<u32>(x * 2 + 0, y * 2 + 1);
    let v2 = spd_reduce_load_mid_mip_4(tex, slice);
    spd_store(pix, v2, 6, slice);

    tex = vec2<u32>(x * 4 + 2, y * 4 + 2);
    pix = vec2<u32>(x * 2 + 1, y * 2 + 1);
    let v3 = spd_reduce_load_mid_mip_4(tex, slice);
    spd_store(pix, v3, 6, slice);

    if mips <= 7 {
        return;
    }
    // no barrier needed, working on values only from the same thread

    let v = spd_reduce_4(v0, v1, v2, v3);
    spd_store(vec2<u32>(x, y), v, 7, slice);
    spd_store_intermediate(x, y, v);
}

fn spd_downsample_last_6(x: u32, y: u32, local_invocation_index: u32, mips: u32, num_work_groups: u32, slice: u32) {
    if mips <= 6 {
        return;
    }

    // increase the global atomic counter for the given slice and check if it's the last remaining thread group:
    // terminate if not, continue if yes.
    if spd_exit_workgroup(num_work_groups, local_invocation_index, slice) {
        return;
    }

    // reset the global atomic counter back to 0 for the next spd dispatch
    spd_reset_atomic_counter(slice);

    // After mip 5 there is only a single workgroup left that downsamples the remaining up to 64x64 texels.
    // compute MIP level 6 and 7
    spd_downsample_mips_6_7(x, y, mips, slice);

    // compute MIP level 8, 9, 10, 11
    spd_downsample_last_four(x, y, vec2<u32>(0, 0), local_invocation_index, 8, mips, slice);
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
    ${numMips > 6 ? 'spd_downsample_last_6(xy.x, xy.y, local_invocation_index, mips, num_work_groups, slice);' : ''}
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
fn spd_reduce_4(v0: vec4<SPDScalar>, v1: vec4<SPDScalar>, v2: vec4<SPDScalar>, v3: vec4<SPDScalar>) -> vec4<SPDScalar> {
    return (v0 + v1 + v2 + v3) * SPDScalar(0.25);
}
`;

const SPD_FILTER_MIN = /* wgsl */`
fn spd_reduce_4(v0: vec4<SPDScalar>, v1: vec4<SPDScalar>, v2: vec4<SPDScalar>, v3: vec4<SPDScalar>) -> vec4<SPDScalar> {
    return min(min(v0, v1), min(v2, v3));
}
`;

const SPD_FILTER_MAX = /* wgsl */`
fn spd_reduce_4(v0: vec4<SPDScalar>, v1: vec4<SPDScalar>, v2: vec4<SPDScalar>, v3: vec4<SPDScalar>) -> vec4<SPDScalar> {
    return max(max(v0, v1), max(v2, v3));
}
`;

const SPD_FILTER_MINMAX = /* wgsl */`
fn spd_reduce_4(v0: vec4<SPDScalar>, v1: vec4<SPDScalar>, v2: vec4<SPDScalar>, v3: vec4<SPDScalar>) -> vec4<SPDScalar> {
    let max4 = max(max(v0.xy, v1.xy), max(v2.xy, v3.xy));
    return vec4<SPDScalar>(min(min(v0.x, v1.x), min(v2.x, v3.x)), max(max4.x, max4.y), 0, 0);
}
`;

/**
 * The names of all predefined filters of {@link WebGPUSinglePassDownsampler}.
 * Custom ones can be registered with an instance of {@link WebGPUSinglePassDownsampler} using {@link WebGPUSinglePassDownsampler.registerFilter}.
 */
export enum SPDFilters {
    /**
     * Takes the channel-wise average of 4 pixels.
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

/**
 * A compute pass for downsampling a texture.
 */
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
     * Encodes the configured mipmap generation pass(es) with the given {@link GPUComputePassEncoder}.
     * All bind groups indices used by {@link SPDPass} are reset to `null` to prevent unintentional bindings of internal bind groups for subsequent pipelines encoded in the same {@link GPUComputePassEncoder}.
     * @param computePassEncoder The {@link GPUComputePassEncoder} to encode this mipmap generation pass with.
     * @returns The {@link computePassEncoder}
     */
    encode(computePassEncoder: GPUComputePassEncoder): GPUComputePassEncoder {
        this.passes.forEach(p => p.encode(computePassEncoder));
        computePassEncoder.setBindGroup(0, null);
        computePassEncoder.setBindGroup(1, null);
        return computePassEncoder;
    }

    /**
     * Returns the number of passes that will be encoded by calling this instance's {@link SPDPass.encode} method.
     */
    get numPasses(): number {
        return this.passes.length
    }
}

enum SPDScalarType {
    F32 = 'f32',
    F16 = 'f16',
    I32 = 'i32',
    U32 = 'u32',
}

/**
 * Configuration for {@link WebGPUSinglePassDownsampler.preparePass}.
 */
export interface SPDPassConfig {
    /**
     * The name of the filter to use for downsampling the given texture.
     * Should be one of the filters registered with {@link WebGPUSinglePassDownsampler}.
     * Defaults to {@link SPDFilters.Average}.
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
     * Default to [texture.width - 1 - offset[0], texture.height - 1 - offset[1]].
     */
    size?: [number, number],

    /**
     * The number of mipmaps to generate.
     * Defaults to target.mipLevelCount.
     */
    numMips?: number,

    /**
     * If set to true, will try to use half-precision floats (`f16`) for this combination of texture format and filters.
     * Falls back to full precision, if half precision is requested but not supported by the device (feature 'shader-f16' not enabled).
     * Falls back to full precision, if the texture format is not a float format.
     * Defaults to false.
     */
    halfPrecision?: boolean;
}

interface GPUDownsamplingMeta {
    workgroupOffset: [number, number],
    numWorkGroups: number,
    numMips: number,
    numArrayLayers: number,
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

    /**
     * If set to true, will try to use half-precision floats (`f16`) for this combination of texture format and filters.
     * Falls back to full precision, if half precision is requested but not supported by the device (feature 'shader-f16' not enabled).
     * Falls back to full precision, if the texture format is not a float format.
     * Defaults to false.
     */
    halfPrecision?: boolean,
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

    /**
     * The maximum number of array layers will be downsampled on the {@link device} within a single pass.
     * If a texture has more, downsampling will be split up into multiple passes handling up to this limit of array layers each. 
     * Defaults to device.limits.maxTextureArrayLayers.
     */
    maxArrayLayersPerPass?: number,

    /**
     * The maximum number of mip levels that can be generated on the {@link device} within a single pass.
     * Note that generating more than 6 mip levels per pass is currently not supported on all platforms.
     * Defaults to `Math.min(device.limits.maxStorageTexturesPerShaderStage, 12)`.
     */
    maxMipsPerPass?: number,
}

function sanitizeScalarType(device: GPUDevice, format: GPUTextureFormat, halfPrecision: boolean): SPDScalarType {
    const texelType = format.toLocaleLowerCase().includes('sint') ? SPDScalarType.I32 : (format.toLocaleLowerCase().includes('uint') ? SPDScalarType.U32 : SPDScalarType.F32);
    if (halfPrecision && !device.features.has('shader-f16')) {
        console.warn(`[sanitizeScalarType]: half precision requested but the device feature 'shader-f16' is not enabled, falling back to full precision`);
    }
    if (halfPrecision && texelType !== SPDScalarType.F32) {
        console.warn(`[sanitizeScalarType]: half precision requested for non-float format (${format}, uses ${texelType}), falling back to full precision`);
    }
    return halfPrecision === true && !device.features.has('shader-f16') && texelType === SPDScalarType.F32 ? SPDScalarType.F16 : texelType;
}

class DevicePipelines {
    private device: WeakRef<GPUDevice>;
    private maxMipsPerPass: number;
    private maxArrayLayers: number;
    private internalResourcesBindGroupLayout: GPUBindGroupLayout;
    private internalResourcesBindGroupLayout12?: GPUBindGroupLayout;
    private atomicCounters: Map<number, GPUBuffer>;
    private midMipBuffers: Map<number, GPUBuffer>;
    private pipelines: Map<GPUTextureFormat, Map<SPDScalarType, Map<string, Map<number, SPDPipeline>>>>;

    constructor(device: GPUDevice, maxArrayLayers?: number, maxMipsPerPass?: number) {
        this.device = new WeakRef(device);
        this.maxMipsPerPass = Math.min(device.limits.maxStorageTexturesPerShaderStage, maxMipsPerPass ?? 12);
        this.maxArrayLayers = Math.min(device.limits.maxTextureArrayLayers, maxArrayLayers ?? device.limits.maxTextureArrayLayers);
        this.pipelines = new Map();
        this.atomicCounters = new Map();
        this.midMipBuffers = new Map();

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

        if (this.maxMipsPerPass > 6) {
            this.internalResourcesBindGroupLayout12 = device.createBindGroupLayout({
                entries: [
                    {
                        binding: 0,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: 'uniform',
                            hasDynamicOffset: false,
                            minBindingSize: 16,
                        },
                    },
                    {
                        binding: 1,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: 'storage',
                            hasDynamicOffset: false,
                            minBindingSize: 4,
                        },
                    },
                    {
                        binding: 2,
                        visibility: GPUShaderStage.COMPUTE,
                        buffer: {
                            type: 'storage',
                            hasDynamicOffset: false,
                            minBindingSize: 16 * 64 * 64,
                        },
                    },
                ],
            }); 
        }
    }

    preparePipelines(pipelineConfigs?: Array<SPDPrepareFormatDescriptor>) {
        const device = this.device.deref();
        if (device) {
            pipelineConfigs?.forEach(c => {
                const scalarType = sanitizeScalarType(device, c.format, c.halfPrecision ?? false);
                Array.from(c.filters ?? [SPD_FILTER_AVERAGE]).map(filter => {
                    for (let i = 0; i < this.maxMipsPerPass; ++i) {
                        this.getOrCreatePipeline(c.format, filter, i + 1, scalarType);
                    }
                });
            });
        }
    }

    private createPipeline(targetFormat: GPUTextureFormat, filterCode: string, numMips: number, scalarType: SPDScalarType): SPDPipeline | undefined {
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
                        sampleType: scalarType === SPDScalarType.I32 ? 'sint' : (scalarType === SPDScalarType.U32 ? 'uint' : 'unfilterable-float'),
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

        return new SPDPipeline(
            mipsBindGroupLayout,
            device.createComputePipeline({
                compute: {
                    module: device.createShaderModule({
                        code: makeShaderCode(targetFormat, filterCode, Math.min(numMips, this.maxMipsPerPass), scalarType),
                    }),
                    entryPoint: 'downsample',
                },
                layout: device.createPipelineLayout({
                    bindGroupLayouts: [
                        mipsBindGroupLayout,
                        numMips > 6 ? this.internalResourcesBindGroupLayout12! : this.internalResourcesBindGroupLayout,
                    ],
                }),
            }),
        );
    }

    private getOrCreatePipeline(targetFormat: GPUTextureFormat, filterCode: string, numMipsToCreate: number, scalarType: SPDScalarType): SPDPipeline | undefined {
        if (!this.pipelines.has(targetFormat)) {
            this.pipelines.set(targetFormat, new Map());
        }
        if (!this.pipelines.get(targetFormat)?.has(scalarType)) {
            this.pipelines.get(targetFormat)?.set(scalarType, new Map());
        }
        if (!this.pipelines.get(targetFormat)?.get(scalarType)?.has(filterCode)) {
            this.pipelines.get(targetFormat)?.get(scalarType)?.set(filterCode, new Map());
        }
        if (!this.pipelines.get(targetFormat)?.get(scalarType)?.get(filterCode)?.has(numMipsToCreate)) {
            const pipelines = this.createPipeline(targetFormat, filterCode, numMipsToCreate, scalarType);
            if (pipelines) {
                this.pipelines.get(targetFormat)?.get(scalarType)?.get(filterCode)?.set(numMipsToCreate, pipelines);
            }
        }
        return this.pipelines.get(targetFormat)?.get(scalarType)?.get(filterCode)?.get(numMipsToCreate);
    }

    private getOrCreateAtomicCountersBuffer(device: GPUDevice, numArrayLayers: number): GPUBuffer {
        if (!this.atomicCounters.has(numArrayLayers)) {
            const atomicCountersBuffer = device.createBuffer({
                size: 4 * numArrayLayers,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            device.queue.writeBuffer(atomicCountersBuffer, 0, new Uint32Array(Array(numArrayLayers).fill(0)));
            this.atomicCounters.set(numArrayLayers, atomicCountersBuffer);
        }
        return this.atomicCounters.get(numArrayLayers)!
    }

    private getOrCreateMidMipBuffer(device: GPUDevice, numArrayLayers: number): GPUBuffer {
        if (!this.midMipBuffers.has(numArrayLayers)) {
            this.midMipBuffers.set(numArrayLayers, device.createBuffer({
                size: 16 * 64 * 64 * numArrayLayers,
                usage: GPUBufferUsage.STORAGE,
            }));
        }
        return this.midMipBuffers.get(numArrayLayers)!
    }


    private createMetaBindGroup(device: GPUDevice, meta: GPUDownsamplingMeta, halfPrecision: boolean): GPUBindGroup {
        const metaBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });
        device.queue.writeBuffer(metaBuffer, 0, new Uint32Array([
            ...meta.workgroupOffset,
            meta.numWorkGroups,
            meta.numMips,
        ]));
        if (meta.numMips > 6) {
            const numArrayLayersForPrecision = halfPrecision ? Math.ceil(meta.numArrayLayers / 2) : meta.numArrayLayers;
            return device.createBindGroup({
                layout: this.internalResourcesBindGroupLayout12!,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: metaBuffer,
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.getOrCreateAtomicCountersBuffer(device, numArrayLayersForPrecision),
                        },
                    },
                    {
                        binding: 2,
                        resource: {
                            buffer: this.getOrCreateMidMipBuffer(device, numArrayLayersForPrecision),
                        },
                    },
                ]
            });            
        } else {
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
    }

    preparePass(texture: GPUTexture, target: GPUTexture, filterCode: string, offset: [number, number], size: [number, number], numMipsTotal: number, scalarType: SPDScalarType): SPDPass | undefined {
        const device = this.device.deref();
        if (!device) {
            return undefined;
        }

        const passes = [];
        for (let baseArrayLayer = 0; baseArrayLayer < target.depthOrArrayLayers; baseArrayLayer += this.maxArrayLayers) {
            const numArrayLayersThisPass = Math.min(target.depthOrArrayLayers - baseArrayLayer, this.maxArrayLayers);
            for (let baseMip = 0; baseMip < numMipsTotal - 1; baseMip += this.maxMipsPerPass) {
                const numMipsThisPass = Math.min(numMipsTotal - 1 - baseMip, this.maxMipsPerPass);

                const baseMipOffset = offset.map(o => Math.trunc(o / Math.pow(2, baseMip)));
                const baseMipSize = size.map(s => Math.max(Math.trunc(s / Math.pow(2, baseMip)), 1));
                const workgroupOffset = baseMipOffset.map(o => Math.trunc(o / 64)) as [number, number];
                const dispatchDimensions = baseMipOffset.map((o, i) => Math.trunc((o + baseMipSize[i] - 1) / 64) + 1 - workgroupOffset[i]) as [number, number];
                const numWorkGroups = dispatchDimensions.reduce((product, v) => v * product, 1);

                const metaBindGroup = this.createMetaBindGroup(
                    device,
                    {
                        workgroupOffset,
                        numWorkGroups,
                        numMips: numMipsThisPass,
                        numArrayLayers: numArrayLayersThisPass,
                    },
                    scalarType === SPDScalarType.F16,
                );

                // todo: handle missing pipeline
                const pipeline = this.getOrCreatePipeline(target.format, filterCode, numMipsThisPass, scalarType)!;

                const mipViews = Array(numMipsThisPass + 1).fill(0).map((_, i) => {
                    if (baseMip === 0 && i === 0) {
                        return texture.createView({
                            dimension: '2d-array',
                            baseMipLevel: 0,
                            mipLevelCount: 1,
                            baseArrayLayer,
                            arrayLayerCount: numArrayLayersThisPass,
                        });
                    } else {
                        const mip = baseMip + i;
                        return target.createView({
                            dimension: '2d-array',
                            baseMipLevel: texture === target ? mip : mip - 1,
                            mipLevelCount: 1,
                            baseArrayLayer,
                            arrayLayerCount: numArrayLayersThisPass,
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
                passes.push(new SPDPassInner(pipeline.pipelines, [mipsBindGroup, metaBindGroup], [...dispatchDimensions, numArrayLayersThisPass]));
            }
        }
        return new SPDPass(passes, target);
    }
}

/**
 * Returns the maximum number of mip levels for a given n-dimensional size.
 * @param size The size to compute the maximum number of mip levels for
 * @returns The maximum number of mip levels for the given size
 */
export function maxMipLevelCount(...size: number[]): number {
    return 1 + Math.trunc(Math.log2(Math.max(0, ...size)));
}

/**
 * A helper class for downsampling 2D {@link GPUTexture} (& arrays) using as few passes as possible on a {@link GPUDevice} depending on its {@link GPUSupportedLimits}.
 * Up to 12 mip levels can be generated within a single pass, if {@link GPUSupportedLimits.maxStorageTexturesPerShaderStage} supports it.
 */
export class WebGPUSinglePassDownsampler {
    private filters: Map<string, string>;
    private devicePipelines: WeakMap<GPUDevice, DevicePipelines>;

    /**
     * The set of formats supported by WebGPU SPD.
     * 
     * Note that `bgra8unorm` is only supported if the device feature `bgra8unorm-storage` is enabled.
     */
    readonly supportedFormats: Set<string> = new Set([
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
     * Sets the preferred device limits for {@link WebGPUSinglePassDownsampler} in a given record of limits.
     * Existing preferred device limits are either increased or left untouched.
     * If {@link limits} is undefined, creates a new record of preferred device limits for {@link WebGPUSinglePassDownsampler}.
     * The result can be used to set {@link GPUDeviceDescriptor.requiredLimits} when requesting a device.
     * @param limits A record of device limits set to update with the preferred limits for {@link WebGPUSinglePassDownsampler}
     * @param adapter If this is set, the preferred limits that are set by this function will be clamped to {@link GPUAdapter.limits}.
     * @returns The updated or created set of device limits with all preferred limits for {@link WebGPUSinglePassDownsampler} set
     */
    static setPreferredLimits(limits?: Record<string, number | GPUSize64>, adapter?: GPUAdapter): Record<string, number | GPUSize64> {
        if (!limits) {
            limits = {};
        }
        const maxStorageTexturesPerShaderStage = Math.min(adapter?.limits.maxStorageTexturesPerShaderStage ?? 6, 6);
        limits.maxStorageTexturesPerShaderStage = Math.max(limits.maxStorageTexturesPerShaderStage ?? maxStorageTexturesPerShaderStage, maxStorageTexturesPerShaderStage);
        return limits;
    }

    /**
     * Creates a new {@link WebGPUSinglePassDownsampler}.
     * On its own, {@link WebGPUSinglePassDownsampler} does not allocate any GPU resources.
     * Optionally, prepare GPU resources for a given {@link SPDPrepareDeviceDescriptor}.
     * @param prepareDescriptor An optional descriptor for preparing GPU resources
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
        this.getOrCreateDevicePipelines(prepareDescriptor.device, prepareDescriptor.maxArrayLayersPerPass, prepareDescriptor.maxMipsPerPass)?.preparePipelines(prepareDescriptor?.formats?.map(format => {
            return {
                ...format,
                filters: new Set(Array.from(format.filters ?? []).map(filter => this.filters.get(filter) ?? SPD_FILTER_AVERAGE)),
            };
        }));
    }

    private getOrCreateDevicePipelines(device: GPUDevice, maxArrayLayers?: number, maxMipsPerPass?: number): DevicePipelines | undefined {
        if (!this.devicePipelines.has(device)) {
            this.devicePipelines.set(device, new DevicePipelines(device, maxArrayLayers, maxMipsPerPass));
        }
        return this.devicePipelines.get(device);
    }

    /**
     * Deregisters all resources stored for a given device.
     * @param device The device resources should be deregistered for
     */
    deregisterDevice(device: GPUDevice) {
        this.devicePipelines.delete(device);
    }

    /**
     * Registers a new downsampling filter operation that can be injected into the downsampling shader for new pipelines.
     * 
     * The given WGSL code must (at least) specify a function to reduce four values into one with the following name and signature:
     * 
     *   `spd_reduce_4(v0: vec4<SPDScalar>, v1: vec4<SPDScalar>, v2: vec4<SPDScalar>, v3: vec4<SPDScalar>) -> vec4<SPDScalar>`
     * 
     * @param name The unique name of the filter operation
     * @param wgsl The WGSL code to inject into the downsampling shader as the filter operation
     */
    registerFilter(name: string, wgsl: string) {
        if (this.filters.has(name)) {
            console.warn(`[WebGPUSinglePassDownsampler::registerFilter]: overriding existing filter '${name}'. Previously generated pipelines are not affected.`);
        }
        this.filters.set(name, wgsl);
    }

    /**
     * Prepares a pass to downsample a 2d texture / 2d texture array.
     * The produced {@link SPDPass} can be used multiple times to repeatedly downsampling a texture, e.g., for downsampling the depth buffer each frame. 
     * For one-time use, {@link WebGPUSinglePassDownsampler.generateMipmaps} can be used instead.
     * 
     * By default, the texture is downsampled `texture.mipLevelCount - 1` times using an averaging filter, i.e., 4 pixel values from the parent level are averaged to produce a single pixel in the current mip level.
     * This behavior can be configured using the optional {@link config} parameter.
     * For example, instead of writing the mip levels into the input texture itself, a separate target texture can be specified using {@link SPDPassConfig.target}.
     * Other configuration options include using a different (possibly custom) filter, only downsampling a subregion of the input texture, and limiting the number of mip levels to generate, e.g., if a min-max pyramid is only needed up to a certain tile resolution.
     * If the given filter does not exist, an averaging filter will be used as a fallback.
     * The image region to downsample and the number of mip levels to generate are clamped to the input texture's size, and the output texture's `mipLevelCount`.
     * 
     * Depending on the number of mip levels to generate and the device's `maxStorageTexturesPerShaderStage` limit, the {@link SPDPass} will internally consist of multiple passes, each generating up to `min(maxStorageTexturesPerShaderStage, 12)` mip levels.
     * 
     * @param device The device the {@link SPDPass} should be prepared for
     * @param texture The texture that is to be processed by the {@link SPDPass}. Must support generating a {@link GPUTextureView} with {@link GPUTextureViewDimension:"2d-array"}. Must support {@link GPUTextureUsage.TEXTURE_BINDING}, and, if no other target is given, {@link GPUTextureUsage.STORAGE_BINDING}.
     * @param config The config for the {@link SPDPass}
     * @returns The prepared {@link SPDPass} or undefined if preparation failed or if no mipmaps would be generated.
     * @throws If the {@link GPUTextureFormat} of {@link SPDPassConfig.target} is not supported (does not support {@link GPUStorageTextureAccess:"write-only"} on the given {@link device}).
     * @throws If the size of {@link SPDPassConfig.target} is too small to store the first mip level generated for {@link texture}
     * @throws If {@link texture} or {@link SPDPassConfig.target} is not a 2d texture.
     * @see WebGPUSinglePassDownsampler.generateMipmaps
     * @see WebGPUSinglePassDownsampler.registerFilter
     * @see WebGPUSinglePassDownsampler.setPreferredLimits
     */
    preparePass(device: GPUDevice, texture: GPUTexture, config?: SPDPassConfig): SPDPass | undefined {
        const target = config?.target ?? texture;
        const filter = config?.filter ?? SPDFilters.Average;
        const offset = (config?.offset ?? [0, 0]).map((o, d) => Math.max(0, Math.min(o, (d === 0 ? texture.width : texture.height) - 1))) as [number, number];
        const size = (config?.size ?? [texture.width, texture.height]).map((s, d) => Math.max(0, Math.min(s, (d === 0 ? texture.width : texture.height) - offset[d]))) as [number, number];
        const numMips = Math.min(Math.max(config?.numMips ?? target.mipLevelCount, 0), maxMipLevelCount(...size));

        if (numMips < 2) {
            console.warn(`[WebGPUSinglePassDownsampler::prepare]: no mips to create (numMips = ${numMips})`);
            return undefined;
        }
        if (!this.supportedFormats.has(target.format)) {
            throw new Error(`[WebGPUSinglePassDownsampler::prepare]: format ${target.format} not supported. (Supported formats: ${this.supportedFormats})`);
        }
        if (target.format === 'bgra8unorm' && !device.features.has('bgra8unorm-storage')) {
            throw new Error(`[WebGPUSinglePassDownsampler::prepare]: format ${target.format} not supported without feature 'bgra8unorm-storage' enabled`);
        }
        if (target.width < Math.max(1, Math.floor(size[0] / 2)) || target.height < Math.max(1, Math.floor(size[1] / 2))) {
            throw new Error(`[WebGPUSinglePassDownsampler::prepare]: target too small (${[target.width, target.height]}) for input size ${size}`);
        }
        if (target.dimension !== '2d' || texture.dimension !== '2d') {
            throw new Error('[WebGPUSinglePassDownsampler::prepare]: texture or target is not a 2d texture');
        }
        if (!this.filters.has(filter)) {
            console.warn(`[WebGPUSinglePassDownsampler::prepare]: unknown filter ${filter}, falling back to average`);
        }
        if (filter === SPD_FILTER_MINMAX && target.format.includes('r32')) {
            console.warn(`[WebGPUSinglePassDownsampler::prepare]: filter ${filter} makes no sense for one-component target format ${target.format}`);
        }
        const filterCode = this.filters.get(filter) ?? SPD_FILTER_AVERAGE;
        const scalarType = sanitizeScalarType(device, target.format, config?.halfPrecision ?? false);

        return this.getOrCreateDevicePipelines(device)?.preparePass(texture, target, filterCode, offset, size, numMips, scalarType);
    }

    /**
     * Generates mipmaps for the given texture.
     * For textures that will be downsampled more than once, consider generating a {@link SPDPass} using {@link WebGPUSinglePassDownsampler.preparePass} and calling its {@link SPDPass.encode} method.
     * This way, allocated GPU resources for downsampling the texture can be reused.
     * @param device The device to use for downsampling the texture
     * @param texture The texture to generate mipmaps for. Must support generating a {@link GPUTextureView} with {@link GPUTextureViewDimension:"2d-array"}.
     * @param config The config for mipmap generation
     * @returns True if mipmaps were generated, false otherwise
     * @throws If {@link WebGPUSinglePassDownsampler.preparePass} threw an error.
     * @see WebGPUSinglePassDownsampler.preparePass
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

