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
    inv_input_size: vec2<f32>,
    work_group_offset: vec2<u32>,
    num_work_groups: u32,
    mips: u32,
    padding_0: u32,
    padding_1: u32,
}

// In the original version dst_mip_i is an image2Darray [SPD_MAX_MIP_LEVELS+1], i.e., 12+1, but WGSL doesn't support arrays of textures yet
// Also these are read_write because for mips 7-13, the workgroup reads from mip level 6 - since most formats don't support read_write access in WGSL yet, this is split in two passes and we can just use write access
@group(0) @binding(0) var src_mip_0: texture_2d_array<f32>;
@group(0) @binding(1) var dst_mip_1: texture_storage_2d_array<${format}, write>;
@group(0) @binding(2) var dst_mip_2: texture_storage_2d_array<${format}, write>;
@group(0) @binding(3) var dst_mip_3: texture_storage_2d_array<${format}, write>;
@group(0) @binding(4) var dst_mip_4: texture_storage_2d_array<${format}, write>;
@group(0) @binding(5) var dst_mip_5: texture_storage_2d_array<${format}, write>;
@group(0) @binding(6) var dst_mip_6: texture_storage_2d_array<${format}, write>;

@group(1) @binding(0) var<uniform> downsample_pass_meta : DownsamplePassMeta;

@group(2) @binding(0) var linear_clamp_sampler: sampler;

fn get_mips() -> u32 {
    return downsample_pass_meta.mips;
}

fn get_num_work_groups() -> u32 {
    return downsample_pass_meta.num_work_groups;
}

fn get_work_group_offset() -> vec2<u32> {
    return downsample_pass_meta.work_group_offset;
}

fn get_inv_input_size() -> vec2<f32> {
    return downsample_pass_meta.inv_input_size;
}

fn sample_src_image(uv: vec2<u32>, slice: u32) -> vec4<f32> {
    let tex_coord = vec2<f32>(uv) * get_inv_input_size() + get_inv_input_size();
    let result = textureSampleLevel(src_mip_0, linear_clamp_sampler, vec2<f32>(tex_coord), slice, 0);
    return vec4<f32>(srgb_to_linear(result.x), srgb_to_linear(result.y), srgb_to_linear(result.z), result.w);
}

fn load_src_image(uv: vec2<u32>, slice: u32) -> vec4<f32> {
    return textureLoad(src_mip_0, uv, slice, 0);
}

fn store_src_mip(value: vec4<f32>, uv: vec2<u32>, slice: u32, mip: u32) {
    if (mip == 1) {
        textureStore(dst_mip_1, uv, slice, value);
    } else if (mip == 2) {
        textureStore(dst_mip_2, uv, slice, value);
    } else if (mip == 3) {
        textureStore(dst_mip_3, uv, slice, value);
    } else if (mip == 4) {
        textureStore(dst_mip_4, uv, slice, value);
    } else if (mip == 5) {
        textureStore(dst_mip_5, uv, slice, value);
    } else if (mip == 6) {
        textureStore(dst_mip_6, uv, slice, value);
    }
}

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
    store_src_mip(out_value, pix, slice, mip + 1);
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

fn spd_downsample_mips_0_1_linear(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, mip: u32, slice: u32) {
    var v: array<vec4<f32>, 4>;

    var tex = workgroup_id.xy * 64 + vec2<u32>(x * 2, y * 2);
    var pix = workgroup_id.xy * 32 + vec2<u32>(x, y);
    v[0] = sample_src_image(tex, slice);
    spd_store(pix, v[0], 0, slice);

    tex = workgroup_id.xy * 64 + vec2<u32>(x * 2 + 32, y * 2);
    pix = workgroup_id.xy * 32 + vec2<u32>(x + 16, y);
    v[1] = sample_src_image(tex, slice);
    spd_store(pix, v[1], 0, slice);

    tex = workgroup_id.xy * 64 + vec2<u32>(x * 2, y * 2 + 32);
    pix = workgroup_id.xy * 32 + vec2<u32>(x, y + 16);
    v[2] = sample_src_image(tex, slice);
    spd_store(pix, v[2], 0, slice);

    tex = workgroup_id.xy * 64 + vec2<i32>(x * 2 + 32, y * 2 + 32);
    pix = workgroup_id.xy * 32 + vec2<i32>(x + 16, y + 16);
    v[3] = sample_src_image(tex, slice);
    spd_store(pix, v[3], 0, slice);

    if (mip <= 1) {
        return;
    }

    for (var i = 0u; i < 4u; i++) {
        spd_store_intermediate(x, y, v[i]);
        spd_barrier();
        if (local_invocation_index < 64) {
            v[i] = spd_reduce_intermediate(
                vec2<u32>(x * 2 + 0, y * 2 + 0),
                vec2<u32>(x * 2 + 1, y * 2 + 0),
                vec2<u32>(x * 2 + 0, y * 2 + 1),
                vec2<u32>(x * 2 + 1, y * 2 + 1)
            );
            spd_store(workgroup_id.xy * 16 + vec2<u32>(x + (i % 2) * 8, y + (i / 2) * 8), v[i], 1, slice);
        }
        spd_barrier();
    }

    if (local_invocation_index < 64) {
        spd_store_intermediate(x + 0, y + 0, v[0]);
        spd_store_intermediate(x + 8, y + 0, v[1]);
        spd_store_intermediate(x + 0, y + 8, v[2]);
        spd_store_intermediate(x + 8, y + 8, v[3]);
    }
}

fn spd_downsample_mips_0_1_load(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, mip: u32, slice: u32) {
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

    if (mip <= 1) {
        return;
    }

    for (var i = 0u; i < 4u; i++) {
        spd_store_intermediate(x, y, v[i]);
        spd_barrier();
        if (local_invocation_index < 64) {
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

    if (local_invocation_index < 64) {
        spd_store_intermediate(x + 0, y + 0, v[0]);
        spd_store_intermediate(x + 8, y + 0, v[1]);
        spd_store_intermediate(x + 0, y + 8, v[2]);
        spd_store_intermediate(x + 8, y + 8, v[3]);
    }
}

fn spd_downsample_mip_2(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, mip: u32, slice: u32) {
    if (local_invocation_index < 64u) {
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
    if (local_invocation_index < 16u) {
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
    if (local_invocation_index < 4u) {
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
    if (local_invocation_index < 1u) {
        // x x x x 0 ...
        // 0 ...
        let v = spd_reduce_intermediate(vec2<u32>(0, 0), vec2<u32>(1, 0), vec2<u32>(2, 0), vec2<u32>(3, 0));
        spd_store(workgroup_id.xy, v, mip, slice);
    }
}

fn spd_downsample_next_four(x: u32, y: u32, workgroup_id: vec2<u32>, local_invocation_index: u32, base_mip: u32, mips: u32, slice: u32) {
    if (mips <= base_mip) {
        return;
    }
    spd_barrier();
    spd_downsample_mip_2(x, y, workgroup_id, local_invocation_index, base_mip, slice);

    if (mips <= base_mip + 1) {
        return;
    }
    spd_barrier();
    spd_downsample_mip_3(x, y, workgroup_id, local_invocation_index, base_mip + 1, slice);

    if (mips <= base_mip + 2) {
        return;
    }
    spd_barrier();
    spd_downsample_mip_4(x, y, workgroup_id, local_invocation_index, base_mip + 2, slice);

    if (mips <= base_mip + 3) {
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

    if (mips <= 7) {
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
fn spd_downsample_first_6_linear(workgroup_id: vec2<u32>, local_invocation_index: u32, mips: u32, num_work_groups: u32, slice: u32) {
    let xy = map_to_xy(local_invocation_index);
    spd_downsample_mips_0_1_linear(xy.x, xy.y, workgroup_id, local_invocation_index, mips, slice);
    spd_downsample_next_four(xy.x, xy.y, workgroup_id, local_invocation_index, 2, mips, slice);
}

/// Downsamples a 64x64 tile based on the work group id.
/// If after downsampling it's the last active thread group, computes the remaining MIP levels.
///
/// @param [in] workGroupID             index of the work group / thread group
/// @param [in] localInvocationIndex    index of the thread within the thread group in 1D
/// @param [in] mips                    the number of total MIP levels to compute for the input texture
/// @param [in] numWorkGroups           the total number of dispatched work groups / thread groups for this slice
/// @param [in] slice                   the slice of the input texture
fn spd_downsample_first_6_load(workgroup_id: vec2<u32>, local_invocation_index: u32, mips: u32, num_work_groups: u32, slice: u32) {
    let xy = map_to_xy(local_invocation_index);
    spd_downsample_mips_0_1_load(xy.x, xy.y, workgroup_id, local_invocation_index, mips, slice);
    spd_downsample_next_four(xy.x, xy.y, workgroup_id, local_invocation_index, 2, mips, slice);
}

/// Downsamples a 64x64 tile based on the work group id.
/// If after downsampling it's the last active thread group, computes the remaining MIP levels.
///
/// @param [in] workGroupID             index of the work group / thread group
/// @param [in] localInvocationIndex    index of the thread within the thread group in 1D
/// @param [in] mips                    the number of total MIP levels to compute for the input texture
/// @param [in] numWorkGroups           the total number of dispatched work groups / thread groups for this slice
/// @param [in] slice                   the slice of the input texture
fn spd_downsample_second_6(workgroup_id: vec2<u32>, local_invocation_index: u32, mips: u32, num_work_groups: u32, slice: u32) {
    let xy = map_to_xy(local_invocation_index);
    spd_downsample_mips_6_7(xy.x, xy.y, mips, slice);
    spd_downsample_next_four(xy.x, xy.y, vec2<u32>(0, 0), local_invocation_index, 2, mips - 6, slice);
}

// Entry points -------------------------------------------------------------------------------------------------------

@compute
@workgroup_size(256, 1, 1)
fn downsample_first_6_linear(@builtin(local_invocation_index) local_invocation_index: u32, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    spd_downsample_first_6_linear(
        workgroup_id.xy + get_work_group_offset(),
        local_invocation_index,
        get_mips(),
        get_num_work_groups(),
        workgroup_id.z
    );
}

@compute
@workgroup_size(256, 1, 1)
fn downsample_first_6_load(@builtin(local_invocation_index) local_invocation_index: u32, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    spd_downsample_first_6_load(
        workgroup_id.xy + get_work_group_offset(),
        local_invocation_index,
        get_mips(),
        get_num_work_groups(),
        workgroup_id.z
    );
}

@compute
@workgroup_size(256, 1, 1)
fn downsample_second_6(@builtin(local_invocation_index) local_invocation_index: u32, @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    spd_downsample_second_6(
        workgroup_id.xy + get_work_group_offset(),
        local_invocation_index,
        get_mips(),
        get_num_work_groups(),
        workgroup_id.z
    );
}
