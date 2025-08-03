# Changelog

## [Unreleased]

## [v3.0.0] - 2025-08-03

### Added

- Add support for texture formats enabled by the device feature [texture-formats-tier1](https://www.w3.org/TR/webgpu/#texture-formats-tier1).

### Changed

- Use subgroup built-ins for downsampling by default if the device feature [subgroups](https://www.w3.org/TR/webgpu/#subgroups) is enabled.
- Move texture format `bgra8unorm` out of `WebGPUSinglePassDownsampler::supportedFormats`.
- If the texture format supports it, bind mip 6 as `'read-write'` storage texture instead of duplicating texture data in an extra buffer in case more than 6 mips are generated per pass. 

### Fixed

- Fix handling of barriers for active workgroup counter.
- Cast downsampling weight to concrete scalar type for average filter.
- Fix minor typing issues.

## [v2.0.1] - 2024-06-20

### Fixed

 - Fix handling of cases where a texture's number of array layers exceeds the maximum number of array layers per pass.

## [v2.0.0] - 2024-04-25

### Added

 - Add support for specifying the maximum number of array layers that can be downsampled per pass when configuring the device using `SPDPrepareDeviceDescriptor.maxArrayLayersPerPass`.
 - Add support for specifying the maximum number of mip levels that can be downsampled per pass when configuring the device using `SPDPrepareDeviceDescriptor.maxMipsPerPass`.
 - Add support for using `f16` instead of `f32` during downsampling.

 ### Changed

 - Depending on the limit supported by a device, up to 12 mip levels can be generated within a single pass now.
 - `WebGPUSinglePassDownsampler.setPreferredLimits` now accepts an optional `GPUAdapter` as input to clamp this limit to what the adapter allows.

 ### Fixed

 - Fix handling of integer formats (`i32` and `u32`).

