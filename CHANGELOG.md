# Changelog

## [Unreleased]

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

