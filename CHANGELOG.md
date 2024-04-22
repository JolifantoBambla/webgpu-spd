# Changelog

## [Unreleased]

### Added

 - Added support for specifying the maximum number of array layers that can be downsampled per pass when configuring the device using `SPDPrepareDeviceDescriptor.maxArrayLayers`.

 ### Changed

 - Depending on the limit supported by a device, up to 12 mip levels can be generated within a single pass now.
 - `WebGPUSinglePassDownsampler.setPreferredLimits` now sets the `maxStorageTexturesPerShaderStage` limit to 12 instead of 6 and accepts an optional `GPUAdapter` as input to clamp this limit to what the adapter allows.
