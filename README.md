# WebGPU SPD

A utility library for downsampling 2D GPUTextures in as few passes as possible.

## Docs

See [here](https://jolifantobambla.github.io/webgpu-spd).

## Installation

### From GitHub
```js
import { WebGPUSinglePassDownsampler } from 'https://jolifantobambla.github.io/webgpu-spd/1.0.0/dist/index.js';
```

### NPM
```bash
npm install webgpu-spd
```

## Usage

WebGPU SPD downsamples 2d textures and 2d texture arrays using compute pipelines generating up to 6 mip levels in a single pass (all array layers are processed in the same pass). The maximum number of mip levels that can be generated within a single pass depends on the `maxStorageTexturesPerShaderStage` limit supported by the device used.
Should the number of mip levels requested for a texture exceed this limit, multiple passes, generating up to `min(maxStorageTexturesPerShaderStage, 6)` mip levels each, will be used.
The mip levels generated for a given input texture are either stored in the input texture itself or in a given target texture.
This output texture must support `GPUTextureUsage.STORAGE_BINDING` with access mode `"write-only"`.

#### Generate mipmaps
```js
import { WebGPUSinglePassDownsampler, maxMipLevelCount } from 'webgpu-spd';

const downsampler = new WebGPUSinglePassDownsampler();

const size = [/* size + array layers */];
const texture = device.createTexture({
    size,
    mipLevelCount: maxMipLevelCount(size[0], size[1]),
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
});

// write mip level 0

downsampler.generateMipmaps(device, texture);
```

#### Downsample a texture each frame
```js
import { WebGPUSinglePassDownsampler, SPDFilters } from 'webgpu-spd';

// during setup
const downsampler = new WebGPUSinglePassDownsampler();
const downsampleDepthPass = downsampler.preparePass(device, linearDepthTexture, { filter: SPDFilters.Min }); 

// in render loop
const commandEncoder = device.createCommandEncoder();

const computePassEncoder = commandEncoder.beginComputePass();
downsampleDepthPass.encode(computePassEncoder);
computePassEncoder.end();

device.queue.submit([commandEncoder.finish()]);
```

#### Downsample into target
```js
import { WebGPUSinglePassDownsampler, maxMipLevelCount } from 'webgpu-spd';

const downsampler = new WebGPUSinglePassDownsampler();

const size = [/* width, height, array layers */];
const texture = device.createTexture({
    size,
    mipLevelCount: 1,
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING,
});
const target = device.createTexture({
    size: [size[0] / 2, size[1] / 2, size[2]],
    mipLevelCount: maxMipLevelCount(size[0], size[1]) - 1,
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
});

// write mip level 0

downsampler.generateMipmaps(device, texture, { target });
```

#### Use min-max filter to generate a min-max pyramid for single-channel textures

The `SPDFilters.MinMax` filter provided by WebGPU SPD is a special filter that is meant to be used with input textures using single-channel formats like `"r32float"`, and a target texture using a two-channel format like `"rg32float"`.
After the downsampling pass, the target texture will contain the minimum values in the red channel and the maximum values in the green channel.

```js
import { WebGPUSinglePassDownsampler, SPDFilters, maxMipLevelCount } from 'webgpu-spd';

// during setup
const downsampler = new WebGPUSinglePassDownsampler();
const linearDepth = device.createTexture({
    size: [/* gBuffer size */],
    mipLevelCount: 1,
    format: 'r32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
});
const minMaxDepthPyramid = device.createTexture({
    size: [linearDepth.width / 2, linearDepth.height / 2],
    mipLevelCount: maxMipLevelCount(linearDepth.width, linearDepth.height) - 1
    format: 'rg32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
});
const minMaxDepthPass = downsampler.preparePass(device, linearDepth, {
    target: minMaxDepthPyramid,
    filter: SPDFilters.MinMax,
}); 

// in render loop

// ... write mip level 0 of linearDepth

minMaxDepthPass.encode(computePassEncoder);

```

#### Prepare pipelines for expected formats

In the above examples, GPU resources, like compute pipelines and bindgroup layouts etc., are created on the fly the first time a new configuration of `GPUDevice`, `GPUTextureFormat`, and filter is needed.

WebGPU SPD also supports allocating resources during setup, like this:

```js
import { WebGPUSinglePassDownsampler, SPDFilters } from 'webgpu-spd';

const downsampler = new WebGPUSinglePassDownsampler({ device, formats: [
    { format: 'rgba8unorm' },
    { format: 'r32float', filters: [ SPDFilters.Min ] },
]});

// alternatively call
downsampler.prepareDeviceResources({ device, formats: [
    { format: 'rgba8unorm' },
    { format: 'r32float', filters: [ SPDFilters.Min ] },
]});
```

#### Handling device loss
```js
import { WebGPUSinglePassDownsampler, SPDFilters } from 'webgpu-spd';

const formatConfigs = [
    { format: 'rgba8unorm' },
    { format: 'r32float', filters: [ SPDFilters.Min ] },
];

// on new device
downsampler.deregisterDevice(oldDevice);
downsampler.prepareDeviceResources({ device: newDevice, formats: formatConfig s});
downsampleTexturePass = downsampler.preparePass(newDevice, texture);
```

#### Use custom filters

Custom filters of radius 1 can be registered with WebGPU SPD using `registerFilter`.
The given WGSL code must at least define a reduction function with the following name and signature:

```wgsl
fn spd_reduce_4(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32>
```

For example, a custom filter that only takes a single pixel value out of the four given ones could be implemented and used like this:

```js
import { WebGPUSinglePassDownsampler } from 'webgpu-spd';

const downsampler = new WebGPUSinglePassDownsampler();
downsampler.registerFilter('upperLeft', `
    fn spd_reduce_4(v0: vec4<f32>, v1: vec4<f32>, v2: vec4<f32>, v3: vec4<f32>) -> vec4<f32> {
        return v0;
    }
`);

// ...

downsampler.generateMipmaps(device, texture, { filter: 'upperLeft' });
```

#### Downsample image region

```js
import { WebGPUSinglePassDownsampler } from 'webgpu-spd';

const downsampler = new WebGPUSinglePassDownsampler();

const sizeHalf = [texture.width / 2, texture.height / 2];
downsampler.generateMipmaps(device, texture, { offset: sizeHalf, size: sizeHalf});
```

## Contributions

Contributions are very welcome. If you find a bug, please file an issue [here](https://github.com/JolifantoBambla/webgpu-spd/issues). If you feel some important functionality is missing, feel free to submit a pull request [here](https://github.com/JolifantoBambla/webgpu-spd/pulls).

## Acknowledgements

This library is a WebGPU port of the FidelityFX Single Pass Downsampler (SPD) included in AMD's [FidelityFX-SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK).
