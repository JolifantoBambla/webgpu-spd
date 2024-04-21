# WebGPU SPD

A utility library for downsampling 2D GPUTextures in as few passes as possible.

## Docs

See [[here](https://jolifantobambla.github.io/webgpu-spd-js)]

## Installation

### From GitHub
```js
import { WebGPUSinglePassDownsampler } from 'https://raw.githubusercontent.com/JolifantoBambla/webgpu-spd/1.0.0/dist/index.js';
```

### NPM
```bash
npm install webgpu-spd
```

## Usage

WebGPU SPD downsamples 2d textures and 2d texture arrays using compute pipelines generating up to 6 mip levels in a single pass (all array layers are processed in the same pass). The maximum number of mip levels that can be generated within a single pass depends on `maxStorageTexturesPerShaderStage` limit supported by the device used.
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

#### Create min-max depth pyramid each frame
```js
import { WebGPUSinglePassDownsampler, SPDFilters } from 'webgpu-spd';

// during setup
const downsampler = new WebGPUSinglePassDownsampler();
const minMaxDepthPass = downsampler.preparePass(device, linearDepthTexture, { filter: SPDFilters.MinMax }); 

// in render loop
const commandEncoder = device.createCommandEncoder();

const computePassEncoder = commandEncoder.beginComputePass();
minMaxDepthPass.encode(computePassEncoder);
computePassEncoder.end();

device.queue.submit([commandEncoder.finish()]);
```

#### Downsample into target
```js
import { WebGPUSinglePassDownsampler, maxMipLevelCount } from 'webgpu-spd';

const downsampler = new WebGPUSinglePassDownsampler();

const size = [/* 2d size */];
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

// setup
downsampler.generateMipmaps(device, texture);
const downsampleDepthPass = downsampler.preparePass(device, linearDepthTexture, { filter: SPDFilters.Min });

// in render loop
downsampleDepthPass.encode(computePassEncoder);
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
downsampler.prepareDeviceResources({ device: newDevice, formats: formatConfigs});
downsampleTexturePass = downsampler.preparePass(newDevice, texture);
```

#### Custom filter
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

## Acknoledgements

This library is a WebGPU port of the FidelityFX Single Pass Downsampler (SPD) included in AMD's [FidelityFX-SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK).
