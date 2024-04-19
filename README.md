# WebGPU-SPD

A utility library for downsampling 2D GPUTextures in as few passes as possible.

## Docs

See [[here](https://jolifantobambla.github.io/webgpu-spd-js)]

## Installation

### From GitHub
```js
npm install webgpu-spd
```

### NPM
```bash
npm install webgpu-spd
```

## Usage

#### Generate mipmaps

```js
import { WebGPUSinglePassDownsampler, maxMipLevelCount } from 'webgpu-spd';

const downsampler = new WebGPUSinglePassDownsampler();

const size = [/* 2d size */];
const texture = device.createTexture({
    size,
    mipLevelCount: maxMipLevelCount(...size),
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
});

// write mip level 0

downsampler.generateMipmaps(device, texture);
```

#### Downsample texture each frame

```js
import { WebGPUSinglePassDownsampler, maxMipLevelCount } from 'webgpu-spd';

const downsampler = new WebGPUSinglePassDownsampler();

const size = [/* 2d size */];
const texture = device.createTexture({
    size,
    mipLevelCount: maxMipLevelCount(...size),
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
});

// write mip level 0

downsampler.generateMipmaps(device, texture);
```

#### Use filters

```js
import { WebGPUSinglePassDownsampler, maxMipLevelCount } from 'webgpu-spd';

const downsampler = new WebGPUSinglePassDownsampler();

const size = [/* 2d size */];
const texture = device.createTexture({
    size,
    mipLevelCount: maxMipLevelCount(...size),
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
});

// write mip level 0

downsampler.generateMipmaps(device, texture);
```

#### Downsample image region

```js
import { WebGPUSinglePassDownsampler, maxMipLevelCount } from 'webgpu-spd';

const downsampler = new WebGPUSinglePassDownsampler();

const size = [/* 2d size */];
const texture = device.createTexture({
    size,
    mipLevelCount: maxMipLevelCount(...size),
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
});

// write mip level 0

downsampler.generateMipmaps(device, texture);
```

## Acknoledgements

This library is a WebGPU port of the FidelityFX Single Pass Downsampler (SPD) included in AMD's [FidelityFX-SDK](https://github.com/GPUOpen-LibrariesAndSDKs/FidelityFX-SDK).
