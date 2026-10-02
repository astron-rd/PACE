# BenchmarkResult

Result of a single benchmark run.

### Examples

```json
{
  "version": "0.2.0",
  "application": "dedisp",
  "implementation": "rust",
  "commit": "a529875",
  "hardware": {
    "cpu": {
      "model": "AMD EPYC 7763",
      "cores": 64
    },
    "gpu": {
      "model": "NVIDIA A100 80GB"
    },
    "memory": {
      "gb": 512,
      "mts": 3200
    }
  },
  "timings": {
    "input": 0.42,
    "plan": 0.08,
    "preprocessing": 0.15,
    "execution": 3.71
  }
}
```

### Type: `object`

| Property | Type | Required | Possible values | Description | Examples |
| -------- | ---- | -------- | --------------- | ----------- | -------- |
| version | `string` | ✅ | [`^(0\|[1-9]\d*)\.(0\|[1-9]\d*)\.(0\|[1-9]\d*)(?:-((?:0\|[1-9]\d*\|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0\|[1-9]\d*\|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$`](https://regex101.com/?regex=%5E%280%7C%5B1-9%5D%5Cd%2A%29%5C.%280%7C%5B1-9%5D%5Cd%2A%29%5C.%280%7C%5B1-9%5D%5Cd%2A%29%28%3F%3A-%28%28%3F%3A0%7C%5B1-9%5D%5Cd%2A%7C%5Cd%2A%5Ba-zA-Z-%5D%5B0-9a-zA-Z-%5D%2A%29%28%3F%3A%5C.%28%3F%3A0%7C%5B1-9%5D%5Cd%2A%7C%5Cd%2A%5Ba-zA-Z-%5D%5B0-9a-zA-Z-%5D%2A%29%29%2A%29%29%3F%28%3F%3A%5C%2B%28%5B0-9a-zA-Z-%5D%2B%28%3F%3A%5C.%5B0-9a-zA-Z-%5D%2B%29%2A%29%29%3F%24) | Specification version. |  |
| application | `string` | ✅ | `all-sky` `dedisp` `idg` | Benchmarked application. |  |
| implementation | `string` | ✅ | Length: `string >= 1` | Implementation language or framework. | ```python```, ```rust```, ```c++-openmp```, ```cuda``` |
| commit | `string` | ✅ | Length: `7 <= string <= 40` | Git commit hash. |  |
| hardware | `object` | ✅ | [Hardware](#hardware) | Hardware configuration. |  |
| timings | `object` | ✅ | object | Measurements in seconds per benchmark phase. |  |


---

# Definitions

## Cpu

Processor of the machine that ran the benchmark.

#### Type: `object`

| Property | Type | Required | Possible values | Description |
| -------- | ---- | -------- | --------------- | ----------- |
| model | `string` | ✅ | string | CPU model name. |
| cores | `integer` | ✅ | `0 < x ` | Number of physical cores. |

## Gpu

Graphics card of the machine that ran the benchmark.

#### Type: `object`

| Property | Type | Required | Possible values | Description |
| -------- | ---- | -------- | --------------- | ----------- |
| model | `string` | ✅ | string | GPU model name. |

## Hardware

Machine that ran the benchmark.

#### Type: `object`

| Property | Type | Required | Possible values | Description |
| -------- | ---- | -------- | --------------- | ----------- |
| cpu | `object` | ✅ | [Cpu](#cpu) | CPU configuration. |
| gpu | `object` or `null` | ✅ | [Gpu](#gpu) | GPU configuration. |
| memory | `object` | ✅ | [Memory](#memory) | Memory configuration. |

## Memory

Main memory of the machine that ran the benchmark.

#### Type: `object`

| Property | Type | Required | Possible values | Description |
| -------- | ---- | -------- | --------------- | ----------- |
| gb | `integer` | ✅ | `0 < x ` | Memory size in gigabytes. |
| mts | `integer` | ✅ | `0 < x ` | Memory speed in megatransfers per second. |
