# image_proc_rust

Rust-powered image processing backend for the **Macan Angkasa Suite** (currently used by **Macan Efek** / Macan Image Viewer).

It replaces performance-critical, pixel-by-pixel image effects — grayscale, sepia, and color invert — with a native Rust extension that operates directly on the same `numpy.ndarray` objects OpenCV already produces in Python, parallelized across all CPU cores with [`rayon`](https://crates.io/crates/rayon).

No OpenCV C++ toolchain is required to build or use this module.

---

## Why

Pure-Python/OpenCV pixel loops (or per-pixel NumPy broadcasting) for effects like sepia and manual grayscale conversion are single-threaded and allocate intermediate arrays at every step. `image_proc_rust` does the same math in native, parallelized Rust and hands the result back as a plain NumPy array — no format conversion needed on the Python side.

## Requirements

- Python 3.9 – 3.13
- `numpy` installed in the target Python environment
- Rust toolchain (stable) + [`maturin`](https://www.maturin.rs/) — only needed to **build** the wheel, not to use it

## Installation

### From a prebuilt wheel
```bash
pip install image_proc_rust-<version>-<platform>.whl
```

### Building from source
```bash
pip install maturin numpy
maturin build --release --out dist
pip install dist/*.whl
```

Or for local development (installs directly into the active virtualenv):
```bash
maturin develop --release
```

## API

All functions accept a `numpy.ndarray` of dtype `uint8` and shape `(H, W, 3)` (BGR) or `(H, W, 4)` (BGRA) — the same layout `cv2.imread` / `cv2.cvtColor` already produce. Arrays must be **C-contiguous**; call `np.ascontiguousarray(img)` first if the array came from a slice/crop.

| Function | Signature | Description |
|---|---|---|
| `manual_grayscale` | `(img: ndarray[H,W,3\|4]) -> ndarray[H,W]` | Converts BGR/BGRA to grayscale using the standard luminance formula (`0.299 R + 0.587 G + 0.114 B`). Alpha channel is ignored. |
| `apply_sepia` | `(img: ndarray[H,W,3\|4]) -> ndarray[H,W,3\|4]` | Applies a sepia tone. Output has the same number of channels as the input — alpha, if present, is preserved unchanged. |
| `invert_colors` | `(img: ndarray[H,W,3\|4]) -> ndarray[H,W,3\|4]` | Inverts every channel (equivalent to `cv2.bitwise_not`), including alpha if present. |

### Example

```python
import cv2
import numpy as np
import image_proc_rust

img = cv2.imread("photo.jpg")  # BGR, uint8
img = np.ascontiguousarray(img)

gray = image_proc_rust.manual_grayscale(img)          # (H, W) uint8
sepia = image_proc_rust.apply_sepia(img)               # (H, W, 3) uint8
inverted = image_proc_rust.invert_colors(img)          # (H, W, 3) uint8

cv2.imwrite("gray.jpg", gray)
cv2.imwrite("sepia.jpg", sepia)
cv2.imwrite("inverted.jpg", inverted)
```

### Error handling

Every function raises a Python `ValueError` (not a panic/crash) when:
- the array isn't 3-dimensional or the channel count isn't 3 or 4, or
- the array isn't C-contiguous.

Callers in the Macan Suite (e.g. `macan_efek.py`) check for `image_proc_rust`'s availability at import time and catch any exception from these calls, falling back to the equivalent OpenCV/NumPy implementation automatically — so the app keeps working even if this module isn't installed or built for the current platform.

## Project layout

```
image_proc_rust/
├── Cargo.toml     # pyo3 + numpy + rayon, no OpenCV dependency
├── src/
│   └── lib.rs     # manual_grayscale, apply_sepia, invert_colors
└── .github/workflows/build.yml   # CI: builds wheels for Windows × Python 3.9–3.13
```

## Building wheels in CI

`build.yml` builds release wheels for `windows-latest` across Python 3.9–3.13 using `maturin build --release`
## Versioning notes

- `numpy` crate dependency is pinned to `>=0.21, <0.23` to match the `into_pyarray_bound` API surface this crate currently targets. If you bump the `numpy` crate past `0.23`, check whether `into_pyarray_bound` was renamed/removed upstream and update `lib.rs` accordingly.
- `pyo3` is pinned to `0.22` with the `extension-module` feature enabled.

## License

MIT/Apache-2.0 (dual-licensed), matching `Cargo.toml`.
