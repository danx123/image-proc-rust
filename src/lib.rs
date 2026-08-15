use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray3};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// Validasi bentuk array: harus (H, W, C) dengan C = 3 (BGR) atau 4 (BGRA).
/// Mengembalikan jumlah channel.
#[inline]
fn validate_image<'py>(arr: &numpy::ndarray::ArrayView3<'py, u8>) -> PyResult<usize> {
    let shape = arr.shape();
    let channels = shape[2];
    if channels != 3 && channels != 4 {
        return Err(PyValueError::new_err(
            "Format gambar harus BGR (3 channel) atau BGRA (4 channel)",
        ));
    }
    Ok(channels)
}

/// Ambil slice &[u8] contiguous dari array, atau error yang jelas kalau tidak contiguous
/// (mis. hasil slicing/crop numpy yang belum di-copy).
#[inline]
fn contiguous_slice<'a>(arr: &'a numpy::ndarray::ArrayView3<'a, u8>) -> PyResult<&'a [u8]> {
    arr.as_slice().ok_or_else(|| {
        PyValueError::new_err(
            "Array tidak contiguous — gunakan np.ascontiguousarray(img) sebelum memanggil fungsi ini",
        )
    })
}

/// Grayscale manual sepenuhnya di Rust — 5–10x lebih cepat dari loop Python.
/// Terima numpy array BGR/BGRA (H, W, 3|4) uint8, kembalikan array (H, W) uint8.
#[pyfunction]
fn manual_grayscale<'py>(
    py: Python<'py>,
    img_bgr: PyReadonlyArray3<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let arr = img_bgr.as_array();
    let channels = validate_image(&arr)?;
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    let data = contiguous_slice(&arr)?;

    let mut out = vec![0u8; h * w];
    out.par_iter_mut().enumerate().for_each(|(i, px)| {
        let base = i * channels;
        let b = data[base] as u32;
        let g = data[base + 1] as u32;
        let r = data[base + 2] as u32;
        // Rumus Luminance: 0.299 R + 0.587 G + 0.114 B
        *px = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
    });

    let out = Array2::from_shape_vec((h, w), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py))
}

/// Terapkan sepia tone — setara dengan versi Python (kernel matrix) tapi jauh lebih cepat.
/// Mendukung BGR maupun BGRA (alpha channel dipertahankan apa adanya).
#[pyfunction]
fn apply_sepia<'py>(
    py: Python<'py>,
    img_bgr: PyReadonlyArray3<'py, u8>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let arr = img_bgr.as_array();
    let channels = validate_image(&arr)?;
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    let data = contiguous_slice(&arr)?;

    let mut out = vec![0u8; h * w * channels];
    out.par_chunks_mut(channels)
        .enumerate()
        .for_each(|(i, px)| {
            let base = i * channels;
            let b = data[base] as f32;
            let g = data[base + 1] as f32;
            let r = data[base + 2] as f32;

            px[2] = (0.393 * r + 0.769 * g + 0.189 * b).clamp(0.0, 255.0) as u8; // R
            px[1] = (0.349 * r + 0.686 * g + 0.168 * b).clamp(0.0, 255.0) as u8; // G
            px[0] = (0.272 * r + 0.534 * g + 0.131 * b).clamp(0.0, 255.0) as u8; // B
            if channels == 4 {
                px[3] = data[base + 3]; // pertahankan alpha
            }
        });

    let out = Array3::from_shape_vec((h, w, channels), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py))
}

/// Invert warna gambar (setara cv2.bitwise_not). Mendukung BGR maupun BGRA.
#[pyfunction]
fn invert_colors<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, u8>,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let arr = img.as_array();
    let channels = validate_image(&arr)?;
    let (h, w) = (arr.shape()[0], arr.shape()[1]);
    let data = contiguous_slice(&arr)?;

    let mut out = vec![0u8; data.len()];
    out.par_iter_mut().zip(data.par_iter()).for_each(|(o, &v)| {
        *o = 255 - v;
    });

    let out = Array3::from_shape_vec((h, w, channels), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py))
}

/// Daftarkan semua fungsi ke Python
#[pymodule]
fn image_proc_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(manual_grayscale, m)?)?;
    m.add_function(wrap_pyfunction!(apply_sepia, m)?)?;
    m.add_function(wrap_pyfunction!(invert_colors, m)?)?;
    Ok(())
}
