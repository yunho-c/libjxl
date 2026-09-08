// Copyright (c) the JPEG XL Project Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license in LICENSE.
//! One reference per process; JSONL requests {"path": "decoded.pfm"} on stdin.
use fast_ssim2::{Ssimulacra2Reference, compute_ssimulacra2_strip};
use imgref::ImgVec;
use serde_json::json;
use std::error::Error;
use std::io::{BufRead, Write};

const REVISION: &str = "c3867954c7bec8a761951df9256354b305fd0cff";
type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn header(reader: &mut impl BufRead) -> Result<String> {
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err("truncated PFM header".into());
        }
        let value = line.split('#').next().unwrap().trim();
        if !value.is_empty() {
            return Ok(value.to_string());
        }
    }
}

fn read_pfm(reader: &mut impl BufRead) -> Result<ImgVec<[f32; 3]>> {
    if header(reader)? != "PF" {
        return Err("expected RGB float PFM (PF)".into());
    }
    let dimensions = header(reader)?;
    let dims: Vec<_> = dimensions.split_whitespace().collect();
    if dims.len() != 2 {
        return Err("invalid PFM dimensions".into());
    }
    let width: usize = dims[0].parse()?;
    let height: usize = dims[1].parse()?;
    let pixels = width.checked_mul(height).ok_or("PFM dimensions overflow")?;
    if width == 0 || height == 0 || pixels > 16384 * 16384 {
        return Err("unsupported PFM dimensions".into());
    }
    let scale: f32 = header(reader)?.parse()?;
    if !scale.is_finite() || scale == 0.0 {
        return Err("invalid PFM scale".into());
    }
    let mut data = vec![[0.0; 3]; pixels];
    let mut row = vec![0u8; width * 12];
    for reverse_y in 0..height {
        reader.read_exact(&mut row)?;
        let y = height - 1 - reverse_y;
        for (x, rgb) in row.chunks_exact(12).enumerate() {
            for c in 0..3 {
                let bytes = rgb[c * 4..c * 4 + 4].try_into()?;
                let value = if scale < 0.0 {
                    f32::from_le_bytes(bytes)
                } else {
                    f32::from_be_bytes(bytes)
                } * scale.abs();
                if !value.is_finite() {
                    return Err("non-finite PFM sample".into());
                }
                data[y * width + x][c] = value;
            }
        }
    }
    if reader.read(&mut [0u8; 1])? != 0 {
        return Err("trailing PFM payload".into());
    }
    Ok(ImgVec::new(data, width, height))
}

fn load(path: &str) -> Result<ImgVec<[f32; 3]>> {
    read_pfm(&mut std::io::BufReader::new(std::fs::File::open(path)?))
}

fn emit(value: serde_json::Value) -> Result<()> {
    let mut out = std::io::stdout().lock();
    writeln!(out, "{value}")?;
    out.flush()?;
    Ok(())
}

fn run() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && args[1] == "--version" {
        emit(
            json!({"adapter": env!("CARGO_PKG_VERSION"), "fast_ssim2_revision": REVISION,
            "input": "linear-srgb-f32", "strip_height": 256}),
        )?;
        return Ok(());
    }
    if args.len() != 3 || !["auto", "full", "strip"].contains(&args[2].as_str()) {
        return Err("usage: cjxl-quality-metric REFERENCE.pfm auto|full|strip".into());
    }
    let original = load(&args[1])?;
    let strip = args[2] == "strip"
        || (args[2] == "auto" && original.width() * original.height() > 8_000_000);
    let reference = if strip {
        None
    } else {
        Some(Ssimulacra2Reference::new(original.as_ref())?)
    };
    let mut context = reference.as_ref().map(|r| r.compare_context());
    emit(
        json!({"ready": true, "width": original.width(), "height": original.height(),
        "mode": if strip {"strip-256"} else {"cached-full"},
        "fast_ssim2_revision": REVISION}),
    )?;
    for request in std::io::stdin().lock().lines() {
        let result = (|| -> Result<serde_json::Value> {
            let request: serde_json::Value = serde_json::from_str(&request?)?;
            let path = request["path"].as_str().ok_or("missing path")?;
            let actual = load(path)?;
            if (original.width(), original.height()) != (actual.width(), actual.height()) {
                return Err("mismatched image dimensions".into());
            }
            let score = if let Some(ref r) = reference {
                r.compare_with(context.as_mut().unwrap(), actual.as_ref())?
            } else {
                compute_ssimulacra2_strip(original.as_ref(), actual.as_ref(), 256)?
            };
            if !score.is_finite() {
                return Err("non-finite score".into());
            }
            Ok(json!({"score": score, "path": path}))
        })();
        match result {
            Ok(value) => emit(value)?,
            Err(error) => emit(json!({"error": error.to_string()}))?,
        }
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    fn pfm(scale: f32, values: &[f32]) -> Vec<u8> {
        let mut bytes = format!("PF\n# test\n2 2\n{scale}\n").into_bytes();
        for v in values {
            bytes.extend(if scale < 0.0 {
                v.to_le_bytes()
            } else {
                v.to_be_bytes()
            });
        }
        bytes
    }
    #[test]
    fn endian_scale_orientation() {
        for scale in [-2.0, 2.0] {
            let values: Vec<_> = (0..12).map(|x| x as f32).collect();
            let result = read_pfm(&mut Cursor::new(pfm(scale, &values))).unwrap();
            assert_eq!(result.buf()[0], [12.0, 14.0, 16.0]);
            assert_eq!(result.buf()[2], [0.0, 2.0, 4.0]);
        }
    }
    #[test]
    fn rejects_bad_payloads() {
        assert!(read_pfm(&mut Cursor::new(pfm(0.0, &[0.0; 12]))).is_err());
        assert!(read_pfm(&mut Cursor::new(pfm(-1.0, &[f32::NAN; 12]))).is_err());
        assert!(read_pfm(&mut Cursor::new(pfm(-1.0, &[0.0; 11]))).is_err());
        assert!(read_pfm(&mut Cursor::new(pfm(-1.0, &[0.0; 13]))).is_err());
    }
    #[test]
    fn identical_and_strip_agreement() {
        let source = ImgVec::new(
            (0..128 * 128)
                .map(|n| {
                    let x = (n % 128) as f32 / 128.0;
                    let y = (n / 128) as f32 / 128.0;
                    [x, y, (x + y) * 0.5]
                })
                .collect::<Vec<_>>(),
            128,
            128,
        );
        let actual = ImgVec::new(
            source
                .buf()
                .iter()
                .map(|p| [p[0] * 0.98, p[1] * 0.99, p[2] * 0.97])
                .collect::<Vec<_>>(),
            128,
            128,
        );
        let reference = Ssimulacra2Reference::new(source.as_ref()).unwrap();
        assert!((reference.compare(source.as_ref()).unwrap() - 100.0).abs() < 1e-8);
        let full = reference.compare(actual.as_ref()).unwrap();
        let strip = compute_ssimulacra2_strip(source.as_ref(), actual.as_ref(), 32).unwrap();
        assert!((full - strip).abs() < 1e-4, "{full} != {strip}");
    }

    #[test]
    fn dimensions_are_not_silently_resized() {
        let source = ImgVec::new(vec![[0.3; 3]; 16 * 16], 16, 16);
        let other = ImgVec::new(vec![[0.3; 3]; 16 * 32], 16, 32);
        let reference = Ssimulacra2Reference::new(source.as_ref()).unwrap();
        assert!(reference.compare(other.as_ref()).is_err());
    }
}
