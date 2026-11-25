use std::env;
use std::fs;
use std::path::PathBuf;

fn fnv1a_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let profile = env::var("PROFILE").unwrap_or_default();
    let opt_level = env::var("OPT_LEVEL").unwrap_or_default();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let asm_name = format!("{target_arch}.s");
    let asm_path = manifest_dir.join(&asm_name);
    println!("cargo:rerun-if-changed={asm_name}");
    println!("cargo:rerun-if-changed=lib.rs");
    println!("cargo:rerun-if-changed=build.rs");

    let mut build_id: u64 = 0xcbf2_9ce4_8422_2325;
    build_id = fnv1a_update(build_id, target_arch.as_bytes());
    build_id = fnv1a_update(build_id, &[0xFF]);
    build_id = fnv1a_update(build_id, target_os.as_bytes());
    build_id = fnv1a_update(build_id, &[0xFF]);
    build_id = fnv1a_update(build_id, profile.as_bytes());
    build_id = fnv1a_update(build_id, &[0xFF]);
    build_id = fnv1a_update(build_id, opt_level.as_bytes());
    for p in [
        manifest_dir.join("build.rs"),
        manifest_dir.join("lib.rs"),
        manifest_dir.join("Cargo.toml"),
        asm_path.clone(),
    ] {
        if let Ok(bytes) = fs::read(&p) {
            build_id = fnv1a_update(build_id, &[0xFE]);
            build_id = fnv1a_update(build_id, &bytes);
        }
    }

    let supi_os: u32 = match target_os.as_str() {
        "linux" => 0,
        "windows" => 1,
        "macos" => 2,
        _ => 255,
    };

    let glue = if asm_path.is_file() {
        let escaped = asm_path.to_str().unwrap().replace('\\', "/");
        format!(
            "pub const SUPI_BUILD_ID: u64 = {build_id:#018x};\nstd::arch::global_asm!(\n    \".set SUPI_OS, {supi_os}\",\n    include_str!(\"{escaped}\"),\n    options(raw),\n);\nunsafe extern \"C\" {{\n    pub fn supi_bind(table: *mut usize, abi: u64);\n}}\n"
        )
    } else {
        format!(
            "pub const SUPI_BUILD_ID: u64 = {build_id:#018x};\npub unsafe extern \"C\" fn supi_bind(_table: *mut usize, _abi: u64) {{}}\n"
        )
    };

    fs::write(out_dir.join("glue.rs"), glue).unwrap();
}
