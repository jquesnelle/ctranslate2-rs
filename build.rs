use std::env;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=include/ctranslate2.h");

    #[cfg(all(target_os = "windows", target_env = "msvc"))]
    let cpp_17_flag = "/std:c++17";
    #[cfg(not(all(target_os = "windows", target_env = "msvc")))]
    let cpp_17_flag = "-std=c++17";
    
    cxx_build::bridge("src/lib.rs")
        .include(Path::new("CTranslate2/include"))
        .flag_if_supported(cpp_17_flag)
        .compile("ctranslate2-rs");

    println!("cargo:rustc-link-search={}", env::var("OUT_DIR").unwrap());
    println!("cargo:rustc-link-lib=static=ctranslate2");
    println!("cargo:rustc-link-lib=static=cpu_features");

    env::set_current_dir("CTranslate2").expect("Unable to change directory to CTranslate2");
    _ = std::fs::remove_dir_all("build");
    _ = std::fs::create_dir("build");
    env::set_current_dir("build").expect("Unable to change directory to CTranslate2/build");

    let mut cmd = std::process::Command::new("cmake");
    cmd.arg("..")
        .arg("-DCMAKE_BUILD_TYPE=Release")
        .arg("-DBUILD_SHARED_LIBS=OFF")
        .arg("-DBUILD_CLI=OFF");

    #[cfg(feature = "mkl")]
    cmd.arg("-DWITH_MKL=ON");
    #[cfg(not(feature = "mkl"))]
    cmd.arg("-DWITH_MKL=OFF");

    #[cfg(feature = "dnnl")]
    cmd.arg("-DWITH_DNNL=ON");
    #[cfg(not(feature = "dnnl"))]
    cmd.arg("-DWITH_DNNL=OFF");

    #[cfg(feature = "accelerate")]
    {
        cmd.arg("-DWITH_ACCELERATE=ON");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    #[cfg(feature = "accelerate")]
    
    #[cfg(not(feature = "accelerate"))]
    cmd.arg("-DWITH_ACCELERATE=OFF");

    #[cfg(feature = "openblas")]
    cmd.arg("-DWITH_OPENBLAS=ON");
    #[cfg(not(feature = "openblas"))]
    cmd.arg("-DWITH_OPENBLAS=OFF");

    #[cfg(feature = "ruy")]
    cmd.arg("-DWITH_RUY=ON");
    #[cfg(not(feature = "ruy"))]
    cmd.arg("-DWITH_RUY=OFF");

    #[cfg(feature = "cuda")]
    {
        use cuda_config::*;

        cmd.arg("-DWITH_CUDA=ON");
        cmd.arg("-DCUDA_DYNAMIC_LOADING=ON");

        if cfg!(target_os = "windows") {
            println!(
                "cargo:rustc-link-search=native={}",
                find_cuda_windows().display()
            );
        } else {
            for path in find_cuda() {
                println!("cargo:rustc-link-search=native={}", path.display());
            }
        };
    
        println!("cargo:rustc-link-lib=dylib=cudart_static");
        println!("cargo:rerun-if-env-changed=CUDA_LIBRARY_PATH");
    }
    #[cfg(not(feature = "cuda"))]
    cmd.arg("-DWITH_CUDA=OFF");

    #[cfg(feature = "cudnn")]
    cmd.arg("-DWITH_CUDNN=ON");
    #[cfg(not(feature = "ruy"))]
    cmd.arg("-DWITH_CUDNN=OFF");

    if let Some((_, value)) = env::vars().find(|(key, _)| key == "CUDNN_LIBRARIES") {
        cmd.arg(format!("-DCUDNN_LIBRARIES={value}"));
    }

    if let Some((_, value)) = env::vars().find(|(key, _)| key == "CUDNN_INCLUDE_DIR") {
        cmd.arg(format!("-DCUDNN_INCLUDE_DIR={value}"));
    }

    let code = cmd.status().expect("Failed to generate build script");
    if code.code() != Some(0) {
        panic!("Failed to generate build script");
    }

    let code = std::process::Command::new("cmake")
        .arg("--build")
        .arg(".")
        .arg("--parallel")
        .arg("--config Release")
        .status()
        .expect("Failed to build lib");
    if code.code() != Some(0) {
        panic!("Failed to build lib");
    }

    #[cfg(target_os = "windows")]
    {
        std::fs::copy(
            "Release/ctranslate2.lib",
            format!("{}/ctranslate2.lib", env::var("OUT_DIR").unwrap()),
        )
        .expect("Failed to copy ctranslate2 lib");

        std::fs::copy(
            "third_party/cpu_features/Release/cpu_features.lib",
            format!("{}/cpu_features.lib", env::var("OUT_DIR").unwrap()),
        )
        .expect("Failed to copy cpu_features lib");
    }

    #[cfg(not(target_os = "windows"))]
    {
        std::fs::copy(
            "libctranslate2.a",
            format!("{}/libctranslate2.a", env::var("OUT_DIR").unwrap()),
        )
        .expect("Failed to copy ctranslate2 lib");

        std::fs::copy(
            "third_party/cpu_features/libcpu_features.a",
            format!("{}/libcpu_features.a", env::var("OUT_DIR").unwrap()),
        )
        .expect("Failed to copy cpu_features lib");
    }
}

