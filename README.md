## ctranslate2-rs

Rust bindings for [CTranslate2](https://github.com/OpenNMT/CTranslate2)!
This package directly builds and links the CTranslate2 library into your Rust application, producing an executable with no external dependencies.

### Acceleration

CTranslate2 supports several different acceleration methods, such as CUDA and Accelerate.
Building with a mix of these accelerators is controlled by individual features in the `ctranslate2-rs` package.
For example, to build with CUDA, add `features = ["cuda"]` to your `Cargo.toml` dependency entry for `ctranslate2-rs`.

Note: I have not tried all of the combinations, and it may be that the build breaks for some.
If so, feel free to open an issue!

### Example

The [text generation example](examples/generator) shows off CTranslate2's wide support of popular LLM model formats.
Since it's Rust, the code can be fearlessly multithreaded.
In the example, tokenization and printing is offloaded to a separate `tokio` task to gain optimal throughput while streaming new generated tokens.