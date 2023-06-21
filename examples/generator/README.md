Example of using the `Generator` class to generate text.
Models needs to first be converted using CTranslate2's `ct2-transformers-converter` tool.

For example, to convert and use [StarCoder](https://huggingface.co/bigcode/starcoder):

```sh
pip install ctranslate2
ct2-transformers-converter --model bigcode/starcoder --quantization int8_float16 --output_dir starcoder
cargo run --release -- --model-path starcoder --tokenizer-name bigcode/starcoder "def create_hyperintelligent_ai():"
```

This example has CUDA acceleration turned on by default.