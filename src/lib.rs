#[allow(unused_imports)]
#[allow(dead_code)]
use cxx::UniquePtr;
use std::str::FromStr;

#[cxx::bridge]
pub mod ffi {
    extern "Rust" {
        type GenerateCallbackContext;
    }

    struct GenerationStepResult {
        step: usize,
        batch_id: usize,
        token_id: usize,
        log_prob: f32,
        log_prob_valid: bool,
        is_last: bool,
    }

    struct GenerationResult {
        sequences: UniquePtr<VecVecString>,
        sequence_ids: UniquePtr<VecVecUsize>,
        scores: Vec<f32>,
    }

    struct GenerationOptions {
        // Beam size to use for beam search (set 1 to run greedy search).
        beam_size: usize,
        // Beam search patience factor, as described in https://arxiv.org/abs/2204.05424.
        // The decoding will continue until beam_size*patience hypotheses are finished.
        patience: f32,
        // Exponential penalty applied to the length during beam search.
        // The scores are normalized with:
        //   hypothesis_score /= (hypothesis_length ** length_penalty)
        length_penalty: f32,
        // Penalty applied to the score of previously generated tokens, as described in
        // https://arxiv.org/abs/1909.05858 (set > 1 to penalize).
        repetition_penalty: f32,
        // Prevent repetitions of ngrams with this size (set 0 to disable).
        no_repeat_ngram_size: usize,
        // Disable the generation of the unknown token.
        disable_unk: bool,
        // Disable the generation of some sequences of tokens.
        suppress_sequences: UniquePtr<VecVecString>,
        // Stop the decoding on one of these tokens
        end_token: Vec<String>,
        // If end_token is empty, top on the EOS token
        empty_end_token_means_stop_on_eos_token: bool,
        // Include the end token in the result.
        return_end_token: bool,
        // Max length constraint
        max_length: usize,
        // Min length constraint
        min_length: usize,
        // Randomly sample from the top K candidates (set 0 to sample from the full output distribution).
        sampling_topk: usize,
        // Keep the most probable tokens whose cumulative probability exceeds this value.
        sampling_topp: f32,
        // High temperature increase randomness.
        sampling_temperature: f32,
        // Number of hypotheses to include in the result.
        num_hypotheses: usize,
        // Include scores in the result.
        return_scores: bool,
        // Return alternatives at the first unconstrained decoding position. This is typically
        // used with a prefix to provide alternatives at a specifc location.
        return_alternatives: bool,
        // Minimum probability to expand an alternative.
        min_alternative_expansion_prob: f32,
        // The static prompt will prefix all inputs for this model.
        static_prompt: Vec<String>,
        // Cache the model state after the static prompt and reuse it for future runs using
        // the same static prompt.
        cache_static_prompt: bool,
        // Include the input tokens in the generation result.
        include_prompt_in_result: bool,
    }

    unsafe extern "C++" {
        include!("ctranslate2-rs/include/ctranslate2.h");

        type VecVecString;
        fn at(self: &VecVecString, index: usize) -> Result<Vec<String>>;
        fn push_back(self: Pin<&mut VecVecString>, data: Vec<String>);
        fn clear(self: Pin<&mut VecVecString>);
        fn reserve(self: Pin<&mut VecVecString>, size: usize);
        fn empty(self: &VecVecString) -> bool;
        fn len(self: &VecVecString) -> usize;
        fn new_vec_vec_string() -> UniquePtr<VecVecString>;

        type VecVecUsize;
        fn at(self: &VecVecUsize, index: usize) -> Result<Vec<usize>>;
        fn push_back(self: Pin<&mut VecVecUsize>, data: Vec<usize>);
        fn clear(self: Pin<&mut VecVecUsize>);
        fn reserve(self: Pin<&mut VecVecUsize>, size: usize);
        fn empty(self: &VecVecUsize) -> bool;
        fn len(self: &VecVecUsize) -> usize;
        fn new_vec_vec_usize() -> UniquePtr<VecVecUsize>;

        type GeneratorWrapper;
        fn device(self: &GeneratorWrapper) -> String;
        fn num_replicas(self: &GeneratorWrapper) -> usize;
        fn num_queued_batches(self: &GeneratorWrapper) -> usize;
        fn num_active_batches(self: &GeneratorWrapper) -> usize;
        fn generate_batch(
            self: &GeneratorWrapper,
            tokens: UniquePtr<VecVecString>,
            max_batch_size: usize,
            batch_type_str: &str,
            options: Box<GenerationOptions>,
        ) -> Result<Vec<GenerationResult>>;
        fn generate_batch_with_callback(
            self: &GeneratorWrapper,
            tokens: UniquePtr<VecVecString>,
            max_batch_size: usize,
            batch_type_str: &str,
            options: Box<GenerationOptions>,
            callback: fn(result: GenerationStepResult, context: &GenerateCallbackContext) -> bool,
            context: Box<GenerateCallbackContext>,
        ) -> Result<Vec<GenerationResult>>;
        fn new_generator_wrapper(
            model_path: &str,
            device: &str,
            device_indicies: Vec<i32>,
            compute_type: &str,
            inter_threads: usize,
            intra_threads: usize,
            max_queued_batches: i32,
        ) -> Result<UniquePtr<GeneratorWrapper>>;
    }
}

pub use ffi::{GenerationOptions, GenerationResult, GenerationStepResult};

unsafe impl Sync for ffi::GeneratorWrapper {}
unsafe impl Send for ffi::VecVecString {}
unsafe impl Send for ffi::VecVecUsize {}
unsafe impl Send for ffi::GeneratorWrapper {}

#[derive(Debug)]
pub struct CTranslate2Error(cxx::Exception);

pub fn set_cuda_allocator_to_cub_caching() {
    std::env::set_var("CT2_CUDA_ALLOCATOR", "cub_caching");
}

impl Default for GenerationOptions {
    fn default() -> GenerationOptions {
        GenerationOptions {
            beam_size: 1,
            patience: 1.,
            length_penalty: 1.,
            repetition_penalty: 1.,
            no_repeat_ngram_size: 0,
            disable_unk: false,
            suppress_sequences: ffi::new_vec_vec_string(),
            end_token: Vec::new(),
            empty_end_token_means_stop_on_eos_token: true,
            return_end_token: false,
            max_length: 512,
            min_length: 0,
            sampling_topk: 1,
            sampling_topp: 1.,
            sampling_temperature: 1.,
            num_hypotheses: 1,
            return_scores: false,
            return_alternatives: false,
            min_alternative_expansion_prob: 0.,
            static_prompt: Vec::new(),
            cache_static_prompt: true,
            include_prompt_in_result: true,
        }
    }
}

#[derive(Debug)]
pub struct ParseError(String);

#[derive(Copy, Clone, Debug)]
pub enum Device {
    CPU,
    CUDA,
}

impl ToString for Device {
    fn to_string(&self) -> String {
        match self {
            Device::CPU => "cpu",
            Device::CUDA => "cuda",
        }
        .to_string()
    }
}

impl FromStr for Device {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "cpu" => Ok(Device::CPU),
            "cuda" => Ok(Device::CUDA),
            _ => Err(ParseError(format!("Unknown device {s}"))),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ComputeType {
    Default,
    Auto,
    Float32,
    Int8,
    Int8Float16,
    Int16,
    Float16,
}

impl ToString for ComputeType {
    fn to_string(&self) -> String {
        match self {
            ComputeType::Default => "default",
            ComputeType::Auto => "auto",
            ComputeType::Float32 => "float32",
            ComputeType::Int8 => "int8",
            ComputeType::Int8Float16 => "int8_float16",
            ComputeType::Int16 => "int16",
            ComputeType::Float16 => "float16",
        }
        .to_string()
    }
}

impl FromStr for ComputeType {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "default" => Ok(ComputeType::Default),
            "auto" => Ok(ComputeType::Auto),
            "float32" | "float" => Ok(ComputeType::Float32),
            "int8" => Ok(ComputeType::Int8),
            "int8_float16" => Ok(ComputeType::Int8Float16),
            "float16" => Ok(ComputeType::Float16),
            _ => Err(ParseError(format!("Unknown compute type {s}"))),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum BatchType {
    Examples,
    Tokens,
}

impl ToString for BatchType {
    fn to_string(&self) -> String {
        match self {
            BatchType::Examples => "examples",
            BatchType::Tokens => "tokens",
        }
        .to_string()
    }
}

impl FromStr for BatchType {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "examples" => Ok(BatchType::Examples),
            "tokens" => Ok(BatchType::Tokens),
            _ => Err(ParseError(format!("Unknown batch type {s}"))),
        }
    }
}

impl ffi::VecVecString {
    pub fn new_unique_from(value: Vec<Vec<String>>) -> UniquePtr<ffi::VecVecString> {
        let mut v = ffi::new_vec_vec_string();
        v.pin_mut().reserve(value.len());
        for item in value.into_iter() {
            v.pin_mut().push_back(item);
        }
        v
    }
}

pub struct Generator {
    generator: UniquePtr<ffi::GeneratorWrapper>,
}

pub struct GenerateCallbackContext(Box<dyn Fn(GenerationStepResult) -> bool>);

impl Generator {
    pub fn new(
        model_path: &str,
        device: Device,
        device_indicies: &[i32],
        compute_type: ComputeType,
        inter_threads: usize,
        intra_threads: usize,
        max_queued_batches: i32,
    ) -> Result<Generator, CTranslate2Error> {
        let generator = ffi::new_generator_wrapper(
            model_path,
            &device.to_string(),
            device_indicies.to_vec(),
            &compute_type.to_string(),
            inter_threads,
            intra_threads,
            max_queued_batches,
        )
        .map_err(|ex| CTranslate2Error(ex))?;
        Ok(Generator { generator })
    }

    pub fn device(&self) -> Result<Device, ParseError> {
        self.generator.device().parse()
    }

    pub fn num_replicas(&self) -> usize {
        self.generator.num_replicas()
    }

    pub fn num_queued_batches(&self) -> usize {
        self.generator.num_queued_batches()
    }

    pub fn num_active_batches(&self) -> usize {
        self.generator.num_active_batches()
    }

    pub fn generate_batch<F>(
        &self,
        tokens: Vec<Vec<String>>,
        max_batch_size: usize,
        batch_type: BatchType,
        options: GenerationOptions,
        callback: Option<F>
    ) -> Result<Vec<GenerationResult>, CTranslate2Error> where F: Fn(GenerationStepResult) -> bool + 'static {
        match callback {
            Some(callback) => self.generator
            .generate_batch_with_callback(
                ffi::VecVecString::new_unique_from(tokens),
                max_batch_size,
                &batch_type.to_string(),
                Box::new(options),
                |result: GenerationStepResult, context: &GenerateCallbackContext| context.0(result),
                Box::new(GenerateCallbackContext(Box::new(callback))),
            )
            .map_err(|ex| CTranslate2Error(ex)),
            None => self.generator
            .generate_batch(
                ffi::VecVecString::new_unique_from(tokens),
                max_batch_size,
                &batch_type.to_string(),
                Box::new(options),
            )
            .map_err(|ex| CTranslate2Error(ex))
        }
    }
}
