#[allow(unused_imports)]

use cxx::UniquePtr;

#[cxx::bridge]
mod ffi {
    struct GenerationStepResult {
        step: usize,
        batch_id: usize,
        token_id: usize,
        log_prob: f32,
        is_last: bool
    }

    struct GenerationOptions {
        beam_size: usize,
        patience: f32,
        length_penalty: f32,
        repetition_penalty: f32,
        no_repeat_ngram_size: usize,
        disable_unk: bool,
        suppress_sequence: Vec<String>,
        end_token: Vec<String>,
        return_end_token: bool,
        max_length: usize,
        min_length: usize,
        sampling_topk: usize,
        sampling_topp: f32,
        sampling_temperature: f32,
        num_hypotheses: usize,
        return_scores: bool,
        return_alternatives: bool,
        min_alternative_expansion_prob: f32,
        static_prompt: Vec<String>,
        cache_static_prompt: bool,
        include_prompt_in_result: bool,
    }

    unsafe extern "C++" {
        include!("ctranslate2-rs/include/ctranslate2.h");

        type GeneratorWrapper;
    }
}

pub use ffi::GeneratorWrapper as Generator;
pub use ffi::GenerationOptions;

impl Default for GenerationOptions {
    fn default() -> GenerationOptions {
        GenerationOptions {
            beam_size: 1,
            patience: 1.,
            length_penalty: 1.,
            repetition_penalty: 1.,
            no_repeat_ngram_size: 0,
            disable_unk: false,
            suppress_sequence: Vec::new(),
            end_token: Vec::new(),
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
            include_prompt_in_result: true
        }
    }
}
