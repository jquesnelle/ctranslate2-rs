use clap::Parser;
use colored::*;
use ctranslate2_rs::{
    ffi, BatchType, ComputeType, Device, GenerationOptions, GenerationStepResult, Generator,
};
use std::{
    io::{stdout, Write},
    path::PathBuf,
    time::Instant,
};
use tokenizers::{tokenizer::Tokenizer, FromPretrainedParameters};
use tokio::sync::mpsc;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Prompt
    prompt: String,

    /// Path to converted CTranslate2 model
    #[arg(short, long)]
    model_path: PathBuf,

    /// Name of HuggingFace tokenizer to use
    #[arg(short, long)]
    tokenizer_name: String,

    /// HuggingFace auth token
    #[arg(long)]
    hf_auth_token: Option<String>,

    #[cfg(feature = "cuda")]
    /// Device type to load model on (choices are "cuda" and "cpu")
    #[arg(short, long, default_value = "cuda")]
    device: String,

    #[cfg(not(feature = "cuda"))]
    #[arg(short, long, default_value = "cpu")]
    device: String,
    /// Indicies of devices to load on
    #[arg(long)]
    device_indicies: Option<Vec<i32>>,

    /// Compute type
    #[arg(short, long, default_value = "auto")]
    compute_type: String,

    /// Add special tokens when tokenizing
    #[arg(long, default_value_t = true)]
    add_special_tokens: bool,

    // Top-k sampling
    #[arg(long)]
    top_k: Option<usize>,

    // Top-p sampling
    #[arg(long)]
    top_p: Option<f32>,

    // Temperature sampling
    #[arg(long)]
    temperature: Option<f32>,

    // Repetition penalty
    #[arg(long)]
    repetition_penalty: Option<f32>,

    // Maximum number of new tokens
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,

    // Only output new tokens, nothing else
    #[arg(long)]
    quiet: bool,

    // Replace \n in prompt with literal newline
    #[arg(long, default_value_t = true)]
    convert_prompt_newlines: bool,

    // Ignore EOS token (keep generating)
    #[arg(long)]
    ignore_eos: bool,
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let mut options = FromPretrainedParameters::default();
    options.auth_token = args.hf_auth_token;

    if options.auth_token.is_none() {
        // check to see if the user has logged in using huggingface hub
        if let Some(mut token_path) = home::home_dir() {
            token_path.push(".cache");
            token_path.push("huggingface");
            token_path.push("token");

            options.auth_token = std::fs::read_to_string(token_path).ok();
        }
    }

    if !args.quiet {
        print!("Loading tokenizer {0}...", args.tokenizer_name);
        stdout().flush().unwrap();
    }

    let tokenizer_name = args.tokenizer_name;
    let tokenizer = tokio::task::spawn_blocking(move || {
        match Tokenizer::from_pretrained(&tokenizer_name, Some(options)) {
            Ok(tokenizer) => Some(tokenizer),
            Err(_) => {
                eprintln!("Unable to download tokenizer {0}", &tokenizer_name);
                eprintln!(
                    "To download tokenizers from HuggingFace, pass an auth token via --hf-auth-token or log in with huggingface-cli"
                );
                eprintln!("Get a token here: https://huggingface.co/settings/tokens");
                None
            }
        }
    }).await.unwrap().unwrap();
    if !args.quiet {
        println!(" done.");
    }

    let compute_type: ComputeType = args.compute_type.parse().unwrap();
    let device: Device = args.device.parse().unwrap();
    let device_indicies = match args.device_indicies {
        Some(device_indicies) => device_indicies,
        None => vec![0],
    };

    if !args.quiet {
        print!(
            "Loading {0} to {1}:{2:?} in {3}...",
            args.model_path.display(),
            device.to_string(),
            device_indicies,
            compute_type.to_string()
        );
    }
    stdout().flush().unwrap();
    let generator = Generator::new(
        args.model_path
            .clone()
            .into_os_string()
            .to_str()
            .unwrap_or_default(),
        device,
        &device_indicies,
        compute_type,
        1,
        0,
        1,
    )
    .unwrap();
    if !args.quiet {
        println!(" done.");
    }

    let mut prompt = args.prompt;
    if args.convert_prompt_newlines {
        prompt = prompt.replace("\\n", "\n");
    }

    let tokenized = tokenizer
        .encode(prompt.clone(), args.add_special_tokens)
        .unwrap()
        .get_tokens()
        .to_vec();
    let num_prompt_tokens = tokenized.len();

    let mut options = GenerationOptions::default();
    options.min_length = num_prompt_tokens;
    options.max_length = num_prompt_tokens + args.max_new_tokens;
    if let Some(top_k) = args.top_k {
        options.sampling_topk = top_k;
    }
    if let Some(top_p) = args.top_p {
        options.sampling_topp = top_p;
    }
    if let Some(temperature) = args.temperature {
        options.sampling_temperature = temperature;
    }
    if let Some(repetition_penalty) = args.repetition_penalty {
        options.repetition_penalty = repetition_penalty;
    }
    options.include_prompt_in_result = false;
    if args.ignore_eos {
        let mut config_path = args.model_path.clone();
        config_path.push("config.json");
        let config: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(config_path).unwrap()).unwrap();
        let eos_token = config.get("eos_token").unwrap().as_str().unwrap();
        options.suppress_sequences = ffi::new_vec_vec_string();
        options
            .suppress_sequences
            .as_mut()
            .unwrap()
            .push_back(vec![eos_token.to_string()]);
    }

    print!("{}", prompt.yellow());

    let (tx, mut rx) = mpsc::channel::<usize>(args.max_new_tokens);

    tokio::task::spawn(async move {
        while let Some(token) = rx.recv().await {
            if let Ok(decoded) = tokenizer.decode(vec![token as u32], true) {
                print!("{decoded}");
                stdout().flush().unwrap();
            }
        }
    });

    let start = Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        generator.generate_batch(
            vec![tokenized],
            1,
            BatchType::Examples,
            options,
            Some(move |result: GenerationStepResult| {
                let _ = tx.try_send(result.token_id);
                false
            }),
        )
    })
    .await
    .unwrap()
    .unwrap();
    let duration = start.elapsed();

    let num_generated_tokens = result[0].sequence_ids.at(0).unwrap().len() - num_prompt_tokens;
    let tokens_per_second = num_generated_tokens as f32 / duration.as_secs_f32();

    if !args.quiet {
        println!("");
        println!("Number of generated tokens: {0}", num_generated_tokens);
        println!("Tokens per second: {0}", tokens_per_second);
    }
}
