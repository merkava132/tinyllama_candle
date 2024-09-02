mod data_loading;

use candle_core::{Device, DType, Result, Tensor};
use candle_nn::{VarBuilder, Module, VarMap, Optimizer};
use candle_transformers::models::llama::{Llama, Config, Cache};

use std::path::Path;
use tokenizers::Tokenizer;
use std::time::Instant;

use crate::data_loading::load_and_process_file;


const HIDDEN_SIZE: usize = 64;
const INTERMEDIATE_SIZE: usize = 256;
const NUM_ATTENTION_HEADS: usize = 16;
const NUM_HIDDEN_LAYERS: usize = 8;
const VOCAB_SIZE: usize = 32000;
const NUM_KEY_VALUE_HEADS: usize = 16;
const EPOCHS: usize = 3;
const LEARNING_RATE: f64 = 1e-4;
const CHUNK_SIZE: usize = 16 * 1024 * 1024;

struct TinyLlama {
    model: Llama,
    config: Config,
}

impl TinyLlama {
    fn new(vb: VarBuilder) -> Result<Self> {
        let config = Config {
            hidden_size: HIDDEN_SIZE,
            intermediate_size: INTERMEDIATE_SIZE,
            num_attention_heads: NUM_ATTENTION_HEADS,
            num_hidden_layers: NUM_HIDDEN_LAYERS,
            vocab_size: VOCAB_SIZE,
            num_key_value_heads: NUM_KEY_VALUE_HEADS,
            use_flash_attn: false,
            rope_theta: 10000.0,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            rms_norm_eps: 1e-6,
        };
        let model = Llama::load(vb, &config)?;
        Ok(Self { model, config })
    }
}

impl Module for TinyLlama {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut cache = Cache::new(false, DType::F32, &self.config, input.device())?;
        self.model.forward(input, 0, &mut cache)
    }
}


fn train(model: &mut TinyLlama, varmap: &mut VarMap, device: &Device) -> Result<()> {
    let tokenizer = Tokenizer::from_file("tokenizer.json").expect("Failed to load tokenizer");
    let train_file = Path::new("./tinierstories_dataset/TinyStoriesV2-GPT4-train-10k.txt");
    // let valid_file = Path::new("TinyStoriesV2-GPT4-valid.txt");

    println!("Loading dataset...");
    let start = Instant::now();

    let train_tokens = load_and_process_file(train_file, &tokenizer, CHUNK_SIZE)?;
    // let valid_tokens = load_and_process_file(valid_file, &tokenizer, CHUNK_SIZE)?;

    let duration = start.elapsed();
    println!("Dataset loaded in {:?}", duration);
    println!("Total train tokens: {}", train_tokens.len());
    // println!("Total valid tokens: {}", valid_tokens.len());

    let mut opt = candle_nn::AdamW::new_lr(
        varmap.all_vars(),
        LEARNING_RATE
    )?;

    let total_start = Instant::now();

    for epoch in 0..EPOCHS {
        println!("Starting epoch {}/{}", epoch + 1, EPOCHS);
        let epoch_start = Instant::now();
        let mut total_loss = 0f32;

        for i in 0..train_tokens.len() - 128 {
            let input = Tensor::from_slice(&train_tokens[i..i+128], (128,), device)?;
            let target = Tensor::from_slice(&train_tokens[i+1..i+129], (128,), device)?;

            let logits = model.forward(&input)?;
            let loss = candle_nn::loss::cross_entropy(&logits.squeeze(0)?.transpose(0, 1)?, &target)?;

            opt.backward_step(&loss)?;

            let batch_loss = loss.to_scalar::<f32>()?;
            total_loss += batch_loss;

            if (i + 1) % 1000 == 0 {
                println!("  Step {}: Loss = {:.4}", i + 1, batch_loss);
            }
        }

        let epoch_duration = epoch_start.elapsed();
        println!("Epoch {}: Average loss = {:.4}, Duration: {:?}", 
                 epoch + 1, total_loss / (train_tokens.len() - 128) as f32, epoch_duration);
    }

    let total_duration = total_start.elapsed();
    println!("Total training time: {:?}", total_duration);

    Ok(())
}


fn main() -> Result<()> {
    let device = Device::cuda_if_available(0)?;
    println!("Running on device: {:?}", device);

    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    let mut model = TinyLlama::new(vs)?;

    train(&mut model, &mut varmap, &device)?;

    Ok(())
}