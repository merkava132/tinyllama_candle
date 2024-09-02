use candle_core::{Device, DType, Tensor, Result};
use candle_nn::Optimizer;
use candle_nn::{VarBuilder, Module, VarMap};
use candle_transformers::models::llama::{Llama, Config, Cache};

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use tokenizers::Tokenizer;

const HIDDEN_SIZE: usize = 64;
const INTERMEDIATE_SIZE: usize = 256;
const NUM_ATTENTION_HEADS: usize = 16;
const NUM_HIDDEN_LAYERS: usize = 8;
const VOCAB_SIZE: usize = 32000; // Adjust based on your tokenizer
const NUM_KEY_VALUE_HEADS: usize = 16; // Adjust as needed
const EPOCHS: usize = 3;
const BATCH_SIZE: usize = 16;
const LEARNING_RATE: f64 = 1e-4; // Adjust as needed

#[derive(Debug)]
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
        let (seq_len, batch_size) = input.shape().dims2()?;
        let mut cache = Cache::new(
            false,
            DType::F32,
            &self.config,
            input.device(),
        )?;
        self.model.forward(input, seq_len as usize, &mut cache)
    }
}


pub fn load_dataset(device: &Device) -> Result<(Tensor, Tensor)> {
    let dataset_path = Path::new("tinystories_dataset");
    let tokenizer = Tokenizer::from_file("tokenizer.json").expect("Failed to load tokenizer");

    let mut all_input_ids = Vec::new();
    let mut all_labels = Vec::new();

    for entry in fs::read_dir(dataset_path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            let file = File::open(path)?;
            let reader = BufReader::new(file);

            for line in reader.lines() {
                let line = line?;
                let encoded = tokenizer.encode(line, true).expect("Failed to encode text");
                let input_ids: Vec<i64> = encoded.get_ids().iter().map(|&id| id as i64).collect();
                
                if !input_ids.is_empty() {
                    all_input_ids.extend_from_slice(&input_ids[..input_ids.len() - 1]);
                    all_labels.extend_from_slice(&input_ids[1..]);
                }
            }
        }
    }

    let input_ids_tensor = Tensor::from_slice(&all_input_ids, (all_input_ids.len(),), device)?;
    let labels_tensor = Tensor::from_slice(&all_labels, (all_labels.len(),), device)?;

    Ok((input_ids_tensor, labels_tensor))
}


fn train(model: &mut TinyLlama, varmap: &mut VarMap, device: &Device) -> Result<()> {
    let (input_ids, labels) = load_dataset(&device)?;
    let mut opt = candle_nn::AdamW::new_lr(
        varmap.all_vars(),
        LEARNING_RATE
    )?;

    for epoch in 0..EPOCHS {
        let mut total_loss = 0f32;
        let num_batches = input_ids.dim(0)? / BATCH_SIZE;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * BATCH_SIZE;
            let batch_end = (batch_idx + 1) * BATCH_SIZE;

            let batch_input = input_ids.narrow(0, batch_start, BATCH_SIZE)?;
            let batch_labels = labels.narrow(0, batch_start, BATCH_SIZE)?;

            let logits = model.forward(&batch_input)?;
            let loss = candle_nn::loss::cross_entropy(&logits.transpose(1, 2)?, &batch_labels)?;

            opt.backward_step(&loss)?;

            total_loss += loss.to_scalar::<f32>()?;
        }

        println!("Epoch {}: Average loss = {}", epoch + 1, total_loss / num_batches as f32);
    }

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
