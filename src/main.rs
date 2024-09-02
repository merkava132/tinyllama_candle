use candle::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Module, VarBuilder, VarMap};
use candle_transformers::models::llama::{Cache, Config, Llama};
use tokenizers::Tokenizer;

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

const HIDDEN_SIZE: usize = 64;
const INTERMEDIATE_SIZE: usize = 256;
const NUM_ATTENTION_HEADS: usize = 16;
const NUM_HIDDEN_LAYERS: usize = 8;
const VOCAB_SIZE: usize = 32000;
const NUM_KEY_VALUE_HEADS: usize = 16;
const EPOCHS: usize = 3;
const LEARNING_RATE: f64 = 1e-4;
const BATCH_SIZE: usize = 32;

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
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            use_flash_attn: false,
            bos_token_id: Some(1),
            eos_token_id: Some(2),
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

fn load_pretokenized_dataset(path: PathBuf) -> Result<Vec<u32>> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut tokens = Vec::new();
    let mut buffer = [0u8; 2];
    while reader.read_exact(&mut buffer).is_ok() {
        let token = u16::from_le_bytes(buffer);
        tokens.push(token as u32);
    }
    Ok(tokens)
}

fn valid_loss(
    model: &TinyLlama,
    dataset: &[u32],
    device: &Device,
) -> Result<f64> {
    let seq_len = model.config.seq_len;
    let mut sum_loss = 0f64;
    let mut count = 0usize;

    for chunk in dataset.chunks(seq_len + 1) {
        if chunk.len() < seq_len + 1 {
            continue;
        }
        let input = Tensor::new(&chunk[..seq_len], device)?;
        let target = Tensor::new(&chunk[1..], device)?;
        let logits = model.forward(&input.unsqueeze(0)?)?;
        let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(2)?, &target.flatten()?)?;
        sum_loss += loss.to_vec0::<f32>()? as f64;
        count += 1;
    }

    Ok(sum_loss / count as f64)
}

fn train(model: &mut TinyLlama, varmap: &mut VarMap, device: &Device) -> Result<()> {
    println!("Loading datasets...");
    let start = Instant::now();

    let train_tokens = load_pretokenized_dataset(PathBuf::from("./tinierstories_dataset/data00.bin"))?;
    let valid_tokens = load_pretokenized_dataset(PathBuf::from("./tinierstories_dataset/data01.bin"))?;

    let duration = start.elapsed();
    println!("Datasets loaded in {:?}", duration);
    println!("Total train tokens: {}", train_tokens.len());
    println!("Total valid tokens: {}", valid_tokens.len());

    let mut opt = candle_nn::AdamW::new_lr(varmap.all_vars(), LEARNING_RATE)?;

    let seq_len = model.config.seq_len;
    let total_start = Instant::now();

    for epoch in 0..EPOCHS {
        println!("Starting epoch {}/{}", epoch + 1, EPOCHS);
        let epoch_start = Instant::now();
        let mut sum_loss = 0f64;
        let mut count = 0usize;

        for (batch_index, batch) in train_tokens.chunks(BATCH_SIZE * (seq_len + 1)).enumerate() {
            let xs = Tensor::new(
                batch.chunks(seq_len + 1).map(|c| &c[..seq_len]).collect::<Vec<_>>(),
                device,
            )?;
            let ys = Tensor::new(
                batch.chunks(seq_len + 1).map(|c| &c[1..]).collect::<Vec<_>>(),
                device,
            )?;
            let logits = model.forward(&xs)?;
            let loss = candle_nn::loss::cross_entropy(&logits.flatten_to(2)?, &ys.flatten()?)?;
            opt.backward_step(&loss)?;

            sum_loss += loss.to_vec0::<f32>()? as f64;
            count += 1;

            if (batch_index + 1) % 10 == 0 {
                println!(
                    "  Batch {}: Loss = {:.4}",
                    batch_index + 1,
                    sum_loss / count as f64
                );
            }

            if (batch_index + 1) % 100 == 0 {
                let valid_loss = valid_loss(model, &valid_tokens, device)?;
                println!("  Validation Loss: {:.4}", valid_loss);
            }
        }

        let epoch_duration = epoch_start.elapsed();
        println!(
            "Epoch {}: Average loss = {:.4}, Duration: {:?}",
            epoch + 1,
            sum_loss / count as f64,
            epoch_duration
        );
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