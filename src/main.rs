use candle_core::{Device, DType, Result};
use candle_core::Tensor;
use candle_nn::Optimizer;
use candle_nn::{VarBuilder, Module, VarMap};
use candle_transformers::models::llama::{Llama, Config, Cache};

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;
use tokenizers::Tokenizer;
use std::time::Instant;

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
        println!("Debug: Input shape in forward: {:?}", input.shape());
        let (seq_len, batch_size) = input.shape().dims2()?;
        let mut cache = Cache::new(
            false,
            DType::F32,
            &self.config,
            input.device(),
        )?;

        let output = self.model.forward(input, seq_len as usize, &mut cache)?;
        
        // Assert output dimensions
        assert_eq!(output.shape().dims().len(), 3, "Output tensor should be 3-dimensional");
        assert_eq!(output.shape().dims()[1], seq_len, "Output sequence length should match input");
        assert_eq!(output.shape().dims()[2], VOCAB_SIZE, "Output last dimension should match vocab size");
        
        Ok(output)

    }
}

pub fn load_dataset(device: &Device) -> Result<(Tensor, Tensor)> {
    let dataset_path = Path::new("tinier_stories_dataset");
    let tokenizer = Tokenizer::from_file("tokenizer.json").expect("Failed to load tokenizer");

    let mut all_input_ids = Vec::new();
    let mut all_labels = Vec::new();

    println!("Loading dataset...");
    let start = Instant::now();

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

    let duration = start.elapsed();
    println!("Dataset loaded in {:?}", duration);
    println!("Total tokens: {}", all_input_ids.len());

    let seq_len = 128; // Choose an appropriate sequence length
    let num_batches = all_input_ids.len() / (BATCH_SIZE * seq_len);
    
    let input_ids_tensor = Tensor::from_slice(&all_input_ids[..(num_batches * BATCH_SIZE * seq_len)], (num_batches, BATCH_SIZE, seq_len), device)?;
    let labels_tensor = Tensor::from_slice(&all_labels[..(num_batches * BATCH_SIZE * seq_len)], (num_batches, BATCH_SIZE, seq_len), device)?;        

    // Assert dataset tensor dimensions
    assert_eq!(input_ids_tensor.shape().dims(), labels_tensor.shape().dims(), "Input and label tensors should have the same dimensions");
    assert_eq!(input_ids_tensor.shape().dims().len(), 3, "Dataset tensors should be 3-dimensional");

    Ok((input_ids_tensor, labels_tensor))
}

fn train(model: &mut TinyLlama, varmap: &mut VarMap, device: &Device) -> Result<()> {
    let (input_ids, labels) = load_dataset(&device)?;

    // Assert input and label dimensions
    assert_eq!(input_ids.shape().dims(), labels.shape().dims(), "Input and label tensors should have the same dimensions");


    let mut opt = candle_nn::AdamW::new_lr(
        varmap.all_vars(),
        LEARNING_RATE
    )?;

    let total_start = Instant::now();

    for epoch in 0..EPOCHS {
        println!("Starting epoch {}/{}", epoch + 1, EPOCHS);
        let epoch_start = Instant::now();
        let mut total_loss = 0f32;
        let num_batches = input_ids.dim(0)? / BATCH_SIZE;

        for batch_idx in 0..num_batches {
            let batch_start = batch_idx * BATCH_SIZE;
            let batch_end = (batch_idx + 1) * BATCH_SIZE;

            let batch_input = input_ids.narrow(0, batch_idx, 1)?.squeeze(0)?;
            let batch_labels = labels.narrow(0, batch_idx, 1)?.squeeze(0)?;

            // Assert batch input and label dimensions
            assert_eq!(batch_input.shape().dims(), batch_labels.shape().dims(), "Batch input and label tensors should have the same dimensions");
            assert_eq!(batch_input.shape().dims().len(), 2, "Batch tensors should be 2-dimensional");

            println!("Debug: batch_input shape: {:?}", batch_input.shape());
            let logits = model.forward(&batch_input)?;

            // Assert logits dimensions
            assert_eq!(logits.shape().dims().len(), 3, "Logits tensor should be 3-dimensional");
            assert_eq!(logits.shape().dims()[0], BATCH_SIZE, "Logits first dimension should match batch size");
            assert_eq!(logits.shape().dims()[2], VOCAB_SIZE, "Logits last dimension should match vocab size");

            let loss = candle_nn::loss::cross_entropy(&logits.transpose(1, 2)?, &batch_labels)?;

            // Assert loss dimension
            assert_eq!(loss.shape().dims().len(), 0, "Loss should be a scalar (0-dimensional tensor)");

            opt.backward_step(&loss)?;

            let batch_loss = loss.to_scalar::<f32>()?;
            total_loss += batch_loss;

            if (batch_idx + 1) % 10 == 0 || batch_idx == num_batches - 1 {
                println!("  Batch {}/{}: Loss = {:.4}", batch_idx + 1, num_batches, batch_loss);
            }
        }

        let epoch_duration = epoch_start.elapsed();
        println!("Epoch {}: Average loss = {:.4}, Duration: {:?}", 
                 epoch + 1, total_loss / num_batches as f32, epoch_duration);
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
