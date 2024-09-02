use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tokenizers::Tokenizer;
use candle_core::{Result,Error};

pub fn load_chunks(file_path: &Path, chunk_size: usize) -> Result<Vec<String>> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);
    let mut chunks = Vec::new();
    let mut current_chunk = String::with_capacity(chunk_size);

    for line in reader.lines() {
        let line = line?;
        if current_chunk.len() + line.len() + 1 > chunk_size {
            chunks.push(current_chunk);
            current_chunk = String::with_capacity(chunk_size);
        }
        current_chunk.push_str(&line);
        current_chunk.push('\n');
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.clone());
    }

    Ok(chunks)
}

pub fn process_chunks(chunks: &[String], tokenizer: &Tokenizer) -> Result<Vec<i64>> {
    let mut token_ids = Vec::new();

    for chunk in chunks {
        // let encoded = tokenizer.encode(chunk.as_str(), true)?;
        let encoded = tokenizer.encode(chunk.as_str(), true).map_err(|e| Error::Msg(e.to_string()))?;
        token_ids.extend(encoded.get_ids().iter().map(|&id| id as i64));
    }

    Ok(token_ids)
}

pub fn load_and_process_file(file_path: &Path, tokenizer: &Tokenizer, chunk_size: usize) -> Result<Vec<i64>> {
    // Use shard 0 for the test split, similar to llama2.c
    // https://github.com/karpathy/llama2.c/blob/ce05cc28cf1e3560b873bb21837638a434520a67/tinystories.py#L121
    let path = std::path::PathBuf::from(pretokenized_dir).join("data00.bin");
    let bytes = std::fs::read(path)?;
    // Tokens are encoded as u16.
    let mut tokens = vec![0u16; bytes.len() / 2];
    std::io::Cursor::new(bytes).read_u16_into::<LittleEndian>(&mut tokens)?;
    tokens.into_iter().map(|u| u as u32).collect::<Vec<u32>>()

}

