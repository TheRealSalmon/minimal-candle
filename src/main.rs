mod token_output_stream;

use std::{fs::File, io::Write};

use candle_core::{quantized::gguf_file::Content, Device, Tensor};
use candle_transformers::{
    generation::{LogitsProcessor, Sampling},
    models::quantized_llama::ModelWeights,
    utils::apply_repeat_penalty,
};
use tokenizers::Tokenizer;

use crate::token_output_stream::TokenOutputStream;

#[derive(Clone, Copy)]
pub enum WhichLLM {
    Mistral7B,
    OpenChat7B,
}

impl WhichLLM {
    pub fn model_repo_id(&self) -> String {
        match self {
            Self::Mistral7B => "TheBloke/Mistral-7B-v0.1-GGUF".to_string(),
            Self::OpenChat7B => "TheBloke/openchat_3.5-GGUF".to_string(),
        }
    }

    pub fn model_file_name(&self) -> String {
        match self {
            Self::Mistral7B => "mistral-7b-v0.1.Q4_K_S.gguf".to_string(),
            Self::OpenChat7B => "openchat_3.5.Q4_K_M.gguf".to_string(),
        }
    }

    pub fn tokenizer_repo_id(&self) -> String {
        match self {
            Self::Mistral7B => "mistralai/Mistral-7B-v0.1".to_string(),
            Self::OpenChat7B => "openchat/openchat_3.5".to_string(),
        }
    }

    pub fn process_prompt(&self, prompt: String) -> String {
        match self {
            Self::Mistral7B => format!("[INST] {prompt} [/INST]"),
            Self::OpenChat7B => {
                format!("GPT4 Correct User: {prompt}<|end_of_turn|>GPT4 Correct Assistant:")
            }
        }
    }

    pub fn eos_token(&self) -> String {
        match self {
            Self::Mistral7B => "</s>".to_string(),
            Self::OpenChat7B => "<|end_of_turn|>".to_string(),
        }
    }
}

pub fn load_model(which_llm: WhichLLM, device: &Device) -> anyhow::Result<ModelWeights> {
    let api = hf_hub::api::sync::Api::new()?;
    let model_path = api
        .repo(hf_hub::Repo::with_revision(
            which_llm.model_repo_id(),
            hf_hub::RepoType::Model,
            "main".to_string(),
        ))
        .get(&which_llm.model_file_name())?;

    let mut model_file = File::open(model_path)?;
    let model_file_content = Content::read(&mut model_file)?;
    ModelWeights::from_gguf(model_file_content, &mut model_file, device).map_err(anyhow::Error::msg)
}

pub fn load_token_output_stream(which_llm: WhichLLM) -> anyhow::Result<TokenOutputStream> {
    let api = hf_hub::api::sync::Api::new()?;
    let tokenizer_path = api
        .model(which_llm.tokenizer_repo_id())
        .get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;
    Ok(TokenOutputStream::new(tokenizer))
}

pub fn logits_processor(
    temperature: f64,
    top_k: Option<usize>,
    top_p: Option<f64>,
    seed: u64,
) -> LogitsProcessor {
    let sampling = if temperature <= 0. {
        Sampling::ArgMax
    } else {
        match (top_k, top_p) {
            (None, None) => Sampling::All { temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
        }
    };
    LogitsProcessor::from_sampling(seed, sampling)
}

fn main() -> anyhow::Result<()> {
    // TODO: load from config file
    let which_llm = WhichLLM::OpenChat7B;
    let device = Device::new_metal(0)?;
    let sample_len = 1000;
    let temperature = 0.8;
    let top_p: Option<f64> = None;
    let top_k: Option<usize> = None;
    let seed = 42;
    let repeat_penalty = 1.1;
    let repeat_last_n = 64;

    let mut model = load_model(which_llm, &device)?;
    let mut tos = load_token_output_stream(which_llm)?;

    let mut pre_prompt_tokens = vec![];
    loop {
        let mut logits_processor = logits_processor(temperature, top_k, top_p, seed);

        print!("> ");
        std::io::stdout().flush()?;
        let mut prompt = String::new();
        std::io::stdin().read_line(&mut prompt)?;
        if prompt.ends_with('\n') {
            prompt.pop();
            if prompt.ends_with('\r') {
                prompt.pop();
            }
        }
        prompt = which_llm.process_prompt(prompt);

        let tokens = tos
            .tokenizer()
            .encode(prompt, true)
            .map_err(anyhow::Error::msg)?;
        let prompt_tokens = [&pre_prompt_tokens, tokens.get_ids()].concat();

        let mut all_tokens = vec![];
        let input = Tensor::new(prompt_tokens.as_slice(), &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, 0)?.squeeze(0)?;
        let mut next_token = logits_processor.sample(&logits)?;
        if let Some(mut t) = tos.next_token(next_token)? {
            if t.starts_with(". ") {
                t = t.strip_prefix(". ").unwrap().to_string();
            }
            print!("{t}");
            std::io::stdout().flush()?;
        }
        all_tokens.push(next_token);

        let eos_token = *tos
            .tokenizer()
            .get_vocab(true)
            .get(&which_llm.eos_token())
            .unwrap();
        for index in 0..sample_len {
            let input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
            let logits = model
                .forward(&input, prompt_tokens.len() + index)?
                .squeeze(0)?;
            let start_at = all_tokens.len().saturating_sub(repeat_last_n);
            let logits = apply_repeat_penalty(&logits, repeat_penalty, &all_tokens[start_at..])?;
            next_token = logits_processor.sample(&logits)?;
            all_tokens.push(next_token);
            if let Some(t) = tos.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
            if next_token == eos_token {
                break;
            }
        }
        if let Some(rest) = tos.decode_rest()? {
            println!("{rest}");
        }
        std::io::stdout().flush()?;

        pre_prompt_tokens = [prompt_tokens.as_slice(), all_tokens.as_slice()].concat();
    }
}
