// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use tokenizers::models::bpe::{BpeTrainer, BPE};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{
    AddedToken, OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, Tokenizer,
    Trainer,
};

#[test]
fn trains_a_byte_level_bpe_tokenizer() -> tokenizers::Result<()> {
    let pre_tokenizer = ByteLevel::default().add_prefix_space(false);
    let mut trainer = BpeTrainer::builder()
        .vocab_size(300)
        .min_frequency(1)
        .show_progress(false)
        .special_tokens(vec![AddedToken::from("[UNK]", true)])
        .build();
    let texts = ["red shoes", "red hat", "blue shoes"];
    trainer.feed(texts.iter(), |text| {
        let mut pretokenized = PreTokenizedString::from(text);
        pre_tokenizer.pre_tokenize(&mut pretokenized)?;
        Ok(pretokenized
            .get_splits(OffsetReferential::Original, OffsetType::Byte)
            .into_iter()
            .map(|(word, _, _)| word.to_owned())
            .collect())
    })?;
    let mut model = BPE::builder().unk_token("[UNK]".into()).build()?;
    trainer.train(&mut model)?;
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));

    let json: serde_json::Value = serde_json::from_str(&tokenizer.to_string(false)?)?;
    assert_eq!(json["model"]["type"], "BPE");
    assert_eq!(json["pre_tokenizer"]["type"], "ByteLevel");
    assert_eq!(json["pre_tokenizer"]["add_prefix_space"], false);
    assert!(json["model"]["vocab"]["Ġshoes"].is_u64());
    assert_eq!(
        tokenizer.encode("red shoes", false)?.get_tokens(),
        ["red", "Ġshoes"]
    );
    Ok(())
}
