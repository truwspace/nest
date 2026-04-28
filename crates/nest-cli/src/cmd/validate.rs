//! `nest validate <file>` — full integrity check.

use anyhow::Result;
use std::path::PathBuf;

pub fn run(file: PathBuf) -> Result<()> {
    let data = std::fs::read(&file)?;
    let view = nest_format::NestView::from_bytes(&data)?;
    view.validate_embeddings_values()?;
    let _contract = view.search_contract()?;
    println!("OK: {} is a valid .nest v1 file", file.display());
    println!("  Header checksum:    valid");
    println!(
        "  Section checksums:  {} sections OK",
        view.section_table.len()
    );
    println!("  Footer hash:        valid");
    println!("  Manifest:           valid (contract enforced)");
    println!("  Required sections:  all present");
    println!("  Embedding values:   no NaN/Inf");
    println!("  File hash:          {}", view.file_hash_hex());
    println!("  Content hash:       {}", view.content_hash_hex()?);
    Ok(())
}
