//! `nest inspect <file>` — header, section table, manifest, hashes.

use anyhow::Result;
use std::path::PathBuf;

use super::util::encoding_name;

pub fn run(file: PathBuf) -> Result<()> {
    let data = std::fs::read(&file)?;
    let view = nest_format::NestView::from_bytes(&data)?;
    let magic = std::str::from_utf8(&view.header.magic).unwrap_or("???");
    println!("Magic:        {:?}", magic);
    println!(
        "Version:      {}.{}",
        view.header.version_major, view.header.version_minor
    );
    println!("File size:    {} bytes", view.header.file_size);
    println!("Sections:     {}", view.section_table.len());
    for entry in &view.section_table {
        let name = nest_format::layout::section_name(entry.section_id).unwrap_or("unknown");
        let enc = encoding_name(entry.encoding);
        println!(
            "  0x{:02x} {:<24} encoding={} offset={} size={} checksum={}",
            entry.section_id,
            name,
            enc,
            entry.offset,
            entry.size,
            hex::encode(entry.checksum),
        );
    }
    println!(
        "Manifest:\n{}",
        serde_json::to_string_pretty(&view.manifest)?
    );
    println!("File hash:    {}", view.file_hash_hex());
    println!("Content hash: {}", view.content_hash_hex()?);
    Ok(())
}
