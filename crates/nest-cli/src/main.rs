use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "nest")]
#[command(about = ".nest — Semantic Knowledge Format for Local Agents", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Inspect file metadata, manifest, and section table.
    Inspect { file: PathBuf },
    /// Validate file integrity (magic, checksums, hashes, manifest, contract).
    Validate { file: PathBuf },
    /// Search a `.nest` file with a JSON-array query vector.
    Search {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
    },
    /// Benchmark exact flat search latency.
    Benchmark {
        file: PathBuf,
        #[arg(short, long, default_value = "100")]
        queries: usize,
        #[arg(short, long, default_value = "10")]
        k: i32,
    },
    /// Show file stats.
    Stats { file: PathBuf },
    /// Resolve a `nest://content_hash/chunk_id` citation into the
    /// canonical text and original span for the chunk.
    Cite {
        file: PathBuf,
        /// `nest://<content_hash>/<chunk_id>` URI.
        citation: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Inspect { file } => cmd_inspect(file),
        Commands::Validate { file } => cmd_validate(file),
        Commands::Search { file, query, k } => cmd_search(file, query, k),
        Commands::Benchmark { file, queries, k } => cmd_benchmark(file, queries, k),
        Commands::Stats { file } => cmd_stats(file),
        Commands::Cite { file, citation } => cmd_cite(file, citation),
    }
}

fn cmd_inspect(file: PathBuf) -> Result<()> {
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
        println!(
            "  0x{:02x} {:<24} offset={} size={} checksum={}",
            entry.section_id,
            name,
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

fn cmd_validate(file: PathBuf) -> Result<()> {
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

fn cmd_search(file: PathBuf, query: String, k: i32) -> Result<()> {
    let runtime = nest_runtime::MmapNestFile::open(&file)?;
    let qvec: Vec<f32> =
        serde_json::from_str(&query).map_err(|e| anyhow::anyhow!("invalid query JSON: {}", e))?;
    let result = runtime.search(&qvec, k)?;
    println!("index_type:   {}", result.index_type);
    println!("recall:       {}", result.recall);
    println!("truncated:    {}", result.truncated);
    println!("k_requested:  {}", result.k_requested);
    println!("k_returned:   {}", result.k_returned);
    println!("query_time:   {:.3} ms", result.query_time_ms);
    println!("hits:");
    for (i, hit) in result.hits.iter().enumerate() {
        println!(
            "  [{:3}] chunk_id={} score={:.6} score_type={} source_uri={} offset={}-{} model={} index_type={} reranked={} file_hash={} content_hash={} citation_id={}",
            i + 1,
            hit.chunk_id,
            hit.score,
            hit.score_type,
            hit.source_uri,
            hit.offset_start,
            hit.offset_end,
            hit.embedding_model,
            hit.index_type,
            hit.reranked,
            hit.file_hash,
            hit.content_hash,
            hit.citation_id,
        );
    }
    Ok(())
}

fn cmd_benchmark(file: PathBuf, n_queries: usize, k: i32) -> Result<()> {
    let runtime = nest_runtime::MmapNestFile::open(&file)?;
    let dim = runtime.embedding_dim();

    let mut queries = Vec::with_capacity(n_queries);
    for _ in 0..n_queries {
        let mut q: Vec<f32> = (0..dim).map(|_| rand::random::<f32>()).collect();
        let norm = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut q {
                *x /= norm;
            }
        }
        queries.push(q);
    }

    let mut times = Vec::with_capacity(n_queries);
    for q in &queries {
        let t0 = std::time::Instant::now();
        let res = runtime.search(q, k)?;
        // Recall is contractually 1.0 for exact search; if the runtime ever
        // disagrees, surface it as an error rather than panicking the bench.
        if res.recall != 1.0 {
            return Err(anyhow::anyhow!(
                "exact search returned recall={} (expected 1.0)",
                res.recall
            ));
        }
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p = |q: f64| -> f64 {
        let idx = ((times.len() as f64 - 1.0) * q).round() as usize;
        times[idx]
    };
    let mean = times.iter().sum::<f64>() / times.len() as f64;

    println!("Benchmark: {} random queries, k={}", n_queries, k);
    println!("  mean:   {:.3} ms", mean);
    println!("  p50:    {:.3} ms", p(0.50));
    println!("  p95:    {:.3} ms", p(0.95));
    println!("  p99:    {:.3} ms", p(0.99));
    println!("  recall: 1.0 (exact)");
    Ok(())
}

fn cmd_cite(file: PathBuf, citation: String) -> Result<()> {
    let rest = citation
        .strip_prefix("nest://")
        .ok_or_else(|| anyhow::anyhow!("citation must start with 'nest://': {}", citation))?;
    let (content_hash_part, chunk_id_part) = rest
        .split_once('/')
        .ok_or_else(|| anyhow::anyhow!("citation missing chunk_id: {}", citation))?;

    let chunk_id = match chunk_id_part.split_once('/') {
        Some((first, _)) => first,
        None => chunk_id_part,
    };
    let expected_content_hash =
        format!("sha256:{}", content_hash_part.trim_start_matches("sha256:"));

    let data = std::fs::read(&file)?;
    let view = nest_format::NestView::from_bytes(&data)?;
    let actual_content_hash = view.content_hash_hex()?;
    if actual_content_hash != expected_content_hash {
        anyhow::bail!(
            "content_hash mismatch: citation says {} but file is {}",
            expected_content_hash,
            actual_content_hash
        );
    }

    let n = view.header.n_chunks as usize;
    let ids = nest_format::sections::decode_chunk_ids(
        view.get_section_data(nest_format::layout::SECTION_CHUNK_IDS)?,
        n,
    )?;
    let idx = ids
        .iter()
        .position(|id| id == chunk_id)
        .ok_or_else(|| anyhow::anyhow!("chunk_id {} not found in file", chunk_id))?;

    let texts = nest_format::sections::decode_chunks_canonical(
        view.get_section_data(nest_format::layout::SECTION_CHUNKS_CANONICAL)?,
        n,
    )?;
    let spans = nest_format::sections::decode_chunks_original_spans(
        view.get_section_data(nest_format::layout::SECTION_CHUNKS_ORIGINAL_SPANS)?,
        n,
    )?;

    let span = &spans[idx];
    println!("citation_id:  {}", citation);
    println!("file:         {}", file.display());
    println!("file_hash:    {}", view.file_hash_hex());
    println!("content_hash: {}", actual_content_hash);
    println!("chunk_id:     {}", chunk_id);
    println!("source_uri:   {}", span.source_uri);
    println!("byte_start:   {}", span.byte_start);
    println!("byte_end:     {}", span.byte_end);
    println!("text:");
    println!("{}", texts[idx]);
    Ok(())
}

fn cmd_stats(file: PathBuf) -> Result<()> {
    let data = std::fs::read(&file)?;
    let view = nest_format::NestView::from_bytes(&data)?;
    let size = std::fs::metadata(&file)?.len();
    println!("file:         {}", file.display());
    println!("size:         {} bytes ({:.2} MB)", size, size as f64 / 1e6);
    println!("chunks:       {}", view.header.n_chunks);
    println!("embeddings:   {}", view.header.n_embeddings);
    println!("dim:          {}", view.header.embedding_dim);
    println!("dtype:        {}", view.manifest.dtype);
    println!("metric:       {}", view.manifest.metric);
    println!("score_type:   {}", view.manifest.score_type);
    println!("normalize:    {}", view.manifest.normalize);
    println!("index_type:   {}", view.manifest.index_type);
    println!("model:        {}", view.manifest.embedding_model);
    println!("model_hash:   {}", view.manifest.model_hash);
    println!("chunker:      {}", view.manifest.chunker_version);
    println!("sections:     {}", view.section_table.len());
    for entry in &view.section_table {
        let name = nest_format::layout::section_name(entry.section_id).unwrap_or("unknown");
        println!(
            "  0x{:02x} {:<24} {} bytes",
            entry.section_id, name, entry.size
        );
    }
    println!("file_hash:    {}", view.file_hash_hex());
    println!("content_hash: {}", view.content_hash_hex()?);
    Ok(())
}
