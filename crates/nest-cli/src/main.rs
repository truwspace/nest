use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::process::Command as ProcCommand;

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
    /// Search a `.nest` file with a JSON-array query vector (exact path).
    Search {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
    },
    /// Search by raw text — embeds the query with the model declared in
    /// the manifest, then runs the appropriate vector path. Honors the
    /// declared `index_type` (exact / hnsw / hybrid). Validates the
    /// embedder's model_hash against the manifest before running search;
    /// a mismatch fails with a typed error rather than returning
    /// silently-bad results.
    SearchText {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
        /// Override the embedder script. Default: `python/embed_query.py`.
        #[arg(long)]
        embedder: Option<PathBuf>,
        /// `ef` (HNSW) / candidates-per-path (hybrid). Default: 4*k or 64.
        #[arg(long)]
        candidates: Option<usize>,
        /// Local path to the model snapshot dir. Use this for fully
        /// offline operation: copy the model dir alongside the .nest,
        /// pass --model-path at every search. Without this, the
        /// embedder resolves the model from the sentence-transformers
        /// cache (requires network on first use).
        #[arg(long)]
        model_path: Option<PathBuf>,
        /// Skip model_hash validation. ONLY use when intentionally
        /// running search-text against a corpus whose `model_hash`
        /// is the legacy zero-placeholder (pre-Phase-3 builds). In
        /// that case the search is still cosine-valid IF the user
        /// genuinely uses the same embedding model — but there is
        /// no guarantee. Prefer rebuilding the corpus.
        #[arg(long)]
        skip_model_hash_check: bool,
    },
    /// Force the ANN (HNSW) path. Falls back to exact if the file has
    /// no HNSW section.
    SearchAnn {
        file: PathBuf,
        query: String,
        #[arg(short, long, default_value = "10")]
        k: i32,
        #[arg(long, default_value = "100")]
        ef: usize,
    },
    /// Benchmark exact flat search latency.
    Benchmark {
        file: PathBuf,
        #[arg(short, long, default_value = "100")]
        queries: usize,
        #[arg(short, long, default_value = "10")]
        k: i32,
        /// If set, also benchmark `search_ann` with the given ef.
        #[arg(long)]
        ann: Option<usize>,
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
        Commands::SearchText {
            file,
            query,
            k,
            embedder,
            candidates,
            model_path,
            skip_model_hash_check,
        } => cmd_search_text(
            file,
            query,
            k,
            embedder,
            candidates,
            model_path,
            skip_model_hash_check,
        ),
        Commands::SearchAnn { file, query, k, ef } => cmd_search_ann(file, query, k, ef),
        Commands::Benchmark {
            file,
            queries,
            k,
            ann,
        } => cmd_benchmark(file, queries, k, ann),
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
    print_result(&result);
    Ok(())
}

/// Output schema produced by `python/embed_query.py`.
#[derive(serde::Deserialize)]
struct EmbedderOutput {
    model_hash: String,
    embedding_model: String,
    embedding_dim: usize,
    vector: Vec<f32>,
    /// Full structured fingerprint — included here for diagnostics on
    /// mismatch but not validated field-by-field (the compact
    /// `model_hash` is the source of truth).
    #[serde(default)]
    fingerprint: serde_json::Value,
}

/// Legacy zero placeholder; pre-Phase-3 corpora may have this. Caller
/// can opt out of strict validation via `--skip-model-hash-check` to
/// search them.
const PLACEHOLDER_MODEL_HASH: &str =
    "sha256:0000000000000000000000000000000000000000000000000000000000000000";

#[allow(clippy::too_many_arguments)]
fn cmd_search_text(
    file: PathBuf,
    query: String,
    k: i32,
    embedder: Option<PathBuf>,
    candidates: Option<usize>,
    model_path: Option<PathBuf>,
    skip_model_hash_check: bool,
) -> Result<()> {
    let runtime = nest_runtime::MmapNestFile::open(&file)?;
    let info: serde_json::Value = serde_json::from_str(&runtime.inspect_json()?)?;
    let model = info["manifest"]["embedding_model"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("manifest.embedding_model missing"))?
        .to_string();
    let declared_dim = info["manifest"]["embedding_dim"]
        .as_u64()
        .ok_or_else(|| anyhow::anyhow!("manifest.embedding_dim missing"))?
        as usize;
    let declared_model_hash = info["manifest"]["model_hash"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("manifest.model_hash missing"))?
        .to_string();

    let embedder = embedder.unwrap_or_else(default_embedder_path);
    if !embedder.exists() {
        anyhow::bail!(
            "embedder script not found: {} (override with --embedder)",
            embedder.display()
        );
    }

    eprintln!(
        "[nest] embedding query with {} via {}{}",
        model,
        embedder.display(),
        match &model_path {
            Some(p) => format!(" (--model-path {})", p.display()),
            None => String::new(),
        }
    );
    let mut cmd = ProcCommand::new("python3");
    cmd.arg(&embedder);
    if let Some(p) = &model_path {
        cmd.arg("--model-path").arg(p);
    }
    cmd.arg(&model).arg(&query);
    let out = cmd
        .output()
        .map_err(|e| anyhow::anyhow!("failed to spawn embedder: {} ({})", e, embedder.display()))?;
    if !out.status.success() {
        anyhow::bail!(
            "embedder failed (status={}): {}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let payload: EmbedderOutput = serde_json::from_slice(&out.stdout).map_err(|e| {
        anyhow::anyhow!(
            "invalid embedder output: {} (stdout={:?})",
            e,
            String::from_utf8_lossy(&out.stdout)
        )
    })?;

    // Layer 1: name match (cheap; catches obvious mistakes).
    if payload.embedding_model != model {
        anyhow::bail!(
            "model name mismatch: manifest={}, embedder reports={}",
            model,
            payload.embedding_model
        );
    }
    // Layer 2: dim match (cheap; catches dim collisions).
    if payload.embedding_dim != declared_dim || payload.vector.len() != declared_dim {
        anyhow::bail!(
            "dim mismatch: manifest={}, embedder dim={}, vector len={}",
            declared_dim,
            payload.embedding_dim,
            payload.vector.len()
        );
    }
    // Layer 3: model_hash match (the strict check; catches "same name,
    // same dim, different snapshot" silent failures).
    if !skip_model_hash_check {
        if declared_model_hash == PLACEHOLDER_MODEL_HASH {
            anyhow::bail!(
                "manifest carries the legacy placeholder model_hash ({}). Rebuild \
                 this corpus with a real fingerprint, or pass --skip-model-hash-check \
                 to proceed at your own risk.",
                PLACEHOLDER_MODEL_HASH
            );
        }
        if payload.model_hash != declared_model_hash {
            anyhow::bail!(
                "model_hash mismatch: corpus was built with {}, embedder reports {}\n\
                 fingerprint reported by embedder: {}\n\
                 hint: --model-path PATH to point at the exact snapshot, or rebuild \
                 the corpus with the model you intend to use.",
                declared_model_hash,
                payload.model_hash,
                payload.fingerprint
            );
        }
    }

    let cand = candidates.unwrap_or(((k as usize) * 4).max(64));
    let result = match runtime.declared_index_type() {
        "hnsw" => runtime.search_ann(&payload.vector, k, cand)?,
        "hybrid" => runtime.search_hybrid(&payload.vector, &query, k, cand)?,
        _ => runtime.search(&payload.vector, k)?,
    };
    print_result(&result);
    Ok(())
}

fn cmd_search_ann(file: PathBuf, query: String, k: i32, ef: usize) -> Result<()> {
    let runtime = nest_runtime::MmapNestFile::open(&file)?;
    let qvec: Vec<f32> =
        serde_json::from_str(&query).map_err(|e| anyhow::anyhow!("invalid query JSON: {}", e))?;
    let result = runtime.search_ann(&qvec, k, ef)?;
    print_result(&result);
    Ok(())
}

fn print_result(result: &nest_runtime::SearchResult) {
    println!("index_type:   {}", result.index_type);
    if !result.recall.is_nan() {
        println!("recall:       {}", result.recall);
    } else {
        println!("recall:       (not computed; rerank guarantees real cosine)");
    }
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
}

fn cmd_benchmark(file: PathBuf, n_queries: usize, k: i32, ann_ef: Option<usize>) -> Result<()> {
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

    let exact_times = run_bench(&runtime, &queries, k, "exact", |rt, q| rt.search(q, k))?;
    println!(
        "Exact ({} queries, dim={}, dtype={}, simd={}):",
        n_queries,
        dim,
        runtime.dtype().name(),
        runtime.simd_backend().name()
    );
    print_latency(&exact_times);

    if let Some(ef) = ann_ef {
        if !runtime.has_ann() {
            println!("(no HNSW section — ANN bench skipped)");
            return Ok(());
        }
        let ann_times = run_bench(&runtime, &queries, k, "ann", |rt, q| {
            rt.search_ann(q, k, ef)
        })?;
        println!("ANN ef={} ({} queries):", ef, n_queries);
        print_latency(&ann_times);

        // Recall@k of ANN vs exact, computed on the same queries.
        let mut hits_overlap_total = 0.0f64;
        for q in &queries {
            let exact = runtime.search(q, k)?;
            let approx = runtime.search_ann(q, k, ef)?;
            let exact_set: std::collections::HashSet<&str> =
                exact.hits.iter().map(|h| h.chunk_id.as_str()).collect();
            let overlap = approx
                .hits
                .iter()
                .filter(|h| exact_set.contains(h.chunk_id.as_str()))
                .count();
            hits_overlap_total += overlap as f64 / k as f64;
        }
        println!(
            "  recall@{} (ANN vs exact): {:.4}",
            k,
            hits_overlap_total / n_queries as f64
        );
    }
    Ok(())
}

fn run_bench(
    rt: &nest_runtime::MmapNestFile,
    queries: &[Vec<f32>],
    _k: i32,
    _label: &str,
    mut f: impl FnMut(
        &nest_runtime::MmapNestFile,
        &[f32],
    ) -> Result<nest_runtime::SearchResult, nest_runtime::RuntimeError>,
) -> Result<Vec<f64>> {
    let mut times = Vec::with_capacity(queries.len());
    for q in queries {
        let t0 = std::time::Instant::now();
        f(rt, q)?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok(times)
}

fn print_latency(times: &[f64]) {
    let p = |q: f64| -> f64 {
        let idx = ((times.len() as f64 - 1.0) * q).round() as usize;
        times[idx]
    };
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    println!("  mean:   {:.3} ms", mean);
    println!("  p50:    {:.3} ms", p(0.50));
    println!("  p95:    {:.3} ms", p(0.95));
    println!("  p99:    {:.3} ms", p(0.99));
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
        &view.decoded_section(nest_format::layout::SECTION_CHUNK_IDS)?,
        n,
    )?;
    let idx = ids
        .iter()
        .position(|id| id == chunk_id)
        .ok_or_else(|| anyhow::anyhow!("chunk_id {} not found in file", chunk_id))?;

    let texts = nest_format::sections::decode_chunks_canonical(
        &view.decoded_section(nest_format::layout::SECTION_CHUNKS_CANONICAL)?,
        n,
    )?;
    let spans = nest_format::sections::decode_chunks_original_spans(
        &view.decoded_section(nest_format::layout::SECTION_CHUNKS_ORIGINAL_SPANS)?,
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
    println!("rerank:       {}", view.manifest.rerank_policy);
    println!("model:        {}", view.manifest.embedding_model);
    println!("model_hash:   {}", view.manifest.model_hash);
    println!("chunker:      {}", view.manifest.chunker_version);
    println!("supports_ann: {}", view.manifest.capabilities.supports_ann);
    println!("supports_bm25:{}", view.manifest.capabilities.supports_bm25);
    println!(
        "simd_backend: {}",
        nest_runtime::simd::detect_backend().name()
    );
    println!("sections:     {}", view.section_table.len());
    for entry in &view.section_table {
        let name = nest_format::layout::section_name(entry.section_id).unwrap_or("unknown");
        let enc = encoding_name(entry.encoding);
        println!(
            "  0x{:02x} {:<24} encoding={:<8} {} bytes",
            entry.section_id, name, enc, entry.size
        );
    }
    println!("file_hash:    {}", view.file_hash_hex());
    println!("content_hash: {}", view.content_hash_hex()?);
    Ok(())
}

fn encoding_name(e: u32) -> &'static str {
    match e {
        nest_format::layout::SECTION_ENCODING_RAW => "raw",
        nest_format::layout::SECTION_ENCODING_ZSTD => "zstd",
        nest_format::layout::SECTION_ENCODING_FLOAT16 => "float16",
        nest_format::layout::SECTION_ENCODING_INT8 => "int8",
        _ => "unknown",
    }
}

/// Walk up from CARGO_MANIFEST_DIR / current dir to find python/embed_query.py.
fn default_embedder_path() -> PathBuf {
    let candidates = [
        std::env::current_dir()
            .ok()
            .map(|p| p.join("python").join("embed_query.py")),
        std::env::current_dir()
            .ok()
            .map(|p| p.join("..").join("python").join("embed_query.py")),
    ];
    for c in candidates.into_iter().flatten() {
        if c.exists() {
            return c;
        }
    }
    PathBuf::from("python/embed_query.py")
}
