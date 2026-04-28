//! Pure-Rust HNSW index for approximate nearest neighbor search.
//!
//! This is a minimal, self-contained HNSW implementation tailored to
//! `.nest`'s contract:
//!
//! - Vectors are L2-normalized → distance is `1 - cosine` (smaller = closer).
//! - Index lives in section `0x07` and is bit-equal across rebuilds.
//! - Search returns a candidate set; the runtime reranks with the exact
//!   dot product against the embeddings section so the final score is
//!   the real cosine.
//!
//! On-disk layout (`encoding=raw`, payload version 1):
//!
//! ```text
//!   u32 LE  payload_version = 1
//!   u32 LE  m                 — out-degree at non-zero levels
//!   u32 LE  m_max0            — out-degree at level 0 (typically 2*m)
//!   u32 LE  ef_construction
//!   u32 LE  entry_point       — node id of the entry vertex
//!   u32 LE  max_level         — highest layer with any node (0-based)
//!   u32 LE  n_nodes           — equal to header.n_embeddings
//!   for each node i in 0..n_nodes:
//!       u32 LE  level_i       — top layer this node lives in
//!       for layer in 0..=level_i:
//!           u32 LE  k_i_l     — neighbor count at this layer
//!           u32 LE * k_i_l    — neighbor ids
//! ```
//!
//! Construction uses the standard HNSW algorithm (Malkov & Yashunin, 2018)
//! with a deterministic level distribution so the same input produces the
//! same graph (required for reproducible builds).

use std::collections::{BinaryHeap, HashSet};

use nest_format::Int8EmbeddingsView;
use nest_format::error::NestError;

use crate::error::RuntimeError;

pub const HNSW_PAYLOAD_VERSION: u32 = 1;

/// Default neighbor count at non-zero layers. 16 is a common HNSW sweet
/// spot for ~1M points; for smaller corpora the recall-vs-size curve is
/// flat enough that the default is fine.
pub const DEFAULT_M: usize = 16;
/// Default candidate-list size during construction. Larger = better
/// recall, slower build.
pub const DEFAULT_EF_CONSTRUCTION: usize = 200;

#[derive(Clone, Debug)]
struct Node {
    /// Top layer this node lives in (0-based).
    level: u32,
    /// `neighbors[layer][i]` is the i-th neighbor id at `layer`. Index 0
    /// is the densest layer (level 0).
    neighbors: Vec<Vec<u32>>,
}

/// A built HNSW index. Reads borrow from the on-disk payload at open
/// time; the graph is owned (small relative to embeddings).
pub struct HnswIndex {
    pub m: usize,
    pub m_max0: usize,
    pub ef_construction: usize,
    pub entry_point: u32,
    pub max_level: u32,
    nodes: Vec<Node>,
    /// A snapshot of the f32 vectors used at search time. We copy
    /// because we want the ANN graph to be dtype-independent: f16/i8
    /// runtimes still get the same recall curve. Cost: ~n*dim*4 bytes
    /// of RAM beyond the mmap.
    vectors: Vec<f32>,
    dim: usize,
    n: usize,
    /// `ef_search` default. Caller can override per query.
    pub ef_search: usize,
}

impl HnswIndex {
    /// Build an HNSW index from f32 vectors. Deterministic given the
    /// same `seed`. `vectors` is row-major `n*dim`.
    pub fn build(
        vectors: Vec<f32>,
        n: usize,
        dim: usize,
        m: usize,
        ef_construction: usize,
        seed: u64,
    ) -> Self {
        assert_eq!(vectors.len(), n * dim);
        let m_max0 = m * 2;
        let mut rng = LcgRng::new(seed);

        let mut nodes: Vec<Node> = Vec::with_capacity(n);
        for _ in 0..n {
            let level = sample_level(&mut rng, m);
            let mut neighbors = Vec::with_capacity((level as usize) + 1);
            for _ in 0..=level {
                neighbors.push(Vec::new());
            }
            nodes.push(Node { level, neighbors });
        }

        // Insert nodes in id order for determinism.
        let mut entry_point: u32 = 0;
        let mut max_level: u32 = 0;

        for i in 0..n {
            let level = nodes[i].level;
            if i == 0 {
                entry_point = 0;
                max_level = level;
                continue;
            }

            // 1. Greedy walk from entry_point down to layer (level+1) to
            //    find the closest entry on the layer above this node.
            let q_off = i * dim;
            let q = &vectors[q_off..q_off + dim];
            let mut curr = entry_point;
            for layer in (level + 1..=max_level).rev() {
                curr = greedy_search(curr, q, layer, &nodes, &vectors, dim);
            }

            // 2. From `curr`, do an `ef_construction`-sized search on
            //    each layer from `min(level, max_level)` down to 0;
            //    take top-m as neighbors and link bidirectionally.
            let mut entry = curr;
            let start_layer = level.min(max_level);
            for layer in (0..=start_layer).rev() {
                let mut candidates = layer_search(
                    &[entry],
                    q,
                    layer,
                    ef_construction,
                    &nodes,
                    &vectors,
                    dim,
                    i as u32,
                );
                let m_layer = if layer == 0 { m_max0 } else { m };
                truncate_and_keep_top(&mut candidates, m_layer);
                let neighbor_ids: Vec<u32> = candidates.iter().map(|c| c.id).collect();
                nodes[i].neighbors[layer as usize] = neighbor_ids.clone();

                // Backlinks. Insert i into each neighbor's list and prune
                // if it overflows m_layer.
                for &nbr in &neighbor_ids {
                    let nbr_idx = nbr as usize;
                    let layer_idx = layer as usize;
                    if layer_idx >= nodes[nbr_idx].neighbors.len() {
                        continue;
                    }
                    if nodes[nbr_idx].neighbors[layer_idx].len() >= m_layer {
                        // Prune: keep the m_layer closest of (existing + i).
                        let mut all: Vec<Candidate> = nodes[nbr_idx].neighbors[layer_idx]
                            .iter()
                            .map(|&id| Candidate {
                                id,
                                dist: cosine_dist(
                                    &vectors[(nbr as usize) * dim..(nbr as usize + 1) * dim],
                                    &vectors[(id as usize) * dim..(id as usize + 1) * dim],
                                ),
                            })
                            .collect();
                        all.push(Candidate {
                            id: i as u32,
                            dist: cosine_dist(
                                &vectors[nbr_idx * dim..(nbr_idx + 1) * dim],
                                &vectors[i * dim..(i + 1) * dim],
                            ),
                        });
                        all.sort_by(|a, b| {
                            a.dist
                                .partial_cmp(&b.dist)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        all.truncate(m_layer);
                        nodes[nbr_idx].neighbors[layer_idx] =
                            all.into_iter().map(|c| c.id).collect();
                    } else if !nodes[nbr_idx].neighbors[layer_idx].contains(&(i as u32)) {
                        nodes[nbr_idx].neighbors[layer_idx].push(i as u32);
                    }
                }

                if !candidates.is_empty() {
                    entry = candidates[0].id;
                }
            }

            if level > max_level {
                max_level = level;
                entry_point = i as u32;
            }
        }

        Self {
            m,
            m_max0,
            ef_construction,
            entry_point,
            max_level,
            nodes,
            vectors,
            dim,
            n,
            ef_search: ef_construction,
        }
    }

    /// Build from int8 quantized embeddings. Dequantizes once (lossy) and
    /// hands off to `build`. Recall ends up bounded by quantization
    /// noise; in practice still well above 0.95 @ k=10 for real corpora.
    pub fn build_from_int8(view: &Int8EmbeddingsView<'_>, m: usize, ef: usize, seed: u64) -> Self {
        let mut vectors = vec![0.0f32; view.n * view.dim];
        for i in 0..view.n {
            let scale = view.scale(i);
            let row = view.row(i);
            for j in 0..view.dim {
                vectors[i * view.dim + j] = row[j] as f32 * scale;
            }
        }
        Self::build(vectors, view.n, view.dim, m, ef, seed)
    }

    /// Build from float16 LE bytes. Decodes once into f32 and builds.
    pub fn build_from_f16(
        bytes: &[u8],
        n: usize,
        dim: usize,
        m: usize,
        ef: usize,
        seed: u64,
    ) -> Self {
        let vectors = nest_format::f16_bytes_to_f32(bytes);
        Self::build(vectors, n, dim, m, ef, seed)
    }

    /// Build from raw f32 LE bytes. Copies into an owned buffer.
    pub fn build_from_f32(
        bytes: &[u8],
        n: usize,
        dim: usize,
        m: usize,
        ef: usize,
        seed: u64,
    ) -> Self {
        let mut vectors = vec![0.0f32; n * dim];
        for (i, slot) in vectors.iter_mut().enumerate() {
            let off = i * 4;
            *slot =
                f32::from_le_bytes([bytes[off], bytes[off + 1], bytes[off + 2], bytes[off + 3]]);
        }
        Self::build(vectors, n, dim, m, ef, seed)
    }

    /// Encode the index to bytes for embedding in section `0x07`.
    pub fn to_bytes(&self) -> Vec<u8> {
        // 7 u32 header fields + per-node (level + per-layer (count + ids)).
        let mut out = Vec::with_capacity(7 * 4 + self.n * 16);
        out.extend_from_slice(&HNSW_PAYLOAD_VERSION.to_le_bytes());
        out.extend_from_slice(&(self.m as u32).to_le_bytes());
        out.extend_from_slice(&(self.m_max0 as u32).to_le_bytes());
        out.extend_from_slice(&(self.ef_construction as u32).to_le_bytes());
        out.extend_from_slice(&self.entry_point.to_le_bytes());
        out.extend_from_slice(&self.max_level.to_le_bytes());
        out.extend_from_slice(&(self.n as u32).to_le_bytes());
        for node in &self.nodes {
            out.extend_from_slice(&node.level.to_le_bytes());
            for layer in 0..=node.level {
                let nbrs = &node.neighbors[layer as usize];
                out.extend_from_slice(&(nbrs.len() as u32).to_le_bytes());
                for &id in nbrs {
                    out.extend_from_slice(&id.to_le_bytes());
                }
            }
        }
        out
    }

    /// Parse an HNSW payload from bytes. The vectors are reconstructed
    /// from the embeddings section by the caller; this constructor is
    /// for the on-disk graph only.
    pub fn from_bytes(bytes: &[u8], n: usize, dim: usize) -> Result<Self, RuntimeError> {
        let mut cur = ByteCursor::new(bytes);
        let version = cur.u32()?;
        if version != HNSW_PAYLOAD_VERSION {
            return Err(RuntimeError::Format(NestError::UnsupportedSectionVersion {
                section_id: nest_format::layout::SECTION_HNSW_INDEX,
                version,
            }));
        }
        let m = cur.u32()? as usize;
        let m_max0 = cur.u32()? as usize;
        let ef_construction = cur.u32()? as usize;
        let entry_point = cur.u32()?;
        let max_level = cur.u32()?;
        let n_nodes = cur.u32()? as usize;
        if n_nodes != n {
            return Err(RuntimeError::Format(NestError::MalformedSectionPayload {
                section_id: nest_format::layout::SECTION_HNSW_INDEX,
                reason: format!("node count {} != n_embeddings {}", n_nodes, n),
            }));
        }
        let mut nodes = Vec::with_capacity(n_nodes);
        for _ in 0..n_nodes {
            let level = cur.u32()?;
            let mut neighbors = Vec::with_capacity((level as usize) + 1);
            for _ in 0..=level {
                let k = cur.u32()? as usize;
                let mut ids = Vec::with_capacity(k);
                for _ in 0..k {
                    ids.push(cur.u32()?);
                }
                neighbors.push(ids);
            }
            nodes.push(Node { level, neighbors });
        }
        // The reader must materialize vectors before we can search; until
        // then `vectors` is empty. The runtime sets this via attach_vectors.
        Ok(Self {
            m,
            m_max0,
            ef_construction,
            entry_point,
            max_level,
            nodes,
            vectors: Vec::new(),
            dim,
            n,
            ef_search: ef_construction,
        })
    }

    /// Attach the f32 vectors needed at search time. Called by
    /// `MmapNestFile::open` after the embeddings section is decoded.
    pub fn attach_vectors(&mut self, vectors: Vec<f32>) {
        debug_assert_eq!(vectors.len(), self.n * self.dim);
        self.vectors = vectors;
    }

    /// Search for the `ef` closest candidates to `q`. Returns ids only —
    /// the runtime reranks with the exact dot product to produce the
    /// final cosine score.
    pub fn search(&self, q: &[f32], ef: usize) -> Vec<usize> {
        if self.n == 0 {
            return Vec::new();
        }
        if self.vectors.is_empty() {
            // Index is loaded but vectors haven't been attached. Return
            // an empty candidate set; runtime falls back to exact.
            return Vec::new();
        }
        let mut curr = self.entry_point;
        for layer in (1..=self.max_level).rev() {
            curr = greedy_search(curr, q, layer, &self.nodes, &self.vectors, self.dim);
        }
        let candidates = layer_search(
            &[curr],
            q,
            0,
            ef.max(self.ef_search),
            &self.nodes,
            &self.vectors,
            self.dim,
            u32::MAX,
        );
        candidates.into_iter().map(|c| c.id as usize).collect()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
struct Candidate {
    id: u32,
    /// `1 - cosine`. Smaller = closer.
    dist: f32,
}

impl Eq for Candidate {}

// `BinaryHeap` is a max-heap; we want closest-first / farthest-first
// orderings. Define orderings explicitly.
#[derive(Clone, Copy)]
struct ByDistAsc(Candidate);
#[derive(Clone, Copy)]
struct ByDistDesc(Candidate);

impl PartialEq for ByDistAsc {
    fn eq(&self, other: &Self) -> bool {
        self.0.dist == other.0.dist
    }
}
impl Eq for ByDistAsc {}
impl Ord for ByDistAsc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse so BinaryHeap (max-heap) pops smallest distance.
        other
            .0
            .dist
            .partial_cmp(&self.0.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for ByDistAsc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ByDistDesc {
    fn eq(&self, other: &Self) -> bool {
        self.0.dist == other.0.dist
    }
}
impl Eq for ByDistDesc {}
impl Ord for ByDistDesc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .dist
            .partial_cmp(&other.0.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}
impl PartialOrd for ByDistDesc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    // Vectors are L2-normalized → cosine = dot. Distance = 1 - cosine.
    let mut dot = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
    }
    1.0 - dot
}

/// Greedy descent on a single layer until no neighbor is closer.
fn greedy_search(
    entry: u32,
    q: &[f32],
    layer: u32,
    nodes: &[Node],
    vectors: &[f32],
    dim: usize,
) -> u32 {
    let mut curr = entry;
    let mut curr_dist = cosine_dist(
        q,
        &vectors[(curr as usize) * dim..(curr as usize + 1) * dim],
    );
    loop {
        let layer_idx = layer as usize;
        if layer_idx >= nodes[curr as usize].neighbors.len() {
            return curr;
        }
        let nbrs = &nodes[curr as usize].neighbors[layer_idx];
        let mut best = curr;
        let mut best_dist = curr_dist;
        for &nbr in nbrs {
            let d = cosine_dist(q, &vectors[(nbr as usize) * dim..(nbr as usize + 1) * dim]);
            if d < best_dist {
                best = nbr;
                best_dist = d;
            }
        }
        if best == curr {
            return curr;
        }
        curr = best;
        curr_dist = best_dist;
    }
}

/// Search a single layer with a candidate list of size `ef`. Returns
/// the best `ef` candidates sorted ascending by distance.
#[allow(clippy::too_many_arguments)]
fn layer_search(
    entries: &[u32],
    q: &[f32],
    layer: u32,
    ef: usize,
    nodes: &[Node],
    vectors: &[f32],
    dim: usize,
    skip_id: u32,
) -> Vec<Candidate> {
    let mut visited: HashSet<u32> = HashSet::new();
    // BinaryHeap orderings:
    //   `frontier` — min-heap by distance (closest first to expand).
    //   `result`   — max-heap by distance (so we can prune the farthest).
    let mut frontier: BinaryHeap<ByDistAsc> = BinaryHeap::new();
    let mut result: BinaryHeap<ByDistDesc> = BinaryHeap::new();

    for &e in entries {
        if e == skip_id {
            continue;
        }
        let d = cosine_dist(q, &vectors[(e as usize) * dim..(e as usize + 1) * dim]);
        let c = Candidate { id: e, dist: d };
        frontier.push(ByDistAsc(c));
        result.push(ByDistDesc(c));
        visited.insert(e);
    }

    while let Some(ByDistAsc(curr)) = frontier.pop() {
        let worst_in_result = result.peek().map(|r| r.0.dist).unwrap_or(f32::INFINITY);
        if curr.dist > worst_in_result && result.len() >= ef {
            break;
        }
        let layer_idx = layer as usize;
        if layer_idx >= nodes[curr.id as usize].neighbors.len() {
            continue;
        }
        let nbrs = &nodes[curr.id as usize].neighbors[layer_idx];
        for &nbr in nbrs {
            if nbr == skip_id || !visited.insert(nbr) {
                continue;
            }
            let d = cosine_dist(q, &vectors[(nbr as usize) * dim..(nbr as usize + 1) * dim]);
            let worst = result.peek().map(|r| r.0.dist).unwrap_or(f32::INFINITY);
            if result.len() < ef || d < worst {
                let c = Candidate { id: nbr, dist: d };
                frontier.push(ByDistAsc(c));
                result.push(ByDistDesc(c));
                if result.len() > ef {
                    result.pop();
                }
            }
        }
    }

    let mut out: Vec<Candidate> = result.into_iter().map(|w| w.0).collect();
    out.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn truncate_and_keep_top(candidates: &mut Vec<Candidate>, k: usize) {
    candidates.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(k);
}

/// Geometric level distribution. Deterministic via the LCG state.
fn sample_level(rng: &mut LcgRng, m: usize) -> u32 {
    let m_l = 1.0 / (m as f64).ln();
    let r = rng.next_f64();
    if r <= 0.0 {
        return 0;
    }
    let level = (-(r.ln()) * m_l).floor() as i64;
    level.clamp(0, 31) as u32 // cap at 31 layers
}

/// Tiny LCG (deterministic, no rand dep).
struct LcgRng {
    state: u64,
}
impl LcgRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493),
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f64(&mut self) -> f64 {
        // Map upper 53 bits to [0, 1).
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }
}

/// Light cursor for parsing the on-disk HNSW payload.
struct ByteCursor<'a> {
    buf: &'a [u8],
    pos: usize,
}
impl<'a> ByteCursor<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
    fn u32(&mut self) -> Result<u32, RuntimeError> {
        if self.pos + 4 > self.buf.len() {
            return Err(RuntimeError::Format(NestError::MalformedSectionPayload {
                section_id: nest_format::layout::SECTION_HNSW_INDEX,
                reason: "unexpected EOF".into(),
            }));
        }
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = LcgRng::new(seed);
        let mut v = Vec::with_capacity(n * dim);
        for _ in 0..(n * dim) {
            v.push((rng.next_f64() as f32) - 0.5);
        }
        // L2-normalize each row.
        for i in 0..n {
            let row = &mut v[i * dim..(i + 1) * dim];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }
        v
    }

    #[test]
    fn small_index_recall_against_exact() {
        // 200 random vectors, dim 32. Recall@10 vs exact should be very
        // high — small enough that the graph is fully connected.
        let n = 200;
        let dim = 32;
        let vecs = random_vectors(n, dim, 0xDEAD_BEEF);
        let idx = HnswIndex::build(vecs.clone(), n, dim, 8, 50, 42);

        let q = random_vectors(1, dim, 0xCAFEBABE);
        // Exact top-10.
        let mut exact: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let row = &vecs[i * dim..(i + 1) * dim];
                let mut s = 0.0f32;
                for j in 0..dim {
                    s += q[j] * row[j];
                }
                (i, s)
            })
            .collect();
        exact.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let exact_top: HashSet<usize> = exact.iter().take(10).map(|p| p.0).collect();
        let approx = idx.search(&q, 50);
        let approx_set: HashSet<usize> = approx.into_iter().take(10).collect();
        let overlap = exact_top.intersection(&approx_set).count();
        assert!(
            overlap >= 7,
            "recall@10 too low: {} of 10 (expected >= 7)",
            overlap
        );
    }

    #[test]
    fn serialize_roundtrip() {
        let n = 50;
        let dim = 16;
        let vecs = random_vectors(n, dim, 7);
        let idx = HnswIndex::build(vecs.clone(), n, dim, 8, 30, 42);
        let bytes = idx.to_bytes();
        let mut decoded = HnswIndex::from_bytes(&bytes, n, dim).unwrap();
        decoded.attach_vectors(vecs);
        // Same query should produce a similar candidate set.
        let q: Vec<f32> = vec![1.0 / (dim as f32).sqrt(); dim];
        let a = idx.search(&q, 20);
        let b = decoded.search(&q, 20);
        let a_set: HashSet<usize> = a.into_iter().collect();
        let b_set: HashSet<usize> = b.into_iter().collect();
        assert_eq!(a_set, b_set, "candidate sets must match after roundtrip");
    }

    #[test]
    fn deterministic_build_same_seed() {
        let n = 30;
        let dim = 8;
        let vecs = random_vectors(n, dim, 0xABCD);
        let a = HnswIndex::build(vecs.clone(), n, dim, 4, 20, 123);
        let b = HnswIndex::build(vecs, n, dim, 4, 20, 123);
        assert_eq!(a.to_bytes(), b.to_bytes(), "same seed => same graph");
    }
}
