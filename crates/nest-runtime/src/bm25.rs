//! Tiny BM25 inverted index for the hybrid search path.
//!
//! Tokenization is intentionally simple — lowercase, split on
//! Unicode-aware whitespace + punctuation, drop tokens with `len < 2`.
//! Multilingual corpora (PT-BR, EN, ES) get reasonable behavior without
//! a stemmer or stop list. A future version can pluggable-ize the
//! tokenizer if needed.
//!
//! The index is small enough that we keep it fully decoded in memory at
//! open time. On-disk layout (section `0x08`, payload version 1):
//!
//! ```text
//!   u32 LE  payload_version = 1
//!   f32 LE  k1                (BM25 hyperparameter)
//!   f32 LE  b                 (BM25 hyperparameter)
//!   f32 LE  avgdl             (average doc length)
//!   u32 LE  n_docs
//!   u32 LE  n_terms
//!   for d in 0..n_docs:
//!       u32 LE  dl_d          (document length in tokens)
//!   for t in 0..n_terms:
//!       u32 LE  len_t
//!       bytes   token_t       (UTF-8)
//!       u32 LE  df_t          (document frequency)
//!       for posting in postings_t:
//!           u32 LE  doc_id
//!           u32 LE  tf
//! ```

use std::collections::HashMap;

use nest_format::error::NestError;

use crate::error::RuntimeError;

pub const BM25_PAYLOAD_VERSION: u32 = 1;
pub const DEFAULT_K1: f32 = 1.5;
pub const DEFAULT_B: f32 = 0.75;

#[derive(Clone, Debug)]
struct Posting {
    doc: u32,
    tf: u32,
}

#[derive(Clone, Debug)]
struct TermEntry {
    df: u32,
    postings: Vec<Posting>,
}

pub struct Bm25Index {
    pub k1: f32,
    pub b: f32,
    pub avgdl: f32,
    pub n_docs: usize,
    pub n_terms: usize,
    doc_lengths: Vec<u32>,
    /// term -> entry. HashMap for O(1) query lookup.
    terms: HashMap<String, TermEntry>,
}

impl Bm25Index {
    /// Build a BM25 index from the canonical chunk texts.
    pub fn build(docs: &[String], k1: f32, b: f32) -> Self {
        let n_docs = docs.len();
        let mut doc_lengths = Vec::with_capacity(n_docs);
        let mut term_postings: HashMap<String, Vec<Posting>> = HashMap::new();
        for (doc_id, doc) in docs.iter().enumerate() {
            let tokens = tokenize(doc);
            doc_lengths.push(tokens.len() as u32);
            let mut tf_map: HashMap<&str, u32> = HashMap::new();
            for t in &tokens {
                *tf_map.entry(t.as_str()).or_insert(0) += 1;
            }
            for (term, tf) in tf_map {
                term_postings
                    .entry(term.to_string())
                    .or_default()
                    .push(Posting {
                        doc: doc_id as u32,
                        tf,
                    });
            }
        }
        let total_dl: u64 = doc_lengths.iter().map(|x| *x as u64).sum();
        let avgdl = if n_docs == 0 {
            0.0
        } else {
            total_dl as f32 / n_docs as f32
        };
        // Sort terms alphabetically so the on-disk encoding is reproducible.
        let mut keys: Vec<String> = term_postings.keys().cloned().collect();
        keys.sort();
        let mut terms: HashMap<String, TermEntry> = HashMap::with_capacity(keys.len());
        for key in keys {
            let mut postings = term_postings.remove(&key).unwrap();
            postings.sort_by_key(|p| p.doc);
            terms.insert(
                key,
                TermEntry {
                    df: postings.len() as u32,
                    postings,
                },
            );
        }
        let n_terms = terms.len();
        Self {
            k1,
            b,
            avgdl,
            n_docs,
            n_terms,
            doc_lengths,
            terms,
        }
    }

    /// Encode for storage in section `0x08`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&BM25_PAYLOAD_VERSION.to_le_bytes());
        out.extend_from_slice(&self.k1.to_le_bytes());
        out.extend_from_slice(&self.b.to_le_bytes());
        out.extend_from_slice(&self.avgdl.to_le_bytes());
        out.extend_from_slice(&(self.n_docs as u32).to_le_bytes());
        out.extend_from_slice(&(self.n_terms as u32).to_le_bytes());
        for &dl in &self.doc_lengths {
            out.extend_from_slice(&dl.to_le_bytes());
        }
        // Sorted by term for determinism.
        let mut terms: Vec<(&String, &TermEntry)> = self.terms.iter().collect();
        terms.sort_by(|a, b| a.0.cmp(b.0));
        for (term, entry) in terms {
            let bs = term.as_bytes();
            out.extend_from_slice(&(bs.len() as u32).to_le_bytes());
            out.extend_from_slice(bs);
            out.extend_from_slice(&entry.df.to_le_bytes());
            for p in &entry.postings {
                out.extend_from_slice(&p.doc.to_le_bytes());
                out.extend_from_slice(&p.tf.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RuntimeError> {
        let mut cur = ByteCursor::new(bytes);
        let version = cur.u32()?;
        if version != BM25_PAYLOAD_VERSION {
            return Err(RuntimeError::Format(NestError::UnsupportedSectionVersion {
                section_id: nest_format::layout::SECTION_BM25_INDEX,
                version,
            }));
        }
        let k1 = cur.f32()?;
        let b = cur.f32()?;
        let avgdl = cur.f32()?;
        let n_docs = cur.u32()? as usize;
        let n_terms = cur.u32()? as usize;
        let mut doc_lengths = Vec::with_capacity(n_docs);
        for _ in 0..n_docs {
            doc_lengths.push(cur.u32()?);
        }
        let mut terms: HashMap<String, TermEntry> = HashMap::with_capacity(n_terms);
        for _ in 0..n_terms {
            let len = cur.u32()? as usize;
            let term_bytes = cur.bytes(len)?;
            let term = std::str::from_utf8(term_bytes)
                .map_err(|e| {
                    RuntimeError::Format(NestError::MalformedSectionPayload {
                        section_id: nest_format::layout::SECTION_BM25_INDEX,
                        reason: format!("term utf-8: {}", e),
                    })
                })?
                .to_string();
            let df = cur.u32()?;
            let mut postings = Vec::with_capacity(df as usize);
            for _ in 0..df {
                let doc = cur.u32()?;
                let tf = cur.u32()?;
                postings.push(Posting { doc, tf });
            }
            terms.insert(term, TermEntry { df, postings });
        }
        Ok(Self {
            k1,
            b,
            avgdl,
            n_docs,
            n_terms,
            doc_lengths,
            terms,
        })
    }

    /// Score `query_text` against the corpus, return the top-k `(doc, score)`
    /// pairs. Empty query returns an empty vec.
    pub fn search(&self, query_text: &str, k: usize) -> Vec<(usize, f32)> {
        if k == 0 || self.n_docs == 0 {
            return Vec::new();
        }
        let q_tokens = tokenize(query_text);
        if q_tokens.is_empty() {
            return Vec::new();
        }
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for term in q_tokens {
            let Some(entry) = self.terms.get(&term) else {
                continue;
            };
            let idf =
                ((self.n_docs as f32 - entry.df as f32 + 0.5) / (entry.df as f32 + 0.5) + 1.0).ln();
            for p in &entry.postings {
                let dl = self.doc_lengths[p.doc as usize] as f32;
                let denom =
                    p.tf as f32 + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl.max(1.0));
                let term_score = idf * (p.tf as f32 * (self.k1 + 1.0)) / denom;
                *scores.entry(p.doc).or_insert(0.0) += term_score;
            }
        }
        let mut all: Vec<(usize, f32)> = scores.into_iter().map(|(d, s)| (d as usize, s)).collect();
        all.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        all.truncate(k);
        all
    }
}

/// Reciprocal-rank fusion of two ranked candidate lists. Returns the
/// union sorted by RRF score descending. The runtime then reranks with
/// the exact dot product so the final scores are real cosine.
pub fn rrf_union(a: &[usize], b: &[usize]) -> Vec<usize> {
    const K: f32 = 60.0;
    let mut scores: HashMap<usize, f32> = HashMap::new();
    for (rank, &id) in a.iter().enumerate() {
        *scores.entry(id).or_insert(0.0) += 1.0 / (K + rank as f32 + 1.0);
    }
    for (rank, &id) in b.iter().enumerate() {
        *scores.entry(id).or_insert(0.0) += 1.0 / (K + rank as f32 + 1.0);
    }
    let mut all: Vec<(usize, f32)> = scores.into_iter().collect();
    all.sort_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.0.cmp(&y.0))
    });
    all.into_iter().map(|p| p.0).collect()
}

/// Lowercase, split on non-alphanumerics, drop tokens of length < 2.
/// Unicode-aware: handles PT-BR accents (ã, ç, õ) by virtue of `char`
/// iteration. Simple but effective enough for fake-news / news corpora.
fn tokenize(s: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    for c in s.chars() {
        if c.is_alphanumeric() {
            for low in c.to_lowercase() {
                cur.push(low);
            }
        } else if !cur.is_empty() {
            if cur.chars().count() >= 2 {
                out.push(std::mem::take(&mut cur));
            } else {
                cur.clear();
            }
        }
    }
    if cur.chars().count() >= 2 {
        out.push(cur);
    }
    out
}

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
                section_id: nest_format::layout::SECTION_BM25_INDEX,
                reason: "unexpected EOF (u32)".into(),
            }));
        }
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn f32(&mut self) -> Result<f32, RuntimeError> {
        if self.pos + 4 > self.buf.len() {
            return Err(RuntimeError::Format(NestError::MalformedSectionPayload {
                section_id: nest_format::layout::SECTION_BM25_INDEX,
                reason: "unexpected EOF (f32)".into(),
            }));
        }
        let v = f32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8], RuntimeError> {
        if self.pos + n > self.buf.len() {
            return Err(RuntimeError::Format(NestError::MalformedSectionPayload {
                section_id: nest_format::layout::SECTION_BM25_INDEX,
                reason: format!("unexpected EOF (bytes {})", n),
            }));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokenizer_handles_pt_br() {
        let toks = tokenize("Olá mundo. ESTA é uma frase com acentuação.");
        // "olá", "mundo", "esta", "uma", "frase", "com", "acentuação"
        // (single-char tokens dropped: "é")
        assert!(toks.contains(&"olá".to_string()));
        assert!(toks.contains(&"acentuação".to_string()));
        assert!(!toks.contains(&"é".to_string()));
    }

    #[test]
    fn bm25_finds_relevant_doc() {
        let docs = vec![
            "vacina contra covid no brasil".to_string(),
            "futebol jogos resultados".to_string(),
            "vacinação contra a gripe".to_string(),
            "meteorologia previsão tempo".to_string(),
        ];
        let idx = Bm25Index::build(&docs, DEFAULT_K1, DEFAULT_B);
        let hits = idx.search("vacina covid", 4);
        assert_eq!(hits[0].0, 0, "doc 0 should rank first for 'vacina covid'");
    }

    #[test]
    fn bm25_serialize_roundtrip() {
        let docs = vec![
            "alpha beta gamma".to_string(),
            "alpha delta".to_string(),
            "gamma omega".to_string(),
        ];
        let idx = Bm25Index::build(&docs, DEFAULT_K1, DEFAULT_B);
        let bytes = idx.to_bytes();
        let back = Bm25Index::from_bytes(&bytes).unwrap();
        let a = idx.search("alpha", 3);
        let b = back.search("alpha", 3);
        assert_eq!(a, b);
    }

    #[test]
    fn rrf_union_combines_sources() {
        let a = vec![10, 20, 30];
        let b = vec![20, 40, 50];
        let union = rrf_union(&a, &b);
        // 20 appears in both → ranks first.
        assert_eq!(union[0], 20);
        // All ids appear exactly once.
        let mut s = union.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), union.len());
    }
}
