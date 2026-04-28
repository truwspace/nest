//! Neighbor selection during HNSW construction.
//!
//! Two variants live here:
//!
//! - `select_neighbors_simple` (Algorithm 3 in Malkov-Yashunin 2018):
//!   sort candidates ascending by distance to the inserted node, keep
//!   top-m. Cheap. Picks geometrically-closest neighbors but tends to
//!   create dense clusters that hurt recall.
//!
//! - `select_neighbors_heuristic` (Algorithm 4): iteratively pick the
//!   closest remaining candidate, but reject it if there is already a
//!   chosen neighbor that is closer to the candidate than the candidate
//!   is to the query — that is, the candidate would be redundant. This
//!   gives angular diversity and dramatically improves recall on
//!   non-uniform corpora.
//!
//! Phase 1 wires `select_neighbors_simple` everywhere; Phase 2 swaps in
//! the heuristic where appropriate (build/insert call sites in
//! `super::build`).

use super::{Candidate, cosine_dist};

/// Algorithm 3: keep the `m` closest candidates by distance ascending.
/// Stable: ties preserve input order so the same candidate set + same
/// `m` always returns the same neighbor list.
pub(super) fn select_neighbors_simple(candidates: &[Candidate], m: usize) -> Vec<u32> {
    let mut sorted: Vec<Candidate> = candidates.to_vec();
    sorted.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });
    sorted.truncate(m);
    sorted.into_iter().map(|c| c.id).collect()
}

/// Algorithm 4 (Malkov & Yashunin 2018): heuristic neighbor selection
/// with diversity. Picks neighbors that maximize angular coverage instead
/// of just minimizing distance.
///
/// Inputs:
/// - `candidates`: pre-scored candidates (any order; we re-sort).
/// - `m`: target neighbor count.
/// - `vectors`: row-major f32 vectors of length `n * dim`.
/// - `dim`: vector dimensionality.
/// - `extend_candidates`: if true, expand the candidate set with each
///   candidate's neighbors before selecting — improves recall on sparse
///   layers at the cost of more distance calls. Off by default.
/// - `keep_pruned`: if true, refill the result with the closest pruned
///   candidates when fewer than `m` survive the diversity test. On by
///   default to ensure neighbor lists actually reach `m` even on
///   pathological queries.
///
/// **Not yet wired into `build`/`insert`** — Phase 2. The simple variant
/// remains the live call site to keep Phase 1 a no-op refactor.
#[allow(dead_code)]
pub(super) fn select_neighbors_heuristic(
    candidates: &[Candidate],
    m: usize,
    vectors: &[f32],
    dim: usize,
    extend_candidates: bool,
    keep_pruned: bool,
) -> Vec<u32> {
    if candidates.is_empty() || m == 0 {
        return Vec::new();
    }

    // 1. Working set W = candidates (optionally extended).
    let mut working: Vec<Candidate> = candidates.to_vec();
    if extend_candidates {
        let _ = &mut working; // placeholder — extension uses graph state
        // we don't have here. Implementation deferred to Phase 2 where
        // `build` passes in the graph; for now it's a no-op so the API
        // is stable.
    }
    working.sort_by(|a, b| {
        a.dist
            .partial_cmp(&b.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.cmp(&b.id))
    });

    // 2. Greedily pick: take closest; reject if a chosen neighbor is
    //    strictly closer to candidate than candidate is to query.
    let mut chosen: Vec<Candidate> = Vec::with_capacity(m);
    let mut pruned: Vec<Candidate> = Vec::new();
    for cand in working {
        if chosen.len() >= m {
            break;
        }
        let cand_row = &vectors[(cand.id as usize) * dim..(cand.id as usize + 1) * dim];
        let mut redundant = false;
        for chosen_one in &chosen {
            let chosen_row =
                &vectors[(chosen_one.id as usize) * dim..(chosen_one.id as usize + 1) * dim];
            let d_cc = cosine_dist(cand_row, chosen_row);
            // If chosen is closer to candidate than query is, candidate
            // adds no new coverage.
            if d_cc < cand.dist {
                redundant = true;
                break;
            }
        }
        if redundant {
            pruned.push(cand);
        } else {
            chosen.push(cand);
        }
    }

    // 3. Refill from pruned (closest first) if under m.
    if keep_pruned && chosen.len() < m {
        pruned.sort_by(|a, b| {
            a.dist
                .partial_cmp(&b.dist)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        for cand in pruned {
            if chosen.len() >= m {
                break;
            }
            chosen.push(cand);
        }
    }

    chosen.into_iter().map(|c| c.id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cand(id: u32, dist: f32) -> Candidate {
        Candidate { id, dist }
    }

    #[test]
    fn simple_keeps_top_m_by_distance() {
        let cs = vec![cand(0, 0.5), cand(1, 0.1), cand(2, 0.3), cand(3, 0.2)];
        assert_eq!(select_neighbors_simple(&cs, 2), vec![1, 3]);
        assert_eq!(select_neighbors_simple(&cs, 4), vec![1, 3, 2, 0]);
    }

    #[test]
    fn simple_breaks_ties_by_id() {
        let cs = vec![cand(2, 0.1), cand(0, 0.1), cand(1, 0.1)];
        assert_eq!(select_neighbors_simple(&cs, 2), vec![0, 1]);
    }

    #[test]
    fn heuristic_returns_at_most_m() {
        // Tiny synthetic: 4 vectors in 2D, all unit-norm.
        let vectors = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0];
        let cs = vec![cand(0, 0.0), cand(1, 0.5), cand(2, 1.0), cand(3, 1.5)];
        let picked = select_neighbors_heuristic(&cs, 2, &vectors, 2, false, true);
        assert!(picked.len() <= 2);
        // 0 must always be picked (it's the closest).
        assert!(picked.contains(&0));
    }
}
