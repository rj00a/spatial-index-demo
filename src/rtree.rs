use std::mem;

#[cfg(test)]
use approx::relative_eq;
use arrayvec::ArrayVec;
use ordered_float::OrderedFloat;
use vek::Aabr;

pub struct RTree<T, const MIN: usize, const MAX: usize> {
    root: Node<T, MIN, MAX>,
    // The bufs are put here to reuse their allocations.
    internal_split_buf: InternalBuf<T, MIN, MAX>,
    leaf_split_buf: LeafBuf<T>,
    reinsert_buf: LeafBuf<T>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum QueryAction {
    Continue,
    Break,
}

type InternalBuf<T, const MIN: usize, const MAX: usize> = Vec<(Box<Node<T, MIN, MAX>>, Aabr<f32>)>;
type LeafBuf<T> = Vec<(T, Aabr<f32>)>;

enum Node<T, const MIN: usize, const MAX: usize> {
    Internal(ArrayVec<(Box<Node<T, MIN, MAX>>, Aabr<f32>), MAX>),
    Leaf(ArrayVec<(T, Aabr<f32>), MAX>),
}

impl<T, const MIN: usize, const MAX: usize> RTree<T, MIN, MAX> {
    pub fn new() -> Self {
        assert!(
            MIN >= 2 && MIN <= MAX / 2 && MAX >= 2,
            "invalid R-Tree configuration"
        );

        Self {
            root: Node::Leaf(ArrayVec::new()),
            internal_split_buf: Vec::new(),
            leaf_split_buf: Vec::new(),
            reinsert_buf: Vec::new(),
        }
    }

    pub fn insert(&mut self, data: T, data_aabr: Aabr<f32>) {
        if let InsertResult::Split(new_node) = self.root.insert(
            data,
            data_aabr,
            &mut self.internal_split_buf,
            &mut self.leaf_split_buf,
        ) {
            let root_aabr = self.root.bounds();
            let new_node_aabr = new_node.bounds();

            let old_root = mem::replace(&mut self.root, Node::Internal(ArrayVec::new()));

            match &mut self.root {
                Node::Internal(children) => {
                    children.push((Box::new(old_root), root_aabr));
                    children.push((new_node, new_node_aabr));
                }
                Node::Leaf(_) => unreachable!(),
            }
        }
    }

    pub fn retain(
        &mut self,
        mut collides: impl FnMut(Aabr<f32>) -> bool,
        mut retain: impl FnMut(&mut T, &mut Aabr<f32>) -> bool,
    ) {
        self.root
            .retain(None, &mut collides, &mut retain, &mut self.reinsert_buf);

        if let Node::Internal(children) = &mut self.root {
            if children.len() == 1 {
                let new_root = *children.drain(..).next().unwrap().0;
                self.root = new_root;
            } else if children.is_empty() {
                self.root = Node::Leaf(ArrayVec::new());
            }
        }

        let mut reinsert_buf = mem::take(&mut self.reinsert_buf);

        for (data, data_aabr) in reinsert_buf.drain(..) {
            self.insert(data, data_aabr);
        }

        debug_assert!(self.reinsert_buf.capacity() == 0);
        self.reinsert_buf = reinsert_buf;

        // Don't waste too much memory after a large restructuring.
        self.reinsert_buf.shrink_to(16);
    }

    pub fn query(
        &self,
        mut collides: impl FnMut(Aabr<f32>) -> bool,
        mut callback: impl FnMut(&T, Aabr<f32>) -> QueryAction,
    ) {
        self.root.query(&mut collides, &mut callback);
    }

    pub fn clear(&mut self) {
        self.root = Node::Leaf(ArrayVec::new());
    }

    pub fn depth(&self) -> usize {
        self.root.depth(0)
    }

    /// For the purposes of rendering the R-Tree.
    pub fn visit(&self, mut f: impl FnMut(Aabr<f32>, usize)) {
        self.root.visit(&mut f, 1);
        if self.root.children_count() != 0 {
            f(self.root.bounds(), 0);
        }
    }

    #[cfg(test)]
    fn check_invariants(&self, expected_len: usize) {
        assert!(self.internal_split_buf.is_empty());
        assert!(self.leaf_split_buf.is_empty());
        assert!(self.reinsert_buf.is_empty());

        if let Node::Internal(children) = &self.root {
            assert!(
                children.len() != 1,
                "internal root with a single entry should become the child"
            );
            assert!(!children.is_empty(), "empty internal root should be a leaf");
        }

        let mut len_counter = 0;

        self.root.check_invariants(None, 0, &mut len_counter);

        assert_eq!(
            len_counter, expected_len,
            "unexpected number of entries in rtree"
        )
    }
}

impl<T, const MIN: usize, const MAX: usize> Default for RTree<T, MIN, MAX> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const MIN: usize, const MAX: usize> Node<T, MIN, MAX> {
    fn bounds(&self) -> Aabr<f32> {
        match self {
            Node::Internal(children) => children
                .iter()
                .map(|(_, aabr)| *aabr)
                .reduce(Aabr::union)
                .unwrap(),
            Node::Leaf(children) => children
                .iter()
                .map(|(_, aabr)| *aabr)
                .reduce(Aabr::union)
                .unwrap(),
        }
    }

    fn children_count(&self) -> usize {
        match self {
            Node::Internal(children) => children.len(),
            Node::Leaf(children) => children.len(),
        }
    }

    fn insert(
        &mut self,
        data: T,
        data_aabr: Aabr<f32>,
        internal_split_buf: &mut InternalBuf<T, MIN, MAX>,
        leaf_split_buf: &mut LeafBuf<T>,
    ) -> InsertResult<T, MIN, MAX> {
        match self {
            Self::Internal(children) => {
                let (best_child, best_child_aabr) = {
                    let best = area_insertion_heuristic(data_aabr, children);
                    &mut children[best]
                };

                match best_child.insert(data, data_aabr, internal_split_buf, leaf_split_buf) {
                    InsertResult::Ok => {
                        best_child_aabr.expand_to_contain(data_aabr);
                        InsertResult::Ok
                    }
                    InsertResult::Split(new_node) => {
                        let new_node_aabr = new_node.bounds();
                        *best_child_aabr = best_child.bounds();

                        if children.is_full() {
                            let other = split_node::<_, MIN, MAX>(
                                internal_split_buf,
                                children,
                                (new_node, new_node_aabr),
                            );
                            InsertResult::Split(Box::new(Node::Internal(other)))
                        } else {
                            children.push((new_node, new_node_aabr));
                            InsertResult::Ok
                        }
                    }
                }
            }
            Self::Leaf(children) => {
                if children.is_full() {
                    let other =
                        split_node::<_, MIN, MAX>(leaf_split_buf, children, (data, data_aabr));
                    debug_assert!(other.len() >= MIN);

                    InsertResult::Split(Box::new(Node::Leaf(other)))
                } else {
                    children.push((data, data_aabr));
                    InsertResult::Ok
                }
            }
        }
    }

    fn query(
        &self,
        collides: &mut impl FnMut(Aabr<f32>) -> bool,
        callback: &mut impl FnMut(&T, Aabr<f32>) -> QueryAction,
    ) -> QueryAction {
        match self {
            Node::Internal(children) => {
                for child in children {
                    if collides(child.1) {
                        if let QueryAction::Break = child.0.query(collides, callback) {
                            return QueryAction::Break;
                        }
                    }
                }
            }
            Node::Leaf(children) => {
                for (child, child_aabr) in children {
                    if collides(*child_aabr) {
                        if let QueryAction::Break = callback(child, *child_aabr) {
                            return QueryAction::Break;
                        }
                    }
                }
            }
        }
        QueryAction::Continue
    }

    fn retain(
        &mut self,
        bounds: Option<Aabr<f32>>, // `None` when self is root.
        collides: &mut impl FnMut(Aabr<f32>) -> bool,
        retain: &mut impl FnMut(&mut T, &mut Aabr<f32>) -> bool,
        reinsert_buf: &mut LeafBuf<T>,
    ) -> RetainResult {
        match self {
            Node::Internal(children) => {
                let mut recalculate_bounds = false;

                children.retain(|(child, child_aabr)| {
                    if collides(*child_aabr) {
                        match child.retain(Some(*child_aabr), collides, retain, reinsert_buf) {
                            RetainResult::Ok => true,
                            RetainResult::Deleted => {
                                recalculate_bounds = true;
                                false
                            }
                            RetainResult::ShrunkAabr(new_aabr) => {
                                *child_aabr = new_aabr;
                                recalculate_bounds = true;
                                true
                            }
                        }
                    } else {
                        true
                    }
                });

                if let Some(bounds) = bounds {
                    if children.len() < MIN {
                        for (child, _) in children.drain(..) {
                            child.collect_orphans(reinsert_buf);
                        }
                        RetainResult::Deleted
                    } else if recalculate_bounds {
                        let new_bounds = self.bounds();
                        debug_assert!(bounds.contains_aabr(new_bounds));

                        if bounds != new_bounds {
                            RetainResult::ShrunkAabr(new_bounds)
                        } else {
                            RetainResult::Ok
                        }
                    } else {
                        RetainResult::Ok
                    }
                } else {
                    RetainResult::Ok
                }
            }
            Node::Leaf(children) => {
                let mut recalculate_bounds = false;

                let mut i = 0;
                while i < children.len() {
                    let (child, child_aabr) = &mut children[i];
                    let before = *child_aabr;
                    if collides(before) {
                        if retain(child, child_aabr) {
                            let after = *child_aabr;
                            if before != after {
                                if let Some(bounds) = bounds {
                                    recalculate_bounds = true;
                                    // A child can move within a leaf node without reinsertion
                                    // as long as it does not increase the bounds of the leaf.
                                    if !bounds.contains_aabr(after) {
                                        reinsert_buf.push(children.swap_remove(i));
                                    } else {
                                        i += 1;
                                    }
                                } else {
                                    i += 1;
                                }
                            } else {
                                i += 1;
                            }
                        } else {
                            recalculate_bounds = true;
                            children.swap_remove(i);
                        }
                    } else {
                        i += 1;
                    }
                }

                if let Some(bounds) = bounds {
                    if children.len() < MIN {
                        reinsert_buf.extend(children.drain(..));
                        RetainResult::Deleted
                    } else if recalculate_bounds {
                        let new_bounds = self.bounds();
                        debug_assert!(bounds.contains_aabr(new_bounds));

                        if bounds != new_bounds {
                            RetainResult::ShrunkAabr(new_bounds)
                        } else {
                            RetainResult::Ok
                        }
                    } else {
                        RetainResult::Ok
                    }
                } else {
                    RetainResult::Ok
                }
            }
        }
    }

    fn collect_orphans(self, reinsert_buf: &mut LeafBuf<T>) {
        match self {
            Node::Internal(children) => {
                for (child, _) in children {
                    child.collect_orphans(reinsert_buf);
                }
            }
            Node::Leaf(children) => reinsert_buf.extend(children),
        }
    }

    fn depth(&self, level: usize) -> usize {
        match self {
            Node::Internal(children) => children[0].0.depth(level + 1),
            Node::Leaf(_) => level,
        }
    }

    fn visit(&self, f: &mut impl FnMut(Aabr<f32>, usize), level: usize) {
        match self {
            Node::Internal(children) => {
                for (child, child_aabr) in children {
                    child.visit(f, level + 1);
                    f(*child_aabr, level);
                }
            }
            Node::Leaf(children) => {
                for (_, child_aabr) in children {
                    f(*child_aabr, level);
                }
            }
        }
    }

    #[cfg(test)]
    fn check_invariants(
        &self,
        bounds: Option<Aabr<f32>>,
        depth: usize,
        len_counter: &mut usize,
    ) -> usize {
        let mut child_depth = None;

        match self {
            Node::Internal(children) => {
                assert!(!children.is_empty());

                if let Some(bounds) = bounds {
                    let tight_bounds = self.bounds();
                    assert!(
                        relative_eq!(tight_bounds.min, bounds.min)
                            && relative_eq!(tight_bounds.max, bounds.max),
                        "bounding rectangle for internal node is not tight"
                    );
                }

                for (child, child_aabr) in children {
                    let d = child.check_invariants(Some(*child_aabr), depth + 1, len_counter);
                    if let Some(child_depth) = &mut child_depth {
                        assert_eq!(*child_depth, d, "rtree is not balanced");
                    } else {
                        child_depth = Some(d);
                    }
                }
            }
            Node::Leaf(children) => {
                if let Some(bounds) = bounds {
                    let tight_bounds = self.bounds();
                    assert!(
                        relative_eq!(tight_bounds.min, bounds.min)
                            && relative_eq!(tight_bounds.max, bounds.max),
                        "bounding rectangle for leaf node is not tight"
                    );
                }

                *len_counter += children.len();
                child_depth = Some(depth);
            }
        }

        if let Some(bounds) = bounds {
            assert!(bounds == self.bounds());
        }

        child_depth.unwrap()
    }
}
enum InsertResult<T, const MIN: usize, const MAX: usize> {
    /// No split occurred.
    Ok,
    /// Contains the new node that was split off.
    Split(Box<Node<T, MIN, MAX>>),
}

enum RetainResult {
    /// Nothing changed.
    Ok,
    /// This node must be deleted from its parent.
    Deleted,
    /// This node was not deleted but its AABR was shrunk.
    /// Contains the new AABR.
    ShrunkAabr(Aabr<f32>),
}

fn area_insertion_heuristic<T>(data_aabr: Aabr<f32>, children: &[(T, Aabr<f32>)]) -> usize {
    debug_assert!(
        !children.is_empty(),
        "internal node must have at least one child"
    );

    let mut best = 0;
    let mut best_area_increase = f32::INFINITY;
    let mut best_aabr = Aabr::default();

    for (idx, (_, child_aabr)) in children.iter().enumerate() {
        let area_increase = area(child_aabr.union(data_aabr)) - area(*child_aabr);
        if area_increase < best_area_increase {
            best = idx;
            best_area_increase = area_increase;
            best_aabr = *child_aabr;
        } else if area_increase == best_area_increase && area(*child_aabr) < area(best_aabr) {
            best = idx;
            best_aabr = *child_aabr;
        }
    }

    best
}

/// This heuristic produces better trees than the area heuristic, but at a
/// greater insertion cost.
#[allow(unused)]
fn overlap_insertion_heuristic<T>(data_aabr: Aabr<f32>, children: &[(T, Aabr<f32>)]) -> usize {
    debug_assert!(
        !children.is_empty(),
        "internal node must have at least one child"
    );

    let mut best = 0;
    let mut best_aabr = Aabr::default();
    let mut best_overlap_value = f32::INFINITY;

    for (idx, (_, aabr)) in children.iter().enumerate() {
        let mut base_overlap = 0.0;
        let mut union_overlap = 0.0;
        for (other_idx, (_, other_aabr)) in children.iter().enumerate() {
            if other_idx != idx {
                let int = aabr.intersection(*other_aabr);
                if int.is_valid() {
                    base_overlap += area(int);
                }

                let int = aabr.union(data_aabr).intersection(*other_aabr);
                if int.is_valid() {
                    union_overlap += area(int);
                }
            }
        }

        // The increase in overlap value
        let overlap_value = union_overlap - base_overlap;
        debug_assert!(overlap_value >= 0.0);

        if overlap_value < best_overlap_value {
            best = idx;
            best_aabr = *aabr;
            best_overlap_value = overlap_value;
        } else if overlap_value == best_overlap_value {
            let area_value = area(aabr.union(data_aabr)) - area(*aabr);
            let best_area_value = area(best_aabr.union(data_aabr)) - area(best_aabr);

            if area_value < best_area_value {
                best = idx;
                best_aabr = *aabr;
            }
        }
    }

    best
}

/// Splits a node with `children` being the children of the node being split.
///
/// After returning, `children` contains half the data while the returned
/// `ArrayVec` contains the other half for the new node.
fn split_node<T, const MIN: usize, const MAX: usize>(
    split_buf: &mut Vec<(T, Aabr<f32>)>,
    children: &mut ArrayVec<(T, Aabr<f32>), MAX>,
    data: (T, Aabr<f32>),
) -> ArrayVec<(T, Aabr<f32>), MAX> {
    split_buf.extend(children.take());
    split_buf.push(data);

    let dists = MIN..MAX - MIN + 2;

    let bb = |es: &[(T, Aabr<f32>)]| es.iter().map(|e| e.1).reduce(Aabr::union).unwrap();

    let mut sum_x = 0.0;
    split_buf.sort_unstable_by_key(|e| OrderedFloat(e.1.min.x / 2.0 + e.1.max.x / 2.0));

    for split in dists.clone() {
        sum_x += perimeter(bb(&split_buf[..split])) + perimeter(bb(&split_buf[split..]));
    }

    let mut sum_y = 0.0;
    split_buf.sort_unstable_by_key(|e| OrderedFloat(e.1.min.y / 2.0 + e.1.max.y / 2.0));

    for split in dists.clone() {
        sum_y += perimeter(bb(&split_buf[..split])) + perimeter(bb(&split_buf[split..]));
    }

    // Sort by the winning axis
    split_buf.sort_unstable_by_key(|e| {
        let (min, max) = if sum_x <= sum_y {
            (e.1.min.x, e.1.max.x)
        } else {
            (e.1.min.y, e.1.max.y)
        };
        OrderedFloat(min / 2.0 + max / 2.0)
    });

    let mut best_dist = 0;
    let mut best_overlap_value = f32::INFINITY;
    let mut best_area_value = f32::INFINITY;

    for split in dists {
        let group_1 = bb(&split_buf[..split]);
        let group_2 = bb(&split_buf[split..]);
        let overlap_value = {
            let int = group_1.intersection(group_2);
            if int.is_valid() {
                area(int)
            } else {
                0.0
            }
        };
        let area_value = area(group_1) + area(group_2);

        if overlap_value < best_overlap_value {
            best_overlap_value = overlap_value;
            best_area_value = area_value;
            best_dist = split;
        } else if overlap_value == best_overlap_value && area_value < best_area_value {
            best_area_value = area_value;
            best_dist = split;
        }
    }

    debug_assert!(children.is_empty());
    debug_assert_eq!(split_buf.len(), MAX + 1);

    let mut other = ArrayVec::new();
    other.extend(split_buf.drain(best_dist..));

    children.extend(split_buf.drain(..));

    other
}

fn area(aabr: Aabr<f32>) -> f32 {
    (aabr.max - aabr.min).product()
}

fn perimeter(aabr: Aabr<f32>) -> f32 {
    (aabr.max - aabr.min).sum() * 2.0
}

#[cfg(test)]
mod tests {
    use std::f32::consts::TAU;
    use std::sync::atomic::{AtomicU64, Ordering};

    use rand::Rng;
    use vek::Vec2;

    use super::*;

    fn insert_rand<const MIN: usize, const MAX: usize>(
        rtree: &mut RTree<u64, MIN, MAX>,
    ) -> (u64, Aabr<f32>) {
        static NEXT_UNIQUE_ID: AtomicU64 = AtomicU64::new(0);

        let id = NEXT_UNIQUE_ID.fetch_add(1, Ordering::SeqCst);

        let mut rng = rand::thread_rng();

        let min = Vec2::new(rng.gen(), rng.gen());
        let max = Vec2::new(
            min.x + rng.gen_range(0.003..=0.01),
            min.y + rng.gen_range(0.003..=0.01),
        );

        let aabr = Aabr { min, max };

        rtree.insert(id, aabr);

        (id, aabr)
    }

    #[test]
    fn insert_delete_interleaved() {
        let mut rtree: RTree<u64, 4, 8> = RTree::new();

        for i in 0..5_000 {
            insert_rand(&mut rtree);
            let (id_0, aabr_0) = insert_rand(&mut rtree);

            let mut found = false;
            rtree.retain(
                |aabr| aabr.collides_with_aabr(aabr_0),
                |&mut id, _| {
                    if id == id_0 {
                        assert!(!found);
                        found = true;
                        false
                    } else {
                        true
                    }
                },
            );
            assert!(found);

            rtree.check_invariants(i + 1);
        }
    }

    #[test]
    fn node_underfill() {
        let mut rtree: RTree<u64, 4, 8> = RTree::new();

        for i in 0..5_000 {
            insert_rand(&mut rtree);
            rtree.check_invariants(i + 1);
        }

        let mut delete_count = 0;

        rtree.retain(
            |_| true,
            |_, _| {
                if rand::random() {
                    delete_count += 1;
                    false
                } else {
                    true
                }
            },
        );
        rtree.check_invariants(5_000 - delete_count);

        rtree.clear();
        rtree.check_invariants(0);
    }

    #[test]
    fn movement() {
        let mut rtree: RTree<u64, 4, 8> = RTree::new();

        for _ in 0..5_000 {
            insert_rand(&mut rtree);
        }

        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            rtree.retain(
                |_| true,
                |_, aabr| {
                    let angle = rng.gen_range(0.0..TAU);
                    let v = Vec2::new(angle.cos(), angle.sin()) * 0.003;

                    aabr.min += v;
                    aabr.max += v;
                    assert!(aabr.is_valid());

                    true
                },
            );
            rtree.check_invariants(5_000);
        }
    }
}
