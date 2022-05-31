// TODO: improve insertion heuristic.
// TODO: run clippy

use std::mem;

use approx::relative_eq;
use arrayvec::ArrayVec;
use ordered_float::OrderedFloat;
use vek::Aabr;

pub struct RTree<T, const M: usize> {
    root: Node<T, M>,
    // The bufs are put here to reuse their allocations.
    internal_split_buf: InternalBuf<T, M>,
    leaf_split_buf: LeafBuf<T>,
    reinsert_buf: LeafBuf<T>,
}

type InternalBuf<T, const M: usize> = Vec<(Box<Node<T, M>>, Aabr<f32>)>;
type LeafBuf<T> = Vec<(T, Aabr<f32>)>;

enum Node<T, const M: usize> {
    Internal(ArrayVec<(Box<Node<T, M>>, Aabr<f32>), M>),
    Leaf(ArrayVec<(T, Aabr<f32>), M>),
}

impl<T, const M: usize> Node<T, M> {
    fn is_internal(&self) -> bool {
        match self {
            Node::Internal(_) => true,
            Node::Leaf(_) => false,
        }
    }

    fn is_leaf(&self) -> bool {
        !self.is_internal()
    }

    fn bounds(&self) -> Aabr<f32> {
        match self {
            Node::Internal(children) => children
                .iter()
                .map(|(_, aabr)| *aabr)
                .reduce(|l, r| l.union(r))
                .unwrap(),
            Node::Leaf(children) => children
                .iter()
                .map(|(_, aabr)| *aabr)
                .reduce(|l, r| l.union(r))
                .unwrap(),
        }
    }

    fn insert(
        &mut self,
        data: T,
        data_aabr: Aabr<f32>,
        internal_split_buf: &mut InternalBuf<T, M>,
        leaf_split_buf: &mut LeafBuf<T>,
    ) -> InsertResult<T, M> {
        match self {
            Self::Internal(children) => {
                let children_is_full = children.is_full();

                let (best_child, best_child_aabr) = children
                    .iter_mut()
                    .min_by_key(|(_, child_aabr)| {
                        OrderedFloat(area(child_aabr.union(data_aabr)) - area(*child_aabr));
                    })
                    .expect("internal node must have at least one child");

                match best_child.insert(data, data_aabr, internal_split_buf, leaf_split_buf) {
                    InsertResult::Ok => {
                        best_child_aabr.expand_to_contain(data_aabr);
                        InsertResult::Ok
                    }
                    InsertResult::Split(new_node) => {
                        let new_node_aabr = new_node.bounds();
                        *best_child_aabr = best_child.bounds();

                        if children_is_full {
                            let other = split_node(
                                internal_split_buf,
                                children,
                                (new_node, new_node_aabr),
                                |(_, child_aabr)| *child_aabr,
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
                    let other = split_node(
                        leaf_split_buf,
                        children,
                        (data, data_aabr),
                        |(_, data_aabr)| *data_aabr,
                    );
                    debug_assert!(other.len() >= M / 2);

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
        callback: &mut impl FnMut(&T, Aabr<f32>),
    ) {
        match self {
            Node::Internal(children) => {
                for child in children {
                    if collides(child.1) {
                        child.0.query(collides, callback);
                    }
                }
            }
            Node::Leaf(children) => {
                for (child, child_aabr) in children {
                    if collides(*child_aabr) {
                        callback(child, *child_aabr);
                    }
                }
            }
        }
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
                    if children.len() < M / 2 {
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
                    if children.len() < M / 2 {
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

    fn visit(&self, f: &mut impl FnMut(Aabr<f32>, usize), aabr: Option<Aabr<f32>>, level: usize) {
        if let Some(aabr) = aabr {
            f(aabr, level);
        }

        match self {
            Node::Internal(children) => {
                for (child, child_aabr) in children {
                    child.visit(f, Some(*child_aabr), level + 1);
                }
            }
            Node::Leaf(children) => {
                for (_, child_aabr) in children {
                    f(*child_aabr, level + 1);
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

    fn dbg_print(&self) {
        match self {
            Node::Internal(children) => eprintln!("Internal, #children = {}", children.len()),
            Node::Leaf(children) => eprintln!("Leaf, #children = {}", children.len()),
        }
    }
}

impl<T, const M: usize> RTree<T, M> {
    pub fn new() -> Self {
        Self {
            root: Node::Leaf(ArrayVec::new()),
            internal_split_buf: Vec::new(),
            leaf_split_buf: Vec::new(),
            reinsert_buf: Vec::new(),
        }
    }

    pub fn insert(&mut self, data: T, data_aabr: Aabr<f32>) {
        assert!(M >= 2, "bad max node capacity");

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

        let mut reinsert_buf = mem::replace(&mut self.reinsert_buf, Vec::new());

        for (data, data_aabr) in reinsert_buf.drain(..) {
            self.insert(data, data_aabr);
        }

        debug_assert!(self.reinsert_buf.capacity() == 0);
        self.reinsert_buf = reinsert_buf;

        // Don't waste too much memory after a large restructuring.
        self.reinsert_buf.shrink_to(M * 2);
    }

    pub fn query(
        &self,
        mut collides: impl FnMut(Aabr<f32>) -> bool,
        mut callback: impl FnMut(&T, Aabr<f32>),
    ) {
        self.root.query(&mut collides, &mut callback)
    }

    pub fn clear(&mut self) {
        self.root = Node::Leaf(ArrayVec::new());
    }

    /// For the purposes of rendering the RTree.
    pub fn visit(&self, mut f: impl FnMut(Aabr<f32>, usize)) {
        self.root.visit(&mut f, None, 0);
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

enum InsertResult<T, const M: usize> {
    /// No split occurred.
    Ok,
    /// Contains the new node that was split off.
    Split(Box<Node<T, M>>),
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

/// Splits a node with `children` being the children of the node being split.
///
/// After returning, `children` contains half the data while the returned
/// `ArrayVec` contains the other half for the new node.
fn split_node<T, const M: usize>(
    split_buf: &mut Vec<T>,
    children: &mut ArrayVec<T, M>,
    data: T,
    get_aabr: impl Fn(&T) -> Aabr<f32>,
) -> ArrayVec<T, M> {
    split_buf.extend(children.take());
    split_buf.push(data);

    let groups = (M / 2)..=(M / 2) + M % 2 + 1;

    let bb = |es: &[T]| es.iter().map(&get_aabr).reduce(|l, r| l.union(r)).unwrap();

    let mut sum_x = 0.0;
    split_buf.sort_unstable_by_key(|e| {
        let aabr = get_aabr(e);
        OrderedFloat(aabr.min.x / 2.0 + aabr.max.x / 2.0)
    });

    for cnt in groups.clone() {
        sum_x += perimeter(bb(&split_buf[..cnt])) + perimeter(bb(&split_buf[cnt..]));
    }

    let mut sum_y = 0.0;
    split_buf.sort_unstable_by_key(|e| {
        let aabr = get_aabr(e);
        OrderedFloat(aabr.min.y / 2.0 + aabr.max.y / 2.0)
    });

    for cnt in groups.clone() {
        sum_y += perimeter(bb(&split_buf[..cnt])) + perimeter(bb(&split_buf[cnt..]));
    }

    // Sort by the winning axis
    split_buf.sort_unstable_by_key(|e| {
        let aabr = get_aabr(e);
        let (min, max) = if sum_x <= sum_y {
            (aabr.min.x, aabr.max.x)
        } else {
            (aabr.min.y, aabr.max.y)
        };
        OrderedFloat(min / 2.0 + max / 2.0)
    });

    let mut best_dist = 0;
    let mut best_overlap_value = f32::INFINITY;

    for cnt in groups.clone() {
        let group_1 = bb(&split_buf[..cnt]);
        let group_2 = bb(&split_buf[cnt..]);
        let overlap_value = area(group_1.intersection(group_2));

        if relative_eq!(overlap_value, best_overlap_value) {
            let area_value = area(group_1) + area(group_2);
            let best_area_value =
                area(bb(&split_buf[..best_dist])) + area(bb(&split_buf[best_dist..]));

            if area_value < best_area_value {
                best_dist = cnt;
            }
        } else if overlap_value < best_overlap_value {
            best_overlap_value = overlap_value;
            best_dist = cnt;
        }
    }

    debug_assert!(children.is_empty());
    debug_assert_eq!(split_buf.len(), M + 1);

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

    fn insert_rand<const M: usize>(rtree: &mut RTree<u64, M>) -> (u64, Aabr<f32>) {
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
        let mut rtree: RTree<u64, 8> = RTree::new();

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
        let mut rtree: RTree<u64, 8> = RTree::new();

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
        let mut rtree: RTree<u64, 8> = RTree::new();

        for _ in 0..10_000 {
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
            rtree.check_invariants(10_000);
        }
    }
}
