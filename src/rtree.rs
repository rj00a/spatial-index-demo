use std::mem;

use approx::relative_eq;
use arrayvec::ArrayVec;
use ordered_float::OrderedFloat;
use vek::{Aabr, Vec2};

pub struct RTree<T, const M: usize> {
    root: Node<T, M>,
    len: usize,
    // The bufs are put here to reuse their allocations.
    internal_split_buf: InternalSplitBuf<T, M>,
    leaf_split_buf: Vec<T>,
    reinsert_buf: Vec<T>,
}

type InternalSplitBuf<T, const M: usize> = Vec<(Box<Node<T, M>>, Aabr<f32>)>;

enum Node<T, const M: usize> {
    Internal(ArrayVec<(Box<Node<T, M>>, Aabr<f32>), M>),
    Leaf(ArrayVec<T, M>),
}

impl<T: GetAabr, const M: usize> Node<T, M> {
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
                .map(|t| t.get_aabr())
                .reduce(|l, r| l.union(r))
                .unwrap(),
        }
    }

    fn insert(
        &mut self,
        data: T,
        internal_split_buf: &mut InternalSplitBuf<T, M>,
        leaf_split_buf: &mut Vec<T>,
    ) -> InsertResult<T, M> {
        match self {
            Self::Internal(children) => {
                let data_aabr = data.get_aabr();
                let data_aabr_area = area(data_aabr);

                let children_is_full = children.is_full();

                let (best_child, best_child_aabr) = children
                    .iter_mut()
                    .min_by_key(|(_, child_aabr)| {
                        OrderedFloat(area(child_aabr.union(data_aabr)) - data_aabr_area)
                    })
                    .expect("must have at least one child");

                match best_child.insert(data, internal_split_buf, leaf_split_buf) {
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
                                |e| e.1,
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
                    let other = split_node(leaf_split_buf, children, data, |e| e.get_aabr());
                    InsertResult::Split(Box::new(Node::Leaf(other)))
                } else {
                    children.push(data);
                    InsertResult::Ok
                }
            }
        }
    }

    fn query(&self, collides: &mut impl FnMut(Aabr<f32>) -> bool, callback: &mut impl FnMut(&T)) {
        match self {
            Node::Internal(children) => {
                for child in children {
                    if collides(child.1) {
                        child.0.query(collides, callback);
                    }
                }
            }
            Node::Leaf(children) => {
                for child in children {
                    if collides(child.get_aabr()) {
                        callback(child);
                    }
                }
            }
        }
    }

    /// Returns true if the current node should be removed from its parent.
    fn retain(
        &mut self,
        bounds: Aabr<f32>,
        collides: &mut impl FnMut(Aabr<f32>) -> bool,
        retain: &mut impl FnMut(&mut T) -> bool,
        reinsert_buf: &mut Vec<T>,
    ) -> RetainResult {
        match self {
            Node::Internal(children) => {
                debug_assert!(children.len() >= M / 2);
                let mut recalculate_bounds = false;

                children.retain(|(child, child_aabr)| {
                    if collides(*child_aabr) {
                        match child.retain(*child_aabr, collides, retain, reinsert_buf) {
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
            }
            Node::Leaf(children) => {
                debug_assert!(children.len() >= M / 2);
                let mut recalculate_bounds = false;

                let mut i = 0;
                while i < children.len() {
                    let child = &mut children[i];
                    let before = child.get_aabr();
                    if collides(before) {
                        if retain(child) {
                            let after = child.get_aabr();
                            if before != after {
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
                            recalculate_bounds = true;
                            children.swap_remove(i);
                        }
                    } else {
                        i += 1;
                    }
                }

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
            }
        }
    }

    fn collect_orphans(self, reinsert_buf: &mut Vec<T>) {
        match self {
            Node::Internal(children) => {
                for (child, _) in children {
                    child.collect_orphans(reinsert_buf);
                }
            }
            Node::Leaf(children) => {
                reinsert_buf.extend(children);
            }
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
                for child in children {
                    f(child.get_aabr(), level + 1);
                }
            }
        }
    }
}

pub trait GetAabr {
    fn get_aabr(&self) -> Aabr<f32>;
}

impl<T: GetAabr, const M: usize> RTree<T, M> {
    pub fn new() -> Self {
        Self {
            root: Node::Leaf(ArrayVec::new()),
            len: 0,
            internal_split_buf: Vec::new(),
            leaf_split_buf: Vec::new(),
            reinsert_buf: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn insert(&mut self, data: T) {
        assert!(M >= 2, "bad max node capacity");

        if let InsertResult::Split(new_node) =
            self.root
                .insert(data, &mut self.internal_split_buf, &mut self.leaf_split_buf)
        {
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

        self.len += 1;
    }

    pub fn retain(
        &mut self,
        mut collides: impl FnMut(Aabr<f32>) -> bool,
        mut retain: impl FnMut(&mut T) -> bool,
    ) {
        let root_bounds = Aabr {
            min: -Vec2::broadcast(f32::INFINITY),
            max: Vec2::broadcast(f32::INFINITY),
        };

        self.root.retain(
            root_bounds,
            &mut collides,
            &mut retain,
            &mut self.reinsert_buf,
        );

        if let Node::Internal(children) = &mut self.root {
            if children.len() == 1 {
                let new_root = *children.drain(..).next().unwrap().0;
                self.root = new_root;
            }
        }

        let mut reinsert_buf = mem::replace(&mut self.reinsert_buf, Vec::new());

        for data in reinsert_buf.drain(..) {
            self.insert(data);
        }

        self.reinsert_buf = reinsert_buf;

        // TODO: set len
    }

    pub fn query(&self, mut collides: impl FnMut(Aabr<f32>) -> bool, mut callback: impl FnMut(&T)) {
        self.root.query(&mut collides, &mut callback)
    }

    /// For the purposes of rendering the RTree.
    pub fn visit(&self, mut f: impl FnMut(Aabr<f32>, usize)) {
        self.root.visit(&mut f, None, 0);
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
