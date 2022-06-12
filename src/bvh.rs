use std::mem;

use vek::Aabr;

#[derive(Clone)]
pub struct Bvh<T> {
    internal_nodes: Vec<InternalNode>,
    leaf_nodes: Vec<LeafNode<T>>,
    root: NodeIdx,
}

#[derive(Clone)]
struct InternalNode {
    bb: Aabr<f32>,
    left: NodeIdx,
    right: NodeIdx,
}

#[derive(Clone)]
struct LeafNode<T> {
    bb: Aabr<f32>,
    id: T,
}

type NodeIdx = u32;

impl<T: Send> Bvh<T> {
    pub fn new() -> Self {
        Self {
            internal_nodes: Vec::new(),
            leaf_nodes: Vec::new(),
            root: NodeIdx::MAX,
        }
    }

    pub fn build(&mut self, leaves: impl IntoIterator<Item = (T, Aabr<f32>)>) {
        self.leaf_nodes.clear();
        self.internal_nodes.clear();

        self.leaf_nodes
            .extend(leaves.into_iter().map(|(id, bb)| LeafNode { bb, id }));

        let leaf_count = self.leaf_nodes.len();

        if leaf_count == 0 {
            return;
        }

        self.internal_nodes.reserve_exact(leaf_count - 1);
        self.internal_nodes.resize(
            leaf_count - 1,
            InternalNode {
                bb: Aabr::default(),
                left: NodeIdx::MAX,
                right: NodeIdx::MAX,
            },
        );

        if NodeIdx::try_from(leaf_count)
            .ok()
            .and_then(|count| count.checked_add(count - 1))
            .is_none()
        {
            panic!("too many elements in BVH");
        }

        self.root = build_rec(
            0,
            bb(&self.leaf_nodes),
            &mut self.internal_nodes,
            &mut self.leaf_nodes,
            leaf_count as NodeIdx,
        );

        debug_assert_eq!(self.internal_nodes.len(), self.leaf_nodes.len() - 1);
    }

    pub fn find<C, F, U>(&self, mut collides: C, mut find: F) -> Option<U>
    where
        C: FnMut(Aabr<f32>) -> bool,
        F: FnMut(&T, Aabr<f32>) -> Option<U>,
    {
        if !self.leaf_nodes.is_empty() {
            self.find_rec(self.root, &mut collides, &mut find)
        } else {
            None
        }
    }

    fn find_rec<C, F, U>(&self, idx: NodeIdx, collides: &mut C, find: &mut F) -> Option<U>
    where
        C: FnMut(Aabr<f32>) -> bool,
        F: FnMut(&T, Aabr<f32>) -> Option<U>,
    {
        if idx < self.internal_nodes.len() as NodeIdx {
            let internal = &self.internal_nodes[idx as usize];

            if collides(internal.bb) {
                if let Some(found) = self.find_rec(internal.left, collides, find) {
                    return Some(found);
                }

                if let Some(found) = self.find_rec(internal.right, collides, find) {
                    return Some(found);
                }
            }
        } else {
            let leaf = &self.leaf_nodes[(idx - self.internal_nodes.len() as NodeIdx) as usize];

            if collides(leaf.bb) {
                return find(&leaf.id, leaf.bb);
            }
        }

        None
    }

    pub fn visit(&self, mut f: impl FnMut(Aabr<f32>, usize)) {
        if !self.leaf_nodes.is_empty() {
            self.visit_rec(self.root, 0, &mut f);
        }
    }

    pub fn visit_rec(&self, idx: NodeIdx, depth: usize, f: &mut impl FnMut(Aabr<f32>, usize)) {
        if idx >= self.internal_nodes.len() as NodeIdx {
            let leaf = &self.leaf_nodes[(idx - self.internal_nodes.len() as NodeIdx) as usize];
            f(leaf.bb, depth);
        } else {
            let internal = &self.internal_nodes[idx as usize];

            self.visit_rec(internal.left, depth + 1, f);
            self.visit_rec(internal.right, depth + 1, f);

            f(internal.bb, depth);
        }
    }
}

fn build_rec<T: Send>(
    idx: NodeIdx,
    bounds: Aabr<f32>,
    internal_nodes: &mut [InternalNode],
    leaf_nodes: &mut [LeafNode<T>],
    total_leaf_count: NodeIdx,
) -> NodeIdx {
    debug_assert_eq!(leaf_nodes.len() - 1, internal_nodes.len());

    if leaf_nodes.len() == 1 {
        // Leaf node
        return total_leaf_count - 1 + idx;
    }

    debug_assert!(bounds.is_valid());
    let dims = bounds.max - bounds.min;

    let mut split = if dims.x >= dims.y {
        let mid = bounds.min.x / 2.0 + bounds.max.x / 2.0;
        partition(leaf_nodes, |l| l.bb.min.x / 2.0 + l.bb.max.x / 2.0 <= mid)
    } else {
        let mid = bounds.min.y / 2.0 + bounds.max.y / 2.0;
        partition(leaf_nodes, |l| l.bb.min.y / 2.0 + l.bb.max.y / 2.0 <= mid)
    };

    // Handle edge cases with the partitioning.
    if split == 0 {
        split += 1;
    } else if split == leaf_nodes.len() {
        split -= 1;
    }

    let (leaves_left, leaves_right) = leaf_nodes.split_at_mut(split);

    let (internal_left, internal_right) = internal_nodes.split_at_mut(split);
    let (internal, internal_left) = internal_left.split_last_mut().unwrap();

    let (left, right) = rayon::join(
        || {
            build_rec(
                idx,
                bb(leaves_left),
                internal_left,
                leaves_left,
                total_leaf_count,
            )
        },
        || {
            build_rec(
                idx + split as NodeIdx,
                bb(leaves_right),
                internal_right,
                leaves_right,
                total_leaf_count,
            )
        },
    );

    internal.bb = bounds;
    internal.left = left;
    internal.right = right;

    idx + split as NodeIdx - 1
}

fn partition<T>(s: &mut [T], mut pred: impl FnMut(&T) -> bool) -> usize {
    let mut it = s.iter_mut();
    let mut true_count = 0;

    while let Some(head) = it.find(|x| {
        if pred(x) {
            true_count += 1;
            false
        } else {
            true
        }
    }) {
        if let Some(tail) = it.rfind(|x| pred(x)) {
            mem::swap(head, tail);
            true_count += 1;
        } else {
            break;
        }
    }
    true_count
}

fn bb<T>(leaves: &[LeafNode<T>]) -> Aabr<f32> {
    leaves
        .iter()
        .map(|l| l.bb)
        .reduce(Aabr::union)
        .expect("leaf slice is empty")
}

impl<T: Send> Default for Bvh<T> {
    fn default() -> Self {
        Self::new()
    }
}