https://user-images.githubusercontent.com/31678482/172040388-4710d5d3-1756-451f-b7ff-d2a8c5ae6fe0.mp4

# Spatial Index Demo

This repository contains an implementation of a parallel [Bounding Volume Hierarchy](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) in Rust for fast spatial queries such as collision detection and ray tracing. At the beginning of each frame, we rebuild the BVH from scratch in parallel rather than update the BVH to accout for any changes. This strategy allows the BVH to be robust in the face of frequently changing objects. I'm especially interested in the use of the BVH for collision detection where construction time is paramount and query performance is a secondary concern. Additionally, the implementation should be optimized for CPUs (My use case is servers without GPUs).

Also included is an in-memory [R*-Tree](https://en.wikipedia.org/wiki/R*-tree) for comparison. This can be more efficient if objects are not changing much and query performance is more of a concern. With dynamic BVHs we can surround each object in a "fattened" bounding volume. We only need to reinsert the object if it's real bounding volume moves outside the fattened bounding volume.

The implemented BVHs only support two dimensional rectangles for clarity, but generalizations should be fairly straightforward.

# Controls
* **W**: Spawn objects
* **E**: Toggle BVH rendering
* **R**: Delete all objects
* **T**: Toggle object rendering
* **Space**: Pause/Unpause time
* **Escape**: Close the program

# Running

Simply run `cargo run -r`. You'll also need SDL and SDL_ttf installed.

# References

The BVH is inspired by
* Wächter, Carsten, and Alexander Keller. "Instant ray tracing: The bounding interval hierarchy." Rendering Techniques 2006 (2006): 139-149.

While the R-Tree is based on
* Guttman, Antonin. "R-trees: A dynamic index structure for spatial searching." Proceedings of the 1984 ACM SIGMOD international conference on Management of data. 1984.
* Beckmann, Norbert, et al. "The R*-tree: An efficient and robust access method for points and rectangles." Proceedings of the 1990 ACM SIGMOD international conference on Management of data. 1990.
