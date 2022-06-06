https://user-images.githubusercontent.com/31678482/172040388-4710d5d3-1756-451f-b7ff-d2a8c5ae6fe0.mp4

# Dynamic R-Tree Demo

This repo contains an implementation of an in-memory [R-Tree](https://en.wikipedia.org/wiki/R-tree) (specifically an [R*-Tree](https://en.wikipedia.org/wiki/R*-tree)). This is used for fast spatial queries such as "Find the nearest McDonalds." or "Find all entities that collide with this bounding box." or "Find all geometry that intersects with this ray."

R-Trees and similar data structures are great for static data which does not change much after insertion. But what if our entries are constantly moving? This is especially relevant in games where entities could change their location arbitrarily. If we're not careful, the cost of updating the R-Tree every frame could become prohibitively expensive.

To account for moving objects, we have a few tricks we can use to reduce the amount of work we need to do:
* The bounding boxes of objects are "fattened" in the direction of their velocity. Objects are only adjusted in the R-Tree when their real bounding box is not completely contained within the fattened bounding box. This decreases query performance slightly, but greatly reduces the number of reinsertions that need to happen every frame.
* If an object's position changes in the R-Tree and the bounds of its leaf node are not increased, then the object does not need to be reinserted.

In this demo, objects turn red when they intersect with another object. Without the R-Tree we would have to check every object exhaustively for a collision, which is quite slow.

## Why not use a grid?

For a situation as simple as this, a grid-based spatial partition would probably work fine. However, grids don't work well when there is a large discrepancy in the size and density of objects in the scene (see the "teapot in a stadium" problem). Additionally, grids can use a lot of memory if the scene is large.

# Conclusion

With the R-Tree enabled, I was able to reach over 90,000 objects loaded simultaneously before the framerate dipped below 144 FPS on my modest machine. (This is without the overhead of rendering)

To improve performance further, the `retain` function could be parallelized or a bulk-insert routine could be created, since entries are reinserted sequentially at the end of the `retain` function.

Another potential improvement is to store all nodes in a [slab](https://docs.rs/slab/latest/slab/). This could increase cache locality and remove the overhead associated with malloc. I originally went with this approach but stopped once I ran into lifetime and borrowing issues.

This implementation only supports two dimensions for the sake of clarity. However, a generalization to higher dimensions should be fairly straightforward.

# Controls
* **W**: Spawn objects
* **E**: Toggle R-Tree rendering
* **R**: Delete all objects
* **T**: Toggle R-Tree
* **Y**: Toggle object rendering
* **Space**: Pause/Unpause time
* **Escape**: Close the program

# Running

Simply run `cargo run -r`. You'll also need SDL and SDL_ttf installed.

# References

* Guttman, Antonin. "R-trees: A dynamic index structure for spatial searching." Proceedings of the 1984 ACM SIGMOD international conference on Management of data. 1984.
* Beckmann, Norbert, et al. "The R*-tree: An efficient and robust access method for points and rectangles." Proceedings of the 1990 ACM SIGMOD international conference on Management of data. 1990.
