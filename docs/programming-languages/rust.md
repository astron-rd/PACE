# Intro

Rust is a systems programming language that offers high performance combined
with a powerful high-level typing and ownership system which lets you
confidently write reliable and efficient software.

Rust is an interesting language for the PACE project because of the following
properties:

- Native compiled binaries that offer performance rivalling C and C++.
- Memory safety semantics that grant C++-like memory flexibility without using a
  garbage collector.
- Iterator semantics that make it easy to write astronomy code without getting
  lost in `for` loops.
- Trivial parallelization using said iterators.
- Memory and data-race safe parallelization if parallel iterators are not
  applicable.
- Excellent build system and library ecosystem.
- Easy FFI integration with C and C++ code.

## Parallelization

Rust contains many different concepts to organize parallelism both in the standard library and through (library) crates. Below we briefly discuss some of the most powerful concepts followed by a brief section on some current crates.

### Thread scope

Within Rust the lifetime of objects as well as exclusive ownership need to be guaranteed. This allows Rust to provide memory safety without a garbage collector. However, the semantics of moving objects into threads complicates returning that object to the caller once the thread has joined. Similarly, providing a reference to an object for a thread might result in the reference outlasting the lifetime of the object.

To address this thread scopes are introduced. Both in several crates as well as in the standard library. With these scopes the lifetime of threads is guaranteed and so objects can be used within them safely. The lifetime of all threads spawned within the scope is guaranteed to be over when the scope closes. Should the threads fail to join the scope will block indefinitely.

```Rust
fn processor(work: &mut [_]) {
    thread::scope(|s| {
        for item in work.into_iter() {
            s.spawn(move || {
                process(item);
            });
        }
    });
}
```

### Chunk

Rust has many built in functions to operate on slices that work equally well for Arrays and Vectors. The most important return an iterator of sub-slices with the last chunk being smaller if the work is not dividable. These functions are `chunks` and `chunks_mut`. Additionally methods like `chunks_exact` and `rchunks` exist.

These mechanism of work division in Rust are highly favorable to distribute the work across threads and nodes (sharding).

```Rust
fn processor(work: &mut [_]) {
    let n_workers = num_cpus::get();
    let chunks: Vec<&mut [_]> = work.chunks_mut(n_workers).collect();
    std::thread::scope(|s| {
        for chunk in chunks.into_iter() {
            s.spawn(move || {
                process(chunk);
            });
        }
    });
}
```

Work distributed in chunks can better utilize CPU L1 and L2 caches by tuning the chunk size to the CPU heuristics. Additionally, it allows to support tiling implementations of dense computations.

### Channels

Channels are FIFO message queues using tx & rx pairs with additional primitives. Beyond their basic function they allow to signal termination through the closing of the tx channels. Channels can be cloned allowing for multi-producer and or multi-producer setups. Effectively, channels combine queues with other language constructs like conditions or atomic variables in one cohesive package.

```Rust
#![feature(mpmc_channel)] // Required because std::mpmc is still in nightly.

fn main() {
    // Create a multiple-producer, multiple-consumer channel pair
    let (tx, rx) = std::sync::mpmc::channel();

    // Create a thread to put work into the channel
    std::thread::spawn(move || {
        for i in 0..15 {
            tx.send(i).expect("receiver hung up");
        }
    });

    // Create multiple worker threads to consume the work
    std::thread::scope(|s| {
        for worker in 0..5 {
            let cloned_rx = rx.clone();
            s.spawn(move || {
                while let Ok(value) = cloned_rx.recv() {
                    println!("worker {}: {}", worker, value);
                }
            });
        }
    });
}
```

### Iterators

Through Rust [extension traits](https://rust-lang.github.io/rfcs/0445-extension-trait-conventions.html) we can add additional operators to existing types. Used by [Rayon](https://docs.rs/rayon/latest/rayon/) to add parallel iterators that can be applied to slices, Arrays and Vectors. Notable new iterators include `par_iter` for element-wise iteration and `par_chunks` for chunked iteration.

```Rust
fn processor(work: &mut [_]) {
    let n_workers = num_cpus::get();
    work.par_chunks_mut(n_workers).for_each(|band| {
        process(band);
    });
}
```

### Parallelization Crates

Below is an overview of some common crates that can be used to parallelize workloads. This list is not exhaustive and will not be updated (2026-10-20).

General threading & parallelization crates:

1. [Crossbeam](https://docs.rs/crossbeam/latest/crossbeam/)
1. [Rayon](https://docs.rs/rayon/latest/rayon/)

Specialized concurrency constructs:

1. [parking_lot](https://docs.rs/parking_lot/latest/parking_lot/)
1. [spin](https://docs.rs/spin/latest/spin/)
1. [thread_local](https://docs.rs/thread_local/latest/thread_local/)
1. [ringbuf](https://docs.rs/ringbuf/latest/ringbuf/)
1. [dashmap](https://docs.rs/dashmap/latest/dashmap/)

Overview of Channel crates:

| Crate | SPSC | MPSC | SPMC | MPMC |
| ----------------------------------------------- | ---- | ---- | ---- | ----------- |
| [std::sync](https://doc.rust-lang.org/std/sync) | | X | | X (nightly) |
| [crossbeam](https://crates.io/crates/crossbeam) | | | | X |
| [Flume](https://crates.io/crates/flume) | | X | | X |
| [Kanal](https://crates.io/crates/kanal) | | | | X |
| [oneshot](https://crates.io/crates/oneshot) | X | | | |
| [crossfire](https://crates.io/crates/crossfire) | X | X | | X |
| [bus](https://crates.io/crates/bus) | | X | | |
