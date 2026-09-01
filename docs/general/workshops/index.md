# PACE progress workshop report

**Location:** Researchable HQ, Groningen

**Date:** April 23, 2026

## Executive Summary

The PACE project (Post-Correlation Acceleration for Astronomical Data Processing Efficiency) convened an in-person workshop to share progress on the project.

The workshop combined technical presentations with collaborative discussion and planning. It was a strong opportunity to review progress, exchange ideas, agree on next steps, and meet other Researchable colleagues over coffee and a pleasant lunch.

## Presentations

### Introduction to ASTRON and the PACE project

**Bram Veenboer** presented an overview of ASTRON and the PACE project.

Highlights:
- Introduction to ASTRON’s mission and research priorities.
- Framing PACE as a cross-disciplinary effort to improve astronomical data processing efficiency.
- Connecting the project roadmap to ASTRON's requirements and the team’s collaborative goals.

### CI/CD and Continuous Benchmarking

**Bastiaan Haaksema** presented the integration of automated benchmarking into the development pipeline.

Highlights:
- Use of the Bencher platform to track performance metrics over time.
- Monitoring latency and regression detection across implementations in Rust and Python.
- Ensuring performance changes are visible and actionable as the codebase evolves.

### Fourier-Domain Dedispersion in C++

**Mick Veldhuis** discussed efforts to accelerate pulsar signal processing.

Highlights:
- Moving dedispersion into the frequency domain to shift from memory-bandwidth limits to compute-bound workloads.
- Leveraging OpenMP for multi-core parallelization and future accelerator offloading.
- Using xtensor to provide NumPy-style multi-dimensional array syntax and simpler I/O in C++.

### Image-Domain Gridding (IDG) in Rust

**Vivian Huzen** presented a Rust implementation of the image-domain gridder.

Highlights:
- Achieved approximately 2x speedup over the reference implementation.
- Demonstrated the benefits of Rust’s memory safety and high-level abstractions.
- Employed `ndarray` for array operations and `rayon` for parallel work-stealing iterators.

### All-Sky Imaging in Python (Jax)

**Corne Lukken** demonstrated the power of Jax for high-performance astronomical imaging.

Highlights:
- Retained standard NumPy-style syntax while gaining JIT acceleration.
- Observed up to a 2800x speedup on a 7900 XTX GPU compared with simple Python baselines.
- Showed dramatic energy efficiency improvements.

[Download slides](https://nextcloud.dantalion.nl/s/dTHeJ2t5JgmXKeR/download)

## Discussion topics

The workshop included focused discussion sessions on several technical areas.

### Data formats

The group evaluated formats suitable for inter-language exchange and high-performance I/O:
- Parquet
- Safetensors
- HDF5
- Apache Avro

### Multi-node scaling

Participants explored approaches for scaling beyond a single machine:
- Frameworks: Jax, Dask, and Spark
- Algorithmic approaches: lockless and non-blocking algorithms
- Architectural paradigms: Dataflow versus task-based pipeline design

### GPU offloading

The team discussed practical experience with GPU-accelerated workloads and the broader implications for performance and energy efficiency.

## Workshop reflections

The day was a successful opportunity for the team to come together in person:
- connecting with colleagues,
- discussing progress and new ideas,
- aligning on next steps,
- strengthening collaboration across ASTRON and Researchable.

Meeting other Researchable people over coffee and lunch added value beyond the technical sessions, making the workshop both productive and enjoyable.

## Conclusion

Overall, the workshop was a successful event. It reinforced the project’s momentum, clarified priorities for the next phase, and fostered stronger team cohesion. The team left with a shared roadmap for the next phase and clear priorities for the coming months.
