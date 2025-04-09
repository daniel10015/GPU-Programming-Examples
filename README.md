# GPU Algorithms Examples
Normally I would do this in C++. I stumbled across OpenAI Triton on the GPU-Mode discord. 
Curious about it, I thought this would be a good time to try it out.

The Triton compiler does automatic optimizations that usually a CUDA programmer would do by hand. So I decided to learn a bit of it writing the following algorithms using Triton:
## Algorithms
1. Finding the End of a Linked List 
    - Runtime: O(log(n)) ([paper](https://rsim.cs.illinois.edu/arch/qual_papers/systems/3.pdf))

## Dependencies
- pytorch
  - Quick install with `pip install pytorch`
- triton
  - Quick install with `pip install triton`

## Results
I ran these on an NVIDIA GTX 1650

- Finding the End of a Linked List:

  Here are a couple plots I took of the same run. I tried iterating through in one pass but it became highly ineffient compared to one-pass, so this is all there is. `Custom` is the implementation on the CPU with tensors, and the number in `Triton <number>` represents the block_size.
  ![Finding the End of a Linked List results](images/benchmarks/fast_eof_ll/Finding%20the%20End%20of%20a%20Linked%20List%20Performance(1).png)
  ![Finding the End of a Linked List results](images/benchmarks/fast_eof_ll/Finding%20the%20End%20of%20a%20Linked%20List%20Performance(2).png)

### CPU Behavior
  The CPU impl GB/s would plateau if the links were next to each other, but to simulate a "real" linked list I used a random order. 

  I used `torch.randperm(n)` to generate the linked list (LL), which results in a random permutation of $n$ integers. Each element in the LL is stored as an `int32`, requiring 4 bytes. On most CPUs, the cache line size is 64 bytes, meaning up to 16 int32s can be stored per cache line. 
  
  As $n$ increases beyond 128, spatial locality begins to degrade in the CPU implementation due to the random memory access pattern, each pointer jump in the list likely lands on a new cache line. In the best case, there is 1 miss per cache line.
  
  Assuming 64 byte cache lines the approximate cache misses scale linearly with data size: $2^{k}$, where $k=max(0, \beta-6)$ such that $\beta=log_{2}(4n)$ (note that $k$ represents the number of cache lines). While the exponential is easier to compute, here is it rewritten in terms of $n$ instead of the number of cache lines, $\lceil\frac{n}{16}\rceil$. At n=128 the cache misses would be about 8 at the best case, and at n=256 the approximate best case is 16. This continues to double. 
  
  As the working set exceeds the capacity of L1 and L2, etc. caches, the hardware must evict older lines to bring in new ones. In the worst case, this leads to a cache thrashing scenario where each new access causes an eviction, and future accesses revisit evicted data. This creates a "round-robin" effect where the cache cannot retain even recently used lines, amplifying the temporal locality problem. For example, with 64-byte lines and 4-byte nodes, the cache could cycle through roughly 15 distinct cache lines before revisiting the first — and if the working set is large enough, that line may already have been evicted. 

  ### GPU Behavior
  On the GPU, the SIMT execution model used with the algorithm makes it so at the best case is the number of iterations is $log_{2}n$, assuming all threads operate in parallel. In reality, as $n$ gets larger, the runtime scales *slightly* linearly, roughly $\lceil\frac{n}{threads}\rceil log_{2}n$. However, in practice the coefficient on the linear term is small because the number of concurrently running threads is often very high.