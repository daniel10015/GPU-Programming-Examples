# GPU Algorithms Examples
Normally I would do this in C++. I stumbled across OpenAI Triton on the GPU-Mode discord. 
Curious about it, I thought this would be a good time to try it out.

The Triton compiler does automatic optimizations that usually a CUDA programmer would do by hand. So I decided to write these algorithms in Triton to learn more about it. I will not test Triton, but instead test the runtime of the parallel algorithms with their single-threaded counterpart on the CPU. 

## Algorithms
1. Finding the End of a Linked List 
    - Runtime: O(log(n)) ([paper](https://rsim.cs.illinois.edu/arch/qual_papers/systems/3.pdf))

## Dependencies
- pytorch
  - Quick install with `pip install pytorch`
- triton
  - Quick install with `pip install triton`

## Running the tests
Run the script from the project root with: 

`python src/main.py`

## Results
I ran these on an NVIDIA GTX 1650

### Finding the End of a Linked List:

  Here are a couple plots I took of the same run. I tried iterating through in one pass but it became highly ineffient compared to one-pass, so this is all there is. `Custom` is the implementation on the CPU with tensors, and the number in `Triton <number>` represents the block_size.
  ![Finding the End of a Linked List results](images/benchmarks/fast_eof_ll/Finding%20the%20End%20of%20a%20Linked%20List%20Performance(1).png)
  ![Finding the End of a Linked List results](images/benchmarks/fast_eof_ll/Finding%20the%20End%20of%20a%20Linked%20List%20Performance(2).png)

  To simulate a more realistic linked list structure, for benchmarks I randomized the node order using `torch.randperm(n)`. This removes spatial locality and forces the CPU to load scattered memory addresses. If the links were adjacent, the CPU throughput in GB/s would plateau, but here, randomness results in significantly worse performance.

### CPU Behavior
  Each node in the linked list is an `int32` (4 bytes). Most CPUs have 64-byte cache lines, which means 16 such elements can fit in one line.

  As $n$ increases beyond 128, spatial locality degrades due to the randomized access pattern. Each pointer jump likely lands on a different cache line. In the best case, there’s one miss per cache line.

  Assuming 64-byte cache lines, the number of cache misses scales linearly with the size of the list. You can express this as:

  $2^{k}$, where $k=max(0, \beta-6)$ and $\beta=log_{2}(4n)$

  This expression is equivalent to:

  $\lceil\frac{n}{16}\rceil$
  
  For example:

  - $n=128$, there are ~8 cache lines accessed

  - $n=256$, there are ~16 cache lines accessed
  
  As the working set exceeds the capacity of L1, L2, etc., the CPU must evict older lines to load new ones. This leads to cache thrashing&mdash;when each access evicts data that may be needed again soon. This creates a "round-robin" effect where the cache cycles through lines too quickly to reuse them efficiently. With 64-byte lines and 4-byte nodes, the CPU may cycle through ~15 cache lines before revisiting a previously evicted one.

  ### GPU Behavior
  On the GPU, the SIMT execution model used with the algorithm qis advantageous because of the number of threads available. 
  
  In the ideal case, the number of iterations to reach the end of the list is:

  $log_{2}n$ 
  
  This assumes all threads operate independently and in parallel. In practice, as $n$ gets larger, the runtime scales *slightly* linearly to about:
  
  $\lceil\frac{n}{threads}\rceil log_{2}n$ 
   
  However, in practice the coefficient on the linear term is small because the number of concurrently running threads is often very high.