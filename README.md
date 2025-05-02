# GPU Algorithms Examples
Normally I would do this in C++. I stumbled across OpenAI Triton on the GPU MODE discord. 
Curious about it, I thought this would be a good time to try it out.

The Triton compiler does automatic optimizations that usually a CUDA programmer would do by hand. So I decided to write these algorithms in Triton to learn more about it. I will explore triton and test the runtime of the parallel algorithms with their single-threaded counterpart on the CPU. 

## Algorithms
I am simplifying the runtimes for readability, in reality they are much more complex. For further reading check out this paper for [parallel algorithms](https://www.cs.cmu.edu/~guyb/papers/BM04.pdf)
1. Finding the End of a Linked List 
    - Runtime: $O(log(n))$ ([paper](https://rsim.cs.illinois.edu/arch/qual_papers/systems/3.pdf))
    - Results: [End Of Linked List](End_of_Linked_List.md)
    - Run: `python src/End_of_Linked_List.py`
2. Vector sum (e.g. `sum([1,2,3,4,5]) = 15`)
    - Runtime: $O(log(n))$

## Dependencies
- pytorch
  - Quick install with `pip install pytorch`
- triton
  - Quick install with `pip install triton`

source code is located in `src/[algorithm].py`