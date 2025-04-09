from typing import Tuple # for < python 3.9 
from torch import Tensor, randperm, empty, long

# such that the start points to somewhere, no circular dependencies, 
# all nodes are reached, and the last node is the end
def GenerateLinkedList(size: int, cudaTensor: bool) -> Tuple[Tensor, int]:
  # Generate a random permutation of node indices
  perm = randperm(size)
  
  # Allocate a tensor to represent the linked list
  device = 'cuda' if cudaTensor else 'cpu'
  linked_list = empty(size, dtype=long, device=device)

  # For each node in the permutation, link it to the next
  # The last one will point to -1 (end of list)
  # NOTE: perm[0] is the head
  for i in range(size - 1):
    linked_list[perm[i]] = perm[i + 1]
  
  linked_list[perm[-1]] = -1  # end of list

  return linked_list, perm[0].item()
