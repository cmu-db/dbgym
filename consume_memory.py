import sys
import mmap
import os

if len(sys.argv) != 2:
    print("Usage: python consume_memory.py <memory in GB>")
    sys.exit(1)

gb = int(sys.argv[1])
# Calculate the number of bytes in the specified number of gigabytes
size = gb * 1024**3

# Use mmap to allocate memory
# Note: mmap.mmap(-1, size) creates an anonymous map, so it's not backed by a file but by swap space.
with mmap.mmap(-1, size) as m:
    # Touch the memory to ensure it's actually allocated
    # We do this by writing an int in the range 0-255 to each page interval
    for i in range(0, size, mmap.PAGESIZE):
        m[i] = 0  # Correctly use an integer for assignment
    
    print(f"Allocated {gb} GB of memory. Press enter to exit.")
    input()
