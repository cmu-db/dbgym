import sys

if len(sys.argv) != 2:
    print("Usage: python consume_memory.py <memory in GB>")
    sys.exit(1)

gb = int(sys.argv[1])
memory = ' ' * (gb * 1024 ** 3)
input("Allocated {} GB of memory. Press enter to exit.".format(gb))