import re
import ast
import random
import numpy as np  # Advanced Lib 1
import matplotlib.pyplot as plt # Advanced Lib 2
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Requirement: Generator Function
def color_generator():
    """
    A generator that yields a random RGB color tuple.
    """
    while True:
        yield (random.random(), random.random(), random.random())

class Item:
    """
    Represents a physical 3D item with packing constraints.
    """
    def __init__(self, item_id, length, width, height, weight, bin_reqs=None):
        self.name = f"Item_{item_id}"
        # Store dimensions as [Length(x), Width(y), Height(z)]
        self.dims = np.array([length, width, height]) 
        self.weight = weight
        self.bin_reqs = bin_reqs if bin_reqs is not None else [] # List of allowed bin indices
        self.position = None # (x, y, z) coordinates inside the bin

    @property
    def volume(self):
        return np.prod(self.dims)

    # Requirement: Operator Overloading
    def __lt__(self, other):
        """Allows sorting items by volume (Smallest to Largest)."""
        return self.volume < other.volume

    # Requirement: __str__
    def __str__(self):
        return f"Item({self.name}, {self.dims[0]}x{self.dims[1]}x{self.dims[2]})"

class Bin:
    """
    Represents the container.
    Relationship: Composition (Bin HAS Items).
    """
    def __init__(self, bin_id, length, width, height):
        self.bin_id = bin_id
        self.dims = np.array([length, width, height])
        self.items = [] # Mutable list to hold Item objects
        self.capacity_vol = np.prod(self.dims)

    def add_item(self, item):
        self.items.append(item)

    def get_utilization(self):
        if not self.items:
            return 0.0
        # Requirement: Map and Lambda
        item_volumes = list(map(lambda x: x.volume, self.items))
        total_used = np.sum(item_volumes)
        return (total_used / self.capacity_vol) * 100

def parse_custom_format(filename):
    """
    Parses the specific 'items.csv' format containing metadata and fixed-width tables.
    Requirement: Data I/O and Exception Handling.
    """
    bins = []
    items = []
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        # 1. Parse Metadata for Bins
        # Looking for line: # Bin dimensions (L * W * H): [(800,800,800),...]
        for line in lines:
            if line.startswith("# Bin dimensions"):
                # Extract the list part: [(800,800,800),...]
                match = re.search(r':\s*(\[.*\])', line)
                if match:
                    dims_list_str = match.group(1)
                    # Use AST to safely evaluate the string list of tuples
                    dims_list = ast.literal_eval(dims_list_str)
                    
                    # Create Bin objects
                    for i, (l, w, h) in enumerate(dims_list):
                        bins.append(Bin(i, l, w, h))
                break
        
        # 2. Parse Items
        # Skip metadata and look for data rows
        # Data starts after the separator line '----'
        start_parsing = False
        for line in lines:
            stripped = line.strip()
            
            # Detect separator line to start parsing next lines
            if stripped.startswith("----"):
                start_parsing = True
                continue
            
            if not start_parsing:
                continue
            
            if not stripped:
                continue # Skip empty lines
                
            # Parse columns
            # Format: id  quantity  length  width  height  weight  bin reqs
            parts = stripped.split()
            
            # Basic validation
            if len(parts) < 6:
                continue

            # Extract fields
            i_id = parts[0]
            quantity = int(parts[1])
            length = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            weight = float(parts[5])
            
            # Handle Bin Reqs (optional column)
            bin_reqs = []
            if len(parts) > 6:
                # "0,2" -> [0, 2]
                reqs_str = parts[6]
                bin_reqs = [int(x) for x in reqs_str.split(',')]
            
            # Create 'quantity' number of items
            for q in range(quantity):
                # Unique name for each instance
                item_obj = Item(f"{i_id}_{q}", length, width, height, weight, bin_reqs)
                items.append(item_obj)

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"Error parsing file: {e}")
        
    return bins, items

def pack_bin_naive(bin_obj, available_items):
    """
    Attempts to pack items into the specific bin.
    Returns the list of items successfully packed.
    """
    # Filter items that fit criteria:
    # 1. Must be allowed in this bin (bin_reqs empty or contains bin_id)
    candidates = []
    for item in available_items:
        if not item.bin_reqs or bin_obj.bin_id in item.bin_reqs:
            candidates.append(item)
            
    # Sort largest to smallest
    candidates.sort(reverse=True)
    
    packed_items = []
    
    # Cursor positions
    x, y, z = 0, 0, 0
    max_h_in_row = 0
    
    for item in candidates:
        w, h, d = item.dims
        
        # Check if fits in X
        if x + w <= bin_obj.dims[0]:
             # Check if fits in Y and Z (Simplified logic)
            if y + h <= bin_obj.dims[1] and z + d <= bin_obj.dims[2]:
                item.position = (x, y, z)
                bin_obj.add_item(item)
                packed_items.append(item)
                
                # Advance cursor
                x += w
                max_h_in_row = max(max_h_in_row, h)
            else:
                pass # Doesn't fit in current row/layer
        else:
            # X overflow, move to next Row (Y)
            x = 0
            y += max_h_in_row
            max_h_in_row = 0
            
            # Check Y overflow
            if y >= bin_obj.dims[1]:
                # Move to next Layer (Z) - Simplified: assume layer height matches item
                y = 0
                z += h # Crude approximation for naive packer
                
            # Re-check placement after moving cursor
            if x + w <= bin_obj.dims[0] and y + h <= bin_obj.dims[1] and z + d <= bin_obj.dims[2]:
                item.position = (x, y, z)
                bin_obj.add_item(item)
                packed_items.append(item)
                x += w
                max_h_in_row = max(max_h_in_row, h)
    
    return packed_items

def visualize_packing(bin_list):
    """
    Visualizes multiple bins.
    """
    for bin_obj in bin_list:
        if not bin_obj.items:
            continue
            
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        W, H, D = bin_obj.dims
        
        # Draw Bin Frame
        # Bottom
        ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], [0, 0, 0, 0, 0], 'k-', lw=2)
        # Top
        ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0], [D, D, D, D, D], 'k-', lw=2)
        # Verticals
        for vx in [0, W]:
            for vy in [0, H]:
                ax.plot([vx, vx], [vy, vy], [0, D], 'k-', lw=2)

        colors = color_generator()

        for item in bin_obj.items:
            x, y, z = item.position
            dx, dy, dz = item.dims
            
            # Draw item
            ax.bar3d(x, y, z, dx, dy, dz, alpha=0.6, edgecolor='k', color=next(colors))

        ax.set_title(f"Bin ID: {bin_obj.bin_id} (Util: {bin_obj.get_utilization():.2f}%)")
        ax.set_xlabel("Length")
        ax.set_ylabel("Width")
        ax.set_zlabel("Height")
        
        # Scaling to keep aspect ratio roughly correct
        max_dim = max(W, H, D)
        ax.set_xlim(0, max_dim)
        ax.set_ylim(0, max_dim)
        ax.set_zlim(0, max_dim)
        
        plt.show()

if __name__ == "__main__":
    # 1. Parse File
    filename = input("Please enter the name of the file you want to pack: ")
    print(f"Reading {filename}...")
    bins, all_items = parse_custom_format(filename)
    
    if bins and all_items:
        print(f"Found {len(bins)} bins and {len(all_items)} items to pack.")
        
        # 2. Pack Bins Sequentially
        remaining_items = all_items.copy()
        
        for current_bin in bins:
            print(f"\nPacking Bin {current_bin.bin_id} (Size: {current_bin.dims})...")
            
            # Try to pack items into this bin
            packed = pack_bin_naive(current_bin, remaining_items)
            
            print(f"  -> Packed {len(packed)} items.")
            
            # Remove packed items from the remaining list so they aren't packed twice
            for p in packed:
                if p in remaining_items:
                    remaining_items.remove(p)
                    
        # 3. Summary
        print("\n--- Final Summary ---")
        total_packed = len(all_items) - len(remaining_items)
        print(f"Total Items Packed: {total_packed}/{len(all_items)}")
        
        # 4. Visualize
        visualize_packing(bins)
        
    else:
        print("Could not load data. Check file format.")
