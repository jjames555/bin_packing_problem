import pytest
import numpy as np
from main import Item, Bin

# Test 1: Verify volume calculation and Operator Overloading
def test_item_volume_and_comparison():
    item1 = Item("Small", 1, 1, 1, 10) # Vol 1
    item2 = Item("Big", 2, 2, 2, 20)   # Vol 8
    
    assert item1.volume == 1
    assert item2.volume == 8
    
    # Test __lt__ overloading
    assert item1 < item2

# Test 2: Verify Bin capacity logic
def test_bin_add_item():
    bin_obj = Bin(1, 10, 10, 10)
    item = Item("Box", 5, 5, 5, 50)
    item.position = (0,0,0) # Manually set for this unit test
    
    bin_obj.add_item(item)
    
    assert len(bin_obj.items) == 1
    # Utilization should be 125 / 1000 = 12.5%
    assert bin_obj.get_utilization() == 12.5
