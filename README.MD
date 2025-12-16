

````markdown
# 3D Bin Packing Visualizer

This project implements a naive heuristic for packing 3D items into a container (Bin). It calculates volume utilization and visualizes the result using `matplotlib`.


Joneil: jjames5@stevens.edu
Logan Tenner: ltenner@stevens.edu


## Requirements
* Python 3.8+
* numpy
* matplotlib
* pytest

## Installation
1. Install dependencies:
   ```bash
   pip install numpy matplotlib pytest
````

## Usage

1.  Ensure `items.txt` and `items2.txt` are in the same directory. This file contains the dimensions of items to pack. Alternatively, you can create your own items.txt file with your own items and dimensions.
2.  Run the main program:
    ```bash
    python main.py
    ```
3.  Enter the name of the items file to pack.
4.  A window will appear showing the 3D representation of the packed bin.
5.  Console output will show the packing list and efficiency percentage.

## Running Tests

To verify the integrity of the classes:

```bash
pytest test_packing.py
```

## Features Implemented

  * **OOP:** Item and Bin classes (Composition).
  * **Visuals:** 3D rendering with Matplotlib.
  * **Math:** Numpy used for vector and volume calculations.
  * **Safety:** Exception handling for file I/O and bin overflow.

<!-- end list -->

```

---

### Verification of Requirements

1.  **Classes:** `Item` and `Bin` defined.
2.  **Functions:** `load_items_from_csv` and `visualize_packing`.
3.  **Advanced Libraries:** `numpy` (math/arrays), `matplotlib` (plotting).
4.  **Exceptions & Tests:** `try/except` in file loading; `raise ValueError` in packing loop. `test_packing.py` contains 2 tests.
5.  **Data I/O:** Reads from `items.csv`.
6.  **Loops:** `for` loop (iterating items), `while` loop (in color generator and packing logic), `if` statements throughout.
7.  **Docstrings:** Included in all classes/functions.
8.  **README:** Provided above.
9.  **Pythonic "Pick 4":**
    * **Map/Lambda:** Used in `get_utilization`.
    * **Comprehension:** List comprehension used in `load_items_from_csv`.
    * **Generator:** `color_generator` function.
    * **Operator Overloading:** `__lt__` in `Item`.
    * **`__str__`**: Implemented in `Item`.
    * **`__name__`**: Used in `main.py`.
