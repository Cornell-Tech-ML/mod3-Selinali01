# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py


# Optimization script:

```bash
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (163)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (163)
-------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                              |
        out: Storage,                                                                      |
        out_shape: Shape,                                                                  |
        out_strides: Strides,                                                              |
        in_storage: Storage,                                                               |
        in_shape: Shape,                                                                   |
        in_strides: Strides,                                                               |
    ) -> None:                                                                             |
        # TODO: Implement for Task 3.1.                                                    |
        if list(in_shape) == list(out_shape) and list(in_strides) == list(out_strides):    |
            for i in prange(len(out)):-----------------------------------------------------| #2
                out[i] = fn(in_storage[i])                                                 |
        else:                                                                              |
            for i in prange(len(out)):-----------------------------------------------------| #3
                out_index = np.zeros(MAX_DIMS, np.int32)-----------------------------------| #0
                in_index = np.zeros(MAX_DIMS, np.int32)------------------------------------| #1
                to_index(i, out_shape, out_index)                                          |
                broadcast_index(out_index, out_shape, in_shape, in_index)                  |
                data = in_storage[index_to_position(in_index, in_strides)]                 |
                map_data = fn(data)                                                        |
                out[index_to_position(out_index, out_strides)] = map_data                  |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--0 has the following loops fused into it:
   +--1 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial, fused with loop(s): 1)



Parallel region 0 (loop #3) had 1 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (177) is hoisted
 out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (178) is hoisted
 out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: in_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (210)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (210)
---------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                          |
        out: Storage,                                                                  |
        out_shape: Shape,                                                              |
        out_strides: Strides,                                                          |
        a_storage: Storage,                                                            |
        a_shape: Shape,                                                                |
        a_strides: Strides,                                                            |
        b_storage: Storage,                                                            |
        b_shape: Shape,                                                                |
        b_strides: Strides,                                                            |
    ) -> None:                                                                         |
        # TODO: Implement for Task 3.1.                                                |
        if (list(a_shape) == list(b_shape) == list(out_shape) and                      |
            list(a_strides) == list(b_strides) == list(out_strides)):                  |
            for i in prange(len(out)):-------------------------------------------------| #7
                out[i] = fn(a_storage[i], b_storage[i])                                |
        else:                                                                          |
            # Handle broadcasting case                                                 |
            for i in prange(len(out)):-------------------------------------------------| #8
                # Create index buffers                                                 |
                out_index = np.zeros(MAX_DIMS, np.int32)-------------------------------| #4
                a_index = np.zeros(MAX_DIMS, np.int32)---------------------------------| #5
                b_index = np.zeros(MAX_DIMS, np.int32)---------------------------------| #6
                                                                                       |
                # Convert position to indices and handle broadcasting                  |
                to_index(i, out_shape, out_index)                                      |
                broadcast_index(out_index, out_shape, a_shape, a_index)                |
                broadcast_index(out_index, out_shape, b_shape, b_index)                |
                                                                                       |
                # Get data and apply function                                          |
                a_data = a_storage[index_to_position(a_index, a_strides)]              |
                b_data = b_storage[index_to_position(b_index, b_strides)]              |
                out[index_to_position(out_index, out_strides)] = fn(a_data, b_data)    |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...

Fused loop summary:
+--4 has the following loops fused into it:
   +--5 (fused)
   +--6 (fused)
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial, fused with loop(s): 5, 6)



Parallel region 0 (loop #8) had 2 loop(s) fused and 1 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (230) is hoisted
 out of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (231) is hoisted
 out of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (232) is hoisted
 out of the parallel loop labelled #8 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (268)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (268)
-----------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                   |
        out: Storage,                                                              |
        out_shape: Shape,                                                          |
        out_strides: Strides,                                                      |
        a_storage: Storage,                                                        |
        a_shape: Shape,                                                            |
        a_strides: Strides,                                                        |
        reduce_dim: int,                                                           |
    ) -> None:                                                                     |
        # TODO: Implement for Task 3.1.                                            |
        for i in prange(len(out)):-------------------------------------------------| #11
            out_index = np.zeros(MAX_DIMS, np.int32)-------------------------------| #9
            to_index(i, out_shape, out_index)                                      |
                                                                                   |
            # Initial value                                                        |
            acc = out[index_to_position(out_index, out_strides)]                   |
                                                                                   |
            # Iterate over the reduction dimension                                 |
            for j in range(a_shape[reduce_dim]):                                   |
                # Copy out_index for broadcasting                                  |
                a_index = np.zeros(MAX_DIMS, np.int32)-----------------------------| #10
                for k in range(len(out_shape)):                                    |
                    a_index[k] = out_index[k]                                      |
                a_index[reduce_dim] = j                                            |
                                                                                   |
                # Apply reduction                                                  |
                acc = fn(acc, a_storage[index_to_position(a_index, a_strides)])    |
                                                                                   |
            # Store result                                                         |
            out[index_to_position(out_index, out_strides)] = acc                   |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #11, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...

+--11 is a parallel loop
   +--9 --> rewritten as a serial loop
   +--10 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--9 (parallel)
   +--10 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--11 (parallel)
   +--9 (serial)
   +--10 (serial)



Parallel region 0 (loop #11) had 0 loop(s) fused and 2 loop(s) serialized as
part of the larger parallel loop (#11).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (288) is hoisted
 out of the parallel loop labelled #11 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (279) is hoisted
 out of the parallel loop labelled #11 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.zeros(MAX_DIMS, np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (302)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /Users/selinali/workspace/mod3-Selinali01/minitorch/fast_ops.py (302)
--------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                      |
    out: Storage,                                                                                 |
    out_shape: Shape,                                                                             |
    out_strides: Strides,                                                                         |
    a_storage: Storage,                                                                           |
    a_shape: Shape,                                                                               |
    a_strides: Strides,                                                                           |
    b_storage: Storage,                                                                           |
    b_shape: Shape,                                                                               |
    b_strides: Strides,                                                                           |
) -> None:                                                                                        |
    """NUMBA tensor matrix multiply function.                                                     |
                                                                                                  |
    Should work for any tensor shapes that broadcast as long as                                   |
                                                                                                  |
    ```                                                                                           |
    assert a_shape[-1] == b_shape[-2]                                                             |
    ```                                                                                           |
                                                                                                  |
    Optimizations:                                                                                |
                                                                                                  |
    * Outer loop in parallel                                                                      |
    * No index buffers or function calls                                                          |
    * Inner loop should have no global writes, 1 multiply.                                        |
                                                                                                  |
                                                                                                  |
    Args:                                                                                         |
    ----                                                                                          |
        out (Storage): storage for `out` tensor                                                   |
        out_shape (Shape): shape for `out` tensor                                                 |
        out_strides (Strides): strides for `out` tensor                                           |
        a_storage (Storage): storage for `a` tensor                                               |
        a_shape (Shape): shape for `a` tensor                                                     |
        a_strides (Strides): strides for `a` tensor                                               |
        b_storage (Storage): storage for `b` tensor                                               |
        b_shape (Shape): shape for `b` tensor                                                     |
        b_strides (Strides): strides for `b` tensor                                               |
                                                                                                  |
    Returns:                                                                                      |
    -------                                                                                       |
        None : Fills in `out`                                                                     |
                                                                                                  |
    """                                                                                           |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                        |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                        |
                                                                                                  |
    # TODO: Implement for Task 3.2.                                                               |
    stride_row_a = a_strides[2]                                                                   |
    stride_col_b = b_strides[1]                                                                   |
    sum_limit = a_shape[-1]                                                                       |
                                                                                                  |
    # Main computation loops                                                                      |
    for idx in prange(out_shape[0]):  # Parallel loop for outer dimension-------------------------| #12
        for row in range(out_shape[1]):                                                           |
            for col in range(out_shape[2]):                                                       |
                # Initialize accumulator                                                          |
                sum_temp = 0.0                                                                    |
                                                                                                  |
                # Calculate base indices for this element                                         |
                a_index = idx * a_batch_stride + row * a_strides[1]                               |
                b_index = idx * b_batch_stride + col * b_strides[2]                               |
                                                                                                  |
                # Inner product loop with stride increments                                       |
                for _ in range(sum_limit):                                                        |
                    sum_temp += a_storage[a_index] * b_storage[b_index]                           |
                    a_index += stride_row_a                                                       |
                    b_index += stride_col_b                                                       |
                                                                                                  |
                # Store result                                                                    |
                out_index = idx * out_strides[0] + row * out_strides[1] + col * out_strides[2]    |
                out[out_index] = sum_temp                                                         |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #12).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```

# Simple dataset
### CPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
Average time: 0.131 seonds / epoch

Epoch   0 | loss 4.7625 | correct  35 | time 0.0000s
Epoch  10 | loss 4.4206 | correct  49 | time 1.2905s
Epoch  20 | loss 1.3740 | correct  50 | time 1.2298s
Epoch  30 | loss 1.7379 | correct  49 | time 1.2115s
Epoch  40 | loss 0.9118 | correct  50 | time 1.2015s
Epoch  50 | loss 0.6813 | correct  50 | time 1.2432s
Epoch  60 | loss 1.2124 | correct  50 | time 1.5120s
Epoch  70 | loss 0.7858 | correct  49 | time 2.1759s
Epoch  80 | loss 0.1791 | correct  50 | time 1.2113s
Epoch  90 | loss 0.7007 | correct  50 | time 1.2038s
Epoch 100 | loss 0.1050 | correct  50 | time 1.2026s
Epoch 110 | loss 0.1396 | correct  50 | time 1.1879s
Epoch 120 | loss 0.1553 | correct  50 | time 1.1789s
Epoch 130 | loss 0.2458 | correct  50 | time 1.1967s
Epoch 140 | loss 0.5070 | correct  50 | time 1.1940s
Epoch 150 | loss 0.6026 | correct  50 | time 1.2265s
Epoch 160 | loss 0.0125 | correct  50 | time 1.9865s
Epoch 170 | loss 0.3542 | correct  50 | time 1.5968s
Epoch 180 | loss 0.1913 | correct  50 | time 1.1930s
Epoch 190 | loss 0.0021 | correct  50 | time 1.1957s
Epoch 200 | loss 0.2608 | correct  50 | time 1.1876s
Epoch 210 | loss 0.1519 | correct  50 | time 1.1794s
Epoch 220 | loss 0.2717 | correct  50 | time 1.1996s
Epoch 230 | loss 0.0453 | correct  50 | time 1.1983s
Epoch 240 | loss 0.3217 | correct  50 | time 1.2000s
Epoch 250 | loss 0.1434 | correct  50 | time 1.4624s
Epoch 260 | loss 0.1047 | correct  50 | time 2.0248s
Epoch 270 | loss 0.0379 | correct  50 | time 1.2120s
Epoch 280 | loss 0.1264 | correct  50 | time 1.1925s
Epoch 290 | loss 0.2770 | correct  50 | time 1.2093s
Epoch 300 | loss 0.1321 | correct  50 | time 1.2026s
Epoch 310 | loss 0.3379 | correct  50 | time 1.1917s
Epoch 320 | loss 0.0646 | correct  50 | time 1.1949s
Epoch 330 | loss 0.0044 | correct  50 | time 1.2162s
Epoch 340 | loss 0.3546 | correct  50 | time 1.1900s
Epoch 350 | loss 0.2271 | correct  50 | time 1.8573s
Epoch 360 | loss 0.0607 | correct  50 | time 1.7347s
Epoch 370 | loss 0.0244 | correct  50 | time 1.2061s
Epoch 380 | loss 0.0204 | correct  50 | time 1.1969s
Epoch 390 | loss 0.1846 | correct  50 | time 1.2098s
Epoch 400 | loss 0.0113 | correct  50 | time 1.2186s
Epoch 410 | loss 0.0213 | correct  50 | time 1.1968s
Epoch 420 | loss 0.0775 | correct  50 | time 1.1897s
Epoch 430 | loss 0.1991 | correct  50 | time 1.1953s
Epoch 440 | loss 0.2306 | correct  50 | time 1.4246s
Epoch 450 | loss 0.0712 | correct  50 | time 2.1552s
Epoch 460 | loss 0.1196 | correct  50 | time 1.1908s
Epoch 470 | loss 0.0417 | correct  50 | time 1.1984s
Epoch 480 | loss 0.0643 | correct  50 | time 1.1997s
Epoch 490 | loss 0.0471 | correct  50 | time 1.1912s
```

### GPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
Average time: 1.546 seonds / epoch

Epoch   0 | loss 7.0900 | correct  38 | time 0.0000s
Epoch  10 | loss 3.1212 | correct  50 | time 15.7574s
Epoch  20 | loss 1.2849 | correct  46 | time 15.7238s
Epoch  30 | loss 0.9300 | correct  50 | time 15.7260s
Epoch  40 | loss 0.3159 | correct  49 | time 15.5065s
Epoch  50 | loss 1.8183 | correct  50 | time 15.5837s
Epoch  60 | loss 1.3735 | correct  50 | time 15.7513s
Epoch  70 | loss 1.2828 | correct  50 | time 16.0571s
Epoch  80 | loss 1.6776 | correct  49 | time 15.6818s
Epoch  90 | loss 0.0627 | correct  50 | time 15.7652s
Epoch 100 | loss 0.1603 | correct  49 | time 15.8443s
Epoch 110 | loss 0.3567 | correct  50 | time 16.0176s
Epoch 120 | loss 0.6018 | correct  50 | time 15.6996s
Epoch 130 | loss 0.9472 | correct  50 | time 16.0030s
Epoch 140 | loss 0.6509 | correct  50 | time 15.8030s
Epoch 150 | loss 0.0038 | correct  50 | time 15.5744s
Epoch 160 | loss 0.0125 | correct  50 | time 15.4448s
Epoch 170 | loss 0.1412 | correct  50 | time 15.6114s
Epoch 180 | loss 0.4811 | correct  50 | time 15.8056s
Epoch 190 | loss 0.0120 | correct  50 | time 15.4917s
Epoch 200 | loss 0.0060 | correct  49 | time 15.5023s
Epoch 210 | loss 0.8588 | correct  50 | time 15.5292s
Epoch 220 | loss 1.2056 | correct  49 | time 15.9457s
Epoch 230 | loss 0.0067 | correct  50 | time 15.7361s
Epoch 240 | loss 0.3567 | correct  50 | time 15.7075s
Epoch 250 | loss 0.0474 | correct  50 | time 15.6985s
Epoch 260 | loss 0.6094 | correct  50 | time 15.7062s
Epoch 270 | loss 0.4350 | correct  50 | time 15.6508s
Epoch 280 | loss 0.6361 | correct  50 | time 15.5147s
Epoch 290 | loss 0.0129 | correct  50 | time 15.4408s
Epoch 300 | loss 0.0088 | correct  50 | time 15.4534s
Epoch 310 | loss 0.4940 | correct  50 | time 15.3601s
Epoch 320 | loss 0.1172 | correct  50 | time 15.3389s
Epoch 330 | loss 0.0821 | correct  50 | time 15.6656s
Epoch 340 | loss 0.0002 | correct  50 | time 15.3390s
Epoch 350 | loss 0.4204 | correct  50 | time 15.3983s
Epoch 360 | loss 0.1331 | correct  50 | time 15.3854s
Epoch 370 | loss 0.1922 | correct  50 | time 15.4823s
Epoch 380 | loss 0.0018 | correct  50 | time 15.3716s
Epoch 390 | loss 0.0254 | correct  50 | time 15.4785s
Epoch 400 | loss 0.0989 | correct  50 | time 15.3256s
Epoch 410 | loss 0.2661 | correct  50 | time 15.4227s
Epoch 420 | loss 1.0378 | correct  49 | time 15.4888s
Epoch 430 | loss 0.0010 | correct  50 | time 15.3883s
Epoch 440 | loss 0.2144 | correct  50 | time 15.4048s
Epoch 450 | loss 0.1883 | correct  50 | time 15.3953s
Epoch 460 | loss 0.0336 | correct  50 | time 15.2965s
Epoch 470 | loss 0.0015 | correct  50 | time 15.4001s
Epoch 480 | loss 0.0651 | correct  50 | time 15.3576s
Epoch 490 | loss 0.1821 | correct  50 | time 15.4158s
```

# XOR dataset
### CPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
Average time: 0.132 seonds / epoch


Epoch   0 | loss 6.8296 | correct  29 | time 0.0000s
Epoch  10 | loss 6.2447 | correct  41 | time 1.3376s
Epoch  20 | loss 3.8040 | correct  45 | time 1.2233s
Epoch  30 | loss 3.8895 | correct  43 | time 1.2113s
Epoch  40 | loss 1.4958 | correct  46 | time 1.2342s
Epoch  50 | loss 1.3840 | correct  47 | time 1.1950s
Epoch  60 | loss 1.1705 | correct  46 | time 1.7277s
Epoch  70 | loss 2.1305 | correct  48 | time 1.9451s
Epoch  80 | loss 1.5707 | correct  47 | time 1.2136s
Epoch  90 | loss 0.8814 | correct  47 | time 1.2310s
Epoch 100 | loss 2.7508 | correct  47 | time 1.2090s
Epoch 110 | loss 0.6619 | correct  48 | time 1.2167s
Epoch 120 | loss 1.2975 | correct  48 | time 1.1975s
Epoch 130 | loss 1.8650 | correct  48 | time 1.1884s
Epoch 140 | loss 1.6000 | correct  49 | time 1.1953s
Epoch 150 | loss 0.4435 | correct  50 | time 1.1838s
Epoch 160 | loss 0.9548 | correct  48 | time 2.0649s
Epoch 170 | loss 2.2341 | correct  50 | time 1.4693s
Epoch 180 | loss 1.8368 | correct  50 | time 1.1941s
Epoch 190 | loss 0.8682 | correct  50 | time 1.1816s
Epoch 200 | loss 0.4202 | correct  49 | time 1.1937s
Epoch 210 | loss 0.1157 | correct  49 | time 1.2210s
Epoch 220 | loss 0.0841 | correct  50 | time 1.1810s
Epoch 230 | loss 1.6767 | correct  49 | time 1.1760s
Epoch 240 | loss 0.4805 | correct  49 | time 1.2057s
Epoch 250 | loss 0.7873 | correct  50 | time 1.5419s
Epoch 260 | loss 0.5304 | correct  49 | time 1.9609s
Epoch 270 | loss 1.2844 | correct  50 | time 1.1805s
Epoch 280 | loss 1.4727 | correct  50 | time 1.2156s
Epoch 290 | loss 0.7695 | correct  50 | time 1.1972s
Epoch 300 | loss 0.1086 | correct  49 | time 1.2263s
Epoch 310 | loss 1.0260 | correct  50 | time 1.1823s
Epoch 320 | loss 1.0376 | correct  50 | time 1.2157s
Epoch 330 | loss 0.1072 | correct  49 | time 1.1881s
Epoch 340 | loss 0.4128 | correct  50 | time 1.1978s
Epoch 350 | loss 0.2315 | correct  49 | time 1.9774s
Epoch 360 | loss 0.5271 | correct  50 | time 1.5637s
Epoch 370 | loss 0.0251 | correct  50 | time 1.1958s
Epoch 380 | loss 0.2018 | correct  49 | time 1.1834s
Epoch 390 | loss 0.0831 | correct  50 | time 1.1913s
Epoch 400 | loss 0.2144 | correct  50 | time 1.1862s
Epoch 410 | loss 0.7835 | correct  50 | time 1.1949s
Epoch 420 | loss 0.1430 | correct  50 | time 1.1903s
Epoch 430 | loss 0.4259 | correct  50 | time 1.1944s
Epoch 440 | loss 0.1183 | correct  50 | time 1.5423s
Epoch 450 | loss 0.1149 | correct  50 | time 2.0341s
Epoch 460 | loss 0.5993 | correct  50 | time 1.1840s
Epoch 470 | loss 0.1403 | correct  50 | time 1.2040s
Epoch 480 | loss 0.0621 | correct  50 | time 1.1885s
Epoch 490 | loss 0.1260 | correct  50 | time 1.1914s
```

### GPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
Average time: 1.517 seonds / epoch

Epoch   0 | loss 6.8717 | correct  21 | time 0.0000s
Epoch  10 | loss 6.7972 | correct  39 | time 15.2245s
Epoch  20 | loss 3.8091 | correct  41 | time 15.3121s
Epoch  30 | loss 3.1117 | correct  42 | time 15.3570s
Epoch  40 | loss 2.3664 | correct  39 | time 15.2853s
Epoch  50 | loss 4.6522 | correct  44 | time 15.3275s
Epoch  60 | loss 2.3318 | correct  45 | time 15.1932s
Epoch  70 | loss 2.7787 | correct  44 | time 15.3466s
Epoch  80 | loss 1.7523 | correct  42 | time 15.2669s
Epoch  90 | loss 4.3526 | correct  47 | time 15.4346s
Epoch 100 | loss 1.1022 | correct  41 | time 15.2575s
Epoch 110 | loss 1.9034 | correct  43 | time 15.4205s
Epoch 120 | loss 2.5893 | correct  47 | time 15.1495s
Epoch 130 | loss 2.3194 | correct  48 | time 15.1046s
Epoch 140 | loss 1.5418 | correct  45 | time 15.2069s
Epoch 150 | loss 4.8134 | correct  47 | time 15.2408s
Epoch 160 | loss 1.5340 | correct  48 | time 15.4623s
Epoch 170 | loss 2.3080 | correct  47 | time 15.2806s
Epoch 180 | loss 1.4307 | correct  47 | time 15.3568s
Epoch 190 | loss 0.3806 | correct  46 | time 15.2252s
Epoch 200 | loss 0.7606 | correct  45 | time 15.2535s
Epoch 210 | loss 0.7625 | correct  48 | time 15.1114s
Epoch 220 | loss 2.8793 | correct  49 | time 15.3622s
Epoch 230 | loss 0.4409 | correct  50 | time 15.3286s
Epoch 240 | loss 2.6634 | correct  45 | time 15.1612s
Epoch 250 | loss 0.2688 | correct  48 | time 15.2938s
Epoch 260 | loss 2.5835 | correct  50 | time 15.2527s
Epoch 270 | loss 1.8425 | correct  48 | time 15.3519s
Epoch 280 | loss 0.6170 | correct  48 | time 15.2348s
Epoch 290 | loss 0.3697 | correct  49 | time 15.4068s
Epoch 300 | loss 0.7708 | correct  48 | time 15.2936s
Epoch 310 | loss 1.4046 | correct  48 | time 15.2343s
Epoch 320 | loss 1.3057 | correct  48 | time 15.0756s
Epoch 330 | loss 1.0185 | correct  49 | time 15.1457s
Epoch 340 | loss 2.5566 | correct  46 | time 15.1762s
Epoch 350 | loss 2.2013 | correct  46 | time 15.1563s
Epoch 360 | loss 0.1216 | correct  50 | time 15.2600s
Epoch 370 | loss 2.2796 | correct  49 | time 15.4166s
Epoch 380 | loss 2.1475 | correct  49 | time 15.4440s
Epoch 390 | loss 1.2708 | correct  48 | time 15.3349s
Epoch 400 | loss 2.1857 | correct  48 | time 15.3507s
Epoch 410 | loss 0.3307 | correct  49 | time 15.1832s
Epoch 420 | loss 2.2671 | correct  49 | time 15.1993s
Epoch 430 | loss 2.2525 | correct  49 | time 15.1771s
Epoch 440 | loss 3.7110 | correct  46 | time 15.0751s
Epoch 450 | loss 2.2261 | correct  49 | time 15.0302s
Epoch 460 | loss 3.2996 | correct  46 | time 15.1991s
Epoch 470 | loss 2.0755 | correct  48 | time 15.1446s
Epoch 480 | loss 1.5304 | correct  48 | time 15.1473s
Epoch 490 | loss 0.7748 | correct  48 | time 15.1938s
```


# Split dataset
## CPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
Average time: 0.135 seonds / epoch


Epoch   0 | loss 6.1296 | correct  33 | time 0.0000s
Epoch  10 | loss 5.5595 | correct  40 | time 2.0923s
Epoch  20 | loss 5.9561 | correct  42 | time 1.1894s
Epoch  30 | loss 7.2009 | correct  43 | time 1.1951s
Epoch  40 | loss 3.7786 | correct  41 | time 1.1959s
Epoch  50 | loss 3.5689 | correct  43 | time 1.1885s
Epoch  60 | loss 3.5575 | correct  42 | time 1.1884s
Epoch  70 | loss 2.0395 | correct  42 | time 1.1918s
Epoch  80 | loss 2.6465 | correct  48 | time 1.1826s
Epoch  90 | loss 2.8864 | correct  49 | time 1.1787s
Epoch 100 | loss 2.9914 | correct  48 | time 2.0135s
Epoch 110 | loss 1.9678 | correct  45 | time 1.5828s
Epoch 120 | loss 1.5089 | correct  48 | time 1.1974s
Epoch 130 | loss 1.9191 | correct  49 | time 1.1829s
Epoch 140 | loss 2.1638 | correct  49 | time 1.1754s
Epoch 150 | loss 0.7450 | correct  48 | time 1.1728s
Epoch 160 | loss 0.3318 | correct  49 | time 1.1682s
Epoch 170 | loss 1.5307 | correct  49 | time 1.1734s
Epoch 180 | loss 1.0976 | correct  49 | time 1.1668s
Epoch 190 | loss 1.1133 | correct  49 | time 1.2789s
Epoch 200 | loss 1.4465 | correct  49 | time 2.2637s
Epoch 210 | loss 1.2597 | correct  49 | time 1.2128s
Epoch 220 | loss 0.2599 | correct  49 | time 1.2037s
Epoch 230 | loss 0.2924 | correct  46 | time 1.1994s
Epoch 240 | loss 2.8482 | correct  50 | time 1.1960s
Epoch 250 | loss 0.6654 | correct  50 | time 1.1889s
Epoch 260 | loss 1.1091 | correct  50 | time 1.1974s
Epoch 270 | loss 1.1300 | correct  50 | time 1.1754s
Epoch 280 | loss 0.0453 | correct  49 | time 1.1786s
Epoch 290 | loss 1.1909 | correct  49 | time 1.6584s
Epoch 300 | loss 0.9607 | correct  49 | time 1.8695s
Epoch 310 | loss 0.5009 | correct  50 | time 1.1963s
Epoch 320 | loss 1.2491 | correct  50 | time 1.1955s
Epoch 330 | loss 0.5308 | correct  50 | time 1.2031s
Epoch 340 | loss 0.3091 | correct  49 | time 1.1817s
Epoch 350 | loss 1.0594 | correct  50 | time 1.1805s
Epoch 360 | loss 1.6310 | correct  48 | time 1.1904s
Epoch 370 | loss 0.2460 | correct  49 | time 1.2035s
Epoch 380 | loss 0.5716 | correct  50 | time 1.2150s
Epoch 390 | loss 0.7917 | correct  50 | time 2.2954s
Epoch 400 | loss 0.7180 | correct  50 | time 1.2416s
Epoch 410 | loss 0.7586 | correct  50 | time 1.1794s
Epoch 420 | loss 0.8625 | correct  50 | time 1.1916s
Epoch 430 | loss 0.5584 | correct  49 | time 1.1951s
Epoch 440 | loss 0.9650 | correct  49 | time 1.2237s
Epoch 450 | loss 0.5168 | correct  50 | time 1.2054s
Epoch 460 | loss 0.4595 | correct  50 | time 1.1969s
Epoch 470 | loss 0.8164 | correct  50 | time 1.1958s
Epoch 480 | loss 0.5588 | correct  50 | time 1.7669s
Epoch 490 | loss 0.0119 | correct  50 | time 1.7624s
```

## GPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
Average time: 1.534 seonds / epoch
Epoch   0 | loss 6.9041 | correct  29 | time 0.0000s
Epoch  10 | loss 5.6570 | correct  30 | time 15.3972s
Epoch  20 | loss 6.3936 | correct  42 | time 15.3158s
Epoch  30 | loss 3.9779 | correct  38 | time 15.3092s
Epoch  40 | loss 3.9673 | correct  40 | time 15.3471s
Epoch  50 | loss 3.4284 | correct  47 | time 15.4489s
Epoch  60 | loss 3.7651 | correct  36 | time 15.3406s
Epoch  70 | loss 4.0707 | correct  43 | time 15.4417s
Epoch  80 | loss 3.1851 | correct  47 | time 15.4567s
Epoch  90 | loss 2.8673 | correct  47 | time 15.5145s
Epoch 100 | loss 4.4299 | correct  44 | time 15.4593s
Epoch 110 | loss 1.7502 | correct  46 | time 15.5059s
Epoch 120 | loss 2.1831 | correct  48 | time 15.6358s
Epoch 130 | loss 3.0583 | correct  48 | time 15.8904s
Epoch 140 | loss 1.6347 | correct  43 | time 15.3696s
Epoch 150 | loss 1.5052 | correct  49 | time 15.5542s
Epoch 160 | loss 1.1193 | correct  48 | time 15.5998s
Epoch 170 | loss 0.9105 | correct  49 | time 15.5376s
Epoch 180 | loss 0.9995 | correct  50 | time 15.6717s
Epoch 190 | loss 0.4005 | correct  49 | time 15.4196s
Epoch 200 | loss 0.4136 | correct  50 | time 15.8006s
Epoch 210 | loss 0.9625 | correct  50 | time 15.6987s
Epoch 220 | loss 1.5117 | correct  49 | time 15.5647s
Epoch 230 | loss 0.9401 | correct  48 | time 15.5969s
Epoch 240 | loss 1.9975 | correct  46 | time 15.5093s
Epoch 250 | loss 4.8215 | correct  49 | time 15.3459s
Epoch 260 | loss 0.3942 | correct  49 | time 15.4311s
Epoch 270 | loss 0.5130 | correct  49 | time 15.5999s
Epoch 280 | loss 0.9940 | correct  48 | time 15.2839s
Epoch 290 | loss 0.4658 | correct  48 | time 15.3279s
Epoch 300 | loss 0.6368 | correct  50 | time 15.4122s
Epoch 310 | loss 1.2327 | correct  49 | time 15.6110s
Epoch 320 | loss 0.6767 | correct  49 | time 15.5082s
Epoch 330 | loss 2.3340 | correct  48 | time 15.6646s
Epoch 340 | loss 1.0215 | correct  50 | time 15.3339s
Epoch 350 | loss 0.2649 | correct  50 | time 15.3755s
Epoch 360 | loss 1.5573 | correct  48 | time 15.2270s
Epoch 370 | loss 1.1532 | correct  47 | time 15.5751s
Epoch 380 | loss 0.8344 | correct  50 | time 15.4169s
Epoch 390 | loss 0.3942 | correct  49 | time 15.3485s
Epoch 400 | loss 0.9577 | correct  50 | time 15.2804s
Epoch 410 | loss 0.7136 | correct  50 | time 15.3883s
Epoch 420 | loss 0.7111 | correct  50 | time 15.3236s
Epoch 430 | loss 4.4896 | correct  44 | time 15.4253s
Epoch 440 | loss 0.9565 | correct  49 | time 15.3335s
Epoch 450 | loss 0.5328 | correct  49 | time 15.2206s
Epoch 460 | loss 0.0443 | correct  47 | time 15.3099s
Epoch 470 | loss 2.0371 | correct  46 | time 15.2391s
Epoch 480 | loss 0.1774 | correct  49 | time 15.2550s
Epoch 490 | loss 1.2919 | correct  48 | time 15.2240s
```


# BIGGER DATASET:
### CPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 250 --DATASET split --RATE 0.05
Average time: 0.402 seonds / epoch


Epoch   0 | loss 32.9153 | correct  24 | time 0.0000s
Epoch  10 | loss 3.4666 | correct  39 | time 3.7903s
Epoch  20 | loss 3.8376 | correct  41 | time 3.7120s
Epoch  30 | loss 6.5789 | correct  45 | time 4.9298s
Epoch  40 | loss 0.8202 | correct  49 | time 3.7413s
Epoch  50 | loss 1.1303 | correct  47 | time 3.7136s
Epoch  60 | loss 1.3793 | correct  48 | time 4.8910s
Epoch  70 | loss 0.3568 | correct  48 | time 3.7397s
Epoch  80 | loss 0.5874 | correct  48 | time 3.6981s
Epoch  90 | loss 0.2264 | correct  48 | time 4.9026s
Epoch 100 | loss 2.7446 | correct  48 | time 3.7546s
Epoch 110 | loss 0.4693 | correct  48 | time 3.7355s
Epoch 120 | loss 0.2691 | correct  48 | time 4.8603s
Epoch 130 | loss 2.0739 | correct  48 | time 3.7018s
Epoch 140 | loss 0.3035 | correct  50 | time 3.7121s
Epoch 150 | loss 0.1153 | correct  48 | time 4.8779s
Epoch 160 | loss 1.1639 | correct  50 | time 3.7216s
Epoch 170 | loss 0.5423 | correct  50 | time 3.7184s
Epoch 180 | loss 0.2335 | correct  48 | time 4.9297s
Epoch 190 | loss 1.2612 | correct  50 | time 3.7210s
Epoch 200 | loss 0.9335 | correct  50 | time 3.7028s
Epoch 210 | loss 0.0748 | correct  48 | time 4.9232s
Epoch 220 | loss 0.0600 | correct  48 | time 3.7274s
Epoch 230 | loss 0.0937 | correct  50 | time 3.6763s
Epoch 240 | loss 0.2829 | correct  50 | time 4.9459s
Epoch 250 | loss 0.1616 | correct  47 | time 3.7470s
Epoch 260 | loss 0.3180 | correct  50 | time 3.7084s
Epoch 270 | loss 1.6756 | correct  48 | time 4.8486s
Epoch 280 | loss 0.2361 | correct  47 | time 3.6661s
Epoch 290 | loss 0.5936 | correct  50 | time 3.7315s
Epoch 300 | loss 1.6548 | correct  48 | time 4.8681s
Epoch 310 | loss 0.0770 | correct  49 | time 3.7201s
Epoch 320 | loss 0.8941 | correct  50 | time 3.7331s
Epoch 330 | loss 0.0470 | correct  50 | time 4.9336s
Epoch 340 | loss 0.8119 | correct  48 | time 3.6933s
Epoch 350 | loss 0.2658 | correct  50 | time 3.7027s
Epoch 360 | loss 0.0572 | correct  50 | time 4.9018s
Epoch 370 | loss 0.0470 | correct  50 | time 3.7126s
Epoch 380 | loss 0.7127 | correct  50 | time 3.6907s
Epoch 390 | loss 1.3405 | correct  48 | time 4.9020s
Epoch 400 | loss 0.5378 | correct  50 | time 3.7586s
Epoch 410 | loss 0.5959 | correct  50 | time 3.7066s
Epoch 420 | loss 0.6908 | correct  49 | time 4.8050s
Epoch 430 | loss 0.0136 | correct  48 | time 3.7262s
Epoch 440 | loss 0.1071 | correct  49 | time 3.7581s
Epoch 450 | loss 0.4452 | correct  50 | time 4.6963s
Epoch 460 | loss 0.1754 | correct  50 | time 3.9433s
Epoch 470 | loss 0.5195 | correct  50 | time 3.7995s
Epoch 480 | loss 0.9898 | correct  49 | time 4.8313s
Epoch 490 | loss 0.0496 | correct  50 | time 3.8699s
```

### GPU
```bash
!cd $DIR; PYTHONPATH=/content/$DIR python3.12 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 250 --DATASET split --RATE 0.05
Average time: 1.670 seonds / epoch

Epoch   0 | loss 19.5811 | correct  32 | time 0.0000s
Epoch  10 | loss 4.5612 | correct  44 | time 16.7855s
Epoch  20 | loss 4.3691 | correct  43 | time 16.6470s
Epoch  30 | loss 1.9083 | correct  50 | time 16.9879s
Epoch  40 | loss 3.6118 | correct  43 | time 16.7361s
Epoch  50 | loss 1.7581 | correct  46 | time 16.7830s
Epoch  60 | loss 0.5868 | correct  46 | time 16.8169s
Epoch  70 | loss 1.4774 | correct  45 | time 16.8297s
Epoch  80 | loss 1.8721 | correct  49 | time 16.8117s
Epoch  90 | loss 0.7866 | correct  47 | time 16.7818s
Epoch 100 | loss 0.6950 | correct  49 | time 16.5759s
Epoch 110 | loss 2.6685 | correct  45 | time 16.7416s
Epoch 120 | loss 2.4672 | correct  47 | time 16.6099s
Epoch 130 | loss 0.5328 | correct  48 | time 16.7619s
Epoch 140 | loss 1.2254 | correct  48 | time 16.8127s
Epoch 150 | loss 2.5286 | correct  44 | time 16.7213s
Epoch 160 | loss 0.6899 | correct  49 | time 16.8323s
Epoch 170 | loss 1.2145 | correct  50 | time 16.7369s
Epoch 180 | loss 0.3960 | correct  49 | time 16.8946s
Epoch 190 | loss 0.3535 | correct  50 | time 16.8878s
Epoch 200 | loss 0.5599 | correct  46 | time 16.8253s
Epoch 210 | loss 0.4004 | correct  49 | time 16.9566s
Epoch 220 | loss 0.5610 | correct  49 | time 16.8480s
Epoch 230 | loss 0.5599 | correct  50 | time 16.6870s
Epoch 240 | loss 2.3339 | correct  44 | time 16.6806s
Epoch 250 | loss 0.2726 | correct  50 | time 16.5602s
Epoch 260 | loss 0.5193 | correct  50 | time 16.6575s
Epoch 270 | loss 0.0881 | correct  50 | time 16.7887s
Epoch 280 | loss 0.7888 | correct  49 | time 16.6359s
Epoch 290 | loss 1.8247 | correct  50 | time 16.8094s
Epoch 300 | loss 1.0253 | correct  50 | time 16.6326s
Epoch 310 | loss 0.6781 | correct  49 | time 16.7132s
Epoch 320 | loss 0.6066 | correct  50 | time 16.6300s
Epoch 330 | loss 1.1063 | correct  48 | time 16.8941s
Epoch 340 | loss 0.0782 | correct  50 | time 16.6851s
Epoch 350 | loss 0.6397 | correct  48 | time 16.7344s
Epoch 360 | loss 0.3332 | correct  49 | time 16.6279s
Epoch 370 | loss 0.0467 | correct  49 | time 16.8518s
Epoch 380 | loss 0.2697 | correct  50 | time 16.6383s
Epoch 390 | loss 0.2224 | correct  44 | time 16.7850s
Epoch 400 | loss 0.1307 | correct  50 | time 16.7998s
Epoch 410 | loss 0.7234 | correct  48 | time 16.6454s
Epoch 420 | loss 0.2831 | correct  50 | time 16.7453s
Epoch 430 | loss 1.0449 | correct  50 | time 16.6332s
Epoch 440 | loss 0.0552 | correct  50 | time 16.6131s
Epoch 450 | loss 0.1237 | correct  50 | time 17.0508s
Epoch 460 | loss 0.3117 | correct  50 | time 17.1299s
Epoch 470 | loss 0.3432 | correct  50 | time 17.0279s
Epoch 480 | loss 0.0896 | correct  50 | time 16.8609s
Epoch 490 | loss 0.7095 | correct  49 | time 16.7409s
```