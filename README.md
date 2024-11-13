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
(.venv) selinali@Selinas-MacBook-Pro mod3-Selinali01 % python project/parallel_check.py
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