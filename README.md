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

## CPU
```bash
Epoch  0  loss  6.356308824408212 correct 26
Epoch  10  loss  5.418607162935366 correct 41
Epoch  20  loss  4.627901407495846 correct 43
Epoch  30  loss  2.943667037127709 correct 46
Epoch  40  loss  2.462070425647282 correct 43
Epoch  50  loss  2.0335613263905357 correct 48
Epoch  60  loss  2.2090463127433164 correct 47
Epoch  70  loss  1.075256610058709 correct 50
Epoch  80  loss  1.3133875428547999 correct 48
Epoch  90  loss  2.8144017911259187 correct 49
Epoch  100  loss  2.738312518528487 correct 48
Epoch  110  loss  1.5404031069756354 correct 49
Epoch  120  loss  1.7163261309959839 correct 48
Epoch  130  loss  1.4216022381460753 correct 50
Epoch  140  loss  1.4048944129824035 correct 50
Epoch  150  loss  1.133235355465042 correct 49
Epoch  160  loss  0.46140588076299255 correct 50
Epoch  170  loss  1.680154005676697 correct 50
Epoch  180  loss  1.7332371840664194 correct 49
Epoch  190  loss  0.6595139171132911 correct 49
Epoch  200  loss  0.770061535287949 correct 48
Epoch  210  loss  0.4279187110414158 correct 48
Epoch  220  loss  1.8965365474645126 correct 49
Epoch  230  loss  0.7923156551871902 correct 49
Epoch  240  loss  0.24453927248631147 correct 50
Epoch  250  loss  0.38454998212056907 correct 50
Epoch  260  loss  0.2736513574951542 correct 50
Epoch  270  loss  0.8685473298357281 correct 50
Epoch  280  loss  0.5135784650165167 correct 49
Epoch  290  loss  0.7462019300498971 correct 50
Epoch  300  loss  0.38144368277599244 correct 50
Epoch  310  loss  0.301577879806474 correct 50
Epoch  320  loss  0.7608557673668518 correct 50
Epoch  330  loss  0.2596929395024636 correct 50
Epoch  340  loss  0.07736251040049995 correct 50
Epoch  350  loss  0.6357074418240116 correct 50
Epoch  360  loss  0.7275226735556826 correct 50
Epoch  370  loss  1.4316786117701619 correct 50
Epoch  380  loss  0.08567330327584094 correct 50
Epoch  390  loss  0.012569998705600754 correct 50
Epoch  400  loss  0.3244040549828151 correct 50
Epoch  410  loss  0.383282147707516 correct 50
Epoch  420  loss  0.05848998751408848 correct 50
Epoch  430  loss  0.6708089577407705 correct 50
Epoch  440  loss  0.3240682647683909 correct 50
Epoch  450  loss  0.4678792897231626 correct 50
Epoch  460  loss  0.10160021677497674 correct 50
Epoch  470  loss  0.7285665175130407 correct 50
Epoch  480  loss  0.7783227006869917 correct 50
Epoch  490  loss  0.5454981802545661 correct 50
```

## GPU
```bash
Epoch  0  loss  4.973002792406712 correct 36
Epoch  10  loss  4.045323738540773 correct 36
Epoch  20  loss  4.323025458446253 correct 40
Epoch  30  loss  5.457657654352095 correct 44
Epoch  40  loss  3.7136164647006433 correct 41
Epoch  50  loss  2.2371209526863693 correct 43
Epoch  60  loss  2.701607923296729 correct 45
Epoch  70  loss  2.621611325735495 correct 47
Epoch  80  loss  2.3028185969460244 correct 48
Epoch  90  loss  2.7761249295060786 correct 47
Epoch  100  loss  0.8579520672611675 correct 47
Epoch  110  loss  2.5210408958497634 correct 47
Epoch  120  loss  2.0508051271552676 correct 50
Epoch  130  loss  1.2755838155523778 correct 48
Epoch  140  loss  0.820576220407177 correct 48
Epoch  150  loss  1.8761923663266318 correct 49
Epoch  160  loss  1.4173484353810646 correct 47
Epoch  170  loss  1.7798601591536931 correct 48
Epoch  180  loss  1.3904857068378456 correct 48
Epoch  190  loss  3.2292068291259115 correct 49
Epoch  200  loss  0.19123106502922493 correct 50
Epoch  210  loss  0.5035749475711648 correct 48
Epoch  220  loss  0.9857686331036803 correct 47
Epoch  230  loss  0.48555122855141 correct 48
Epoch  240  loss  0.06766357027154071 correct 49
Epoch  250  loss  1.271429219064511 correct 50
Epoch  260  loss  2.1894154169780613 correct 48
Epoch  270  loss  0.16113670807396646 correct 49
Epoch  280  loss  0.3692627521906683 correct 49
Epoch  290  loss  1.0363164845667228 correct 50
Epoch  300  loss  1.190568401035131 correct 49
Epoch  310  loss  3.1654294815574886 correct 49
Epoch  320  loss  1.087325170375514 correct 49
Epoch  330  loss  0.6148179851882811 correct 50
Epoch  340  loss  0.3256727332626702 correct 49
Epoch  350  loss  1.0246646403283775 correct 48
Epoch  360  loss  0.713994353537338 correct 48
Epoch  370  loss  0.5156875915997311 correct 50
Epoch  380  loss  0.13607929234998273 correct 47
Epoch  390  loss  0.30662181121893356 correct 50
Epoch  400  loss  0.03951774956110316 correct 49
Epoch  410  loss  0.4537964885972017 correct 50
Epoch  420  loss  0.1256113199075179 correct 50
Epoch  430  loss  0.11111104207713479 correct 46
Epoch  440  loss  0.7635914093175744 correct 50
Epoch  450  loss  0.6455332162035148 correct 49
Epoch  460  loss  1.5897639592699586 correct 49
Epoch  470  loss  1.1164055534133386 correct 47
Epoch  480  loss  1.1667183488658617 correct 47
Epoch  490  loss  0.9791359518204212 correct 49
```