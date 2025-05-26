import unittest
import torch
import sys
import os

# Add the root directory to sys.path to allow importing tensor_ops
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensor_ops import TensorOps

class TestTensorOps(unittest.TestCase):

    # --- Test Arithmetic Operations ---
    def test_add_tensor_tensor(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]])
        t2 = torch.tensor([[5., 6.], [7., 8.]])
        expected = torch.tensor([[6., 8.], [10., 12.]])
        result = TensorOps.add(t1, t2)
        self.assertTrue(torch.equal(result, expected))

    def test_add_tensor_scalar(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]])
        scalar = 10.
        expected = torch.tensor([[11., 12.], [13., 14.]])
        result = TensorOps.add(t1, scalar)
        self.assertTrue(torch.equal(result, expected))

    def test_add_type_error(self):
        t1 = torch.tensor([1., 2.])
        with self.assertRaises(TypeError):
            TensorOps.add(t1, "not_a_tensor_or_scalar") # type: ignore

    def test_divide_tensor_tensor(self):
        t1 = torch.tensor([[10., 20.], [30., 40.]])
        t2 = torch.tensor([[2., 5.], [3., 4.]])
        expected = torch.tensor([[5., 4.], [10., 10.]])
        result = TensorOps.divide(t1, t2)
        self.assertTrue(torch.equal(result, expected))

    def test_divide_tensor_scalar(self):
        t1 = torch.tensor([[10., 20.], [30., 40.]])
        scalar = 10.
        expected = torch.tensor([[1., 2.], [3., 4.]])
        result = TensorOps.divide(t1, scalar)
        self.assertTrue(torch.equal(result, expected))

    def test_divide_by_zero_scalar(self):
        t1 = torch.tensor([[10., 20.], [30., 40.]])
        scalar_zero = 0
        with self.assertRaises(ValueError): # As per TensorOps.divide implementation
            TensorOps.divide(t1, scalar_zero)

    def test_divide_by_zero_tensor(self):
        t1 = torch.tensor([[10., 20.], [30., 40.]])
        t_zero = torch.tensor([[1., 0.], [3., 1.]])
        # TensorOps.divide logs a warning and returns inf/nan from torch.divide
        # We expect torch.divide's behavior.
        expected_output = torch.divide(t1, t_zero) # This will have inf
        result = TensorOps.divide(t1, t_zero)
        self.assertTrue(torch.equal(result, expected_output))
        # Consider capturing logs here if strict warning check is needed.

    # --- Test Matrix and Dot Operations ---
    def test_matmul_valid(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]])
        t2 = torch.tensor([[5., 6.], [7., 8.]])
        expected = torch.matmul(t1, t2)
        result = TensorOps.matmul(t1, t2)
        self.assertTrue(torch.equal(result, expected))

    def test_matmul_shape_mismatch(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]]) # 2x2
        t2_wrong_shape = torch.tensor([[5., 6., 7.], [8., 9., 10.]]) # 2x3, but matmul t1@t2 needs t2 to be 2xN
        # This specific case is fine, t1.shape[1] == t2.shape[0] is not met for t1@t2_wrong_shape
        # if t2_wrong_shape = torch.tensor([[1.,2.],[3.,4.],[5.,6.]]) # 3x2, this would fail
        t2_fail = torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]) # 3x3, this would fail for 2x2 @ 3x3
        with self.assertRaises(ValueError): # As per TensorOps.matmul specific check for 2D
             TensorOps.matmul(t1, t2_fail)

    def test_matmul_ndim_error(self):
        t1 = torch.tensor(1.) # 0-dim
        t2 = torch.tensor([1.,2.]) # 1-dim
        with self.assertRaises(ValueError): # As per TensorOps.matmul ndim check
            TensorOps.matmul(t1,t2)


    # --- Test Reduction Operations ---
    def test_sum_all_elements(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]])
        expected = torch.tensor(10.)
        result = TensorOps.sum(t1)
        self.assertTrue(torch.equal(result, expected))

    def test_sum_along_dimension(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]])
        # Sum along dim 0
        expected_dim0 = torch.tensor([4., 6.])
        result_dim0 = TensorOps.sum(t1, dim=0)
        self.assertTrue(torch.equal(result_dim0, expected_dim0))
        # Sum along dim 1
        expected_dim1 = torch.tensor([3., 7.])
        result_dim1 = TensorOps.sum(t1, dim=1)
        self.assertTrue(torch.equal(result_dim1, expected_dim1))

    def test_sum_keepdim(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]])
        expected_dim0_keepdim = torch.tensor([[4., 6.]])
        result_dim0_keepdim = TensorOps.sum(t1, dim=0, keepdim=True)
        self.assertTrue(torch.equal(result_dim0_keepdim, expected_dim0_keepdim))

    # --- Existing Power and Log tests follow ---
    def test_power_scalar_exponent(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]])
        exponent = 2.0
        expected = torch.tensor([[1., 4.], [9., 16.]])
        result = TensorOps.power(t1, exponent)
        self.assertTrue(torch.equal(result, expected))

        t2 = torch.tensor([1, 2, 3])
        exponent_int = 3
        expected_int = torch.tensor([1, 8, 27])
        result_int = TensorOps.power(t2, exponent_int)
        self.assertTrue(torch.equal(result_int, expected_int.float())) # torch.pow promotes to float

    def test_power_tensor_exponent(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]])
        t_exponent = torch.tensor([[2., 3.], [1., 2.]])
        expected = torch.tensor([[1., 8.], [3., 16.]])
        result = TensorOps.power(t1, t_exponent)
        self.assertTrue(torch.equal(result, expected))

    def test_power_type_error(self):
        t1 = torch.tensor([1., 2.])
        with self.assertRaises(TypeError):
            TensorOps.power(t1, "not_a_number_or_tensor") # type: ignore

        with self.assertRaises(TypeError):
            TensorOps.power("not_a_tensor", 2.0) # type: ignore

    def test_power_runtime_error_shape_mismatch(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]])
        t_exponent_wrong_shape = torch.tensor([2., 3.]) # Shape mismatch for element-wise
        with self.assertRaises(RuntimeError): # PyTorch broadcasts, but if explicit check was added, this would be ValueError
            TensorOps.power(t1, t_exponent_wrong_shape)

    def test_log_valid_inputs(self):
        t1 = torch.tensor([[1., 2.], [3., 4.]])
        expected = torch.log(t1) # Use torch.log directly for expected value
        result = TensorOps.log(t1)
        self.assertTrue(torch.equal(result, expected))

        t2 = torch.tensor([10., 20., 30.])
        expected2 = torch.log(t2)
        result2 = TensorOps.log(t2)
        self.assertTrue(torch.equal(result2, expected2))

    def test_log_non_positive_inputs(self):
        t_with_zero = torch.tensor([1., 0., 3.])
        # Expect NaN for log(0) and -inf for log(negative)
        # torch.log(0) is -inf
        # torch.log(-1) is nan
        expected_zero = torch.log(t_with_zero) # Let torch.log define the exact output (-inf, nan)
        
        # We are primarily testing that our TensorOps.log runs and produces what torch.log would.
        # The warning for non-positive values is logged, not asserted in output here.
        # We could capture warnings if needed, but for now, let's check output.
        result_zero = TensorOps.log(t_with_zero)
        self.assertTrue(torch.equal(result_zero, expected_zero))

        t_with_negative = torch.tensor([1., -2., 3.])
        expected_negative = torch.log(t_with_negative)
        result_negative = TensorOps.log(t_with_negative)
        # Comparing NaNs: torch.equal treats NaNs as equal if they are in the same position.
        self.assertTrue(torch.equal(result_negative, expected_negative)) 

    def test_log_type_error(self):
        with self.assertRaises(TypeError):
            TensorOps.log("not_a_tensor") # type: ignore

if __name__ == '__main__':
    unittest.main()
