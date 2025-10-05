import unittest
import numpy as np
import torch
from derivatives import *

class TestDerivative(unittest.TestCase):
    def test_derivative_basic(self):
        u = [1, 2, 3, 4, 5, 6, 7]
        h = 1
        boundary = [0, 0, 0, 0, 0, 0, 0]

        du_true = [1, 1, 1, 1, 1, 1, 1]

        du_np = derivative(u=np.array(u),
                           h=h,
                           boundary=np.array(boundary),
                           numpy_=True)
        
        du_t = derivative(u=torch.tensor(u),
                           h=h,
                           boundary=torch.tensor(boundary),
                           numpy_=False)
        
        np.testing.assert_equal(du_np, np.array(du_true))
        assert torch.equal(du_t, torch.tensor(du_true))

    def test_derivative_bounded_1(self):
        u = [1, 2, 3, 4, 5, 6, 7]
        h = 1
        boundary = [1, 0, 0, 1, 0, 0, 1]

        du_true = [0, 1, 1, 0, 1, 1, 0]

        du_np = derivative(u=np.array(u),
                           h=h,
                           boundary=np.array(boundary),
                           numpy_=True)
        
        du_t = derivative(u=torch.tensor(u),
                           h=h,
                           boundary=torch.tensor(boundary),
                           numpy_=False)
        
        np.testing.assert_equal(du_np, np.array(du_true))
        assert torch.equal(du_t, torch.tensor(du_true))

    def test_derivative_bounded_2(self):
        u = [1, 2, 3, 4, 5, 6, 7]
        h = 1
        boundary = [1, 0, 0, 0, 1, 0, 1]

        du_true = [0, 1, 1, 1, 0, 0, 0]

        du_np = derivative(u=np.array(u),
                           h=h,
                           boundary=np.array(boundary),
                           numpy_=True)
        
        du_t = derivative(u=torch.tensor(u),
                           h=h,
                           boundary=torch.tensor(boundary),
                           numpy_=False)
        
        np.testing.assert_equal(du_np, np.array(du_true))
        assert torch.equal(du_t, torch.tensor(du_true))

    def test_derivative_h_array(self):
        u = [1, 2, 3, 4, 5, 6, 7]
        h = [1, 2, 3, 4, 3, 2, 1]
        boundary = [0, 0, 0, 0, 0, 0, 0]

        du_true = [1, 2/3, 2/5, 2/7, 2/7, 2/5, 1/2]

        du_np = derivative(u=np.array(u),
                           h=np.array(h),
                           boundary=np.array(boundary),
                           numpy_=True)
        
        du_t = derivative(u=torch.tensor(u),
                           h=torch.tensor(h),
                           boundary=torch.tensor(boundary),
                           numpy_=False)
        
        np.testing.assert_equal(du_np, np.array(du_true))
        assert torch.equal(du_t, torch.tensor(du_true))

    def test_derivative_batched_data(self):
        u = [[1, 2, 3, 4, 5, 6, 7],
             [1, 3, 5, 7, 9, 11, 13]]
        h = [[1, 2, 3, 4, 3, 2, 1],
             [1, 1, 1, 1, 1, 1, 1]]
        boundary = [[0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]

        du_true = [[1, 2/3, 2/5, 2/7, 2/7, 2/5, 1/2],
                   [2, 2, 2, 2, 2, 2, 2]]

        du_np = derivative(u=np.array(u),
                           h=np.array(h),
                           boundary=np.array(boundary),
                           numpy_=True)
        
        du_t = derivative(u=torch.tensor(u),
                           h=torch.tensor(h),
                           boundary=torch.tensor(boundary),
                           numpy_=False)
        
        np.testing.assert_equal(du_np, np.array(du_true))
        assert torch.equal(du_t, torch.tensor(du_true))
        
        
class TestGradient(unittest.TestCase):
    def test_gradient_basic(self):
        u = [[1, 2, 3],
             [1, 2, 3],
             [1, 2, 3]]
        h = 1
        boundary = [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]

        gradu_true = [[[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]],
                   [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]]

        gradu_np = grad(u=np.array(u),
                           dx=h,
                           dy=h,
                           boundary=np.array(boundary),
                           numpy_=True)
        
        gradu_t = grad(u=torch.tensor(u),
                           dx=h,
                           dy=h,
                           boundary=torch.tensor(boundary),
                           numpy_=False)
        
        np.testing.assert_equal(gradu_np[0], np.array(gradu_true[0]))
        np.testing.assert_equal(gradu_np[1], np.array(gradu_true[1]))

        assert torch.equal(gradu_t[0], torch.tensor(gradu_true[0]))
        assert torch.equal(gradu_t[1], torch.tensor(gradu_true[1]))

    def test_gradient_bounded(self):
        u = [[1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5],
             [1, 2, 3, 4, 5]]
        h = 1
        boundary = [[0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]]

        gradu_true = ([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]],
                   [[1, 1, 1, 1, 1],
                   [1, 1, 0, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]])

        gradu_np = grad(u=np.array(u),
                           dx=h,
                           dy=h,
                           boundary=np.array(boundary),
                           numpy_=True)
        
        gradu_t = grad(u=torch.tensor(u),
                           dx=h,
                           dy=h,
                           boundary=torch.tensor(boundary),
                           numpy_=False)
        
        np.testing.assert_equal(gradu_np[0], np.array(gradu_true[0]))
        np.testing.assert_equal(gradu_np[1], np.array(gradu_true[1]))

        assert torch.equal(gradu_t[0], torch.tensor(gradu_true[0]))
        assert torch.equal(gradu_t[1], torch.tensor(gradu_true[1]))

    def test_gradient_vector(self):
        u = [[[1, 2, 3],
             [1, 2, 3],
             [1, 2, 3]],
             [[1, 2, 3],
             [1, 2, 3],
             [1, 2, 3]]]
        h = 1
        boundary = [[[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]],
                    [[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]]]

        gradu_true = ([[[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]],
                       [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]],
                       [[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],
                       [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]])

        gradu_np = grad(u=np.array(u),
                           dx=h,
                           dy=h,
                           boundary=np.array(boundary),
                           numpy_=True)
        
        gradu_t = grad(u=torch.tensor(u),
                           dx=h,
                           dy=h,
                           boundary=torch.tensor(boundary),
                           numpy_=False)
        
        np.testing.assert_equal(gradu_np[0], np.array(gradu_true[0]))
        np.testing.assert_equal(gradu_np[1], np.array(gradu_true[1]))

        assert torch.equal(gradu_t[0], torch.tensor(gradu_true[0]))
        assert torch.equal(gradu_t[1], torch.tensor(gradu_true[1]))

if __name__ == '__main__':
    unittest.main()