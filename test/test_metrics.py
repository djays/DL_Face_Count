import unittest
import numpy as np
import metrics

class TestMetrics(unittest.TestCase):

    def test_mae(self):
        a = np.array([3, 1, 5])
        b = np.array([1, 3, 5])
        self.assertEqual("%.2f" % metrics.calc_mae(a, b), "1.33")



if __name__ == '__main__':
    unittest.main()