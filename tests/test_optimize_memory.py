import pandas as pd
import numpy as np

from src.data_processing import optimize_memory


def test_optimize_memory_reduces_memory_and_changes_types():
    df = pd.DataFrame({
        'int_col': np.array([1, 2, 3, 4], dtype=np.int64),
        'float_col': np.array([1.0, 2.5, 3.1, 4.0], dtype=np.float64),
        'obj_col': ['a', 'b', 'a', 'c'],
    })

    before = df.memory_usage(deep=True).sum()
    out = optimize_memory(df.copy())
    after = out.memory_usage(deep=True).sum()

    assert after <= before
    # integer downcast expected (to any smaller int dtype)
    assert str(out['int_col'].dtype).startswith('int')
    # float should be at most float32
    assert str(out['float_col'].dtype) in ('float32', 'float64')
    # object -> category expected due to low cardinality
    assert str(out['obj_col'].dtype).startswith('category')
