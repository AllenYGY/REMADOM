import json
import os
import numpy as np
import pytest

from remadom.data.export import export_aligned_arrays_blockwise

@pytest.mark.skip(reason="Requires small .h5ad or .zarr test data; provide fixture in project.")
def test_export_and_manifest(tmp_path):
    # Placeholder: integrate with real AnnData fixture in your project
    pass