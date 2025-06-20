from mypackages.eRDF import DataProcessor
import pandas as pd
import os
import numpy as np 
import math

def test_calculate_fq_gq_against_reference(dp: DataProcessor, reference_file: str) -> None:
    """
    Compares the outputs of calculate_fq_gq() to a saved reference.

    Args:
        dp (DataProcessor): An instance of the DataProcessor class.
        reference_file (str): Path to the .npz file with reference fq_sq, gq, fqfit, iqfit.

    Raises:
        AssertionError: If any of the outputs differ beyond floating point tolerance.
    """
    # Load reference
    ref = np.load(reference_file)

    # Get new results
    fq_sq, gq, fqfit, iqfit = dp.calculate_fq_gq()

    # Compare arrays
    assert np.allclose(fq_sq, ref["fq_sq"]), "Mismatch in fq_sq"
    assert np.allclose(gq, ref["gq"]), "Mismatch in gq"
    assert np.isclose(fqfit, ref["fqfit"]), f"Mismatch in fqfit: {fqfit} vs {ref['fqfit']}"
    assert np.isclose(iqfit, ref["iqfit"]), f"Mismatch in iqfit: {iqfit} vs {ref['iqfit']}"

    print("✅ calculate_fq_gq() passed the test against reference output.")

path = r"C:\Users\seccolev\data_processing\data\processed\ePDF\Au\Au_start.csv"

df1 = pd.read_csv(path, header = None)

data = df1[0].values

start = int(data.shape[0]*0.03)
end =  int(data.shape[0]*0.8)

ds = (0.00743649587727647)/(2*math.pi)

Au =  {1: [79, 1]}

dp = DataProcessor(data, 0.8, None, start, end, ds, Au, 0)
fq_sq_new, gq_new, fqfit_new, iqfit_new = dp.calculate_fq_gq()

#np.savez("reference_output.npz", fq_sq=fq_sq_old, gq=gq_old, fqfit=fqfit_old, iqfit=iqfit_old)

# Assuming you already have dp = DataProcessor(...)
test_calculate_fq_gq_against_reference(dp, "reference_output.npz")
