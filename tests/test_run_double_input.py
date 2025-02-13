import pytest
import os

from RosettaPy.common import Mutant, RosettaPyProteinSequence, Chain, Mutation
from thermompnn import ThermoMPNN

pdb='examples/pdbs/4ajy.pdb'
seq=seq=RosettaPyProteinSequence.from_pdb(pdb)

@pytest.mark.parametrize("mode, mutant", [
    ["epistatic", Mutant(
        [Mutation('B', 1, 'M', 'G'), Mutation('B', 2, 'D', 'I')], # MB1G:DB2I
        seq
    )],
])
def test_run_scorer(mode,mutant, tmp_path):
    save_dir=os.path.join('.',"tests/outputs/examples/")
    os.makedirs(save_dir,exist_ok=True)

    out_file_prefix=os.path.join(save_dir, f"test_run_scorer_{mode}_{mutant.raw_mutant_id}{os.path.basename(pdb)}")
    score=ThermoMPNN(mode=mode, pdb=pdb,out=out_file_prefix, threshold=100).score(mutant)
    assert score is None

