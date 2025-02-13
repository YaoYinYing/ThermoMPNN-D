import pytest
import os

from thermompnn import ThermoMPNN


@pytest.mark.parametrize("mode, pdb", [
    ["single", 'examples/pdbs/1VII.pdb'],
    ["additive", 'examples/pdbs/1VII.pdb'],
    ["epistatic", 'examples/pdbs/4ajy.pdb']
])
def test_run_main(mode, pdb, tmp_path):
    save_dir=os.path.join(tmp_path,"tests/outputs/examples/")
    os.makedirs(save_dir,exist_ok=True)

    out_file_prefix=os.path.join(save_dir, f"test_run_main_{mode}_{os.path.basename(pdb)}")

    ThermoMPNN(mode=mode, pdb=pdb,out=out_file_prefix).process()

