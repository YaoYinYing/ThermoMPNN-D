import pytest
import torch

from dataclasses import dataclass, field
from typing import List

from RosettaPy.common import Mutation, Mutant, RosettaPyProteinSequence, Chain

from thermompnn.run import double_mutant2tensor
@pytest.mark.parametrize(
    "mutations, expected_pos, expected_wtAA, expected_mutAA",
    [
        # Example with two mutations:
        (
            [
                # Mutation #1
                # ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
                # G -> index 5, A -> index 0
                # position = 10
                # chain_id is arbitrary here
                Mutation(chain_id="A", position=10, wt_res="G", mut_res="A"),
                # Mutation #2
                # C -> index 1, V -> index 17
                # position = 20
                Mutation(chain_id="A", position=20, wt_res="C", mut_res="V"),
            ],
            # Expected position combos
            [[10, 20]],
            # Expected WT indices (G=5, C=1)
            [[5, 1]],
            # Expected MUT indices (A=0, V=17)
            [[0, 17]],
        )
    ]
)
def test_double_mutant2tensor(mutations, expected_pos, expected_wtAA, expected_mutAA):
    """
    Check that double_mutant2tensor returns the correct tensors
    for exactly two mutations.
    """

    # Create a minimal RosettaPyProteinSequence (not used by the function,
    # but required to initialize Mutant).
    from dataclasses import dataclass

    # We can supply any valid chain data here; the function doesn't use it.
    dummy_protein_seq = RosettaPyProteinSequence(
        chains=[Chain(chain_id="A", sequence="ACDEFGHIKLMNPQRSTVWY")]
    )

    # Create the Mutant object with exactly two mutations
    mutant = Mutant(mutations=mutations, wt_protein_sequence=dummy_protein_seq)

    pos_combos, wtAA, mutAA = double_mutant2tensor(mutant)

    # Convert tensors to Python lists for simple comparison
    assert pos_combos.tolist() == expected_pos
    assert wtAA.tolist() == expected_wtAA
    assert mutAA.tolist() == expected_mutAA
    # Also verify tensor shapes
    assert pos_combos.shape == (1, 2)
    assert wtAA.shape == (1, 2)
    assert mutAA.shape == (1, 2)


@pytest.mark.parametrize(
    "mutations",
    [
        # Less than two
        ([Mutation(chain_id="A", position=1, wt_res="A", mut_res="C")]),
        # More than two
        ([
            Mutation(chain_id="A", position=1, wt_res="A", mut_res="C"),
            Mutation(chain_id="A", position=2, wt_res="C", mut_res="D"),
            Mutation(chain_id="A", position=3, wt_res="E", mut_res="F")
        ]),
    ]
)
def test_double_mutant2tensor_raises_value_error(mutations):
    """
    Check that double_mutant2tensor raises ValueError
    if mutations list is not length 2.
    """
    dummy_protein_seq = RosettaPyProteinSequence(
        chains=[Chain(chain_id="A", sequence="ACDEFGHIKLMNPQRSTVWY")]
    )

    mutant = Mutant(mutations=mutations, wt_protein_sequence=dummy_protein_seq)

    with pytest.raises(ValueError) as exc_info:
        _ = double_mutant2tensor(mutant)
    assert "exactly 2" in str(exc_info.value)
