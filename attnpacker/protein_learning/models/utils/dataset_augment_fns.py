"""Retrieved from https://github.com/MattMcPartlon/AttnPacker"""

from typing import Tuple

from attnpacker.protein_learning.common.data.data_types.protein import Protein
from attnpacker.protein_learning.common.helpers import exists
from attnpacker.protein_learning.features.masking.partition import (
    ChainPartitionGenerator,
)


def impute_cb(decoy: Protein, native: Protein) -> Tuple[Protein, Protein]:
    """Impute beta carbon in data loader"""
    decoy.impute_cb(override=True, exists_ok=True)
    return decoy, native


def partition_chain(
        decoy: Protein, native: Protein, partition_gen: ChainPartitionGenerator = None
) -> Tuple[Protein, Protein]:
    """Partition a chain to appear as a complex"""
    assert exists(partition_gen)
    if not decoy.is_complex:
        _, _, partition = partition_gen.get_chain_partition_info(decoy)
        if len(partition) > 1:
            decoy.make_complex(partition)
            assert decoy.is_complex
    return decoy, native


def len_filter(decoy: Protein, native: Protein, min_len: int = None) -> bool:
    return len(decoy) > min_len
