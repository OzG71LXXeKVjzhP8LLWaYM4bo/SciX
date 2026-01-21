"""Shannon Entropy feature engineering for polymer sequences."""

import re
from math import log2
from typing import Optional

import numpy as np
import pandas as pd


def parse_blocks(sequence: str) -> list[dict]:
    """
    Parse a block sequence into component blocks.

    Examples:
        "ABC" -> [{'monomers': ['A', 'B', 'C'], 'counts': {}}]
        "(A50C20)(B30)" -> [
            {'monomers': ['A', 'C'], 'counts': {'A': 50, 'C': 20}},
            {'monomers': ['B'], 'counts': {'B': 30}}
        ]
    """
    sequence = str(sequence).strip()

    # Check if it's a simple random sequence (e.g., "ABC", "AB")
    if "(" not in sequence:
        monomers = list(sequence)
        return [{"monomers": monomers, "counts": {}}]

    # Parse block notation: (A50C20)(B30) etc.
    blocks = []
    block_pattern = r"\(([^)]+)\)"
    matches = re.findall(block_pattern, sequence)

    for block_str in matches:
        # Parse monomer-count pairs: A50, C20, B12.5, etc.
        monomer_pattern = r"([A-Z])(\d+(?:\.\d+)?)"
        pairs = re.findall(monomer_pattern, block_str)

        monomers = [p[0] for p in pairs]
        counts = {p[0]: float(p[1]) for p in pairs}

        blocks.append({"monomers": monomers, "counts": counts})

    return blocks


def is_random_copolymer(sequence: str) -> bool:
    """Check if sequence represents a random copolymer (no block structure)."""
    return "(" not in str(sequence)


def composition_entropy(row: pd.Series) -> float:
    """
    Calculate Shannon entropy from monomer compositions.

    H = -sum(p_i * log2(p_i)) for all non-zero compositions.
    """
    compositions = [
        row.get("composition_A", 0),
        row.get("composition_B1", 0),
        row.get("composition_B2", 0),
        row.get("composition_C", 0),
    ]

    # Filter out zero/nan compositions
    compositions = [c for c in compositions if c and c > 0]

    if len(compositions) <= 1:
        return 0.0

    # Normalize to ensure sum = 1
    total = sum(compositions)
    if total == 0:
        return 0.0

    probs = [c / total for c in compositions]

    # Shannon entropy
    entropy = -sum(p * log2(p) for p in probs if p > 0)
    return entropy


def block_entropy(sequence: str) -> float:
    """
    Calculate entropy based on block arrangement.

    Higher entropy for more, equally-sized blocks.
    Lower entropy for fewer or unequally-sized blocks.
    """
    blocks = parse_blocks(sequence)

    if len(blocks) <= 1:
        return 0.0

    # Calculate block sizes (sum of monomer counts in each block)
    block_sizes = []
    for block in blocks:
        if block["counts"]:
            size = sum(block["counts"].values())
        else:
            # For random sequences without counts, use monomer count
            size = len(block["monomers"])
        block_sizes.append(size)

    total_size = sum(block_sizes)
    if total_size == 0:
        return 0.0

    # Probability distribution over blocks
    probs = [s / total_size for s in block_sizes]

    # Shannon entropy over block distribution
    entropy = -sum(p * log2(p) for p in probs if p > 0)
    return entropy


def sequence_entropy(row: pd.Series) -> float:
    """
    Calculate entropy based on sequence type and composition.

    For random copolymers: Use composition entropy (high randomness).
    For block copolymers: Use modified entropy based on block structure.
    """
    sequence = str(row.get("block_sequence_theoretical", ""))

    if is_random_copolymer(sequence):
        # Random copolymer: maximum randomness for given composition
        return composition_entropy(row)

    # Block copolymer: reduced entropy due to block structure
    blocks = parse_blocks(sequence)
    n_blocks = len(blocks)

    # Calculate intra-block entropy (mixing within blocks)
    intra_block_entropies = []
    for block in blocks:
        monomers = block["monomers"]
        if len(monomers) > 1 and block["counts"]:
            # Multiple monomers in block - some mixing
            counts = list(block["counts"].values())
            total = sum(counts)
            probs = [c / total for c in counts]
            h = -sum(p * log2(p) for p in probs if p > 0)
            intra_block_entropies.append(h)
        else:
            # Single monomer in block - no mixing
            intra_block_entropies.append(0.0)

    # Weight by block size
    if intra_block_entropies:
        avg_intra_entropy = np.mean(intra_block_entropies)
    else:
        avg_intra_entropy = 0.0

    # Overall sequence entropy: combination of block structure and intra-block mixing
    # More blocks = more ordered = less overall entropy
    structural_penalty = 1.0 / n_blocks if n_blocks > 0 else 1.0

    comp_entropy = composition_entropy(row)
    seq_entropy = structural_penalty * avg_intra_entropy + (1 - structural_penalty) * comp_entropy * 0.5

    return seq_entropy


def randomness_score(row: pd.Series) -> float:
    """
    Combined randomness metric.

    For random copolymers (1 block): High score based on composition entropy.
    For block copolymers: Lower score based on block structure.
    """
    sequence = str(row.get("block_sequence_theoretical", ""))
    n_blocks = row.get("Number of blocks", 1)

    comp_ent = composition_entropy(row)
    blk_ent = block_entropy(sequence)

    if n_blocks == 1:
        # Random copolymer - full compositional randomness
        return comp_ent
    else:
        # Block copolymer - weighted combination
        # More blocks = lower randomness
        weight = 1.0 / n_blocks
        return weight * comp_ent + (1 - weight) * blk_ent * 0.5


def add_entropy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all entropy-based features to the dataframe."""
    df = df.copy()

    df["composition_entropy"] = df.apply(composition_entropy, axis=1)
    df["block_entropy"] = df["block_sequence_theoretical"].apply(block_entropy)
    df["sequence_entropy"] = df.apply(sequence_entropy, axis=1)
    df["randomness_score"] = df.apply(randomness_score, axis=1)

    return df


def get_composition_features() -> list[str]:
    """Return list of composition feature names."""
    return [
        "composition_A",
        "composition_B1",
        "composition_B2",
        "composition_C",
    ]


def get_entropy_features() -> list[str]:
    """Return list of entropy feature names."""
    return [
        "composition_entropy",
        "block_entropy",
        "sequence_entropy",
        "randomness_score",
    ]


def get_structural_features() -> list[str]:
    """Return list of structural feature names (non-composition, non-entropy)."""
    return [
        "Number of blocks",
        "dpn",
        "Dispersity",
        "cLogP_predicted",
        "Target",   # Target molecular weight
        "NMR",      # Actual MW by NMR spectroscopy
        "GPC",      # Actual MW by gel permeation chromatography
    ]


def get_feature_sets() -> dict[str, list[str]]:
    """
    Return feature sets for experiments.

    - Set A (Composition): Base features + compositions
    - Set B (Entropy): Base features + entropy metrics (no raw compositions)
    - Set C (Combined): All features
    """
    structural = get_structural_features()
    composition = get_composition_features()
    entropy = get_entropy_features()

    return {
        "composition": structural + composition,
        "entropy": structural + entropy,
        "combined": structural + composition + entropy,
    }


if __name__ == "__main__":
    # Test feature extraction
    test_sequences = [
        "ABC",  # Random copolymer
        "(A50C20)(B30)",  # 2-block
        "(A25C10)(B30)(A25C10)",  # 3-block
        "(A12.5)(B7.5C5)(A12.5)(B7.5C5)",  # 4-block
        "AB",  # Simple random
    ]

    print("Block parsing tests:")
    for seq in test_sequences:
        blocks = parse_blocks(seq)
        print(f"  {seq} -> {blocks}")

    # Test with sample data
    print("\nEntropy calculations:")
    test_row = pd.Series({
        "composition_A": 0.5,
        "composition_B1": 0.3,
        "composition_B2": 0.0,
        "composition_C": 0.2,
        "block_sequence_theoretical": "ABC",
        "Number of blocks": 1,
    })

    print(f"  Random ABC:")
    print(f"    Composition entropy: {composition_entropy(test_row):.4f}")
    print(f"    Block entropy: {block_entropy('ABC'):.4f}")
    print(f"    Sequence entropy: {sequence_entropy(test_row):.4f}")
    print(f"    Randomness score: {randomness_score(test_row):.4f}")

    test_row2 = pd.Series({
        "composition_A": 0.5,
        "composition_B1": 0.3,
        "composition_B2": 0.0,
        "composition_C": 0.2,
        "block_sequence_theoretical": "(A50C20)(B30)",
        "Number of blocks": 2,
    })

    print(f"\n  Block (A50C20)(B30):")
    print(f"    Composition entropy: {composition_entropy(test_row2):.4f}")
    print(f"    Block entropy: {block_entropy('(A50C20)(B30)'):.4f}")
    print(f"    Sequence entropy: {sequence_entropy(test_row2):.4f}")
    print(f"    Randomness score: {randomness_score(test_row2):.4f}")
