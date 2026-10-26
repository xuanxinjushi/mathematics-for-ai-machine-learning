"""
Verify that element-wise exponential increases matrix rank.

This script verifies the claim that for a rank-1 matrix M = x x^T where x is a 
random vector, applying element-wise exponential e^M typically increases the 
rank significantly. For n=100, we expect rank(e^M) ≈ 17-18.
"""

import numpy as np

def verify_rank_increase(n=100, num_trials=100, seed=42):
    """
    Verify that element-wise exponential increases rank of a rank-1 matrix.
    
    Parameters:
    -----------
    n : int
        Dimension of the random vector x (default: 100)
    num_trials : int
        Number of random trials to run (default: 100)
    seed : int
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict : Dictionary containing statistics about rank(e^M)
    """
    np.random.seed(seed)
    
    ranks = []
    
    print(f"Verifying rank increase for n={n} with {num_trials} trials...")
    print(f"{'='*60}")
    
    for trial in range(num_trials):
        # Generate random vector x
        x = np.random.randn(n)
        
        # Form rank-1 matrix M = x x^T
        M = np.outer(x, x)
        
        # Verify original rank is 1
        rank_M = np.linalg.matrix_rank(M)
        assert rank_M == 1, f"Expected rank(M) = 1, got {rank_M}"
        
        # Apply element-wise exponential
        e_M = np.exp(M)
        
        # Compute rank of e^M
        rank_eM = np.linalg.matrix_rank(e_M)
        ranks.append(rank_eM)
        
        if (trial + 1) % 10 == 0:
            print(f"Trial {trial + 1}/{num_trials}: rank(e^M) = {rank_eM}")
    
    # Compute statistics
    ranks = np.array(ranks)
    mean_rank = np.mean(ranks)
    std_rank = np.std(ranks)
    min_rank = np.min(ranks)
    max_rank = np.max(ranks)
    median_rank = np.median(ranks)
    
    print(f"\n{'='*60}")
    print(f"Results for n = {n}:")
    print(f"  Original rank(M) = 1")
    print(f"  Rank of e^M:")
    print(f"    Mean:   {mean_rank:.2f}")
    print(f"    Median: {median_rank:.2f}")
    print(f"    Std:    {std_rank:.2f}")
    print(f"    Min:    {min_rank}")
    print(f"    Max:    {max_rank}")
    print(f"\n  Expected: rank(e^M) ≈ 17-18")
    print(f"  Observed: rank(e^M) ≈ {mean_rank:.1f} ± {std_rank:.1f}")
    
    # Check if result matches expectation
    if 16 <= mean_rank <= 19:
        print(f"\n✓ Verification PASSED: Mean rank is in expected range [16, 19]")
    else:
        print(f"\n✗ Verification FAILED: Mean rank {mean_rank:.2f} is outside expected range [16, 19]")
    
    return {
        'n': n,
        'num_trials': num_trials,
        'original_rank': 1,
        'mean_rank': mean_rank,
        'median_rank': median_rank,
        'std_rank': std_rank,
        'min_rank': min_rank,
        'max_rank': max_rank,
        'all_ranks': ranks
    }


if __name__ == "__main__":
    # Run verification with default parameters
    results = verify_rank_increase(n=100, num_trials=100)
    
    # Optional: Test with different values of n
    print(f"\n{'='*60}")
    print("Testing with different values of n:")
    print(f"{'='*60}")
    
    for n in [50, 100, 200]:
        results_n = verify_rank_increase(n=n, num_trials=50, seed=42)
        print()
