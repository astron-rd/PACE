import numpy as np

# Simulated parameters
n_sources = 5
n_channels = 4

# Random temp_prod and local_smear_terms per source, channel
np.random.seed(42)
temp_prod = np.random.randn(n_sources, n_channels)
local_smear_terms = np.random.randn(n_sources, n_channels)
angular_smear = np.random.rand(n_sources, n_channels)  # different per source

# Method A: multiply inside the accumulation loop
buffer_A = np.zeros(n_channels)
for s in range(n_sources):
    for ch in range(n_channels):
        buffer_A[ch] += (
            angular_smear[s, ch] * local_smear_terms[s, ch] * temp_prod[s, ch]
        )

# Method B: try to postpone angular_smear
buffer_B = np.zeros(n_channels)
for s in range(n_sources):
    for ch in range(n_channels):
        buffer_B[ch] += local_smear_terms[s, ch] * temp_prod[s, ch]

# Now multiply once afterward with SOME angular_smear value
# For demonstration, we use the angular_smear of the first source (WRONG if they differ)
buffer_B_wrong = buffer_B * angular_smear[0, :]

# Correct way to postpone: keep per-source contributions, multiply each, then sum
unsmeared_per_source = local_smear_terms * temp_prod
buffer_B_correct = np.sum(unsmeared_per_source * angular_smear, axis=0)

# Print differences
print("Method A:", buffer_A)
print("Method B_wrong (single factor at end):", buffer_B_wrong)
print("Method B_correct (per-source multiply after):", buffer_B_correct)

print("\nError if multiplying once at end:", np.max(np.abs(buffer_A - buffer_B_wrong)))
print(
    "Error if multiplying per-source after:",
    np.max(np.abs(buffer_A - buffer_B_correct)),
)
