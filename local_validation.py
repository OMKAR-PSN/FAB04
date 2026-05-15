import numpy as np

from adapters.myteam import Engine


def main():
    rng = np.random.default_rng(42)
    patterns = rng.normal(size=(16, 64))
    agent = Engine(patterns, {"beta": 1.0})

    tests = {}

    clean_query = patterns[0]
    noisy_query = patterns[0] + 0.25 * rng.normal(size=64)
    masked_query = patterns[1].copy()
    masked_query[:20] = 0.0
    extreme_query = patterns[2] + 3.0 * rng.normal(size=64)
    nan_query = patterns[3].copy()
    nan_query[0] = np.nan

    tests["clean"] = agent.check_local_properties(clean_query)
    tests["noisy"] = agent.check_local_properties(noisy_query)
    tests["masked"] = agent.check_local_properties(masked_query)
    tests["extreme"] = agent.check_local_properties(extreme_query)
    nan_result = agent.predict_precision(nan_query)
    tests["nan_fallback"] = {
        "shape_ok": nan_result.shape == (64,),
        "finite_ok": bool(np.all(np.isfinite(nan_result))),
    }

    shuffle = rng.permutation(64)
    shuffled_patterns = patterns[:, shuffle]
    shuffled_agent = Engine(shuffled_patterns, {"beta": 1.0})
    shuffled_query = noisy_query[shuffle]
    pi_original = agent.predict_precision(noisy_query)
    pi_shuffled = shuffled_agent.predict_precision(shuffled_query)

    tests["permutation_invariance"] = {
        "shape_ok": pi_shuffled.shape == (64,),
        "finite_ok": bool(np.all(np.isfinite(pi_shuffled))),
        "same_sort_profile": bool(np.allclose(np.sort(pi_original), np.sort(pi_shuffled), atol=1e-6)),
    }

    print("Local validation summary:")
    for name, result in tests.items():
        print(name, result)


if __name__ == "__main__":
    main()