from omr import classify_densities

def check(vals, expected_answer, expected_status, choices=("A","B","C","D")):
    answer, status, confidence = classify_densities(
        vals, answer_key_mode=False, choices=choices
    )
    assert answer == expected_answer, (vals, answer, status)
    assert status == expected_status, (vals, answer, status)
    return confidence

# Strong obvious mark.
check([0.07, 0.31, 0.06, 0.05], "B", "clear")

# Light mark: old logic could call this unclear.
# It should now choose A because A clearly dominates the rest.
check([0.115, 0.052, 0.048, 0.050], "A", "clear")

# Two genuinely close marks -> multiple.
check([0.228, 0.218, 0.050, 0.047], None, "multiple")

# Another close pair with a small absolute difference -> multiple.
check([0.145, 0.126, 0.046, 0.045], None, "multiple")

# Truly empty row stays blank.
check([0.052, 0.049, 0.050, 0.048], None, "blank")

# More than four choices remains supported.
check([0.05, 0.06, 0.075, 0.19, 0.055, 0.05], "D", "clear",
      choices=("A","B","C","D","E","F"))

print("Dominance fallback tests passed.")
