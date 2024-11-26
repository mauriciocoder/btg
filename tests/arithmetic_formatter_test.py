from exploratory_data_analysis.arithmetic_formatter import arithmetic_arranger


def test_arithmetic_arranger():
    assert (
        arithmetic_arranger(["1 + 1", "1 + 1", "1 + 1", "1 + 1", "1 + 1", "1 + 1"])
        == "Error: Too many problems."
    )
    assert arithmetic_arranger(["1 * 1"]) == "Error: Operator must be '+' or '-'."
    assert arithmetic_arranger(["1 + a"]) == "Error: Numbers must only contain digits."
    assert (
        arithmetic_arranger(["10000 + 1"])
        == "Error: Numbers cannot be more than four digits."
    )
    assert (
        arithmetic_arranger(["3801 - 2", "123 + 49"])
        == "  3801      123\n-    2    +  49\n------    -----"
    )
    assert (
        arithmetic_arranger(["1 + 2", "1 - 9380"])
        == "  1         1\n+ 2    - 9380\n---    ------"
    )
    assert (
        arithmetic_arranger(
            ["32 - 698", "1 - 3801", "45 + 43", "123 + 49", "988 + 40"], True
        )
        == "   32         1      45      123      988\n- 698    - 3801    + 43    +  49    +  40\n-----    ------    ----    -----    -----\n -666     -3800      88      172     1028"
    )
