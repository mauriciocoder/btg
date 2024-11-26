# https://www.freecodecamp.org/learn/scientific-computing-with-python/build-an-arithmetic-formatter-project/build-an-arithmetic-formatter-project
from typing import List


def arithmetic_arranger(problems: List[str], show_answers: bool = False) -> str:
    if len(problems) > 5:
        return "Error: Too many problems."
    first_row = ""
    second_row = ""
    third_row = ""
    fourth_row = ""
    for problem in problems:
        if problem.count("+") != 1 and problem.count("-") != 1:
            return "Error: Operator must be '+' or '-'."
        operand = "+" if problem.count("+") == 1 else "-"
        try:
            num1, num2 = map(int, problem.split(operand))
        except ValueError:
            return "Error: Numbers must only contain digits."
        if num1 >= 1e4 or num2 >= 1e4:
            return "Error: Numbers cannot be more than four digits."
        num1_len = len(str(num1))
        num2_len = len(str(num2))
        problem_row_size = (num1_len if num1 > num2 else num2_len) + 2
        suffix = 4 * " "
        first_row += str(num1).rjust(problem_row_size) + suffix
        second_row += operand + str(num2).rjust(problem_row_size - 1) + suffix
        third_row += problem_row_size * "-" + suffix
        if show_answers:
            fourth_row += (
                str(num1 + num2 if operand == "+" else num1 - num2).rjust(
                    problem_row_size
                )
                + suffix
            )
    rows = first_row.rstrip() + "\n" + second_row.rstrip() + "\n" + third_row.rstrip()
    return rows if not show_answers else rows + "\n" + fourth_row.rstrip()
