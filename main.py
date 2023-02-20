from src.multiple_regression import MultipleRegression
from src.autoregression_solver import AutoregressionSolver

variant = 22
number_known_values = 10
number_of_predictions = 10
regression = MultipleRegression


def get_quality_of_autoregression(solver: AutoregressionSolver):
    quality_of_autoregression = 0
    try:
        quality_of_autoregression = solver.execute()
    except (Exception,):
        pass
    return quality_of_autoregression


if __name__ == "__main__":
    count = 100
    avg = 0
    for _ in range(count):
        solver = AutoregressionSolver()
        solver.set_variant(variant)
        solver.set_number_known_values(number_known_values)
        solver.set_number_of_predictions(number_of_predictions)
        solver.set_regression(regression)

        quality_of_autoregression = get_quality_of_autoregression(solver)
        avg += quality_of_autoregression

    avg /= count
    percent = round(avg*100, 2)
    print(f"Качество авторегрессии: {percent}%")
