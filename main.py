from src.multiple_regression import MultipleRegression
from src.autoregression_solver import AutoregressionSolver

variant = 3
number_known_values = 2
number_of_predictions = 10
regression = MultipleRegression

if __name__ == "__main__":
    solver = AutoregressionSolver()
    solver.set_variant(variant)
    solver.set_number_known_values(number_known_values)
    solver.set_number_of_predictions(number_of_predictions)
    solver.set_regression(MultipleRegression)

    quality_of_autoregression = solver.execute()
    print(f"quality_of_autoregression: {quality_of_autoregression}")
