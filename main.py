from src.multiple_regression import MultipleRegression
from src.autoregression_solve import AutoregressionSolve

variant = 3
number_known_values = 2
number_of_predictions = 10
regression = MultipleRegression

if __name__ == "__main__":
    solve = AutoregressionSolve()
    solve.set_variant(variant)
    solve.set_number_known_values(number_known_values)
    solve.set_number_of_predictions(number_of_predictions)
    solve.set_regression(MultipleRegression)

    quality_of_autoregression = solve.execute()
    print(f"quality_of_autoregression: {quality_of_autoregression}")
