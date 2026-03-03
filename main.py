from src.ingestion import load_raw_data
from src.preprocessing import preprocess_jhu_data
from src.feature_engineering import create_national_aggregation
from src.benchmarking import benchmark_models
from src.benchmarking import rolling_cross_validation

def main():
    df = load_raw_data()
    df = preprocess_jhu_data(df)
    national = create_national_aggregation(df)

    cv_mae = rolling_cross_validation(national)
    print(f"\nRolling Cross-Validation MAE: {cv_mae:.2f}")

    results, forecast, residuals, model, lower_ci, upper_ci = benchmark_models(national)

    print("\nMODEL BENCHMARK RESULTS\n")
    print(results)


if __name__ == "__main__":
    main()