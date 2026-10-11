def main():

    from src.data import load_data, clean_data, time_split
    from src.baselines import forecast_by_group
    from src.inventory import simulate_inventory
    from src.selection import (load_forecasts, combine_forecasts,
                            build_selected_plan, plan_summary, top_risks)

    df = clean_data(load_data("data/raw/train.csv"))
    train, val = time_split(df, horizon=28)

    pool = combine_forecasts(
        forecast_by_group(train, "naive"),
        forecast_by_group(train, "seasonal_naive"),
        forecast_by_group(train, "moving_average"),
        load_forecasts("data/processed/forecasts_classical.csv"),
    )
    plan = build_selected_plan(val, pool, simulate_inventory(train))
    print(plan_summary(plan)); 
    print(top_risks(plan))
    plan.to_csv("data/processed/inventory_plan.csv", index=False)
    print("Saved plan to data/processed/inventory_plan.csv")

if __name__ == "__main__":
    main()