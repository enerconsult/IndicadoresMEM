from pydataxm.pydataxm import ReadDB
import datetime as dt

def main():
    # Initialize the API object
    # No arguments needed for public access
    objetoAPI = ReadDB()

    # Define dates (last 30 days)
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=30)

    print("Fetching 'Precio_Bolsa_Nacional' (National Spot Price)...")

    try:
        # Request data for 'PrecBolsNaci' (Precio Bolsa Nacional)
        # Filters: Entity type 'Sistema' (System-wide)
        # Date range: 30 days
        df = objetoAPI.request_data(
            "PrecBolsNaci", 
            "Sistema", 
            start_date.date(), 
            end_date.date()
        )

        if df is not None and not df.empty:
            print("\n---------- Data Retrieved Successfully ----------")
            print(f"Metric: Precio Bolsa Nacional")
            print(f"Total records: {len(df)}")
            print("\nLast 5 records:")
            print(df.tail())
            print("\nColumns available:", df.columns.tolist())
        else:
            print("No data returned for 'PrecBolsNaci'.")
            
        # Example of how to see other available variables
        print("\n\n---------- Available Metrics (Sample) ----------")
        print("Fetching list of all available metrics...")
        df_vars = objetoAPI.get_collections()
        
        # Define some interesting keywords to filter (using roots to avoid accent issues)
        keywords = ['Deman', 'Genera', 'Precio', 'Embalse']
        
        for key in keywords:
            print(f"\n--- Metrics containing '{key}' ---")
            # Filter and show ID, Name, Entity, MaxDays
            filtered = df_vars[df_vars['MetricName'].str.contains(key, case=False, na=False)]
            print(filtered[['MetricId', 'MetricName', 'Entity', 'MaxDays']].head(5).to_string(index=False))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
