import pandas as pd
def count_value(df,  value):
    # Calculate counts and percentages of 'na' values for each column
    na_counts = df.isin([value]).sum()
    na_percentages = (na_counts / len(df)) * 100

    # Combine counts and percentages into a DataFrame
    na_summary = pd.DataFrame({
        'Count': na_counts,
        'Percentage': na_percentages
    })

    return na_summary
