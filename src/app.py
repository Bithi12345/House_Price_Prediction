import joblib 
import pandas as pd

test_data = {
    'No. Beds': 2.0,
    'No. Baths': 2.0,
    'Area': 735.0,
    'Type_n': 0,
    'Region_n': 3,
    'Sub-region_n': 432
}

sample_house_price_df = pd.DataFrame([test_data])

print("Sample test data ")
print(sample_house_price_df)