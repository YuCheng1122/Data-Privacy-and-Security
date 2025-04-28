import pandas as pd
import numpy as np

def generalize_age(age):
    if pd.isnull(age):
        return np.nan
    return (age // 10) * 10  # 年齡泛化成10歲一段

def apply_k_anonymity(df, quasi_identifiers, k=5):
    df_gen = df.copy()

    # Step 1: Generalization
    if 'AGE' in quasi_identifiers:
        df_gen['AGE'] = df_gen['AGE'].apply(generalize_age)

    # Step 2: 小城市標成 'Other'
    if 'CITY' in quasi_identifiers:
        city_counts = df_gen['CITY'].value_counts()
        small_cities = city_counts[city_counts < k].index
        df_gen['CITY'] = df_gen['CITY'].apply(lambda x: 'Other' if x in small_cities else x)

    # Step 3: Group by quasi-identifiers and check group size
    group_sizes = df_gen.groupby(quasi_identifiers).size()
    valid_groups = group_sizes[group_sizes >= k].index

    # Step 4: 過濾掉小於 k 的 group
    def is_valid_group(row):
        return tuple(row[q] for q in quasi_identifiers) in valid_groups

    filtered_df = df_gen[df_gen.apply(is_valid_group, axis=1)]

    return filtered_df
