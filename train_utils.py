import pandas as pd

def select_few_shot_target(target_df, scenario, num_few_shot):
    """
    Select few-shot target samples based on RACE_HISP groups.
    
    - If scenario == 'all': select up to num_few_shot samples from each unique RACE_HISP group.
    - Otherwise (e.g., 'nhw', 'nha', 'hisp'):
         Use a predefined mapping:
            'nhw'  -> select from groups 2 and 3.
            'nha'  -> select from groups 1 and 3.
            'hisp' -> select from groups 1 and 2.
    """
    target_df = target_df.copy()
    target_df["RACE_HISP"] = target_df["RACE_HISP"].astype(int)
    if scenario.lower() == 'all':
        groups = sorted(target_df["RACE_HISP"].unique())
    else:
        mapping = {
            'nhw': [2, 3],
            'nha': [1, 3],
            'hisp': [1, 2]
        }
        groups = mapping.get(scenario.lower(), [])
    few_shot_samples = []
    for g in groups:
        group_samples = target_df[target_df["RACE_HISP"] == g]
        sample_size = min(num_few_shot, len(group_samples))
        if sample_size > 0:
            few_shot_samples.append(group_samples.sample(n=sample_size, random_state=42))
    if few_shot_samples:
        return pd.concat(few_shot_samples)
    else:
        return target_df.iloc[0:0]  # return empty DataFrame if no groups found