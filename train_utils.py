import pandas as pd

def select_few_shot_target(target_df, scenario, num_few_shot):
    """
    Select few-shot target samples based on RACE_HISP.
    """
    df = target_df.copy()
    df["RACE_HISP"] = df["RACE_HISP"].astype(int)

    if scenario.lower() == 'all':
        groups = sorted(df["RACE_HISP"].unique())
    else:
        mapping = {'nhw':[2,3], 'nha':[1,3], 'hisp':[1,2]}
        groups = mapping.get(scenario.lower(), [])

    samples = []
    for g in groups:
        grp = df[df["RACE_HISP"]==g]
        k   = min(num_few_shot, len(grp))
        if k>0:
            samples.append(grp.sample(n=k, random_state=42))
    return pd.concat(samples) if samples else df.iloc[0:0]