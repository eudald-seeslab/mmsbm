def get_activity(df, act_items):
    return df.merge(act_items, on="item")
