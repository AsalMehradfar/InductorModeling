import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(df: pd.DataFrame, 
                  train_ratio=0.8, 
                  val_ratio=0.1, 
                  test_ratio=0.1, 
                  random_seed=42):
    """
    Splits the full DataFrame into train/val/test sets based on ratios.
    
    Returns:
        train_df, val_df, test_df
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Step 1: Train vs (Val+Test)
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_seed,
        shuffle=True
    )

    # Step 2: Val vs Test (split temp_df equally or per ratios)
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        random_state=random_seed,
        shuffle=True
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

