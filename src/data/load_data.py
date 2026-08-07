"""
Load raw Home Credit Default Risk tables from data/raw/.

Expected files (from Kaggle):
    application_train.csv / application_test.csv
    bureau.csv
    bureau_balance.csv
    previous_application.csv
    installments_payments.csv
    credit_card_balance.csv
    POS_CASH_balance.csv
"""
from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def load_application_data(split: str = "train") -> pd.DataFrame:
    """Load application_train.csv or application_test.csv."""
    filename = f"application_{split}.csv"
    path = RAW_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"{filename} not found in {RAW_DATA_DIR}. "
            "Download the dataset from Kaggle and place it there."
        )
    return pd.read_csv(path)


def load_bureau_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / "bureau.csv")


def load_previous_application_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / "previous_application.csv")


def load_installments_data() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_DIR / "installments_payments.csv")
