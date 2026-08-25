"""ETL: Load cleaned UCI data into PostgreSQL warehouse."""
import pandas as pd
import sqlalchemy
from pathlib import Path

from src.config import settings


def get_engine():
    """Create SQLAlchemy engine."""
    return sqlalchemy.create_engine(settings.DATABASE_URL)


def create_schema(engine):
    """Run schema SQL files."""
    schema_dir = Path(__file__).parent.parent.parent / "sql" / "schema"
    for sql_file in sorted(schema_dir.glob("*.sql")):
        print(f"  Running {sql_file.name}...")
        engine.execute(sqlalchemy.text(sql_file.read_text()))
    print("  Schema created.")


def load_customers(df: pd.DataFrame, engine):
    """Load customers table."""
    customers = df[df["HasCustomerID"]].groupby("CustomerID").agg(
        country=("Country", "first"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
    ).reset_index()

    customers.columns = ["customer_id", "country", "first_purchase", "last_purchase", "lifetime_days"]
    customers["lifetime_days"] = (customers["last_purchase"] - customers["first_purchase"]).dt.days

    customers.to_sql("customers", engine, if_exists="replace", index=False)
    print(f"  Loaded {len(customers):,} customers")
    return customers


def load_products(df: pd.DataFrame, engine):
    """Load products table."""
    products = df[["StockCode", "Description"]].drop_duplicates(subset=["StockCode"])
    products.columns = ["stock_code", "description"]
    products["product_id"] = range(1, len(products) + 1)
    products = products[["product_id", "stock_code", "description"]]

    products.to_sql("products", engine, if_exists="replace", index=False)
    print(f"  Loaded {len(products):,} products")
    return products


def load_orders(df: pd.DataFrame, engine):
    """Load orders table."""
    orders = df.groupby("InvoiceNo").agg(
        customer_id=("CustomerID", "first"),
        invoice_date=("InvoiceDate", "first"),
        country=("Country", "first"),
        is_cancelled=("IsCancellation", "first"),
    ).reset_index()

    orders.columns = ["order_id", "customer_id", "invoice_date", "country", "is_cancelled"]

    # Only orders with valid customer IDs
    orders = orders[orders["customer_id"].notna()]
    orders["customer_id"] = orders["customer_id"].astype(int)

    orders.to_sql("orders", engine, if_exists="replace", index=False)
    print(f"  Loaded {len(orders):,} orders")
    return orders


def load_order_items(df: pd.DataFrame, engine, products: pd.DataFrame):
    """Load order_items table."""
    # Filter to valid transactions only
    valid = df[(df["HasCustomerID"]) & (df["Revenue"] > 0)]

    # Map stock_code to product_id
    code_to_id = dict(zip(products["stock_code"], products["product_id"]))
    valid = valid.copy()
    valid["product_id"] = valid["StockCode"].map(code_to_id)

    items = valid[["InvoiceNo", "product_id", "Quantity", "UnitPrice", "Revenue"]].copy()
    items.columns = ["order_id", "product_id", "quantity", "unit_price", "revenue"]
    items = items.dropna(subset=["product_id"])
    items["product_id"] = items["product_id"].astype(int)

    items.to_sql("order_items", engine, if_exists="replace", index=False)
    print(f"  Loaded {len(items):,} order items")
    return items


def run_etl():
    """Run full ETL pipeline."""
    from src.preprocessing.cleaning import load_raw, convert_types, derive_flags, remove_duplicates

    print("Loading and cleaning data...")
    df = load_raw()
    df = convert_types(df)
    df = derive_flags(df)
    df = remove_duplicates(df)

    print("\nConnecting to database...")
    engine = get_engine()

    print("\nCreating schema...")
    create_schema(engine)

    print("\nLoading tables...")
    customers = load_customers(df, engine)
    products = load_products(df, engine)
    orders = load_orders(df, engine)
    items = load_order_items(df, engine, products)

    print("\nETL complete!")
    return {"customers": customers, "products": products, "orders": orders, "items": items}


if __name__ == "__main__":
    run_etl()
