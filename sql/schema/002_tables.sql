-- Warehouse tables for RetailAI
-- Derived from UCI Online Retail dataset

DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- Customers
CREATE TABLE customers (
    customer_id     INTEGER PRIMARY KEY,
    country         TEXT NOT NULL,
    first_purchase  TIMESTAMP,
    last_purchase   TIMESTAMP,
    lifetime_days   INTEGER
);

-- Products
CREATE TABLE products (
    product_id      SERIAL PRIMARY KEY,
    stock_code      TEXT UNIQUE NOT NULL,
    description     TEXT
);

-- Orders (invoices)
CREATE TABLE orders (
    order_id        TEXT PRIMARY KEY,
    customer_id     INTEGER REFERENCES customers(customer_id),
    invoice_date    TIMESTAMP NOT NULL,
    country         TEXT NOT NULL,
    is_cancelled    BOOLEAN DEFAULT FALSE
);

-- Order line items
CREATE TABLE order_items (
    order_item_id   SERIAL PRIMARY KEY,
    order_id        TEXT REFERENCES orders(order_id),
    product_id      INTEGER REFERENCES products(product_id),
    quantity        INTEGER NOT NULL,
    unit_price      NUMERIC(12,2) NOT NULL,
    revenue         NUMERIC(14,2) NOT NULL
);
