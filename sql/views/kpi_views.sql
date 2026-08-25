-- KPI Views for analytical layer

-- Customer metrics (RFM base)
CREATE OR REPLACE VIEW v_customer_metrics AS
SELECT
    o.customer_id,
    c.country,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(oi.order_item_id) AS total_items,
    SUM(oi.revenue) AS total_revenue,
    AVG(oi.revenue) AS avg_line_value,
    SUM(oi.revenue) / COUNT(DISTINCT o.order_id) AS avg_order_value,
    MIN(o.invoice_date) AS first_order_date,
    MAX(o.invoice_date) AS last_order_date,
    EXTRACT(EPOCH FROM (MAX(o.invoice_date) - MIN(o.invoice_date))) / 86400 AS customer_lifetime_days
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN customers c ON o.customer_id = c.customer_id
WHERE oi.revenue > 0
GROUP BY o.customer_id, c.country;

-- Daily revenue metrics
CREATE OR REPLACE VIEW v_daily_metrics AS
SELECT
    DATE(o.invoice_date) AS order_date,
    COUNT(DISTINCT o.order_id) AS orders,
    COUNT(DISTINCT o.customer_id) AS customers,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.revenue) AS revenue
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE oi.revenue > 0
GROUP BY DATE(o.invoice_date)
ORDER BY order_date;

-- Product metrics
CREATE OR REPLACE VIEW v_product_metrics AS
SELECT
    p.product_id,
    p.stock_code,
    p.description,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    SUM(oi.quantity) AS units_sold,
    SUM(oi.revenue) AS total_revenue,
    AVG(oi.unit_price) AS avg_price,
    COUNT(DISTINCT o.customer_id) AS unique_customers
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE oi.revenue > 0
GROUP BY p.product_id, p.stock_code, p.description;
