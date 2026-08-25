-- Revenue Analytics
-- Revenue by month, country, and product

CREATE OR REPLACE VIEW v_monthly_revenue AS
SELECT
    DATE_TRUNC('month', o.invoice_date) AS month,
    SUM(oi.revenue) AS total_revenue,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT o.customer_id) AS total_customers,
    SUM(oi.quantity) AS total_units,
    ROUND(SUM(oi.revenue) / COUNT(DISTINCT o.order_id), 2) AS aov
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE oi.revenue > 0
GROUP BY DATE_TRUNC('month', o.invoice_date)
ORDER BY month;

CREATE OR REPLACE VIEW v_country_revenue AS
SELECT
    o.country,
    SUM(oi.revenue) AS total_revenue,
    COUNT(DISTINCT o.order_id) AS total_orders,
    COUNT(DISTINCT o.customer_id) AS total_customers,
    ROUND(AVG(oi.revenue), 2) AS avg_item_revenue
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE oi.revenue > 0
GROUP BY o.country
ORDER BY total_revenue DESC;

CREATE OR REPLACE VIEW v_product_revenue AS
SELECT
    p.stock_code,
    p.description,
    SUM(oi.revenue) AS total_revenue,
    SUM(oi.quantity) AS units_sold,
    COUNT(DISTINCT oi.order_id) AS orders,
    COUNT(DISTINCT o.customer_id) AS unique_customers,
    ROUND(AVG(oi.unit_price), 2) AS avg_price
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE oi.revenue > 0
GROUP BY p.stock_code, p.description
ORDER BY total_revenue DESC;
