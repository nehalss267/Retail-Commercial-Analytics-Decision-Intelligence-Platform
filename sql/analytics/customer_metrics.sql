-- Customer Metrics Analytics

CREATE OR REPLACE VIEW v_customer_analytics AS
SELECT
    c.customer_id,
    c.country,
    c.first_purchase_date,
    c.last_purchase_date,
    c.customer_lifetime_days,
    COUNT(DISTINCT o.order_id) AS total_orders,
    SUM(oi.revenue) AS total_revenue,
    SUM(oi.quantity) AS total_items,
    ROUND(AVG(order_totals.order_revenue), 2) AS avg_order_value,
    ROUND(SUM(oi.revenue) / (c.customer_lifetime_days + 1), 4) AS revenue_per_day,
    COUNT(DISTINCT p.stock_code) AS unique_products
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
LEFT JOIN (
    SELECT order_id, SUM(revenue) AS order_revenue
    FROM order_items
    GROUP BY order_id
) order_totals ON o.order_id = order_totals.order_id
WHERE oi.revenue > 0
GROUP BY c.customer_id, c.country, c.first_purchase_date,
         c.last_purchase_date, c.customer_lifetime_days;

CREATE OR REPLACE VIEW v_customer_rfm AS
SELECT
    customer_id,
    EXTRACT(DAY FROM (SELECT MAX(invoice_date) FROM orders) - last_purchase_date) AS recency_days,
    total_orders AS frequency,
    total_revenue AS monetary
FROM v_customer_analytics;
