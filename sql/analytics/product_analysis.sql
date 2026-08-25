-- Product Analysis

CREATE OR REPLACE VIEW v_product_performance AS
SELECT
    p.stock_code,
    p.description,
    SUM(oi.quantity) AS total_units,
    SUM(oi.revenue) AS total_revenue,
    COUNT(DISTINCT oi.order_id) AS total_orders,
    COUNT(DISTINCT o.customer_id) AS unique_customers,
    ROUND(AVG(oi.unit_price), 2) AS avg_unit_price,
    ROUND(MIN(oi.unit_price), 2) AS min_price,
    ROUND(MAX(oi.unit_price), 2) AS max_price,
    ROUND(SUM(oi.revenue) / COUNT(DISTINCT o.customer_id), 2) AS revenue_per_customer,
    ROUND(COUNT(DISTINCT o.customer_id)::numeric / COUNT(DISTINCT oi.order_id) * 100, 2) AS customer_pct_per_order
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
JOIN orders o ON oi.order_id = o.order_id
WHERE oi.revenue > 0
GROUP BY p.stock_code, p.description
HAVING SUM(oi.revenue) > 0
ORDER BY total_revenue DESC;

CREATE OR REPLACE VIEW v_product_category_summary AS
SELECT
    CASE
        WHEN SUM(oi.revenue) > 100000 THEN 'Top Tier'
        WHEN SUM(oi.revenue) > 10000 THEN 'Mid Tier'
        ELSE 'Long Tail'
    END AS tier,
    COUNT(DISTINCT p.stock_code) AS n_products,
    ROUND(SUM(oi.revenue), 2) AS total_revenue,
    ROUND(AVG(oi.revenue), 2) AS avg_revenue_per_item
FROM products p
JOIN order_items oi ON p.product_id = oi.product_id
WHERE oi.revenue > 0
GROUP BY tier
ORDER BY total_revenue DESC;
