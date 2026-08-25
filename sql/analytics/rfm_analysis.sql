-- RFM Analysis
-- Recency: days since last purchase
-- Frequency: number of orders
-- Monetary: total revenue

CREATE OR REPLACE VIEW v_rfm AS
WITH rfm_calc AS (
    SELECT
        o.customer_id,
        EXTRACT(EPOCH FROM (MAX(o.invoice_date) - CURRENT_TIMESTAMP)) / 86400 AS recency_days,
        COUNT(DISTINCT o.order_id) AS frequency,
        SUM(oi.revenue) AS monetary
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE oi.revenue > 0 AND o.customer_id IS NOT NULL
    GROUP BY o.customer_id
),
rfm_scored AS (
    SELECT
        customer_id,
        recency_days,
        frequency,
        monetary,
        NTILE(5) OVER (ORDER BY recency_days DESC) AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC) AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC) AS m_score
    FROM rfm_calc
)
SELECT
    customer_id,
    recency_days,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    (r_score + f_score + m_score) AS rfm_total,
    CASE
        WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
        WHEN r_score >= 3 AND f_score >= 3 THEN 'Loyal'
        WHEN r_score >= 4 AND f_score <= 2 THEN 'New Customers'
        WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
        WHEN r_score <= 2 AND f_score <= 2 THEN 'Lost'
        ELSE 'Regular'
    END AS segment
FROM rfm_scored;
