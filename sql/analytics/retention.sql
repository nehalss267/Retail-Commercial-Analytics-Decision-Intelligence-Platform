-- Retention Analysis
-- Cohort-based retention rates

CREATE OR REPLACE VIEW v_cohort_retention AS
WITH cohort AS (
    SELECT
        customer_id,
        DATE_TRUNC('month', MIN(invoice_date)) AS cohort_month
    FROM orders
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
),
activity AS (
    SELECT
        o.customer_id,
        DATE_TRUNC('month', o.invoice_date) AS activity_month
    FROM orders o
    WHERE o.customer_id IS NOT NULL
)
SELECT
    c.cohort_month,
    EXTRACT(MONTH FROM AGE(a.activity_month, c.cohort_month)) AS months_since,
    COUNT(DISTINCT a.customer_id) AS active_customers,
    COUNT(DISTINCT c.customer_id) AS cohort_size,
    ROUND(COUNT(DISTINCT a.customer_id)::numeric / COUNT(DISTINCT c.customer_id) * 100, 2) AS retention_rate
FROM cohort c
JOIN activity a ON c.customer_id = a.customer_id
GROUP BY c.cohort_month, months_since
ORDER BY c.cohort_month, months_since;
