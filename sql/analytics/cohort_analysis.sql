-- Cohort Analysis
-- Monthly cohorts based on first purchase date

CREATE OR REPLACE VIEW v_cohort_analysis AS
WITH cohort AS (
    SELECT
        o.customer_id,
        DATE_TRUNC('month', MIN(o.invoice_date)) AS cohort_month,
        DATE_TRUNC('month', o.invoice_date) AS order_month
    FROM orders o
    JOIN order_items oi ON o.order_id = oi.order_id
    WHERE oi.revenue > 0 AND o.customer_id IS NOT NULL
    GROUP BY o.customer_id, DATE_TRUNC('month', o.invoice_date)
),
cohort_size AS (
    SELECT
        cohort_month,
        COUNT(DISTINCT customer_id) AS cohort_customers
    FROM cohort
    WHERE order_month = cohort_month
    GROUP BY cohort_month
),
cohort_periods AS (
    SELECT
        c.cohort_month,
        c.order_month,
        EXTRACT(YEAR FROM AGE(c.order_month, c.cohort_month)) * 12
            + EXTRACT(MONTH FROM AGE(c.order_month, c.cohort_month)) AS period_number,
        COUNT(DISTINCT c.customer_id) AS active_customers
    FROM cohort c
    GROUP BY c.cohort_month, c.order_month
)
SELECT
    cp.cohort_month,
    cs.cohort_customers,
    cp.period_number,
    cp.active_customers,
    ROUND(cp.active_customers::numeric / cs.cohort_customers * 100, 2) AS retention_pct
FROM cohort_periods cp
JOIN cohort_size cs ON cp.cohort_month = cs.cohort_month
ORDER BY cp.cohort_month, cp.period_number;
