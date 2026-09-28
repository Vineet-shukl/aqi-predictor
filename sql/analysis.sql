-- DuckDB queries against the city_day table created by sql/run_analysis.py.
-- Seasons follow the air-pollution convention used in this project:
-- Winter Dec-Feb, Summer Mar-May, Monsoon Jun-Sep, Post-monsoon Oct-Nov.

-- name: city_pm25_ranking
SELECT
    city,
    COUNT(pm25) AS n_days,
    ROUND(AVG(pm25), 2) AS mean_pm25,
    ROUND(MEDIAN(pm25), 2) AS median_pm25,
    ROUND(AVG(aqi), 2) AS mean_aqi
FROM city_day
GROUP BY city
HAVING COUNT(pm25) >= 30
ORDER BY mean_pm25 DESC;

-- name: monthly_trend
SELECT
    MONTH(date) AS month,
    ROUND(AVG(pm25), 2) AS mean_pm25,
    ROUND(MEDIAN(pm25), 2) AS median_pm25,
    ROUND(AVG(pm10), 2) AS mean_pm10,
    ROUND(AVG(no2), 2) AS mean_no2,
    ROUND(AVG(co), 3) AS mean_co,
    COUNT(pm25) AS n_pm25
FROM city_day
GROUP BY MONTH(date)
ORDER BY month;

-- name: seasonal_trend
SELECT
    CASE
        WHEN MONTH(date) IN (12, 1, 2) THEN 'Winter'
        WHEN MONTH(date) IN (3, 4, 5) THEN 'Summer'
        WHEN MONTH(date) IN (6, 7, 8, 9) THEN 'Monsoon'
        ELSE 'Post-monsoon'
    END AS season,
    ROUND(AVG(pm25), 2) AS mean_pm25,
    ROUND(MEDIAN(pm25), 2) AS median_pm25,
    COUNT(pm25) AS n_days
FROM city_day
GROUP BY season
ORDER BY mean_pm25 DESC;

-- name: yearly_trend
SELECT
    YEAR(date) AS year,
    ROUND(AVG(pm25), 2) AS mean_pm25,
    ROUND(AVG(aqi), 2) AS mean_aqi,
    COUNT(pm25) AS n_pm25,
    COUNT(DISTINCT city) AS n_cities
FROM city_day
GROUP BY YEAR(date)
ORDER BY year;

-- name: pollutant_correlations
SELECT
    ROUND(CORR(pm25, pm10), 3) AS pm25_pm10,
    ROUND(CORR(pm25, no2), 3) AS pm25_no2,
    ROUND(CORR(pm25, no), 3) AS pm25_no,
    ROUND(CORR(pm25, nox), 3) AS pm25_nox,
    ROUND(CORR(pm25, co), 3) AS pm25_co,
    ROUND(CORR(pm25, so2), 3) AS pm25_so2,
    ROUND(CORR(pm25, o3), 3) AS pm25_o3,
    ROUND(CORR(pm25, nh3), 3) AS pm25_nh3,
    ROUND(CORR(pm25, aqi), 3) AS pm25_aqi
FROM city_day;

-- name: aqi_bucket_share
SELECT
    aqi_bucket,
    COUNT(*) AS n_days,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM city_day
WHERE aqi_bucket IS NOT NULL
GROUP BY aqi_bucket
ORDER BY n_days DESC;

-- name: missingness
SELECT
    ROUND(100.0 * COUNT(*) FILTER (WHERE pm25 IS NULL) / COUNT(*), 2) AS pm25_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE pm10 IS NULL) / COUNT(*), 2) AS pm10_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE no2 IS NULL) / COUNT(*), 2) AS no2_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE co IS NULL) / COUNT(*), 2) AS co_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE so2 IS NULL) / COUNT(*), 2) AS so2_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE o3 IS NULL) / COUNT(*), 2) AS o3_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE nh3 IS NULL) / COUNT(*), 2) AS nh3_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE xylene IS NULL) / COUNT(*), 2) AS xylene_pct,
    ROUND(100.0 * COUNT(*) FILTER (WHERE aqi IS NULL) / COUNT(*), 2) AS aqi_pct
FROM city_day;

-- name: city_season_pm25
SELECT
    city,
    CASE
        WHEN MONTH(date) IN (12, 1, 2) THEN 'Winter'
        WHEN MONTH(date) IN (3, 4, 5) THEN 'Summer'
        WHEN MONTH(date) IN (6, 7, 8, 9) THEN 'Monsoon'
        ELSE 'Post-monsoon'
    END AS season,
    ROUND(AVG(pm25), 2) AS mean_pm25,
    COUNT(pm25) AS n_days
FROM city_day
GROUP BY city, season
HAVING COUNT(pm25) >= 20
ORDER BY city, season;
