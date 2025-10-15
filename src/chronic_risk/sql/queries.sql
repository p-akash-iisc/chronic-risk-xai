-- Example SQL demonstrating GROUP BY, JOIN, and window functions
-- 1) Basic aggregation
SELECT age, AVG(glucose) AS avg_glucose, COUNT(*) AS n
FROM patients
GROUP BY age;

-- 2) Window function: cumulative count by age
SELECT age, glucose,
       COUNT(*) OVER (PARTITION BY age ORDER BY glucose ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cum_cnt
FROM patients;

-- 3) Join example (patients with demographics)
SELECT p.age, p.glucose, d.sex
FROM patients p
LEFT JOIN demographics d ON p.patient_id = d.patient_id;
