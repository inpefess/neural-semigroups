.open test_output.db
PRAGMA journal_mode=WAL
;
ALTER TABLE mace_output
ADD COLUMN model_exists INTEGER DEFAULT 0
;
ALTER TABLE mace_output
ADD COLUMN timeout INTEGER DEFAULT 0
;
ALTER TABLE mace_output
ADD COLUMN unsatisfiability INTEGER DEFAULT 0
;
UPDATE mace_output
SET model_exists = 1
WHERE errors LIKE '% process % exit (max_models) %'
;
UPDATE mace_output
SET timeout = 1
WHERE errors LIKE '% process % exit (max_sec_no) %'
;
UPDATE mace_output
SET unsatisfiability = 1
WHERE errors LIKE '%NOTE: unsatisfiability detected on input.%'
;
.exit
