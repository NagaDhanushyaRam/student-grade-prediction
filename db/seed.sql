PRAGMA foreign_keys = ON;
BEGIN TRANSACTION;

-- Departments
INSERT OR IGNORE INTO departments(name) VALUES
  ('CS'),
  ('Marketing'),
  ('Media'),
  ('Business');

-- (Optional) default admin created by the app; if you prefer SQL seeding:
-- NOTE: Use your app’s helper to hash the password, then paste the hash below.
-- INSERT INTO users(name,email,password_hash,role)
-- SELECT 'Admin', 'admin@university.edu', '<bcrypt_sha256_hash_here>', 'ADMIN'
-- WHERE NOT EXISTS (SELECT 1 FROM users WHERE email='admin@university.edu');

-- (Optional) sample teacher tied to an existing user (by email)
-- INSERT INTO users(name,email,password_hash,role)
-- SELECT 'Alice Teacher','alice@university.edu','<bcrypt_hash>','TEACHER'
-- WHERE NOT EXISTS (SELECT 1 FROM users WHERE email='alice@university.edu');
-- INSERT OR IGNORE INTO teachers(user_id, department_id)
-- SELECT u.user_id, d.department_id
-- FROM users u JOIN departments d ON d.name='Computer Science'
-- WHERE u.email='alice@university.edu';

-- (Optional) sample student tied to an existing user (by email)
-- INSERT INTO users(name,email,password_hash,role)
-- SELECT 'John Student','john@university.edu','<bcrypt_hash>','STUDENT'
-- WHERE NOT EXISTS (SELECT 1 FROM users WHERE email='john@university.edu');
-- INSERT OR IGNORE INTO students(user_id, department_id, academic_year, gender)
-- SELECT u.user_id, d.department_id, '2025-26', 'Male'
-- FROM users u JOIN departments d ON d.name='Computer Science'
-- WHERE u.email='john@university.edu';

COMMIT;
