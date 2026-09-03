PLAN_LIMITS = {
    "free":       {"max_employees": 10,      "price": 0},
    "starter":    {"max_employees": 50,      "price": 999},
    "pro":        {"max_employees": 200,     "price": 2499},
    "enterprise": {"max_employees": 999999,  "price": 0},
}

MASTER_DB_CONFIG = {
    "host": "10.8.21.52",
    "port": 5432,
    "user": "postgres",
    "password": "daljeet@123",
    "dbname": "attendai_master"
}

POSTGRES_ADMIN_CONFIG = {
    "host": "10.8.21.52",
    "port": 5432,
    "user": "postgres",
    "password": "daljeet@123",
    "dbname": "postgres"
}