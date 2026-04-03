import sqlite3


def create_user_table():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS users(
        username TEXT,
        password TEXT
    )"""
    )
    conn.commit()
    conn.close()


def add_user(user, pwd):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("INSERT INTO users VALUES(?,?)", (user, pwd))
    conn.commit()
    conn.close()


def login_user(user, pwd):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
    data = c.fetchone()
    conn.close()
    return data
