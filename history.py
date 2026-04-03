import sqlite3
import datetime


def create_history():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS history(
                username TEXT,
                plant TEXT,
                disease TEXT,
                confidence REAL,
                date TEXT
    )"""
    )
    conn.commit()
    conn.close()


def add_history(username, plant, disease, confidence):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO history VALUES (?,?,?,?,?)",
        (username, plant, disease, confidence, str(datetime.datetime.now())),
    )
    conn.commit()
    conn.close()


def get_history(username):
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM history WHERE username=?", (username,))
    data = c.fetchall()
    conn.close()
    return data


def get_all_history():
    conn = sqlite3.connect("database.db")
    c = conn.cursor()
    c.execute("SELECT * FROM history")
    data = c.fetchall()
    conn.close()
    return data
