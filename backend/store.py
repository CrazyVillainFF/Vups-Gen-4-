import sqlite3
from pathlib import Path
from typing import Protocol
from .config import Settings
SCHEMA=[
 "CREATE TABLE IF NOT EXISTS conversations (id TEXT PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)",
 "CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant')), content TEXT NOT NULL, skill TEXT NOT NULL DEFAULT 'auto', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, FOREIGN KEY(conversation_id) REFERENCES conversations(id))",
 "CREATE INDEX IF NOT EXISTS idx_messages_conversation_id_id ON messages(conversation_id,id)"]
class Store(Protocol):
 def save_turn(self, conversation_id:str, user_message:str, assistant_message:str, skill:str)->None: ...
 def get_messages(self, conversation_id:str)->list[dict]: ...
 def ping(self)->bool: ...
class SQLiteStore:
 def __init__(self,database_url:str):
  self.path=database_url.removeprefix("sqlite:///")
  if self.path != ":memory:": Path(self.path).parent.mkdir(parents=True,exist_ok=True)
  with self._connect() as db:
   for sql in SCHEMA: db.execute(sql)
 def _connect(self):
  db=sqlite3.connect(self.path,check_same_thread=False); db.row_factory=sqlite3.Row; return db
 def save_turn(self,conversation_id,user_message,assistant_message,skill):
  with self._connect() as db:
   db.execute("INSERT OR IGNORE INTO conversations(id) VALUES (?)",(conversation_id,))
   db.executemany("INSERT INTO messages(conversation_id,role,content,skill) VALUES (?,?,?,?)",[(conversation_id,"user",user_message,skill),(conversation_id,"assistant",assistant_message,skill)])
   db.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?",(conversation_id,))
 def get_messages(self,conversation_id):
  with self._connect() as db:
   rows=db.execute("SELECT role,content,skill,created_at FROM messages WHERE conversation_id=? ORDER BY id",(conversation_id,)).fetchall()
   return [dict(r) for r in rows]
 def ping(self):
  with self._connect() as db: db.execute("SELECT 1").fetchone()
  return True
class TursoStore:
 def __init__(self,url,auth_token):
  from libsql_client import create_client_sync
  self.client=create_client_sync(url=url,auth_token=auth_token)
  for sql in SCHEMA: self.client.execute(sql)
 def save_turn(self,conversation_id,user_message,assistant_message,skill):
  self.client.execute("INSERT OR IGNORE INTO conversations(id) VALUES (?)",[conversation_id])
  for role,content in [("user",user_message),("assistant",assistant_message)]: self.client.execute("INSERT INTO messages(conversation_id,role,content,skill) VALUES (?,?,?,?)",[conversation_id,role,content,skill])
  self.client.execute("UPDATE conversations SET updated_at=CURRENT_TIMESTAMP WHERE id=?",[conversation_id])
 def get_messages(self,conversation_id):
  result=self.client.execute("SELECT role,content,skill,created_at FROM messages WHERE conversation_id=? ORDER BY id",[conversation_id])
  return [dict(zip(("role","content","skill","created_at"),row)) for row in result.rows]
 def ping(self): self.client.execute("SELECT 1"); return True
def create_store(config:Settings)->Store:
 return TursoStore(config.turso_database_url,config.turso_auth_token) if config.turso_enabled else SQLiteStore(config.database_url)
