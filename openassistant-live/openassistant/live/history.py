"""OpenAssistant metadata index and Pi session transcript reader."""

import json
import sqlite3
import time
import uuid
from pathlib import Path


class Conversations:
    def __init__(self, directory, workspace):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / "conversations.sqlite"
        self.workspace = str(Path(workspace).resolve())
        with self.connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS conversations ("
                "id TEXT PRIMARY KEY, thread TEXT UNIQUE, title TEXT NOT NULL, "
                "created REAL NOT NULL, updated REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS tasks ("
                "conversation TEXT PRIMARY KEY, state TEXT, detail TEXT, started REAL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS title_state ("
                "conversation TEXT PRIMARY KEY, completed INTEGER NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS images (conversation TEXT, item TEXT, id TEXT, "
                "path TEXT, alt TEXT, mime TEXT, data BLOB, "
                "PRIMARY KEY (conversation, item, path))"
            )
            db.execute("CREATE TABLE IF NOT EXISTS hidden_chats (id TEXT PRIMARY KEY)")
        self.path.chmod(0o600)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def save_image(self, conversation, item, identifier, path, alt, mime, data):
        self.get(conversation)
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO images VALUES (?, ?, ?, ?, ?, ?, ?)",
                (conversation, item, identifier, path, alt, mime, data),
            )

    def image_list(self, conversation, item):
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, path, alt FROM images WHERE conversation=? AND item=?",
                (conversation, item),
            ).fetchall()
        return [dict(row) for row in rows]

    def image(self, conversation, identifier):
        with self.connect() as db:
            return db.execute(
                "SELECT mime, data FROM images WHERE conversation=? AND id=? LIMIT 1",
                (conversation, identifier),
            ).fetchone()

    def create(self):
        identifier, now = str(uuid.uuid4()), time.time()
        with self.connect() as db:
            db.execute(
                "INSERT INTO conversations VALUES (?, NULL, 'New conversation', ?, ?)",
                (identifier, now, now),
            )
        return self.get(identifier)

    def get(self, identifier):
        if not isinstance(identifier, str):
            raise ValueError("Unknown conversation")
        with self.connect() as db:
            row = db.execute("SELECT * FROM conversations WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise ValueError("Unknown conversation")
        return dict(row)

    def listing(self, offset=0, query=""):
        if type(offset) is not int or not 0 <= offset <= 100000:
            raise ValueError("Invalid conversation page")
        if not isinstance(query, str) or len(query) > 200:
            raise ValueError("Invalid conversation search")
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, title, updated FROM conversations "
                "WHERE NOT EXISTS (SELECT 1 FROM hidden_chats "
                "WHERE hidden_chats.id=conversations.id) "
                "AND instr(lower(title), lower(?)) > 0 "
                "ORDER BY updated DESC, id LIMIT 51 OFFSET ?",
                (query, offset),
            ).fetchall()
        return {
            "chats": [{**dict(row), "task": self.task_status(row["id"])} for row in rows[:50]],
            "next": offset + 50 if len(rows) > 50 else None,
        }

    def hide(self, identifier):
        self.get(identifier)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO hidden_chats VALUES (?)", (identifier,))

    def update_task(self, identifier, state, detail, started):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO tasks VALUES (?, ?, ?, ?)",
                (identifier, state, detail, started),
            )

    def task_status(self, identifier):
        with self.connect() as db:
            row = db.execute(
                "SELECT state, detail, started FROM tasks WHERE conversation=?", (identifier,)
            ).fetchone()
        return dict(row) if row else None

    def interrupt_tasks(self):
        with self.connect() as db:
            db.execute(
                "UPDATE tasks SET state='interrupted', detail='Server restarted. Check completed "
                "actions before continuing.' WHERE state IN ('working', 'approval', 'input')"
            )

    def saved(self, identifier, thread, prompt):
        title = " ".join((prompt or "Conversation").split())[:100] or "Conversation"
        with self.connect() as db:
            db.execute(
                "UPDATE conversations SET thread=?, "
                "title=CASE WHEN thread IS NULL THEN ? ELSE title END, updated=? WHERE id=?",
                (thread, title, time.time(), identifier),
            )

    def touch(self, identifier):
        with self.connect() as db:
            db.execute("UPDATE conversations SET updated=? WHERE id=?", (time.time(), identifier))

    def completed_exchange(self, identifier):
        with self.connect() as db:
            db.execute(
                "INSERT INTO title_state VALUES (?, 1) ON CONFLICT(conversation) "
                "DO UPDATE SET completed=completed+1",
                (identifier,),
            )
            return db.execute(
                "SELECT completed FROM title_state WHERE conversation=?", (identifier,)
            ).fetchone()[0]

    def set_title_if_current(self, identifier, version, title):
        with self.connect() as db:
            result = db.execute(
                "UPDATE conversations SET title=? WHERE id=? AND EXISTS "
                "(SELECT 1 FROM title_state WHERE conversation=? AND completed=?)",
                (title, identifier, identifier, version),
            )
            return result.rowcount == 1

    def transcript(self, identifier):
        thread = self.get(identifier)["thread"]
        if not thread:
            return {"messages": [], "truncated": False}
        session_dir = (self.path.parent / "pi-sessions").resolve()
        path = Path(thread).resolve()
        if path.parent != session_dir or path.suffix != ".jsonl" or not path.is_file():
            raise ValueError("Unknown Pi session")
        messages = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if len(line) > 2_000_000:
                    raise ValueError("Session entry exceeds size limit")
                item = json.loads(line)
                message = item.get("message", {})
                role = message.get("role")
                if item.get("type") != "message" or role not in ("user", "assistant"):
                    continue
                content = message.get("content", [])
                if isinstance(content, str):
                    text = content
                else:
                    text = "\n".join(
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    )
                if text:
                    entry = {
                        "id": item.get("id", str(uuid.uuid4())),
                        "who": "You" if role == "user" else "OpenAssistant",
                        "text": text[:32000],
                    }
                    if role == "assistant":
                        with self.connect() as db:
                            rows = db.execute(
                                "SELECT id, path, alt FROM images WHERE conversation=?",
                                (identifier,),
                            ).fetchall()
                        images = [dict(row) for row in rows if f"({row['path']})" in text]
                        if images:
                            entry["images"] = images
                    messages.append(entry)
        result, size = [], 0
        for message in reversed(messages):
            if len(result) >= 100 or size + len(message["text"]) > 200000:
                break
            result.append(message)
            size += len(message["text"])
        return {"messages": list(reversed(result)), "truncated": len(result) < len(messages)}
