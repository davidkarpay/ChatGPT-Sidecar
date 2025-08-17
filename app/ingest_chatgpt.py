
import json, hashlib
from pathlib import Path
from .db import DB
from .text import chunk_text

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def flatten_conversation(title: str, messages: list) -> str:
    lines = [f"# {title}\n\n"]
    for m in messages:
        role = (m.get('author') or {}).get('role') or m.get('role') or 'unknown'
        if isinstance(m.get('content'), dict) and 'parts' in m['content']:
            parts = m['content']['parts']
            text = "\n".join(p if isinstance(p, str) else str(p) for p in parts)
        else:
            content = m.get('content', '')
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        t = m.get('create_time') or m.get('update_time') or ''
        lines.append(f"[{t}] {role.upper()}:\n{text}\n")
    return "\n".join(lines)

def ingest_export(root: str, project_id: int = None, chunk_chars: int = 1200, overlap: int = 120) -> int:
    rootp = Path(root)
    convo_file = rootp / "conversations.json"
    if not convo_file.exists():
        raise FileNotFoundError(f"conversations.json not found under {root}")
    convos = json.loads(convo_file.read_text(encoding='utf-8'))

    count = 0
    with DB() as db:
        for c in convos:
            title = c.get('title', 'Untitled')
            if 'messages' in c and isinstance(c['messages'], list):
                msgs = c['messages']
            elif 'mapping' in c and isinstance(c['mapping'], dict):
                nodes = list(c['mapping'].values())
                nodes.sort(key=lambda n: (n.get('message', {}).get('create_time', 0) or 0))
                msgs = [n.get('message') for n in nodes if n.get('message')]
            else:
                msgs = []

            text = flatten_conversation(title, msgs)
            fp = sha256(text)

            doc_id = db.upsert_document(
                user_id=1, title=title, doc_type='chatgpt_export', fingerprint=fp, metadata={"export_id": c.get('id')}
            )
            if project_id:
                db.link_project_document(project_id, doc_id)

            chunks = chunk_text(text, size=chunk_chars, overlap=overlap)
            db.insert_chunks(doc_id, chunks)
            count += 1
    return count
