import json, hashlib
import logging
from pathlib import Path
from .db import DB
from .text import chunk_text

logger = logging.getLogger(__name__)

def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_get_create_time(node):
    """Safely extract create_time from a conversation node, handling None values."""
    if not node or not isinstance(node, dict):
        return 0
    
    message = node.get('message')
    if not message or not isinstance(message, dict):
        return 0
    
    return message.get('create_time', 0) or 0

def flatten_conversation(title: str, messages: list) -> str:
    lines = [f"# {title}\n\n"]
    for m in messages:
        if not m or not isinstance(m, dict):
            continue
            
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

def ingest_export_multi_layer(root: str, project_id: int = None) -> dict:
    """
    Ingest ChatGPT export with multi-layer chunking strategy.
    Creates three indexes: precision (400 chars), balanced (1200 chars), context (4800 chars).
    """
    rootp = Path(root)
    convo_file = rootp / "conversations.json"
    if not convo_file.exists():
        raise FileNotFoundError(f"conversations.json not found under {root}")
    convos = json.loads(convo_file.read_text(encoding='utf-8'))

    # Define the three layers
    layers = {
        'precision': {'size': 400, 'overlap': 40},
        'balanced': {'size': 1200, 'overlap': 120},
        'context': {'size': 4800, 'overlap': 480}
    }
    
    layer_counts = {layer: 0 for layer in layers}
    total_docs = 0

    with DB() as db:
        for i, c in enumerate(convos):
            try:
                title = c.get('title', 'Untitled')
                
                # Extract messages with better error handling
                if 'messages' in c and isinstance(c['messages'], list):
                    msgs = c['messages']
                elif 'mapping' in c and isinstance(c['mapping'], dict):
                    nodes = list(c['mapping'].values())
                    nodes.sort(key=safe_get_create_time)
                    msgs = [n.get('message') for n in nodes if n.get('message')]
                else:
                    logger.warning(f"Conversation {i} has no valid message structure, skipping")
                    continue

                # Skip conversations with no valid messages
                if not msgs:
                    logger.warning(f"Conversation {i} '{title}' has no messages, skipping")
                    continue

                text = flatten_conversation(title, msgs)
                if not text or len(text.strip()) < 10:
                    logger.warning(f"Conversation {i} '{title}' has insufficient content, skipping")
                    continue

                fp = sha256(text)

                # Create document entry once
                doc_id = db.upsert_document(
                    user_id=1, title=title, doc_type='chatgpt_export_multi', 
                    fingerprint=fp, metadata={"export_id": c.get('id'), "layers": list(layers.keys())}
                )
                if project_id:
                    db.link_project_document(project_id, doc_id)

                # Generate chunks for each layer
                for layer_name, config in layers.items():
                    chunks = chunk_text(text, size=config['size'], overlap=config['overlap'])
                    if not chunks:
                        logger.warning(f"No chunks generated for {layer_name} layer in conversation '{title}'")
                        continue
                        
                    # Add layer metadata to each chunk
                    for chunk in chunks:
                        chunk['layer'] = layer_name
                    
                    db.insert_chunks(doc_id, chunks)
                    layer_counts[layer_name] += len(chunks)
                
                total_docs += 1
                
                # Log progress for large imports
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(convos)} conversations")
                    
            except Exception as e:
                logger.error(f"Error processing conversation {i}: {e}")
                continue

    return {
        'conversations_processed': total_docs,
        'chunks_by_layer': layer_counts,
        'total_chunks': sum(layer_counts.values())
    }