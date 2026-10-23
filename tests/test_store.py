from backend.store import SQLiteStore
def test_turn_persists(tmp_path):
 store=SQLiteStore(f"sqlite:///{tmp_path/'test.db'}"); store.save_turn("c1","hello","hi","auto"); rows=store.get_messages("c1")
 assert [r["role"] for r in rows]==["user","assistant"] and [r["content"] for r in rows]==["hello","hi"]
def test_unknown_conversation_empty(tmp_path): assert SQLiteStore(f"sqlite:///{tmp_path/'test.db'}").get_messages("absent")==[]
