# === role_manager.py ===

class RoleManager:
    current_responder = None
    observers = []

    @classmethod
    def set_responder(cls, name):
        cls.current_responder = name
        print(f"🎙️ [RoleManager] Responder: {name}")

    @classmethod
    def add_observer(cls, name):
        if name not in cls.observers:
            cls.observers.append(name)
            print(f"👂 [RoleManager] Observer added: {name}")

    @classmethod
    def get_current_roles(cls):
        return {
            "responder": cls.current_responder,
            "observers": cls.observers
        }
