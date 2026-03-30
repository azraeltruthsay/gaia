try:
    from gaia_core.models.model_pool import get_model_pool
except ImportError:
    get_model_pool = lambda: None

if __name__ == "__main__":
    mp = get_model_pool()
    if mp is None:
        print("Model pool not available; cannot register dev model")
    else:
        mp.register_dev_model("azrael")
