import platform
import os
import logging
import multiprocessing

logger = logging.getLogger("GAIA.HardwareOptimization")

def optimize_config_for_hardware(config, force_high_performance=False):
    """
    Automatically adjust system-level performance settings based on hardware.
    Updates config in-place. Set force_high_performance=True to push aggressive settings.
    """
    try:
        system = platform.system().lower()
        cpu_count = multiprocessing.cpu_count()
        config.n_threads = min(config.n_threads or cpu_count, cpu_count)

        logger.info(f"🧠 Optimizing for {system} with {cpu_count} cores")

        if force_high_performance:
            config.n_ctx = 4096
            config.n_threads = cpu_count
            config.low_vram_mode = False
            config.use_mmap = True
            config.gpu_layers = max(10, config.gpu_layers or 20)
            logger.info("🚀 Force high performance mode enabled")
        else:
            if system == "windows":
                config.use_mmap = False
                config.low_vram_mode = True
                config.n_ctx = 2048

            elif system == "linux":
                config.use_mmap = True
                config.low_vram_mode = False
                config.n_ctx = 4096

            elif system == "darwin":
                config.use_mmap = False
                config.low_vram_mode = True
                config.n_ctx = 2048

            else:
                logger.warning("⚠️ Unknown platform; using safe defaults")

            if hasattr(config, "gpu_layers"):
                config.gpu_layers = max(1, config.gpu_layers or 5)

        logger.debug(
            f"⚙️ Config after optimization: n_ctx={config.n_ctx}, threads={config.n_threads}, "
            f"mmap={config.use_mmap}, gpu_layers={getattr(config, 'gpu_layers', 'N/A')}"
        )

    except Exception as e:
        logger.error(f"❌ Hardware optimization failed: {e}", exc_info=True)
