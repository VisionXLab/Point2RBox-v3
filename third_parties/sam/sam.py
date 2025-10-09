try:
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError:
        raise ImportError("Please install MobileSAM: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")