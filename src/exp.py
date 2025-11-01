class BasicExp:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = self._build_model()
        self.device = cfg.device



    def _build_model(self):
        raise NotImplementedError






