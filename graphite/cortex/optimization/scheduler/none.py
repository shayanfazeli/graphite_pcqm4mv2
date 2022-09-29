class NoScheduler:
    def __init__(self, optimizer):
        pass

    def step(self):
        pass

    def state_dict(self):
        return dict(data='nothing')

    def load_state_dict(self, data):
        return