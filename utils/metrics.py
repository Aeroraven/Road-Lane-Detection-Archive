class ArvnAvgMeter:
    def __init__(self):
        self.samples = 0
        self.value = 0.0

    def add_value_raw(self, val):
        self.samples += 1
        self.value += val

    def get_value(self):
        if self.samples == 0:
            return 0.0
        return self.value / self.samples

class ArvnAccuMeter:
    def __init__(self):
        self.samples = 0
        self.value = 0.0

    def add_value_raw(self, val):
        self.samples += 1
        self.value += val

    def get_value(self):
        if self.samples == 0:
            return 0.0
        return str(int(self.value)//60)+":"+str(int(self.value)%60)
