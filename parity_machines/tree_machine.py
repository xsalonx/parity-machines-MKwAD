import numpy as np
from parity_machines import ParityMachine

class TreeParityMachine(ParityMachine):
    def __init__(self, n, k, l, update_rule='hebbian', seed=None):
        super().__init__(k, n, seed=seed)
        self.L = l
        self.update_rule = update_rule
        np.random.seed(seed)
        self._r()

    def _r(self):
        self.consecutive_updates = 0
        self.W = np.random.randint(-self.L, self.L + 1, size=(self.k, self.n))
    
    def generate_input(self):
        return np.random.randint(-self.L, self.L + 1, size=(self.k,  self.n))
	
    def forward(self, x):
        self.x = x.reshape((self.k, self.n))
        self.sgn = np.sign((self.W * self.x).sum(axis=1))
        self.y = self.sgn.prod()
        return self.y
	
    def __call__(self, x):
        return self.forward(x)

    def update(self, y_other):
        if (self.y == y_other):
            self.consecutive_updates += 1
            if self.update_rule == 'hebbian':
                self.hebbian(y_other)
            elif self.update_rule == 'anti_hebbian':
                self.anti_hebbian(y_other)
            elif self.update_rule == 'random_walk':
                self.random_walk(y_other)
            else:
                raise Exception("Invalid update rule. Valid update rules are: " + 
                "\'hebbian\', \'anti_hebbian\' and \'random_walk\'.")
        else:
            self.consecutive_updates = 0	

    def synced(self):
        return self.consecutive_updates > self.n + self.k + self.L

    def synced_p2(self):
        """Higher accuracy of syncronization than for mathod 'synced'"""
        return self.consecutive_updates > self.n * self.k + self.L
            
    def get_key(self):
        return self.W.tobytes()
    
    def sync_level(self):
        return self.consecutive_updates / (self.n * self.k + self.L)
            
    def hebbian(self, y_other):
        self.W += self.x * self.y * (self.sgn.reshape((-1, 1)) == self.y) * (self.y == y_other)
        self.W = np.clip(self.W, -self.L, self.L+1)

    def anti_hebbian(self, y_other):
        self.W -= self.x * self.y * (self.sgn.reshape((-1, 1)) == self.y) * (self.y == y_other)
        self.W = np.clip(self.W, -self.L, self.L+1)

    def random_walk(self, y_other):
        self.W -= self.x * (self.sgn.reshape((-1, 1)) == self.y) * (self.y == y_other)
        self.W = np.clip(self.W, -self.L, self.L+1)
