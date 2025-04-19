class PID:
    def __init__(self, kp, ki, kd, dt, u_min, u_max):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.ierror = 0
        self.y_prev = 0
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max

    def get_u(self, y, y_set):
        error = y_set - y
        self.ierror += error * self.dt
        derror = (y - self.y_prev) / self.dt
        u = self.kp * error + self.ki * self.ierror + self.kd * derror
        u = self.kp * error + self.ki * self.ierror + self.kd * derror
        if u > self.u_max:
            u = self.u_max
        elif u < self.u_min:
            u = self.u_min
        self.y_prev = y
        return u
