import numpy as np

class KF(object):
    def __init__(this, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")
        else:
            this.n = F.shape[1]
            this.m = H.shape[1]

            this.F = F
            this.H = H
            this.B = 0 if B is None else B
            this.Q = np.eye(this.n) if Q is None else Q
            this.R = np.eye(this.n) if R is None else R
            this.P = np.eye(this.n) if P is None else P
            this.x = np.zeros((this.n, 1)) if x0 is None else x0

    def predict(this, u = 0):
        this.x = np.dot(this.F, this.x) + np.dot(this.B, u)
        this.P = np.dot(np.dot(this.F, this.P), this.F.T) + this.Q
        return this.x

    def update(this, z):
        y = z - np.dot(this.H, this.x)
        S = this.R + np.dot(this.H, np.dot(this.P, this.H.T))
        K = np.dot(np.dot(this.P, this.H.T), np.linalg.inv(S))
        this.x = this.x + np.dot(K, y)
        I = np.eye(this.n)
        this.P = np.dot(np.dot(I - np.dot(K, this.H), this.P), 
        	(I - np.dot(K, this.H)).T) + np.dot(np.dot(K, this.R), K.T)

def example():
	dt = 1.0/60
	F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
	H = np.array([1, 0, 0]).reshape(1, 3)
	Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
	R = np.array([0.5]).reshape(1, 1)

	x = np.linspace(-10, 10, 100)
	measurements = - (x**2 + 2*x - 2)  + np.random.normal(0, 2, 100)

	kf = KF(F = F, H = H, Q = Q, R = R)
	predictions = []

	for z in measurements:
		predictions.append(np.dot(H,  kf.predict())[0])
		kf.update(z)

	import matplotlib.pyplot as plt
	plt.plot(range(len(measurements)), measurements, label = 'Measure')
	plt.plot(range(len(predictions)), np.array(predictions), label = 'KF Predict')
	plt.legend()
	plt.show()

if __name__ == '__main__':
    example()