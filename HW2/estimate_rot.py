#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates 
#roll pitch and yaw using an extended kalman filter

from scipy import io
import os
import numpy as np


def quat_average(q, q0): # checked # q:(7,4) q0:(4,)
    qt = q0
    r, c = q.shape #r=7,
    epsilon = 0.0001
    error = np.zeros((r,3)) #(7,3)
    for _ in range(1000):
        for i in range(r):
            qi_error = normalize_quaternion(multiply_quaternions(q[i, :], inverse_quaternion(qt))) # (52)
            ev_error = quat2vec(qi_error) # (3,)
            if np.linalg.norm(ev_error) == 0: # not rotate
                error[i:] = np.zeros(3)
            else:
                error[i,:] = (-np.pi + np.mod(np.linalg.norm(ev_error) + np.pi, 2 * np.pi)) / np.linalg.norm(ev_error) * ev_error
        error_mean = np.mean(error, axis=0)
        qt = normalize_quaternion(multiply_quaternions(vec2quat(error_mean), qt))
        if np.linalg.norm(error_mean) < epsilon:
            return qt, error
        error = np.zeros((r,3))


def multiply_quaternions(q, r):  # checked
    t = np.zeros(4)
    t[0] = r[0]*q[0] - r[1]*q[1] - r[2]*q[2] - r[3]*q[3]
    t[1] = r[0]*q[1] + r[1]*q[0] - r[2]*q[3] + r[3]*q[2]
    t[2] = r[0]*q[2] + r[1]*q[3] + r[2]*q[0] - r[3]*q[1]
    t[3] = r[0]*q[3] - r[1]*q[2] + r[2]*q[1] + r[3]*q[0]
    return t

def inverse_quaternion(q): # checked
    t = np.zeros(4) 
    norm = np.power(norm_quaternion(q), 2)
    t[0] = q[0] / norm
    t[1] = -q[1] / norm
    t[2] = -q[2] / norm
    t[3] = -q[3] / norm
    return t

def norm_quaternion(q):  # checked
    t = np.sqrt(np.sum(np.power(q, 2)))
    return t

def normalize_quaternion(q): # checked
    return q/norm_quaternion(q)

# euler angle refers to roll, pitch and yaw
def quat2euler(q):  # given transformation
    r = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
    p = np.arcsin(2*(q[0]*q[2] - q[3]*q[1]))
    y = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))
    return r, p, y

def quat2euler_vectorized(q):  # given transformation
    r = np.arctan2(2*(q[:,0]*q[:,1]+q[:,2]*q[:,3]), 1 - 2*(q[:,1]**2 + q[:,2]**2))
    p = np.arcsin(2*(q[:,0]*q[:,2] - q[:,3]*q[:,1]))
    y = np.arctan2(2*(q[:,0]*q[:,3]+q[:,1]*q[:,2]), 1 - 2*(q[:,2]**2 + q[:,3]**2))
    return r, p, y

def vec2quat(r):  # right
    r = r/2.0
    q = np.zeros(4)
    q[0] = np.cos(np.linalg.norm(r))
    if np.linalg.norm(r) == 0:
        temp = np.zeros(3)
    else:
        temp = r/np.linalg.norm(r)* np.sin(np.linalg.norm(r))
    q[1:4] = temp
    return q

def quat2vec(q):  # right
    qs = q[0]
    qv = q[1:4]
    if np.linalg.norm(qv) == 0:
        v = np.zeros(3)
    else:
        v = 2*qv/np.linalg.norm(qv)*np.arccos(qs/np.linalg.norm(q))
    return v

#================================================================================

def gaussian_update(q, P, Q):  # checked
    """
    Section 3.1
    """
    n,c = P.shape # n=3,c=3
    S = np.linalg.cholesky(P+Q)
    left_vec = S * np.sqrt(2*n)
    right_vec = -S * np.sqrt(2*n)
    vec = np.hstack((left_vec, right_vec)) # hstack:axis=1 #(3,6)
    X = np.zeros((2*n, 4)) # (6,4)

    for i in range(2*n): # 6
        qW = vec2quat(vec[:, i]) # (4,)
        X[i, :] = multiply_quaternions(q, qW) 
    # add mean, 2n+1 sigma points in total
    X = np.vstack((q, X)) #(7,4) # add itself
    return X


def sigma_update(X, g, dt): # X:(7,4) g:(3,) dt:scalar
    n = X.shape[0] # n=7
    Y = np.zeros((n,4)) # (7,4)
    # compute delta quaternion
    q_delta = vec2quat(g*dt) #(4,)
    for i in range(n):
        # project sigma points by process model
        q = X[i] #(4,)
        Y[i] = multiply_quaternions(q, q_delta) # q_del * q_w * q
    return Y #(7,4)

def estimate_rot(data_num=1):
    filename = os.path.join(os.path.dirname(__file__),
                            "imu/imuRaw" + str(data_num) + ".mat")
    imu = io.loadmat(filename)  # changed
    imu_vals = imu['vals']
    imu_ts = imu['ts']
    imu_ts = np.array(imu['ts']).T  # (5000,1)

    Vref = 3300

    acc_x = -np.array(imu_vals[0]) # IMU Ax and Ay direction is flipped !
    acc_y = -np.array(imu_vals[1])
    acc_z = np.array(imu_vals[2])
    acc = np.array([acc_x, acc_y, acc_z]).T

    acc_sensitivity = 330.0
    acc_scale_factor = Vref/1023.0/acc_sensitivity
    acc_bias = np.mean(acc[:10], axis=0) - np.array([0,0,1])/acc_scale_factor
    acc = (acc-acc_bias)*acc_scale_factor

    gyro_x = np.array(imu_vals[4]) # angular rates are out of order !
    gyro_y = np.array(imu_vals[5])
    gyro_z = np.array(imu_vals[3])
    gyro = np.array([gyro_x, gyro_y, gyro_z]).T

    gyro_bias = np.mean(gyro[:10], axis=0)
    gyro_sensitivity = 3.33
    gyro_scale_factor = Vref/1023/gyro_sensitivity
    gyro = (gyro-gyro_bias)*gyro_scale_factor*(np.pi/180)

    imu_vals = np.hstack((acc,gyro)) # updated imu_vals  (5000,6)

    P = 0.1*np.identity(3)
    Q = 2*np.identity(3)
    R = 2*np.identity(3)
    qt = np.array([1, 0, 0, 0])
    time = imu_vals.shape[0] #5000

    predicted_q = qt

    for i in range(time):

        acc = imu_vals[i,:3]  # (3, )
        gyro = imu_vals[i,3:] # (3, )

        X = gaussian_update(qt, P, Q)  # (7,4)
        if i == time-1:
            dt = imu_ts[-1] - imu_ts[-2])
        else:
            dt = imu_ts[i+1] - imu_ts[i]
        # Process model
        Y = sigma_update(X, gyro, dt) #(7,4)
        # compute mean
        x_k_bar, error = quat_average(Y, qt) # 38,39 # (4,)  error:(7,3)
        # compute covariance (in vector)
        P_k_bar = np.zeros((3, 3)) # (3,3)
        for i in range(7):
            P_k_bar += np.outer(error[i,:], error[i,:])
        P_k_bar /= 7

        # measurement model
        g = np.array([0, 0, 0, 1]) # unchange? # down

        Z = np.zeros((7, 3)) # vector quaternions
        for i in range(7):
            # compute predicted acceleration
            q = Y[i]
            Z[i] = multiply_quaternions(multiply_quaternions(inverse_quaternion(q), g), q)[1:] # rotate from body frame to global frame
    
        # measurement mean
        z_k_bar = np.mean(Z, axis=0) #(3,)
        z_k_bar /= np.linalg.norm(z_k_bar)

        # measurement cov and correlation
        Pzz = np.zeros((3, 3))
        Pxz = np.zeros((3, 3))
        Z_err = Z - z_k_bar
        for i in range(7):
            Pzz += np.outer(Z_err[i,:], Z_err[i,:])
            Pxz += np.outer(error[i,:], Z_err[i,:])
        Pzz /= 7
        Pxz /= 7
        # innovation
        acc /= np.linalg.norm(acc)
        vk = acc - z_k_bar # 44
        Pvv = Pzz + R      # 45
        # compute Kalman gain
        K = np.dot(Pxz,np.linalg.inv(Pvv)) # 72 # (3,3)
        # update
        q_gain = vec2quat(K.dot(vk))  # 74
        q_update = multiply_quaternions(q_gain,x_k_bar) # transition
        P_update = P_k_bar - K.dot(Pvv).dot(K.T) # 75
        P = P_update
        qt = q_update

        predicted_q = np.vstack((predicted_q, qt))

    roll = np.zeros(np.shape(predicted_q)[0])
    pitch = np.zeros(np.shape(predicted_q)[0])
    yaw = np.zeros(np.shape(predicted_q)[0])

    for i in range(np.shape(predicted_q)[0]):
        roll[i], pitch[i], yaw[i] = quat2euler(predicted_q[i])
    return roll, pitch, yaw

a,b,c = estimate_rot(1)
