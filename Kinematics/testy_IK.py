from kinematic_model import EEZYbotARM_Mk2

FakeDaddy = EEZYbotARM_Mk2(initial_q1=0, initial_q2=70, initial_q3=-100)
FakeDaddy.plot()  


x = 240  # mm
y = 85  # mm
z = 200  # mm

a1, a2, a3 = FakeDaddy.inverseKinematics(x, y, z)

print('To move the end effector to the cartesian position (mm) x={}, y={}, z={}, the robot arm joint angles (degrees)  are q1 = {}, q2= {}, q3 = {}'.format(x, y, z, a1, a2, a3))

FakeDaddy.updateJointAngles(q1=a1, q2=a2, q3=a3)
FakeDaddy.plot()