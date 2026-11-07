from kinematic_model import EEZYbotARM_Mk2
RealDaddy = EEZYbotARM_Mk2(initial_q1=0, initial_q2=70, initial_q3=-100)
RealDaddy.plot()

a1 = 20  # joint angle 1
a2 = 80  # joint angle 2
a3 = -90  # joint angle 3

x, y, z = RealDaddy.forwardKinematics(q1=a1, q2=a2, q3=a3)

# Print the result
print('With joint angles(degrees) q1={}, q2={}, q3={}, the cartesian position of the end effector(mm) is x={}, y={}, z={}'.format(a1, a2, a3, x, y, z))

# Visualise the new joint angles
RealDaddy.updateJointAngles(q1=a1, q2=a2, q3=a3)
RealDaddy.plot()