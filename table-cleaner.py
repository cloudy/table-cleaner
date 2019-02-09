#!/usr/bin/env python
#AUTHOR: Joe Cloud

import rospy
import intera_interface
import cv2
import numpy as np

from geometry_msgs.msg import (
        PoseStamped,
        Pose,
        Point,
        Quaternion,
)

from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from intera_core_msgs.srv import (
        SolvePositionIK,
        SolvePositionIKRequest,
)

# Color values in HSV
BLUELOWER = np.array([40, 100, 100])
BLUEUPPER = np.array([130, 255, 255])

# Determines noise clear for morph
KERNELOPEN = np.ones((5,5))
KERNELCLOSE = np.ones((5,5))

# Camera calibration values - Specific to C930e 
CAMERAMATRIX = np.array([[506.857008, 0.000000, 311.541447], 
                         [0.000000, 511.072198, 257.798417], 
                         [0.000000, 0.000000, 1.000000]])
DISTORTION = np.array([0.047441, -0.104070, 0.006161, 0.000338, 0.000000])

CARTIM = [[178, 448], [173, 355]] #[[170, 446], [196, 380]] # [[XX],[YY]] of the calibration points on table
CARTBOT = [[-0.3,0.3], [-0.4,-0.8]] # [[XX],[YY]] for the cartesian EE table values
GOAL = [600,300] # Drop off point of cylinders
ZLOW = -0.065 # Pick up height
ZHIGH = 0.26 # Drop off height (to reach over lip of box)

def main():
    rospy.init_node('table_cleaner')
    
    limb = intera_interface.Limb('right')
    gripper = intera_interface.Gripper('right')
    limb.move_to_neutral()
    gripper.open()
    limb.set_joint_position_speed(0.18)
    
    current_pose = limb.endpoint_pose()
    desired_pose = Pose()
    desired_pose.orientation = current_pose['orientation']

    cap = cv2.VideoCapture(-1)
    if not cap.isOpened():
        exit(1)
    
    blocks = detectBlock(cap)
    while(len(blocks) > 0):
        #go to x,y,z+HIGH of block[0]
        desired_pose.position.x = blocks[0][0]
        desired_pose.position.y = blocks[0][1]
        desired_pose.position.z = ZHIGH
        joint_angles = ik(limb, desired_pose.position, desired_pose.orientation)
        limb.move_to_joint_positions(joint_angles)
        #reduce to x,y,z of block[0]
        desired_pose.position.z = ZLOW
        joint_angles = ik(limb, desired_pose.position, desired_pose.orientation)
        limb.move_to_joint_positions(joint_angles)
        #close gripper
        gripper.close()
        desired_pose.position.z = ZHIGH/3
        joint_angles = ik(limb, desired_pose.position, desired_pose.orientation)
        limb.move_to_joint_positions(joint_angles)
        #move to goal x,y,z+HIGH
        gl = pixelsToCartesian(*GOAL)
        desired_pose.position.x = gl[0]
        desired_pose.position.y = gl[1]
        desired_pose.position.z = ZHIGH
        joint_angles = ik(limb, desired_pose.position, desired_pose.orientation)
        limb.move_to_joint_positions(joint_angles)
        #open gripper
        gripper.open()
        blocks = detectBlock(cap)

# Filters blocks out of image and returns a list of x-y pairs in relation to the end-effector
def detectBlock(cap):
    for i in range(5): cap.grab() # Disregard old frames

    ret_val, im = cap.read()
    while not ret_val: # In case an image is not captured
        ret_val, im = cap.read()

    und_im = cv2.undistort(im, CAMERAMATRIX, DISTORTION) # Remove distortions
    imHSV = cv2.cvtColor(und_im, cv2.COLOR_BGR2HSV)
        
    mask = cv2.inRange(imHSV, BLUELOWER, BLUEUPPER) # Masking out blue cylinders
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNELOPEN)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, KERNELCLOSE)
    
    _, conts, h = cv2.findContours(mask_close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(und_im, conts, -1, (255, 255, 0), 1) # Helpful for visualization
        
    centers = [getCenter(*cv2.boundingRect(c)) for c in conts] # Calc center of each cylinder
    return [pixelsToCartesian(*c) for c in centers] # Return centers (in cartesian instead of pixels)

# Takes pixel representation of table/centroids and converts it to cartesian space
# (in reference to robot end-effector). Originally by Michail Theofanidis, adapted by Joe Cloud
def pixelsToCartesian(cx, cy):
    a_y = (CARTBOT[1][0]-CARTBOT[1][1])/(CARTIM[1][1]-CARTIM[1][0])
    b_y = CARTBOT[1][1]-a_y*CARTIM[1][0]
    y = a_y*cy+b_y
    a_x = (CARTBOT[0][0]-CARTBOT[0][1])/(CARTIM[0][1]-CARTIM[0][0])
    b_x = CARTBOT[0][1]-a_x*CARTIM[0][0]
    x = a_x*cx+b_x
    return (x,y)

def getCenter(x, y, w, h):
    return (((int)(x + 0.5*w)), ((int)(y + 0.5*h)))


# Adapted from Intera Examples by Michail Theofanidis, further modified by Joe Cloud
def ik(limb, coordinates, orientation):
    angles=limb.joint_angles()
    ns = "ExternalTools/right/PositionKinematicsNode/IKService"
    iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
    ikreq = SolvePositionIKRequest()
    hdr = Header(stamp=rospy.Time.now(), frame_id='base')
    poses = {
            'right': PoseStamped(
                header=hdr,
		pose=Pose(
		position=Point(
		x=coordinates.x,
		y=coordinates.y,
		z=coordinates.z,
		),
		orientation=Quaternion(
		x=orientation.x,
		y=orientation.y,
		z=orientation.z,
		w=orientation.w,),
		),),}

    ikreq.pose_stamp.append(poses['right'])
    ikreq.tip_names.append('right_hand')
    ikreq.seed_mode = ikreq.SEED_USER
    
    seed = JointState()
    seed.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']
    seed.position = [angles[a] for a in seed.name]
    ikreq.seed_angles.append(seed)
    
    # Optimize the null space in terms of joint configuration
    ikreq.use_nullspace_goal.append(True)
    goal = JointState()
    goal.name = ['right_j2']
    goal.position = [0]
    ikreq.nullspace_goal.append(goal)
    ikreq.nullspace_gain.append(0.4)
    
    try:
    	rospy.wait_for_service(ns, 5.0)
    	resp = iksvc(ikreq)
    except (rospy.ServiceException, rospy.ROSException), e:
    	rospy.logerr("Service call failed: %s" % (e,))
    
    limb_joints = angles 
    # Check if result valid, and type of seed ultimately used to get solution
    if (resp.result_type[0] > 0):
    	seed_str = {ikreq.SEED_USER: 'User Provided Seed',
    		    ikreq.SEED_CURRENT: 'Current Joint Angles',
    		    ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
    		    }.get(resp.result_type[0], 'None')
    	limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
    
    return limb_joints

if __name__ == '__main__':
    main()

