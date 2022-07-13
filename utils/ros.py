from typing import List, Tuple

from geometry_msgs.msg import Point, Quaternion, Pose


def point_msg_to_list(point_msg: Point) -> List[float]:
    """
    Convert point msg to a float list.
    Args:
        point_msg (Point): point msg

    Returns:
        List[float]: converted point list
    """
    return [point_msg.x, point_msg.y, point_msg.z]


def quat_msg_to_list(quat_msg: Quaternion) -> List[float]:
    """
    Convert quaternion msg to a float list.
    Args:
        quat_msg (Quaternion): quaternion msg

    Returns:
        List[float]: converted quaternion list
    """
    return [quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w]


def pose_msg_to_point_quat_list(
    pose_msg: Pose,
) -> Tuple[List[float], List[float]]:
    """
    Convert pose msg to a tuple of point list and quaternion list.
    Args:
        pose_msg (Pose): pose msg

    Returns:
        List[float]: converted point list
        List[float]: converted quaternion list
    """
    point_msg = pose_msg.position
    point_list = point_msg_to_list(point_msg)
    quat_msg = pose_msg.orientation
    quat_list = quat_msg_to_list(quat_msg)
    return point_list, quat_list
