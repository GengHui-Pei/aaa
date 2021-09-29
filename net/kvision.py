import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import torch,cv2
from mpl_toolkits.mplot3d import Axes3D
__CLASSES__ = ['_background_', 'hand', 'A', 'B', 'C', 'D',
               'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
               'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
               'X', 'Y']
def show_point_cloud(point,color=[0,0,1]):
    # point = point.numpy()
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point))
    # point_cloud.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # point_cloud.paint_uniform_color(color)
    return point_cloud

def show_3d_hm(heatmap): # [1, 21, 64, 64, 64]
    x, y, z = heatmap.shape
    gridxx, gridyy, gridzz = np.mgrid[:x, :y, :z]
    point = np.concatenate((gridxx[..., np.newaxis, np.newaxis],
                            gridyy[..., np.newaxis, np.newaxis],
                            gridzz[..., np.newaxis, np.newaxis]), axis=3).reshape(-1,3)
    point = point[heatmap.reshape(-1)>50]
    print(point.shape)
    color = heatmap.reshape(-1)[heatmap.reshape(-1)>50][:,None].numpy()@np.array([[1., 0., 1.]])/255.
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point))
    point_cloud.colors = o3d.utility.Vector3dVector(color)
    return point_cloud

def show_RGB(rgb,depth_val=0):
    gridyy, gridxx = np.mgrid[:rgb.shape[0], :rgb.shape[1]]
    depth_bg = np.zeros_like(gridxx)[..., np.newaxis] +depth_val
    point = np.concatenate((gridxx[..., np.newaxis], gridyy[..., np.newaxis], depth_bg), axis=2).reshape(-1,3)
    point_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(point[rgb.reshape(-1,3)[:,0]>50]))
    point_cloud.colors = o3d.utility.Vector3dVector(rgb.reshape(-1,3)[rgb.reshape(-1,3)[:,0]>50]/-255.+1)
    return point_cloud
def show_joint(joint):
    # lines = np.array([[0, 4], [4, 3], [3, 2], [2, 1],
    #                   [0, 8], [8, 7], [7, 6], [6, 5],
    #                   [0, 12], [12, 11], [11, 10], [10, 9],
    #                   [0, 16], [16, 15], [15, 14], [14, 13],
    #                   [0, 20], [20, 19], [19, 18], [18, 17],
    #                   ])
    # lines = np.array([[0, 1], [1, 2], [2, 3],
    #                   [0, 4], [4, 5], [5, 6],
    #                   [0, 7], [7, 8], [8, 9],
    #                   [0, 10], [10, 11], [11, 12],
    #                   [0, 13], [13, 14], [14, 15]
    #                   ])
    lines = np.array([[0, 1], [1, 2], [2, 3], [3, 4],
                      [0, 5], [5, 6], [6, 7], [7, 8],
                      [0, 9], [9, 10], [10, 11], [11, 12],
                      [0, 13], [13, 14], [14, 15], [15, 16],
                      [0, 17], [17, 18], [18, 19], [19, 20],
                      ])
    color_hand_joints = [  # [1.0, 0.0, 0.0],
        [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
        [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
        [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
        [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
        [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little
    # joint[:,2] = joint[:,2]+593
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(joint), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector([color_hand_joints[i] for i in range(len(lines))])
    line_set.points = o3d.utility.Vector3dVector(joint)
    return line_set
def show_RGBD(rgb,depth):
    color_raw = o3d.geometry.Image(rgb)
    depth_raw = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw,convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the pointcloud will be upside down
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    return pcd
def show_mesh(f,v,color=1):
    mesh = o3d.geometry.TriangleMesh()
    mesh.triangles = o3d.utility.Vector3iVector(f)
    mesh.vertices = o3d.utility.Vector3dVector(v)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([228 / 255, 178*color / 255, 148 / 255])
    return mesh
def show_o3d(*karge,size):
    lines = np.array([[0, 1], [0, 2], [0, 3]])
    color = [[1,0,0],[0,1,0],[0,0,1]] # {x:r, y: g, z: b}
    coord_xyz = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [0, 2, 0],
                         [0, 0, 1]])*size
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(coord_xyz), lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector([color[i] for i in range(len(lines))])
    line_set.points = o3d.utility.Vector3dVector(coord_xyz)
    # line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.visualization.draw_geometries([line_set,*karge])
lines = np.array([[0, 1],[1, 2],[2, 3],[3, 4],
                  [0, 5],[5, 6],[6, 7],[7, 8],
                  [0, 9],[9, 10],[10, 11],[11, 12],
                  [0, 13],[13, 14],[14, 15],[15, 16],
                  [0, 17],[17, 18],[18, 19],[19, 20],
                  ])
color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little
def depth_to_point(depth_bg):

    gridyy, gridxx = np.mgrid[:depth_bg.shape[0], :depth_bg.shape[1]]
    depth_bg = depth_bg[..., np.newaxis]
    bg_xyz = np.concatenate((gridxx[..., np.newaxis], gridyy[..., np.newaxis], depth_bg), axis=2)
    bg_xyz_resh_all = np.reshape(bg_xyz, (-1, 3))
    bg_xyz_resh = bg_xyz_resh_all[bg_xyz_resh_all[:, 2] > 0]
    return bg_xyz_resh

def draw_2d_skeleton(ax,est_pose_uv, markersize=8, linewidth=2):
    pose_cam_xyz = est_pose_uv
    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1]
                 , '.', markersize=markersize, color = color_hand_joints[joint_ind])
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1],
                     lineWidth=linewidth, color = color_hand_joints[joint_ind])
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                     linewidth=linewidth, color = color_hand_joints[joint_ind])
import inspect

def retrieve_name_ex(var):
    stacks = inspect.stack()
    try:
        callFunc = stacks[1].function
        code = stacks[2].code_context[0]
        startIndex = code.index(callFunc)
        startIndex = code.index("(", startIndex + len(callFunc)) + 1
        endIndex = code.index(")", startIndex)
        return code[startIndex:endIndex].strip()
    except:
        return ""

def kprint(var):
    print("--------------------\n{} = {}".format(retrieve_name_ex(var),var))

def skeleton_inter2mano(joint3d):
    #                           画图中关节点分布编号规则
    #               Mano                   |         InterHand
    # ----------------------------------------------------------------------
    #        16   12                       #        12   8
    #   20   |    |    8                   #   16   |    |    4
    #   |    |    |    |                   #   |    |    |    |
    #   19   15   11   7         4         #   17   13   9   5         0
    #   |    |    |    |        /          #   |    |    |    |        /
    #   18   14   10   6       3           #   18   14   10   6       1
    #   |    |    |    |      /            #   |    |    |    |      /
    #   17   13   9    5     2             #   19   15   11    7     2
    #    \   |    |   /    /               #    \   |    |   /    /
    #      \  \   /       1                #      \  \   /       3
    #        \  /                          #        \  /
    #          0                           #         20
    # ----------------------------------------------------------------------
    joint3d[:21,:] = joint3d[[20,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,19,18,17,16],:]
    return joint3d

def skeleton_mano2inter(joint3d):
    #                           画图中关节点分布编号规则
    #               Mano                   #         InterHand
    # ----------------------------------------------------------------------
    #        16   12                       #        12   8
    #   20   |    |    8                   #   16   |    |    4
    #   |    |    |    |                   #   |    |    |    |
    #   19   15   11   7         4         #   17   13   9   5         0
    #   |    |    |    |        /          #   |    |    |    |        /
    #   18   14   10   6       3           #   18   14   10   6       1
    #   |    |    |    |      /            #   |    |    |    |      /
    #   17   13   9    5     2             #   19   15   11    7     2
    #    \   |    |   /    /               #    \   |    |   /    /
    #      \  \   /       1                #      \  \   /       3
    #        \  /                          #        \  /
    #          0                           #         20
    # ----------------------------------------------------------------------
    joint3d[:21,:] = joint3d[[4,3,2,1,8,7,6,5,12,11,10,9,16,15,14,13,20,19,18,17,0],:]
    return joint3d

def draw1_2d_skeleton(image, pose_uv):
    """
    :param image: H x W x 3
    :param pose_uv: 21 x 2
    wrist,
    thumb_mcp, thumb_pip, thumb_dip, thumb_tip
    index_mcp, index_pip, index_dip, index_tip,
    middle_mcp, middle_pip, middle_dip, middle_tip,
    ring_mcp, ring_pip, ring_dip, ring_tip,
    little_mcp, little_pip, little_dip, little_tip
    :return:
    """
    assert pose_uv.shape[0] == 21
    skeleton_overlay = image.copy()

    marker_sz = 2
    line_wd = 1
    root_ind = 0

    for joint_ind in range(pose_uv.shape[0]):
        joint = pose_uv[joint_ind, 0].astype('int32'), pose_uv[joint_ind, 1].astype('int32')
        cv2.circle(
            skeleton_overlay, joint,
            radius=marker_sz, color=color_hand_joints[joint_ind] * np.array(255), thickness=-1,
            lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            root_joint = pose_uv[root_ind, 0].astype('int32'), pose_uv[root_ind, 1].astype('int32')
            cv2.line(
                skeleton_overlay, root_joint, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)
        else:
            joint_2 = pose_uv[joint_ind - 1, 0].astype('int32'), pose_uv[joint_ind - 1, 1].astype('int32')
            cv2.line(
                skeleton_overlay, joint_2, joint,
                color=color_hand_joints[joint_ind] * np.array(255), thickness=int(line_wd),
                lineType=cv2.CV_AA if cv2.__version__.startswith('2') else cv2.LINE_AA)

    return skeleton_overlay

def draw1_3d_skeleton(pose_cam_xyz, image_size):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    fig = plt.figure()
    fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(111, projection='3d')
    marker_sz = 10
    line_wd = 4

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 2],
                pose_cam_xyz[joint_ind:joint_ind + 1, 1], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 2], pose_cam_xyz[[0, joint_ind], 1],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 2],
                    pose_cam_xyz[[joint_ind - 1, joint_ind], 1], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)

    # ax.axis('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(90, 0)  # x z

    ret = fig2data(fig)  # H x W x 4
    plt.close(fig)
    return ret
def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf
def draw_3d_skeleton(pose_cam_xyz,ss,name=None):
    """
    :param pose_cam_xyz: 21 x 3
    :param image_size: H, W
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21

    fig = plt.figure(1, figsize=(16, 16))
    # fig.set_size_inches(float(image_size[0]) / fig.dpi, float(image_size[1]) / fig.dpi, forward=True)

    ax = plt.subplot(ss, projection='3d')
    marker_sz = 10
    line_wd = 3

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)

        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], lineWidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    pose_cam_xyz[[joint_ind - 1, joint_ind], 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)
    # ax.scatter(np.array(1),np.array(1),np.array(1),s=1)
    # ax.scatter(np.array(-1), np.array(-1), np.array(-1),s=1)
    # ax.axis('equal')
    ax.set_title(name)
    ax.set_xlabel('X')
    # ax.set_xlim3d(-0.5, 2)
    ax.set_ylabel('Y')
    # ax.set_ylim3d(-1.5, 1)
    ax.set_zlabel('Z')
    # ax.set_zlim3d(-1.5,1)
    # plt.show()
    # ax.set_xlim3d(-1, 1)
    # ax.set_ylim3d(-1, 1)
    # ax.set_zlim3d(-1, 1)
    ax.view_init(elev=-75, azim=-90)
    # ax.view_init(elev=-90, azim=-90)
    # ret = fig2data(fig)  # H x W x 4
    # plt.close(fig)
    return fig
def world2hand(kp_coord_xyz,viewRotation):
    rx, ry, rz, scale, tx, ty, tz  = viewRotation
    kp_coord_xyz_t = kp_coord_xyz + torch.tensor([[tx,ty,tz]])
    kp_coord_xyz_t_s = kp_coord_xyz_t / scale
    local_joint_rz = Stn(kp_coord_xyz_t_s).r(torch.zeros(1), torch.zeros(1), rz)
    local_joint_rz_rx = Stn(local_joint_rz).r(rx, torch.zeros(1), torch.zeros(1))
    local_joint_xyz = Stn(local_joint_rz_rx).r(torch.zeros(1), ry, torch.zeros(1))
    return local_joint_xyz



def local2pix(local,r,t,s):
    local_ = torch.cat((local, torch.ones((local.size(0), 21, 1)).cuda()), -1)
    cx, sx, cy, sy, cz, sz = torch.cos(r[:, 0]), torch.sin(r[:, 0]), \
                             torch.cos(r[:, 1]), torch.sin(r[:, 1]), \
                             torch.cos(r[:, 2]), torch.sin(r[:, 2])
    zero = torch.zeros_like(cx)
    one = torch.ones_like(cx)
    s = s.view(local.size(0),1,1)
    t = t.view(local.size(0), 1, 3)
    RTS = torch.stack((s[:, 0, 0] * (cy * cz + sx * sy * sz), -cy * sz + sx * sy * cz, sy * cx, zero,
                       cx * sz, s[:, 0, 0] * (cx * cz), -sx, zero,
                       -sy * cz + cy * sx * sz, sy * sz + cy * sx * cz, s[:, 0, 0] * (cy * cx), zero,
                       t[:, 0, 0], t[:, 0, 1], t[:, 0, 2], one), -1).view(-1, 4, 4)
    return (local_@RTS)[:,:,:3]

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return img_coord

def world2pix(world_joint):
    # for rgb image
    K = torch.tensor([[607.92271, 0, 314.78337],
                    [0, 607.88192, 236.42484],
                    [0, 0, 1]])
    # External camera parameters
    T = torch.tensor([-24.0381, -0.4563, -1.2326]).reshape([3,1])
    rotationMatrix = torch.tensor([[ 0.99992395, -0.00304166, -0.01195165],
                                  [ 0.00297816,  0.99998137, -0.00532784],
                                  [ 0.01196763,  0.00529184,  0.99991438]])
    # pix_joint = np.dot(K, np.dot(np.transpose(rotationMatrix), np.transpose(world_joint) - T))
    pix_joint = K @ (rotationMatrix @ np.transpose(world_joint)-T)
    pix_joint = pix_joint / pix_joint[-1, ...]
    pix_joint = np.transpose(pix_joint)
    # for k in range(21):
    #     pix_joint[k, :] = pix_joint[k, :] / pix_joint[k, 2]
    # # pix_joint = pix_joint / pix_joint[-1, ...]
    # print('pix_jointaaaaaaa',pix_joint)

    crop_center_rgb = pix_joint[9, :2]
    crop_size_rgb = torch.max(torch.abs(pix_joint[:, :2] - crop_center_rgb) + 10)
    crop_pix_joint = pix_joint - pix_joint[9, :]
    crop_pix_joint[:, :2] += crop_size_rgb
    crop_pix_joint = crop_pix_joint * (128 / crop_size_rgb)
    return crop_pix_joint

def pix2norm(pix_joint):
    return pix_joint / 128 - 1

def show_test(rgb,local_joint_xyz,pix_joint,view,i):

    draw_3d_skeleton(local_joint_xyz, 241+i)
    plt.title('local_joint')

    draw_3d_skeleton(pix_joint, 242+i)
    plt.title(f'x:{view[0]:.2f}| '+
              f'y:{view[1]:.2f}| '+
              f'z:{view[2]:.2f}| '+
              '\n'+
              f'tx:{view[3]:.2f}| '+
              f'ty:{view[4]:.2f}| '+
              f'tz:{view[5]:.2f}| '+
              '\n'+
              f's:{view[6]:.2f}'
              )

    plt.subplot(243+i)
    plt.title('r_s_t_joint')
    draw_2d_skeleton(pix_joint)
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.gca().invert_yaxis()

    plt.subplot(244+i)
    plt.title('pix_joint')
    plt.imshow(rgb)
    draw_2d_skeleton(pix_joint * 128 + 128)

def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(1).transpose(1, -1).squeeze(-1)

def kquat2mat(pred_r):
    # bs, num_p, _ = pred_c.size()
    bs = pred_r.size(0)
    num_p = 1
    pred_r = pred_r.view(bs,1,5)
    return torch.cat((((1.0 - 2.0 * (pred_r[:, :, 2] ** 2 + pred_r[:, :, 3] ** 2))*(pred_r[:,:,4]+1)).view(bs, num_p, 1),
                      (2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] - 2.0 * pred_r[:, :, 0] * pred_r[:, :, 3]).view(bs,num_p,1),
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3]).view(bs,num_p,1),

                      ((2.0 * pred_r[:, :, 1] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 3] * pred_r[:, :, 0])*(pred_r[:,:,4]+1)).view(bs,num_p,1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 3] ** 2)).view(bs, num_p, 1), \
                      (-2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,num_p,1), \

                      ((-2.0 * pred_r[:, :, 0] * pred_r[:, :, 2] + 2.0 * pred_r[:, :, 1] * pred_r[:, :, 3])*(pred_r[:,:,4]+1)).view(bs,num_p,1), \
                      (2.0 * pred_r[:, :, 0] * pred_r[:, :, 1] + 2.0 * pred_r[:, :, 2] * pred_r[:, :, 3]).view(bs,num_p,1), \
                      (1.0 - 2.0 * (pred_r[:, :, 1] ** 2 + pred_r[:, :, 2] ** 2)).view(bs, num_p, 1)),dim=2).contiguous().view(bs * num_p, 3, 3)

def save2_joint_img(rgb,pix_joint):
    sss = torch.zeros_like(rgb.cpu().transpose(3, 1).transpose(1, 2))
    pix_joint = pix_joint * 128 + 128

    for i in range(rgb.size(0)):
        skeleton_overlay = draw1_2d_skeleton((rgb[i] * 255+127.5).cpu().transpose(2, 0).transpose(0, 1).to(torch.uint8),pix_joint[i].numpy())/255.
        sss[i] = torch.from_numpy(skeleton_overlay)

    return BHWC_to_BCHW(sss)

def cv2_show_bboxs(img,bbox):
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)


def plot_results(pil_img,boxes,labels):
    plt.imshow(pil_img[..., [2, 1, 0]])
    ax = plt.gca()
    for label, (xmin, ymin, width, height) in zip(labels, boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), width, height,fill=False,color='r', linewidth=3))
        ax.text(xmin, ymin, __CLASSES__[label], fontsize=15,bbox=dict(facecolor='yellow', alpha=0.5))

def transform_points(points,
                     matrix,
                     translate=True):
    """
    Returns points rotated by a homogeneous
    transformation matrix.

    If points are (n, 2) matrix must be (3, 3)
    If points are (n, 3) matrix must be (4, 4)

    Parameters
    ----------
    points : (n, D) float
      Points where D is 2 or 3
    matrix : (3, 3) or (4, 4) float
      Homogeneous rotation matrix
    translate : bool
      Apply translation from matrix or not

    Returns
    ----------
    transformed : (n, d) float
      Transformed points
    """
    points = np.asanyarray(
        points, dtype=np.float64)
    # no points no cry
    if len(points) == 0:
        return points.copy()

    matrix = np.asanyarray(matrix, dtype=np.float64)
    if (len(points.shape) != 2 or
            (points.shape[1] + 1 != matrix.shape[1])):
        raise ValueError('matrix shape ({}) doesn\'t match points ({})'.format(
            matrix.shape,
            points.shape))

    # check to see if we've been passed an identity matrix
    identity = np.abs(matrix - np.eye(matrix.shape[0])).max()
    if identity < 1e-8:
        return np.ascontiguousarray(points.copy())

    dimension = points.shape[1]
    column = np.zeros(len(points)) + int(bool(translate))
    stacked = np.column_stack((points, column))
    transformed = np.dot(matrix, stacked.T).T[:, :dimension]
    transformed = np.ascontiguousarray(transformed)
    return transformed

# def save_joint_img(rgb,local_joint_xyz,viewRotation):
#     sss = torch.zeros_like(rgb.cpu().transpose(3, 1).transpose(1, 2))
#     for i in range(rgb.size(0)):
#         world_joint = hand2world(local_joint_xyz[i], viewRotation[i])
#         crop_pix_joint = world2pix(world_joint)
#         skeleton_overlay = draw1_2d_skeleton((rgb[i] * 255).cpu().transpose(2, 0).transpose(0, 1).to(torch.uint8),crop_pix_joint.numpy())/255.
#         sss[i] = torch.from_numpy(skeleton_overlay)
#
#     return BHWC_to_BCHW(sss)

def calc_joint_error(pre_joint, gt_joint):
    return np.sqrt(np.sum((pre_joint - gt_joint) ** 2))


