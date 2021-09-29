# -*- coding: UTF-8 -*-
import numpy as np
from natsort import natsorted
import os.path as osp
import os,json,math
import torch,random,cv2
# from .render import Rander_hand
import matplotlib.pyplot as plt

def estimateSimilarityUmeyama(SourceHom, TargetHom):
    # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)  # SourceCentroid.shape = (1, 3)
    TargetCentroid = np.mean(TargetHom[:3, :], axis=1)  # TargetCentroid.shape = (1, 3)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()
    CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints
    if np.isnan(CovMatrix).any():
        print('nPoints:', nPoints)
        print(SourceHom.shape)
        print(TargetHom.shape)
        raise RuntimeError('There are NANs in the input.')

    U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
    d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
    if d:
        D[-1] = -D[-1]
        U[:, -1] = -U[:, -1]
    # rotation
    Rotation = np.matmul(U, Vh)
    # scale
    varP = np.var(SourceHom[:3, :], axis=1).sum()
    Scale = 1 / varP * np.sum(D)
    # translation
    Translation = TargetHom[:3, :].mean(axis=1) - SourceHom[:3, :].mean(axis=1).dot(Scale*Rotation.T)
    # transformation matrix
    OutTransform = np.identity(4)
    OutTransform[:3, :3] = Scale * Rotation
    OutTransform[:3, 3] = Translation

    return Scale, Rotation, Translation, OutTransform

def estimateSimilarityTransform(source, target, verbose=False):
#     """ Add RANSAC algorithm to account for outliers.
#     添加RANSAC算法以解决异常值。
#     """
    #  source = (N, 3),   target =  (N, 3)
    assert source.shape[0] == target.shape[0], 'Source and Target must have same number of points.'
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([target.shape[0], 1])]))
    SourceCentroid = np.mean(SourceHom[:3, :], axis=1)
    nPoints = SourceHom.shape[1]
    CenteredSource = SourceHom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
    SourceDiameter = 2 * np.amax(np.linalg.norm(CenteredSource, axis=0))
    InlierT = SourceDiameter / 10.0
    maxIter = 128
    confidence = 0.99

    if verbose:
        print('Inlier threshold: ', InlierT)
        print('Max number of iterations: ', maxIter)

    BestInlierRatio = 0
    BestInlierIdx = np.arange(nPoints)

    for i in range(0, maxIter):
        RandIdx = np.random.randint(nPoints, size=5)
        Scale, _, _, OutTransform = estimateSimilarityUmeyama(SourceHom[:, RandIdx], TargetHom[:, RandIdx])
        PassThreshold = Scale * InlierT
        Diff = TargetHom - np.matmul(OutTransform, SourceHom)
        ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
        InlierIdx = np.where(ResidualVec < PassThreshold)[0]
        nInliers = InlierIdx.shape[0]
        InlierRatio = nInliers / nPoints
        # update best hypothesis
        if InlierRatio > BestInlierRatio:
            BestInlierRatio = InlierRatio
            BestInlierIdx = InlierIdx
        if verbose:
            print('Iteration: ', i)
            print('Inlier ratio: ', BestInlierRatio)
        # early break
        if (1 - (1 - BestInlierRatio ** 5) ** i) > confidence:
            break
    if BestInlierRatio < 0.1:
        print('[ WARN ] - Something is wrong. Small BestInlierRatio: ', BestInlierRatio)
        return None, None, None, None

    SourceInliersHom = SourceHom[:, BestInlierIdx]  # SourceInliersHom =  (4, 654)
    TargetInliersHom = TargetHom[:, BestInlierIdx]  # TargetInliersHom =  (4, 654)
    Scale, Rotation, Translation, OutTransform = estimateSimilarityUmeyama(SourceInliersHom, TargetInliersHom)

    if verbose:
        print('BestInlierRatio:', BestInlierRatio)
        print('Rotation:\n', Rotation)
        print('Translation:\n', Translation)
        print('Scale:', Scale)

    return Scale, Rotation, Translation, OutTransform

def get_frame_imgPaths(dir_path, choosen_img_num):
    """
    根据kn视角下，对象名字、手势id、选定的时刻编号，得到统一时刻四个视角的图片读取路径
    """

    kn_path = dir_path + '/Kinect/color'
    rs0_path = dir_path + '/Realsense/camera0/color_frame'
    rs1_path = dir_path + '/Realsense/camera1/color_frame'
    rs2_path = dir_path + '/Realsense/camera2/color_frame'

    rs0_img_names = [img for img in natsorted(os.listdir(rs0_path)) if '.jpg' in img]
    rs1_img_names = [img for img in natsorted(os.listdir(rs1_path)) if '.jpg' in img]
    rs2_img_names = [img for img in natsorted(os.listdir(rs2_path)) if '.jpg' in img]

    kn_img_time = np.array([int(choosen_img_num)])
    rs0_img_time = np.array([int(osp.splitext(t)[0]) for t in rs0_img_names])
    rs1_img_time = np.array([int(osp.splitext(t)[0]) for t in rs1_img_names])
    rs2_img_time = np.array([int(osp.splitext(t)[0]) for t in rs2_img_names])

    rs0_match_mat = rs0_img_time - kn_img_time
    rs1_match_mat = rs1_img_time - kn_img_time
    rs2_match_mat = rs2_img_time - kn_img_time
    rs0_match_id = np.argmin(np.abs(rs0_match_mat), axis=0)
    rs1_match_id = np.argmin(np.abs(rs1_match_mat), axis=0)
    rs2_match_id = np.argmin(np.abs(rs2_match_mat), axis=0)

    kn_path = kn_path + '/' + choosen_img_num + '.jpg'
    rs0_path = rs0_path + '/' + rs0_img_names[int(rs0_match_id)]
    rs1_path = rs1_path + '/' + rs1_img_names[int(rs1_match_id)]
    rs2_path = rs2_path + '/' + rs2_img_names[int(rs2_match_id)]

    return [kn_path, rs0_path, rs1_path, rs2_path]

def prep_hand_annot(data_root, subject_name, gesture_id):
    '''
    @数据预处理：将方法中smplx-->box_joint-->orgImg_joint-->cam_joint的弱投影过程
                转换为强投影矩阵 mano2cam0, smplx2cam0
    Param data_root: 数据集的root_path
    Param subject_name: ['gyt,...]
    Param gesture_id: ('1', '45')

    saved data sturcture: --> .json
                    dict {subject_name:
                            {gesture_id:
                               {time_id: {"img_path":img_4_cam_paths,
                                          "shape": shape.numpy(),
                                          "pose": pose.numpy(),
                                          "mano2cam0": mano2cam0,
                                          "smplx2cam0": smplx2cam0}
                               }
                            }
                          }
    '''
    # AmbHand_Dataset/gyt/1/right
    dir_path = os.path.join(data_root, '%s/%s/right' % (subject_name, gesture_id))
    # AmbHand_Dataset/gyt/1/right/Kinect/color
    kn_dir = os.path.join(dir_path, 'Kinect/color')
    # ['1610438321163', '1610438321231',...,'1610438330430']
    kn_time_ids = [os.path.splitext(name)[0] for name in natsorted(os.listdir(kn_dir)) if '.jpg' in name]
    hand_data_frame,dict_time,dict_getsure = {},{},{}
    for time_id in kn_time_ids:
        print(time_id)
        # img_4_cam_paths:['AmbHand_Dataset/gyt/1/right/Kinect/color/1610438321163.jpg',
        #                  'AmbHand_Dataset/gyt/1/right/Realsense/camera0/color_frame/1610438321157.jpg',
        #                  'AmbHand_Dataset/gyt/1/right/Realsense/camera1/color_frame/1610438321184.jpg',
        #                  'AmbHand_Dataset/gyt/1/right/Realsense/camera2/color_frame/1610438321161.jpg']
        img_4_cam_paths = get_frame_imgPaths(dir_path=dir_path,choosen_img_num=time_id)  # kn, rs0, rs1, rs2
        kn_hand_annot = \
        np.load(img_4_cam_paths[0].replace('AmbHand_Dataset', 'NewDataset_annotations').replace('jpg', 'npy'),
                allow_pickle=True).item()['right_hand']
        shape = torch.from_numpy(kn_hand_annot['pred_hand_shape'])  # (1, 10)
        pose = torch.from_numpy(kn_hand_annot['pred_hand_pose'])  # (1, 48)
        trans = torch.from_numpy(kn_hand_annot['mano_trans'])[None, :]  # model-->mano (1, 3)
        cam = kn_hand_annot['pred_hand_cam']  # List[3]:(scale,trans_x,trans_y)
        bboxes = [np.load(img_4_cam_paths[i].replace('AmbHand_Dataset', 'NewDataset_annotations').replace('jpg', 'npy'),
                          allow_pickle=True).item()['right_hand']['bbox'] for i in range(4)]  # List[boxe1, boxe2, boxe3, boxe4] [xmin,ymin,width,height]

        #####################################
        # Calc camera0_joints
        ####################################
        RH_S = Rander_hand(model_path='../ktools/model', smplx_or_mano='smplx', # smlpx模型做手腕
                          hand_pose=pose, shape=shape, color=[248, 197, 183],
                         img_paths=img_4_cam_paths,
                         save_path='synthetic_datasets/2')
        org_vertices = RH_S.hand_mesh.vertices
        RH_S.hand_mesh.vertices = RH_S.org_camera_extrinsics(RH_S.hand_mesh.vertices, cam, bboxes, 960, 540)

        #####################################
        # Calc model to camera0(kn)  ex_mat
        ####################################
        _, _, _, smplx2cam0 = estimateSimilarityTransform(org_vertices, RH_S.hand_mesh.vertices)

        #####################################
        # Calc mano to camera0(kn)  ex_mat
        ####################################
        RH_M = Rander_hand(model_path='../ktools/model', smplx_or_mano='mano', # smlpx模型做手腕
                          hand_pose=pose, shape=shape, color=[248, 197, 183],
                         img_paths=img_4_cam_paths,
                         save_path='synthetic_datasets/2')
        _, _, _, mano2cam0 = estimateSimilarityTransform(RH_M.mesh.vertices, RH_S.hand_mesh.vertices)
        #####################################
        # Saved hand_annot -- .json
        ####################################
        # data_structure: samed by InterHand
        dict_time[time_id]= {"shape": shape[0].numpy().tolist(),
                            "pose": pose[0].numpy().tolist(),
                            "mano2cam0": mano2cam0.tolist(),
                            "smplx2cam0": smplx2cam0.tolist()}
        dict_getsure[gesture_id] = dict_time
        hand_data_frame = {subject_name:dict_getsure}
    os.makedirs(data_root.replace("Dataset","annotations_1")+'/'+subject_name+'/'+gesture_id,exist_ok=True)
    with open(data_root.replace("Dataset","annotations_1")+'/'+subject_name+'/'+gesture_id+'/'+'annot.json', 'w') as f:
        json.dump(hand_data_frame, f)
    print(f'Have been convert in '+ data_root.replace("Dataset","annotations_1")+'/'+subject_name+'/'+gesture_id+'/'+'annot.json')

def data_analysis(data):
    '''
    @ 分析, 并可视化数据的分布情况
    Param data: (N, M) N为样本数, M为特征个数
    '''
    print(f'mean:{np.around(data.mean(0),2)}')
    print(f'max:{np.around(data.max(0),2)} \n min:{np.around(data.min(0),2)}')
    fig = plt.figure(figsize=(16, 8))
    ax = list()
    num = len(data[0])
    for i in range(num):
        ax.append(fig.add_subplot(int(math.sqrt(num))+1,int(math.sqrt(num))+1,i+1))
        ax[i].hist(data[:,i])

def random_crop(image, size=256):
    h, w = image.shape[:2]
    y = np.random.randint(0, h-size)
    x = np.random.randint(0, w-size)
    image = image[y:y + size, x:x + size, :]
    return image

class Generate_hand_annot():
    """
        Load GYT标注的AmbHand数据集中的关键帧
        分析每一种手势下的数据分布情况，并进行随机采样

        Parameters
        ----------
        pose: numpy.array (45,21,48).
        shape: numpy.array (45,21,10).

        Functions
        ----------
        self.random_color
        self.random_background
        self.random_poseq
        self.random_shape
        self.random_3d_transform

        Returns
        -------
        result : array, tuple, dict, etc.
            Data stored in the file. For ``.npz`` files, the returned instance
            of NpzFile class must be closed to avoid leaking file descriptors.
    """
    def __init__(self,data_root):
        gyt_annot = np.load(data_root, allow_pickle=True).item()  # (45,21,48)
        self.pose = gyt_annot['pose']
        self.shape = gyt_annot['shape']

    def random_color(self,lock=False):
        if lock:
            return np.array([240/255,160/255,230/255])
        return np.array([random.randint(200,255)/255,
                        random.randint(140,180)/255,
                        random.randint(170,255)/255])

    def random_background(self,bg_path):
        img_paths = [f'{bg_path}/eva.jpg',
                     f'{bg_path}/eva2.jpg',
                     f'{bg_path}/eva2.jpg',
                     f'{bg_path}/eva2.jpg',]
        return img_paths

    def random_pose(self,gesture,lock=False,lock_root=False):
        pose = self.pose[gesture,:,:]
        max_pose = np.around(pose.max(0), 2)  # self.pose (21,48)
        mean_pose = np.around(pose.mean(0), 2)
        min_pose = np.around(pose.min(0), 2)
        if random.random()>0.5:
            random_pose = mean_pose \
                           + (max_pose-mean_pose)*np.abs(np.random.randn(48))/3\
                           + (min_pose-mean_pose)*np.abs(np.random.randn(48))/3
        else:
            random_pose = np.linspace(pose[random.randint(0,20)],
                                      pose[random.randint(0,20)],
                                      5)[random.randint(0,4)]
        if lock:
            random_pose = mean_pose
        if lock_root:
            random_pose[:4] = mean_pose[:4]*0
        return torch.tensor(random_pose,dtype=torch.float32)[None,:]

    def random_pose_0(self,model_path):
        pose_0 = np.load(f'{model_path}/pose_0.npy', allow_pickle=True)  # (45,48)
        max_pose = np.around(pose_0.max(0), 2)  # self.pose (21,48)
        mean_pose = np.around(pose_0.mean(0), 2)
        min_pose = np.around(pose_0.min(0), 2)

        if random.random() > 0.5:
            random_pose = mean_pose \
                          + (max_pose - mean_pose) * np.abs(np.random.randn(48)) / 3 \
                          + (min_pose - mean_pose) * np.abs(np.random.randn(48)) / 3
        else:
            random_pose = np.linspace(pose_0[random.randint(0, 20)],
                                      pose_0[random.randint(0, 20)],
                                      5)[random.randint(0, 4)]
        return torch.tensor(random_pose,dtype=torch.float32)[None,:]

    def random_shape(self,gesture,lock=False):
        shape = self.shape[gesture,:,:]
        max_shape = np.around(shape.max(0), 2)
        mean_shape = np.around(shape.mean(0), 2)
        min_shape = np.around(shape.min(0), 2)
        if lock:
            max_shape = mean_shape
            min_shape = mean_shape
        random_shape = mean_shape \
                       + (max_shape - mean_shape) * np.abs(np.random.randn(10)) \
                       + (min_shape - mean_shape) * np.abs(np.random.randn(10))
        return torch.tensor(random_shape,dtype=torch.float32)[None,:]

    def random_3d_transform(self,lock=False):
        if lock:
            rand_x = 0
            rand_y=0
            rand_z = 0
        else:
            rand_x = random.random() * 2 - 1 # [-1,1]
            rand_y = random.random() * 2 - 1  # [-1,1]
            rand_z = random.random() * 2 - 1  # [-1,1]
        # smplx2cam0 = torch.tensor([[9.1861e+02, 2.1362e-07, -5.2160e-06, 50*rand_x],
        #                            [-2.1362e-07, 9.1861e+02, 2.6464e-05, -20+40*rand_y],
        #                            [5.2160e-06, -2.6464e-05, 9.1861e+02, 640+50*rand_z],
        #                            [0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00]])
        mano2cam0 = torch.tensor([[924.,   0.,   0., 0+20*rand_x],
                                  [  0., 924.,   0., -10+20*rand_y],
                                  [  0.,   0., 924., 640+50*rand_z],
                                  [  0.,   0.,   0., 1.]])

        return mano2cam0

def get_aug_config():
    trans_factor = 0.15
    scale_factor = 0.25
    rot_factor = 45
    color_factor = 0.2

    trans = [np.random.uniform(-trans_factor, trans_factor), np.random.uniform(-trans_factor, trans_factor)]
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    do_flip = random.random() <= 0.5
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = np.array([random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)])

    return trans, scale, rot, do_flip, color_scale

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans
def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]
def generate_patch_image(cvimg, bbox,  scale, rot, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    # if do_flip:
    #     img = img[:, ::-1, :]
    #     bb_c_x = img_width - bb_c_x - 1

    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot,
                                        inv=True)

    return img_patch, trans, inv_trans
def augmentation(img, bbox, joint_coord, joint_valid, hand_type, mode, joint_type):
    img = img.copy();
    joint_coord = joint_coord.copy();

    original_img_shape = img.shape
    joint_num = len(joint_coord)

    if mode == 'train':
        trans, scale, rot, do_flip, color_scale = get_aug_config()
    else:
        trans, scale, rot, do_flip, color_scale = [0, 0], 1.0, 0.0, False, np.array([1, 1, 1])

    bbox[0] = bbox[0] + bbox[2] * trans[0]
    bbox[1] = bbox[1] + bbox[3] * trans[1]
    img, trans, inv_trans = generate_patch_image(img, bbox, do_flip, scale, rot, input_img_shape)
    img = np.clip(img * color_scale[None, None, :], 0, 255)

    if do_flip:
        joint_coord[:, 0] = original_img_shape[1] - joint_coord[:, 0] - 1
        joint_coord[joint_type['right']], joint_coord[joint_type['left']] = joint_coord[joint_type['left']].copy(), \
                                                                            joint_coord[joint_type['right']].copy()
        joint_valid[joint_type['right']], joint_valid[joint_type['left']] = joint_valid[joint_type['left']].copy(), \
                                                                            joint_valid[joint_type['right']].copy()
        hand_type[0], hand_type[1] = hand_type[1].copy(), hand_type[0].copy()
    for i in range(joint_num):
        joint_coord[i, :2] = trans_point2d(joint_coord[i, :2], trans)
        joint_valid[i] = joint_valid[i] * (joint_coord[i, 0] >= 0) * (joint_coord[i, 0] < input_img_shape[1]) * (
                    joint_coord[i, 1] >= 0) * (joint_coord[i, 1] < input_img_shape[0])

    return img, joint_coord, joint_valid, hand_type, inv_trans
def get_bbox(joint_img):
    x_img = joint_img[:, 0]
    y_img = joint_img[:, 1]
    xmin = min(x_img)
    ymin = min(y_img)
    xmax = max(x_img)
    ymax = max(y_img)

    x_center = (xmin + xmax) / 2.
    width = xmax - xmin
    y_center = (ymin + ymax) / 2.
    height = ymax - ymin
    if width>=height: height=width
    xmin = x_center - 0.5 * width * 1.4
    xmax = x_center + 0.5 * width * 1.4
    ymin = y_center - 0.5 * height * 1.4
    ymax = y_center + 0.5 * height * 1.4

    bbox = np.array([int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)]).astype(np.int)
    # bbox = np.array([xmin, ymin, xmax - xmin, ymax - ymin]).astype(np.float32)
    return bbox
def random_crop(image, size=256):
    h, w = image.shape[:2]
    y = np.random.randint(0, h-size)
    x = np.random.randint(0, w-size)
    image = image[y:y + size, x:x + size, :]
    return image

from .config import cfg
def random_bg_img(hand_img:np.array,idx):
    '''

    :param hand_img:  [256, 256, 3] (0, 255) BGR
    :param idx:
    :return: [256, 256, 3] (0, 255)
    '''
    # hand_img = hand_img[...,[2,1,0]]
    img_path = cfg.background_img
    # img_path = '/home/water/PycharmProjects/1_hand_pose_estimation/InterHand2.6M/ktools/background_img'
    bg_img = cv2.imread(f'{img_path}/{idx}.jpg')[...,[2,1,0]]
    random_crop_img = random_crop(bg_img, 256)
    # render_mask = np.zeros_like(hand_img) # (256,256,3)

    # for i in range(256):
    #     for j in range(256):
    #         if (all(hand_img[i][j]>255*0.7)): render_mask[i][j]=np.array([0,0,0])
    #         else: render_mask[i][j]=np.array([1,1,1])
    render_mask = np.where(hand_img>255*0.7,1,0)
    render_mask= render_mask[:,:,0]*render_mask[:,:,1]*render_mask[:,:,2]
    render_mask = np.repeat(render_mask[:,:,None], 3, axis=2)
    # plt.subplot(231)
    # plt.imshow(hand_img[...,[2,1,0]]/255.)
    # plt.subplot(232)
    # plt.imshow((1 - render_mask))
    # plt.subplot(233)
    # plt.imshow(hand_img[...,[2,1,0]]/255. * (1 - render_mask))
    # plt.subplot(234)
    # plt.imshow(random_crop_img[...,[2,1,0]])
    # plt.subplot(235)
    # plt.imshow(random_crop_img[...,[2,1,0]] * (render_mask)/255.)
    # plt.subplot(236)
    # plt.imshow(hand_img[...,[2,1,0]] * (1 - render_mask)/255.+
    #            random_crop_img[...,[2,1,0]] * (render_mask)/255.)
    # plt.show()
    return hand_img * (1 - render_mask) + random_crop_img * (render_mask)
def transfer_3d_points(source, source_refs, target_refs):
    """gyt
    Args:
        source: 源点簇, (N, 3)
        source_refs: 源代表点簇, (< N, 3)
        target_refs: 目标代表点簇, (< N, 3)

    Returns: 转换后源点簇, (N, 3)

    """
    n = source.shape[0]
    s, r, t, ot = estimateSimilarityTransform(source_refs, target_refs)
    x = np.swapaxes(np.concatenate([source, np.ones([n, 1])], axis=1), axis1=0, axis2=1)
    x = np.matmul(ot, x)
    x = np.swapaxes(x, axis1=0, axis2=1)[:, 0:3]

    return x

def cam2pix(cam_coord: np.array, cam_in: str) -> np.array:
    """
    将相机坐标系下的坐标转换为pix坐标系, cam_in: 为对应的相机内参idx
    :param cam_coord: (num, 3)
    :param cam_in: str -->('kn','rs0','rs1','rs2')
    :param self.multi_cam_in_param:
        List [kn_in: np.array(4,4),
             rs0_in: np.array(4,4),
             rs1_in: np.array(4,4),
             rs2_in: np.array(4,4),
             rs3_in: np.array(4,4)]

    :return: pix_coord: (num, 3)
    """
    multi_cam_in_param = np.array([[[1.1025e+03, 0.0000e+00, 4.5852e+02],
                                         [0.0000e+00, 1.0997e+03, 3.1442e+02],
                                         [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                                        [[6.1292e+02, 0.0000e+00, 3.2631e+02],
                                         [0.0000e+00, 6.1238e+02, 2.3663e+02],
                                         [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                                        [[6.1571e+02, 0.0000e+00, 3.1968e+02],
                                         [0.0000e+00, 6.1578e+02, 2.4240e+02],
                                         [0.0000e+00, 0.0000e+00, 1.0000e+00]],
                                        [[6.1599e+02, 0.0000e+00, 3.2635e+02],
                                         [0.0000e+00, 6.1613e+02, 2.3546e+02],
                                         [0.0000e+00, 0.0000e+00, 1.0000e+00]]], dtype=np.float)  # [kn,rs1,rs2,...]
    cam_idx = {'kn': 0, 'rs0': 1, 'rs1': 2, 'rs2': 3}
    fx = multi_cam_in_param[cam_idx[cam_in]][0, 0]
    fy = multi_cam_in_param[cam_idx[cam_in]][1, 1]
    cx = multi_cam_in_param[cam_idx[cam_in]][0, 2]
    cy = multi_cam_in_param[cam_idx[cam_in]][1, 2]
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * fx + cx
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * fy + cy
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord