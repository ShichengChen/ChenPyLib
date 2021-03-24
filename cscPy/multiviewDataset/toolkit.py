from cscPy.mano.network.manolayer import MANO_SMPL
import os,pickle
from cscPy.globalCamera.util import fetch_all_sequences,load_rgb_maps,load_depth_maps,get_cameras_from_dir,visualize_better_qulity_depth_map
from cscPy.globalCamera.camera import CameraIntrinsics,perspective_projection,perspective_back_projection
from cscPy.globalCamera.constant import Constant
from cscPy.mano.network import net
from cscPy.mano.network.utils import AxisRotMat
from cscPy.mano.dataloader.MFjointsDataloader import MF3D
import numpy as np
import torch
import cv2

class MultiviewDatasetDemo():
    def __init__(self,manoPath='/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl',
                 file_path="/media/csc/Seagate Backup Plus Drive/dataset/7-14-1-2/mlresults/7-14-1-2_3_2result_32.pkl",
                 loadMode=True,

    ):
        self.mano_right = MANO_SMPL(manoPath, ncomps=45)

        baseDir = file_path[:file_path.rfind('/', 0, file_path.rfind('/'))]
        self.date = baseDir[baseDir.rfind('/') + 1:]
        calib_path = os.path.join(baseDir, 'calib.pkl')
        print("baseDir", baseDir)
        with open(calib_path, 'rb') as f:
            camera_pose_map = pickle.load(f)
        rgb_paths, depth_paths = fetch_all_sequences(os.path.join(baseDir, "1_840412062035_depth.bin"))
        self.rgb_seqs = [load_rgb_maps(p)[100:] for p in rgb_paths]
        self.depth_seqs = [load_depth_maps(p)[100:] for p in depth_paths]


        cam_list = get_cameras_from_dir(baseDir[:baseDir.rfind('/')], baseDir[baseDir.rfind('/') + 1:], "1")
        self.cam_list = cam_list
        cam_list.sort()
        print('cam_list', cam_list)
        camera, camera_pose = [], []
        for camera_ser in cam_list:
            camera.append(CameraIntrinsics[camera_ser])
            camera_pose.append(camera_pose_map[camera_ser])
        for i in range(4):
            if (np.allclose(camera_pose[i], np.eye(4))):
                rootcameraidx = i
        self.camera_pose,self.camera=camera_pose,camera
        self.rootcameraidx=rootcameraidx

        joints = np.load(os.path.join(baseDir, "mlresults", self.date + '-joints.npy'))
        self.N=joints.shape[0]
        self.joints=joints.reshape(self.N,21,4,1).astype(np.float32)

        joints4view = np.ones((4, self.N, 21, 4, 1)).astype(np.int64)
        for dev_idx, rs_dev in enumerate(cam_list):
            inv = np.linalg.inv(camera_pose[dev_idx])
            joints4view[dev_idx] = inv @ self.joints
        self.joints4view=joints4view


        if(loadMode):
            train_dataset = MF3D(file_path=file_path, adjust=0, onlygt=False, usedirect=True)
            self.train_dataset=train_dataset
            model = net.VPoser(inshape=train_dataset.inshape)
            model = model.to('cuda')
            checkpoint = torch.load('/home/csc/hand_seg/dataset/estimateMano/train/' + self.date + 'iknet.pt')
            model.load_state_dict(checkpoint['iknet'])
            model.eval()
            self.model=model


    def getImgs(self,idx):
        color=[]
        for iv in range(4):color.append(self.rgb_seqs[iv][idx].copy())
        color = np.hstack(color)
        return color
    def getImgsList(self,idx,facemask=True):
        color=[]
        facelist = [[], [], [(250, 0, 424, 100)], [(436, 0, 640, 200)]]
        for iv in range(4):
            img=self.rgb_seqs[iv][idx].copy()
            if facemask:
                for (x, y, x1, y1) in facelist[iv]:
                    mask2 = np.zeros_like(img).astype(np.bool)
                    mask2[y:y1, x:x1, :] = True
                    # mask2[mask, :] = False
                    rgbcopy = img.copy()
                    img = cv2.blur(img, (40, 40))
                    img[mask2 == False] = rgbcopy[mask2 == False]
            color.append(img)


        return color
    def getDepth(self,idx):
        dms = []
        for iv in range(4): dms.append(self.depth_seqs[iv][idx].copy())
        dms = np.hstack(dms)
        return dms
    def getBetterDepth(self,idx):
        dlist = []
        for iv in range(4):
            depth = self.depth_seqs[iv][idx].copy()
            dlist.append(visualize_better_qulity_depth_map(depth))
        return np.hstack(dlist)

    def getManoVertex(self,idx):
        ids, (joints_gt, scale, joint_root, direct)=self.train_dataset.__getitem__(idx)
        direct = direct.to('cuda')
        joints_gt = joints_gt.to('cuda')
        jd = torch.cat([joints_gt, direct], dim=0).reshape(1,41,3)
        results = self.model(jd)
        vertex, joint_pre = \
            self.mano_right.get_mano_vertices(results['pose_aa'][:, 0:1, :],
                                         results['pose_aa'][:, 1:, :], results['shape'],
                                         results['scale'], results['transition'],
                                         pose_type='euler', mmcp_center=False)
        vertex=vertex.cpu()
        scale=scale.cpu()
        joint_root=joint_root.cpu()
        vertices = (vertex * scale + joint_root)[0].cpu().detach().numpy() * 1000
        vertices = np.concatenate([vertices, np.ones([vertices.shape[0], 1])], axis=1)
        vertices = np.expand_dims(vertices, axis=-1)
        self.vertices=vertices
        return vertices

    def get4viewManovertices(self,idx):
        vertices=self.getManoVertex(idx)
        vertices4view=np.zeros([4,778,4,1])
        for iv in range(4):
            inv = np.linalg.inv(self.camera_pose[iv])
            vertices4view[iv] = (inv @ vertices)
        self.vertices4view=vertices4view
        return vertices4view

    def render4mesh(self,idx,ratio=1):
        #the ratio=10 can make the rendered image be black
        vertices4view=self.get4viewManovertices(idx)
        import trimesh
        import pyrender
        from pyrender import RenderFlags
        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
        np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [3, 3])
        recolorlist=[]
        for iv in range(4):
            xyz=vertices4view[iv,:778,:3,0].copy()
            cv = xyz @ np_rot_x
            tmesh = trimesh.Trimesh(vertices=cv / 1000*ratio, faces=self.mano_right.faces)
            # tmesh.visual.vertex_colors = [.9, .7, .7, 1]
            # tmesh.show()
            mesh = pyrender.Mesh.from_trimesh(tmesh)
            scene = pyrender.Scene()
            scene.add(mesh)
            pycamera = pyrender.IntrinsicsCamera(self.camera[iv].fx, self.camera[iv].fy, self.camera[iv].cx, self.camera[iv].cy, znear=0.0001,
                                                 zfar=3000)
            ccamera_pose = self.camera_pose[self.rootcameraidx]
            scene.add(pycamera, pose=ccamera_pose)
            light = pyrender.SpotLight(color=np.ones(3), intensity=2.0, innerConeAngle=np.pi / 16.0)
            # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            scene.add(light, pose=ccamera_pose)
            r = pyrender.OffscreenRenderer(640, 480)
            # flags = RenderFlags.SHADOWS_DIRECTIONAL
            recolor, depth = r.render(scene)
            # cv2.imshow("depth", depth)
            recolorlist.append(recolor[:, :, :3])
        meshcolor = np.hstack(recolorlist)
        return meshcolor


    def drawMesh(self,idx):
        recolor=self.render4mesh(idx)
        color=np.hstack(self.getImgsList(idx))
        recolor[recolor == 255] = color[recolor == 255]
        c = cv2.addWeighted(color, 0.1, recolor, 0.9, 0.0)
        return c

    def drawMask(self,idx):
        #call after get4viewManovertices and
        self.getManoVertex(idx)
        sampleForMask=-1
        self.get4viewCloud(idx,sampleN=sampleForMask,disprio=1.4)
        vertex=self.vertices.copy()
        cloud=self.cloud.copy()
        vertex = vertex.reshape(778, 4,1)
        cloud = cloud.reshape(sampleForMask, 4,1)
        dcloud = np.min(np.sqrt(np.sum((cloud[:,:3].reshape(sampleForMask, 1, 3) - vertex[:,:3].reshape(1, 778, 3)) ** 2, axis=2)), axis=1)
        dcloud = dcloud.reshape(sampleForMask)
        cloud=cloud[dcloud < 10]
        assert (cloud.shape[0]>0),"nothing left for draw"
        cloud4v = np.ones([4, cloud.shape[0], 4, 1])
        background = np.ones([4, 480, 640, 3]).astype(np.uint8) * 255
        for iv in range(4):
            inv = np.linalg.inv(self.camera_pose[iv])
            cloud4v[iv] = inv @ cloud
            for i in range(cloud4v.shape[1]):
                uvd = perspective_projection(cloud4v[iv, i, :3, 0], self.camera[iv]).astype(np.int64)
                background[iv] = cv2.circle(background[iv], tuple(uvd[:2]), 1, (0, 0, 0))
        background = np.concatenate([background[0], background[1], background[2], background[3]], axis=1)
        return background





    def get4viewCloud(self,idx,sampleN=1000,disprio=1.1):
        pointlist=[]

        for iv in range(4):
            dm=self.depth_seqs[iv][idx].copy()
            u, v = np.meshgrid(range(640), range(480))
            u, v, d = u.reshape(-1), v.reshape(-1), dm.reshape(-1)
            uvd = np.transpose(np.stack([u.astype(np.float32), v.astype(np.float32), d]))
            xyz_center=self.joints4view[iv,idx,5,:3,0].copy()
            wristjoint = (self.joints4view[iv,idx,0,:3,0]).copy()
            tipjoint = (self.joints4view[iv,idx,8,:3,0]).copy()
            dis1 = np.sqrt(np.sum((xyz_center - wristjoint) ** 2))
            dis2 = np.sqrt(np.sum((xyz_center - tipjoint) ** 2))
            dis = max(dis1, dis2)

            cloud = perspective_back_projection(uvd, self.camera[iv]).squeeze()
            cloud = np.dstack((cloud[:, 0], cloud[:, 1], cloud[:, 2], np.ones(cloud.shape[0]))).squeeze()
            validIndicies = (np.abs(cloud[:, 0] - xyz_center[0]) < dis * disprio) & \
                            (np.abs(cloud[:, 1] - xyz_center[1]) < dis * disprio) & \
                            (np.abs(cloud[:, 2] - xyz_center[2]) < dis * disprio)
            cloud = cloud[validIndicies, :]

            cloud = self.camera_pose[iv] @ np.expand_dims(cloud, -1)

            pointlist.append(cloud.copy().astype(np.float32))

        cloud = np.concatenate(pointlist, axis=0).reshape(-1, 4, 1)

        randInidices = np.random.permutation(np.arange(len(cloud)))
        if(sampleN>=1):
            cloud = cloud[randInidices[:sampleN, ], :]
            cloud4v = np.ones([4, sampleN, 4, 1])
        else:
            cloud = cloud[:, :]
            cloud4v = np.ones([4, cloud.shape[0], 4, 1])
        #save cloud after sampling
        self.cloud=cloud


        for iv in range(4):
            inv = np.linalg.inv(self.camera_pose[iv])
            cloud4v[iv] = inv @ cloud
        self.cloud4v=cloud4v
        return cloud4v
    def drawCloud4view(self,idx,sampleN=1000,view=4,depthInfluenceColor=False):
        assert (view==1 or view==4),"only support 4 and 1 view"
        background = np.ones([4, 480, 640, 3]).astype(np.uint8) * 255
        cloud4v=self.get4viewCloud(idx,sampleN)
        for iv in range(4):
            vis = np.zeros([480, 640])
            for i in range(cloud4v.shape[1]):
                uvd=perspective_projection(cloud4v[iv,i,:3,0],self.camera[iv]).astype(np.int64)
                if(uvd[1]<0 or uvd[0]<0 or uvd[0]>=640 or uvd[1]>=480 or vis[uvd[1],uvd[0]]==255):continue
                #if(iv==0):print('ori', uvd[-1])
                dist=255
                if(depthInfluenceColor):dist = int(np.clip(np.array([(uvd[-1] - 500)])*3, 50, 200).astype(int)[0])
                #if(iv==0):print('after',int(dist))
                vis=cv2.circle(vis, tuple(uvd[:2]), 1, (255),-1)

                background[iv]=cv2.circle(background[iv],tuple(uvd[:2]),1,(dist, 0, 200-dist))
                #background[iv]=cv2.circle(background[iv],tuple(uvd[:2]),1,(dist, dist, dist))
        if(view==4):
            background = np.concatenate([background[0], background[1], background[2], background[3]], axis=1)
        else:
            background = background[0]
        return background

    def drawSelfRotationCloudView0(self,idx,degree,depthInfluenceColor=False):
        rot=AxisRotMat(degree,[1,0,0])
        c=self.cloud4v[0].copy()
        wrist=self.joints4view[0,idx,5:6,:3,0]
        c[:,:3,0]-=wrist
        c=rot@c
        c[:,:3,0]+=wrist
        background = np.ones([480, 640, 3]).astype(np.uint8) * 255
        vis = np.zeros([480, 640])
        for i in range(c.shape[0]):
            uvd = perspective_projection(c[i, :3, 0], self.camera[0]).astype(np.int64)
            if (uvd[1] < 0 or uvd[0] < 0 or uvd[0] >= 640 or uvd[1] >= 480 or vis[uvd[1], uvd[0]] == 255): continue
            vis = cv2.circle(vis, tuple(uvd[:2]), 1, (255), -1)
            dist = 255
            if (depthInfluenceColor): dist = int(np.clip(np.array([(uvd[-1] - 500)]) * 3, 50, 200).astype(int)[0])
            background = cv2.circle(background, tuple(uvd[:2]), 1, (dist, 0, 200-dist))
            #background=cv2.circle(background,tuple(uvd[:2]),1,(dist, dist, dist))
        return background


    def drawPose4view(self,idx,view=4):
        assert (view == 1 or view == 4), "only support 4 and 1 view"
        lineidx = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
        uvdlist = []
        imgs=self.getImgsList(idx)
        for iv in range(4):
            ujoints=self.joints4view[iv,idx,:21,:3,0].copy()
            for jdx in range(21):
                rgbuvd = perspective_projection(ujoints[jdx], self.camera[iv]).astype(int)[:2]
                uvdlist.append(rgbuvd)

                color=np.array(Constant.joint_color[jdx]).astype(int)
                imgs[iv] = cv2.circle(imgs[iv], tuple(rgbuvd), 3, color.tolist(), -1)
                if (jdx in lineidx):
                    imgs[iv] = cv2.line(imgs[iv], tuple(rgbuvd), tuple(uvdlist[-2]), color.tolist(), thickness=2)
        if(view==1):imgs = imgs[0].copy()
        else:imgs = np.hstack(imgs)
        return imgs








