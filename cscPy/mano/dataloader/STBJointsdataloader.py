from __future__ import print_function, unicode_literals

import sys

import scipy.io
import scipy.misc

#sys.path.append('..')
import pickle
import os
from torch.utils.data import Dataset
from cscPy.mano.network.utils import *
from cscPy.mano.network.manolayer import VPoser,MANO_SMPL
import tqdm


# SET THIS to where RHD is located on your machine
path_to_db = './RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/shicheng/RHD_published_v2/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/home/csc/dataset/RHD_published_v2/'
if (not os.path.exists(path_to_db)):
    path_to_db = '/mnt/ssd/csc/RHD_published_v2/'
if (not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/csc/RHD_published_v2/'
    #os.environ["DISPLAY"] = "localhost:11.0"

set = 'training'


def getDir(joint_gt):
    joints=joint_gt.copy()
    manoidx = [1, 2, 3, 17,  4, 5, 6, 18,  7, 8, 9, 20,  10,11, 12, 19, 13,14, 15, 16]
    manopdx = [0, 1, 2, 3,   0, 4, 5, 6,   0, 7, 8, 9,   0,10, 11, 12,   0,13, 14, 15]
    for idx in range(len(manoidx)):
        ci = manoidx[idx]
        pi = manopdx[idx]
        joints[ci] = joints[ci] - joints[pi]
    joints[0] = 0
    return joints
def getDis(joint_gt):
    joints=joint_gt.copy()
    manoidx = [1, 2, 3, 17,  4, 5, 6, 18,  7, 8, 9, 20,  10,11, 12, 19, 13,14, 15, 16]
    manopdx = [0, 1, 2, 3,   0, 4, 5, 6,   0, 7, 8, 9,   0,10, 11, 12,   0,13, 14, 15]
    for idx in range(len(manoidx)):
        ci = manoidx[idx]
        pi = manopdx[idx]
        joints[ci] = np.linalg.norm(joints[ci] - joints[pi])
    joints[0]=0
    return joints

class RHDJointsDateset(Dataset):
    def __init__(self, mode='training',path_name=path_to_db):
        self.mode = mode
        self.num_samples=0
        self.path_name=path_name
        with open(os.path.join(self.path_name, self.mode, 'anno_%s.pickle' % self.mode), 'rb') as fi:
            self.anno_all = pickle.load(fi)
            self.num_samples = len(self.anno_all.items())
        print('RHDDateset3D self.num_samples',self.num_samples)
        print("mode",mode)


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx,righthand=True):
        if(idx==20500 or idx==28140):idx=0
        anno=self.anno_all[idx]
        kp_coord_xyz = anno['xyz']  # x, y, z coordinates of the keypoints, in meters


        if(righthand==True):
            joints = kp_coord_xyz[-21:, :]*1000
        else:
            joints = kp_coord_xyz[21:, :]*1000
        joints = joints[RHD2Bighand_skeidx, :][Bighand2mano_skeidx, :]

        #print(joints)

        root = joints[4:5, :].copy()
        joints = joints - root  # relative coords in metric coords
        scale=np.array([np.sqrt(np.sum((joints[5]-joints[4])**2))])
        joints/=scale

        # dum = np.zeros([21, 3])
        # dum[4], dum[7] = [0, 1, 0], [-1, 1, 0]
        # r = wristRot(joints, dum)
        # joints = ((r @ joints.T).T).reshape(21, 3).astype(np.float32)
	
	if(righthand==False):joints[:, 0] *= -1  # from left hand to right hand


        dirjoints=getDir(joints)
        disjoints=getDis(joints)
        a,b,c=disjoints.reshape(1,21,3),dirjoints.reshape(1,21,3),joints.reshape(1,21,3)
        out={}
        out['f']=np.concatenate([c,b],axis=0).reshape(2,21,3)
        out['scale']=scale.astype(np.float32)
        out['root']=root.reshape(3).astype(np.float32)
        return out


path_to_db = '/mnt/data/shicheng/STB/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/home/csc/dataset/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/mnt/ssd/csc/STB/'
if(not os.path.exists(path_to_db)):
    path_to_db = '/mnt/data/csc/STB/'

imageNum = 1500

class STBDateset3D(Dataset):
    def __init__(self,names=['B2Counting', 'B2Random', 'B3Counting', 'B3Random', 'B4Counting', 'B4Random',
                          'B5Counting', 'B5Random', 'B6Counting', 'B6Random', 'B1Counting', 'B1Random']):
        self.sequences = ['B2Counting', 'B2Random', 'B3Counting', 'B3Random', 'B4Counting', 'B4Random',
                          'B5Counting', 'B5Random', 'B6Counting', 'B6Random', 'B1Counting', 'B1Random']
        self.handPara=[]
        for seq in self.sequences:
            if(seq not in names):continue
            #print(seq)
            matfile = '%slabels/%s_SK.mat' % (path_to_db, seq)
            data = scipy.io.loadmat(matfile)
            self.handPara.append(data['handPara'])

    def __len__(self):
        return len(self.handPara)*imageNum

    def __getitem__(self, idx):
        folder_idx=idx//imageNum
        id=idx%imageNum

        kp_coord_xyz = self.handPara[folder_idx][:, :, id]

        joints=kp_coord_xyz.transpose(1, 0)
        joints = joints.astype(np.float32)
        #print(joints)

        wrist_xyz = joints[16:17, :] + 1.43 * (joints[0:1, :] - joints[16:17, :])
        joints = np.concatenate([wrist_xyz, joints[1:, :]], axis=0)
        joints = joints[STB2Bighand_skeidx, :][Bighand2mano_skeidx, :]

        root = joints[4:5, :].copy()
        joints-=root
        joints[:,0]*=-1 #from left hand to right hand

        scale=np.array([np.sqrt(np.sum((joints[5]-joints[4])**2))])
        joints/=scale

        # print(scale,joints)

        # dum = np.zeros([21, 3])
        # dum[4], dum[7] = [0, 1, 0], [-1, 1, 0]
        # r = wristRot(joints, dum)
        # joints = ((r @ joints.T).T).reshape(21, 3).astype(np.float32)

        dirjoints = getDir(joints)
        disjoints = getDis(joints)
        a, b, c = disjoints.reshape(1, 21, 3), dirjoints.reshape(1, 21, 3), joints.reshape(1, 21, 3)
        out = {}
        out['f'] = np.concatenate([c,b], axis=0).reshape(2,21,3)
        out['scale'] = scale.astype(np.float32)
        out['root'] = root.reshape(3).astype(np.float32)
        return out

if __name__ == '__main__':

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    lr = 1e-3
    device = 'cuda'
    model=VPoser(inshape=21*2,dim=3).to(device)
    mylist=[]
    mylist.append({'params': model.parameters(), 'weight_decay': 1e-6})
    optimizer = torch.optim.Adam(mylist, lr=lr)

    def _init_fn(worker_id):
        np.random.seed(worker_id)

    names=['B2Counting', 'B2Random', 'B3Counting', 'B3Random', 'B4Counting', 'B4Random',
                          'B5Counting', 'B5Random', 'B6Counting', 'B6Random', 'B1Counting', 'B1Random']
    #for name in names:
    for name in names:

        train_dataset=STBDateset3D(names=name)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, num_workers=4, shuffle=False,
                                                   worker_init_fn=_init_fn,)
        mano_right = MANO_SMPL('/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl', ncomps=45,
                               bighandorder=False)
        minn = 5
        allepochs = 500
        n2m = 1
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=allepochs // 3, gamma=0.1)
        for epoch in tqdm.tqdm(range(allepochs)):
            aveloss, aveloss2 = [], []
            model.eval()
            cnt = 0


            for idx, (out) in enumerate(train_loader):
                cnt += 1
                jd = out['f'].to(device)
                n = jd.shape[0]
                # print(jd.shape)
                joints_gt=jd[:,0,...]
                # print('joints_gt.shape',joints_gt.shape)
                jd=jd.reshape(n,-1)
                scale = out['scale'].to(device)
                joint_root = out['root'].to(device)

                # print('jd',jd)
                # print('joints_gt',joints_gt)

                results = model(jd)

                vertex, joint_pre = \
                    mano_right.get_mano_vertices(results['pose_aa'][:, 0:1, :],
                                                 results['pose_aa'][:, 1:, :], results['shape'],
                                                 results['scale'], results['transition'],
                                                 pose_type='euler', mmcp_center=False)
                if (epoch==1000):
                    np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
                    np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [1, 3, 3])
                    vertex,scale,joint_root=vertex.reshape(n,778,3),scale.reshape(n,1,1),joint_root.reshape(n,1,3)
                    vertices = (vertex * scale + joint_root)[0].cpu().detach().numpy()
                    faces = mano_right.faces
                    import trimesh
                    mesh = trimesh.Trimesh(vertices=vertices, faces=mano_right.faces)
                    mesh.visual.vertex_colors = [.9, .7, .7, 1]
                    mesh.show()

                # print('joint_pre',joint_pre)
                # print('joints_gt',joints_gt)

                joint_pre = joint_pre.view([n, 21, 3])
                joints_gt = joints_gt.reshape([n, 21, 3])
                jointdif = (torch.sqrt(torch.sum((joint_pre - joints_gt) ** 2, dim=2))).view(n, 21)
                eucDistance = jointdif * scale.view(n, 1)

                dloss = torch.mean(eucDistance)

                aveloss2.append(float(torch.mean(jointdif[:, 1:] * scale.view(n, 1))))

                pose_regular = torch.mean(torch.norm((results['pose_aa'][:, 1:, :]).reshape(-1, 3), dim=1)) * 1e-2 + \
                               torch.mean(torch.norm((results['pose_aa'][:, 0:1, :]).reshape(-1, 3), dim=1)) * 1e-3

                shape_regular = torch.mean(torch.sum(results['shape'] ** 2, dim=1)) * 1e-4

                loss = dloss + pose_regular + shape_regular
                aveloss.append(float(dloss))
                if (idx % 200 == 0):
                    print('epoch:{} iteration:{} loss:{:.2f}'.
                          format(epoch, idx, loss.item() * n2m))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # out=invAdjust(out)
            scheduler.step()
            if (np.mean(aveloss) * n2m < minn):
                savename = name+'iknet.pt'
                print('save model', savename)

                minn = min(minn, np.mean(aveloss) * n2m)
                torch.save({
                    'epoch': epoch,
                    'epe': minn * n2m,
                    'iknet': model.state_dict(),
                    'optimizer': optimizer.state_dict()}, savename)
            minn = min(minn, np.mean(aveloss) * n2m)
            print("ave aveloss:", np.mean(aveloss) * n2m, 'minn:', minn, 'no wrist', np.mean(aveloss2) * n2m)



