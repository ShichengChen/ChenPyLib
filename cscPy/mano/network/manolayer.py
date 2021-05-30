import pickle

import torch.nn as nn

from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.mano.network.Const import boneSpace

class MANO_SMPL(nn.Module):
    def __init__(self, mano_pkl_path, ncomps = 10, flat_hand_mean=False,cuda=True,oriorder=False,device='cuda'):
        super(MANO_SMPL, self).__init__()
        self.oriorder=oriorder
        # Load the MANO_RIGHT.pkl
        with open(mano_pkl_path, 'rb') as f:
            model = pickle.load(f, encoding='latin1')

        faces_mano = np.array(model['f'], dtype=int)

        # Add new faces for the wrist part and let mano model waterproof
        # for MANO_RIGHT.pkl
        faces_addition = np.array([[38, 122, 92], [214, 79, 78], [239, 234, 122],
                        [122, 118, 239], [215, 108, 79], [279, 118, 117],
                        [117, 119, 279], [119, 108, 215], [120, 108, 119],
                        [119, 215, 279], [214, 215, 79], [118, 279, 239],
                        [121, 214, 78], [122, 234, 92]])
        self.faces = np.concatenate((faces_mano, faces_addition), axis=0)

        self.flat_hand_mean = flat_hand_mean

        self.is_cuda = (torch.cuda.is_available() and cuda)

        np_v_template = np.array(model['v_template'], dtype=np.float)
        np_v_template = torch.from_numpy(np_v_template).float()
        #print('np_v_template',np_v_template.shape) #np_v_template torch.Size([778, 3])

        self.size = [np_v_template.shape[0], 3]
        np_shapedirs = np.array(model['shapedirs'], dtype=np.float)
        self.num_betas = np_shapedirs.shape[-1]
        np_shapedirs = np.reshape(np_shapedirs, [-1, self.num_betas]).T
        #print('np_shapedirs',np_shapedirs.shape)#np_shapedirs (10, 2334)

        np_shapedirs = torch.from_numpy(np_shapedirs).float()

        # Adding new joints for the fingertips. Original MANO model provide only 16 skeleton joints.
        np_J_regressor = model['J_regressor'].T.toarray()
        np_J_addition = np.zeros((778, 5))
        np_J_addition[745][0] = 1
        np_J_addition[333][1] = 1
        np_J_addition[444][2] = 1
        np_J_addition[555][3] = 1
        np_J_addition[672][4] = 1
        np_J_regressor = np.concatenate((np_J_regressor, np_J_addition), axis=1)
        np_J_regressor = torch.from_numpy(np_J_regressor).float()

        np_hand_component = np.array(model['hands_components'], dtype=np.float)[:ncomps]
        np_hand_component = torch.from_numpy(np_hand_component).float()

        #print("np_hand_component",np_hand_component.shape)

        np_hand_mean = np.array(model['hands_mean'], dtype=np.float)[np.newaxis,:]
        if self.flat_hand_mean:
            np_hand_mean = np.zeros_like(np_hand_mean)
        np_hand_mean = torch.from_numpy(np_hand_mean).float()

        np_posedirs = np.array(model['posedirs'], dtype=np.float)
        num_pose_basis = np_posedirs.shape[-1]
        np_posedirs = np.reshape(np_posedirs, [-1, num_pose_basis]).T
        np_posedirs = torch.from_numpy(np_posedirs).float()

        self.parents = np.array(model['kintree_table'])[0].astype(np.int32)
        #print('self.parents',self.parents)

        np_weights = np.array(model['weights'], dtype=np.float)
        vertex_count = np_weights.shape[0]
        vertex_component = np_weights.shape[1]
        np_weights = torch.from_numpy(np_weights).float().reshape(-1, vertex_count, vertex_component)

        e3 = torch.eye(3).float()

        np_rot_x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float)
        np_rot_x = np.reshape(np.tile(np_rot_x, [1, 1]), [1, 3, 3])
        self.base_rot_mat_x = Variable(torch.from_numpy(np_rot_x).float())

        joint_x = torch.matmul(np_v_template[:, 0], np_J_regressor)
        joint_y = torch.matmul(np_v_template[:, 1], np_J_regressor)
        joint_z = torch.matmul(np_v_template[:, 2], np_J_regressor)
        self.tjoints = torch.stack([joint_x, joint_y, joint_z, torch.ones_like(joint_x)], dim=1).numpy()
        self.J = torch.stack([joint_x, joint_y, joint_z], dim=1).numpy()
        self.bJ=torch.tensor(self.J.reshape(1,21,3),dtype=torch.float32)

        # joints = torch.stack([joint_x, joint_y, joint_z], dim=1).numpy()
        # # joints = joints[mano2bighand_skeidx, :]
        # jidx = [[1, 2, 3, 17], [4, 5, 6, 18], [10, 11, 12, 19], [7, 8, 9, 20], [13, 14, 15, 16]]
        # # index,middle,ring,pinky,thumb
        # # [69, 45, 30, 33, 29, 30, 33, 50, 30, 37, 37, 30, 33, 35, 33, 35, 37, 39, 39, 39, 35]
        # for idxs in jidx:
        #     a = np.sqrt(np.sum((joints[idxs[0]] - joints[idxs[1]]) ** 2))
        #     b = np.sqrt(np.sum((joints[idxs[1]] - joints[idxs[2]]) ** 2))
        #     c = np.sqrt(np.sum((joints[idxs[2]] - joints[idxs[3]]) ** 2))
        #     print(a, a / b, c / b)
        # print(np.sqrt(np.sum((joints[13] - joints[0]) ** 2)))


        if self.is_cuda and device=='cuda':
            np_v_template = np_v_template.cuda()
            np_shapedirs = np_shapedirs.cuda()
            np_J_regressor = np_J_regressor.cuda()
            np_hand_component = np_hand_component.cuda()
            np_hand_mean = np_hand_mean.cuda()
            np_posedirs = np_posedirs.cuda()
            e3 = e3.cuda()
            np_weights = np_weights.cuda()
            self.base_rot_mat_x = self.base_rot_mat_x.cuda()

        '''
        np_hand_component torch.Size([45, 45])
        np_v_template torch.Size([778, 3])
        np_shapedirs torch.Size([10, 2334])
        np_J_regressor torch.Size([778, 21])
        np_hand_component torch.Size([45, 45])
        np_hand_mean torch.Size([1, 45])
        np_posedirs torch.Size([135, 2334])
        weight torch.Size([1, 778, 16])
        '''

        self.register_buffer('v_template', np_v_template)
        self.register_buffer('shapedirs', np_shapedirs)
        self.register_buffer('J_regressor', np_J_regressor)
        self.register_buffer('hands_comp', np_hand_component)
        self.register_buffer('hands_mean', np_hand_mean)
        self.register_buffer('posedirs', np_posedirs)
        self.register_buffer('e3', e3)
        self.register_buffer('weight', np_weights)


    def getTemplate(self,beta,zero_wrist=False):
        v_shaped = torch.matmul(beta*10, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)
        if(zero_wrist):J-=J[:,0:1,:].clone()
        return J





    def forward(self, beta, theta, wrist_euler, pose_type, get_skin=False,external_transition=None):
        assert pose_type in ['pca', 'euler', 'rot_matrix'], print('The type of pose input should be pca, euler or rot_matrix')
        num_batch = beta.shape[0]
        # print("num_batch",num_batch)

        v_shaped = torch.matmul(beta, self.shapedirs).view(-1, self.size[0], self.size[1]) + self.v_template
        Jx = torch.matmul(v_shaped[:, :, 0], self.J_regressor)
        Jy = torch.matmul(v_shaped[:, :, 1], self.J_regressor)
        Jz = torch.matmul(v_shaped[:, :, 2], self.J_regressor)
        J = torch.stack([Jx, Jy, Jz], dim=2)
        self.CJ=J.clone()
        #print("J.shape",J.shape)

        #global_rot = self.batch_rodrigues(wrist_euler).view(-1, 1, 3, 3)

        # pose_type should be 'pca' or 'euler' here
        if pose_type == 'pca':
            euler_pose = theta.mm(self.hands_comp) + self.hands_mean
            Rs = self.batch_rodrigues(euler_pose.contiguous().view(-1, 3))
            #print('Rs',Rs)
            global_rot = self.batch_rodrigues(wrist_euler.view(-1, 3)).view(-1, 1, 3, 3)
            #print("global_rot",global_rot)
        elif pose_type == 'euler':
            euler_pose = theta
            Rs = self.batch_rodrigues(euler_pose.contiguous().view(-1, 3)).view(-1, 15, 3, 3)
            global_rot = self.batch_rodrigues(wrist_euler.view(-1, 3)).view(-1, 1, 3, 3)
        else:
            Rs = theta.view(num_batch, 15, 3, 3)
            global_rot = wrist_euler.view(num_batch, 1, 3, 3)

        Rs = Rs.view(-1, 15, 3, 3)
        pose_feature = (Rs[:, :, :, :]).sub(1.0, self.e3).view(-1, 135)
        v_posed = v_shaped + torch.matmul(pose_feature, self.posedirs).view(-1, self.size[0], self.size[1])

        self.J_transformed, A = self.batch_global_rigid_transformation(torch.cat([global_rot, Rs], dim=1), J[:, :16, :], self.parents,JsAll=J.clone())

        weight = self.weight.repeat(num_batch, 1, 1)
        W = weight.view(num_batch, -1, 16)
        T = torch.matmul(W, A.view(num_batch, 16, 16)).view(num_batch, -1, 4, 4)

        ones_homo = torch.ones(num_batch, v_posed.shape[1], 1)
        if self.is_cuda:
            ones_homo = ones_homo.cuda()
        v_posed_homo = torch.cat([v_posed, ones_homo], dim=2)
        v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, -1))

        verts = v_homo[:, :, :3, 0]
        joint_x = torch.matmul(verts[:, :, 0], self.J_regressor)
        joint_y = torch.matmul(verts[:, :, 1], self.J_regressor)
        joint_z = torch.matmul(verts[:, :, 2], self.J_regressor)
        joints = torch.stack([joint_x, joint_y, joint_z], dim=2)

        if get_skin:
            return verts, joints, Rs
        else:
            return joints

    def get_mano_vertices(self, wrist_euler, pose, shape, scale, translation, pose_type = 'pca', mmcp_center = False,external_transition=None):
        """
        :param wrist_euler: mano wrist rotation params in euler representation [batch_size, 3]
        :param pose: mano articulation params [batch_size, 45] or pca pose [batch_size, ncomps]
        :param shape: mano shape params [batch_size, 10]
        :param cam: mano scale and translation params [batch_size, 3]
        :return: vertices: mano vertices Nx778x3,
                 joints: 3d joints in BigHand skeleton indexing Nx21x3
        """

        # apply parameters on the model
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float)
        if not isinstance(translation, torch.Tensor):
            translation = torch.tensor(translation, dtype=torch.float)
        if not isinstance(wrist_euler, torch.Tensor):
            wrist_euler = torch.tensor(wrist_euler, dtype=torch.float)
        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose, dtype=torch.float)
        if not isinstance(shape, torch.Tensor):
            shape = torch.tensor(shape, dtype=torch.float)

        if self.is_cuda:
            translation = translation.cuda()
            scale = scale.cuda()
            shape = shape.cuda()
            pose = pose.cuda()
            wrist_euler = wrist_euler.cuda()

        #
        if pose_type == 'pca':
            pose = pose.clamp(-2.,2.)
            #shape = shape.clamp(-0.03, 0.03)

        verts, joints, Rs = self.forward(shape, pose, wrist_euler, pose_type, get_skin=True,external_transition=external_transition)

        scale = scale.contiguous().view(-1, 1, 1)
        trans = translation.contiguous().view(-1, 1, 3)

        verts = scale * verts
        verts = trans + verts
        joints = scale * joints
        joints = trans + joints

        if(self.oriorder==False):joints = joints[:, mano2bighand_skeidx, :]

        # mmcp is 3th joint in bighand order
        if mmcp_center:
            mmcp = joints[:, 3, :].clone().unsqueeze(1)
            verts -= mmcp
            joints -= mmcp

        #verts = torch.matmul(verts, self.base_rot_mat_x)

        joints = joints # convert to mm

        return verts, joints

    def quat2mat(self, quat):
        """Convert quaternion coefficients to rotation matrix.
        Args:
            quat: size = [B, 4] 4 <===>(w, x, y, z)
        Returns:
            Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
        """
        norm_quat = quat
        norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
        w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

        B = quat.size(0)

        w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z

        rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                              2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                              2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)

        return rotMat

    def batch_rodrigues(self, theta):
        l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
        angle = torch.unsqueeze(l1norm, -1)
        normalized = torch.div(theta, angle)
        angle = angle * 0.5
        v_cos = torch.cos(angle)
        v_sin = torch.sin(angle)
        quat = self.quat2mat(torch.cat([v_cos, v_sin * normalized], dim=1))
        return quat

    def batch_global_rigid_transformation(self, Rs, Js, parent,JsAll=None):
        N = Rs.shape[0]
        root_rotation = Rs[:, 0, :, :]
        Js = torch.unsqueeze(Js, -1)

        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            ones_homo = Variable(torch.ones(N, 1, 1))
            if self.is_cuda:
                ones_homo = ones_homo.cuda()
            t_homo = torch.cat([t, ones_homo], dim=1)
            return torch.cat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        newjs=JsAll.clone().reshape(N,21,3)

        newjsones=torch.ones([N,21,1]).to(Rs.device)
        newjs=torch.cat([newjs,newjsones],dim=2).reshape(N,21,4,1)
        orijs = newjs.clone().reshape(N,21,4)
        transidx=[2,3,17,   5, 6, 18,    8, 9, 20,    11, 12, 19,   14, 15, 16]
        transpdx=[1,2,3,    4, 5, 6,     7, 8, 9,     10, 11, 12,   13, 14, 15]
        #manopdx=[1,2,3,   4,5, 6,    7,8, 9,    10,11, 12,   13,14, 15]
        #parent: 012 045 078 01011 01314
        cpidx=[1,4,7,10,13]
        for i in range(len(cpidx)):
            a=minusHomoVectors(orijs[:, cpidx[i]],orijs[:, 0]).reshape(N,4,1)
            newjs[:,cpidx[i]]=(A0@a)


        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)

            a = minusHomoVectors(orijs[:,transidx[i-1]], orijs[:,transpdx[i-1]]).reshape(N,4,1)
            newjs[:,transidx[i-1]]=(res_here@a)
            results.append(res_here)

        self.newjs=newjs
        results = torch.stack(results, dim=1)

        new_J = results[:, :, :3, 3] #did not use later
        ones_homo = Variable(torch.zeros(N, 16, 1, 1))
        if self.is_cuda:ones_homo = ones_homo.cuda()
        Js_w0 = torch.cat([Js, ones_homo], dim=2)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        A = results - init_bone

        return new_J, A




    def matchTemplate2JointsGreedy(self,joint_gt:np.ndarray,tempJ=None,restrainFingerDOF=0):
        #restrainFingerDOF=1 forward
        #restrainFingerDOF=2 backward
        #restrainFingerDOF=3 backward+finger angle
        N = joint_gt.shape[0]
        joint_gt = joint_gt.reshape(N, 21, 3)
        if (not torch.is_tensor(joint_gt)):
            joint_gt = torch.tensor(joint_gt, device='cpu', dtype=torch.float32)
        device = joint_gt.device

        # first make wrist to zero

        orijoint_gt=joint_gt.clone()
        oriWrist = orijoint_gt[:, 0:1, :].clone()
        joint_gt = joint_gt- oriWrist.clone()

        transformG = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                                 1).reshape(N,
                                                                                                            16, 4, 4)
        transformL = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                                 1).reshape(N,
                                                                                                            16, 4, 4)
        transformLmano = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                                 1).reshape(N,
                                                                                                            16, 4, 4)
        transformG[:, 0, :3, 3] = joint_gt[:, 0].clone()
        transformL[:, 0, :3, 3] = joint_gt[:, 0].clone()
        transformLmano[:, 0, :3, 3] = joint_gt[:, 0].clone()



        if(tempJ is None):
            tempJ = self.bJ.reshape(1, 21, 3).clone().repeat(N, 1, 1).reshape(N, 21, 3).to(device)
        else:
            #print("use external template")
            if(not torch.is_tensor(tempJ)):tempJ=torch.tensor(tempJ,dtype=torch.float32,device=device)
            if(len(tempJ.shape)==3):
                tempJ=tempJ.reshape(N, 21, 3)
            else:
                tempJ = tempJ.reshape(1, 21, 3).clone().repeat(N, 1, 1).reshape(N, 21, 3)
        tempJori = tempJ.clone()
        tempJ = tempJ - tempJori[:, 0:1, :]
        tempJori = tempJori - tempJori[:, 0:1, :].clone()

        R = wristRotTorch(tempJ, joint_gt)
        transformG[:, 0, :3, :3] = R
        transformL[:, 0, :3, :3] = R
        transformLmano[:, 0, :3, :3] = R

        #print(joint_gt,tempJ)
        assert (torch.sum(joint_gt[:,0]-tempJ[:,0])<1e-5),"wrist joint should be same!"+str(torch.sum(joint_gt[:,0]-tempJ[:,0]))

        childern = [[1, 2, 3, 17, 4, 5, 6, 18, 7, 8, 9, 20, 10, 11, 12, 19, 13, 14, 15, 16],
                    [2, 3, 17], [3, 17], [17],
                    [5, 6, 18], [6, 18], [18],
                    [8, 9, 20], [9, 20], [20],
                    [11, 12, 19], [12, 19], [19],
                    [14, 15, 16], [15, 16], [16]]

        for child in childern[0]:
            t1 = (tempJ[:,child] - tempJ[:,0]).reshape(N,3,1)
            tempJ[:,child] = (transformL[:,0] @ getHomo3D(t1)).reshape(N,4,1)[:,:-1,0]


        if restrainFingerDOF==0:pass
        elif restrainFingerDOF==1:
            palmNorm = unit_vector(torch.cross(tempJ[:, 4] - tempJ[:, 0], tempJ[:, 7] - tempJ[:, 4], dim=1))
            palmd = -torch.sum((tempJ[:, 0]*palmNorm).reshape(N,3),dim=1,keepdim=True)
            #palmPlane=torch.cat([palmNorm,palmd],dim=1).reshape(N,4)
            palmHorizon=unit_vector(tempJ[:,1]-tempJ[:,7])
            self.mcpjoints = joint_gt.clone()
        elif restrainFingerDOF==2:
            joint_gt=self.restrainFingerDirectly(joint_gt)
        elif restrainFingerDOF==3:
            #have both restrainFingerDirectly and finger angle
            joint_gt,loss=self.restrainFingerAngle(joint_gt)
        else:
            assert False,"wrong restrainFingerDOF"
        # cpidx = [0, 1, 4, 7, 10, 13]
        # for i in range(len(cpidx)):
        #     print("mcp dis", cpidx[i], torch.mean(torch.norm(tempJ[:,cpidx[i]] - joint_gt[:,cpidx[i]],dim=1)) * 1000)



        manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        jidx = [[0], [1, 2, 3, 17], [4, 5, 6, 18], [10, 11, 12, 19], [7, 8, 9, 20], [13, 14, 15, 16]]
        mcpidx=[1,4,10,7]
        ratio = []
        for idx, i in enumerate(manoidx):
            pi = manopdx[idx]
            v0 = (tempJ[:,i] - tempJ[:,pi]).reshape(N,3)
            v1 = (joint_gt[:,i] - tempJ[:,pi]).reshape(N,3)

            # print('ratio',pi,i,torch.mean(torch.norm(v0)/torch.norm(joint_gt[:,i]-joint_gt[:,pi])))
            # ratio.append(np.linalg.norm(v0) / np.linalg.norm(v1))

            if(pi in mcpidx and restrainFingerDOF==1):
                dis,projectedPoint=projectPoint2Plane(points=joint_gt[:,i],planeNorm=palmNorm,planeD=palmd)

                self.mcpjoints[:,i]=projectedPoint.clone().reshape(N,3)
                vp=(projectedPoint - tempJ[:,pi]).reshape(N,3)
                mask=(torch.norm(vp,dim=1)>torch.norm(v0,dim=1)*0.5)
                N2=torch.sum(mask)
                if(N2>0):
                    pr=getRotationBetweenTwoVector(v0[mask],vp[mask])
                    rotedHarizon=(pr.reshape(N2,3,3)@palmHorizon[mask].reshape(N2,3,1))
                    assert rotedHarizon.shape==(N2,3,1)
                    fingerNorm=rotedHarizon.reshape(N2,3)
                    #fingerbaseD=v1.clone()
                    #fingerNorm=unit_vector(torch.cross(rotedHarizon,fingerbaseD,dim=1).reshape(N,3))
                    FingerD=-torch.sum(fingerNorm*tempJ[mask,i],dim=1,keepdim=True)
                    dis1,projectedjoint1 = projectPoint2Plane(points=joint_gt[mask, manoidx[idx+1]], planeNorm=fingerNorm, planeD=FingerD)
                    dis2,projectedjoint2 = projectPoint2Plane(points=joint_gt[mask, manoidx[idx+2]], planeNorm=fingerNorm, planeD=FingerD)
                    joint_gt[mask, manoidx[idx + 1]]=projectedjoint1
                    joint_gt[mask, manoidx[idx + 2]]=projectedjoint2



            tr = torch.eye(4, dtype=torch.float32,device=device).reshape(1, 4, 4).repeat(N,1,1)
            r = getRotationBetweenTwoVector(v0, v1)
            tr[:,:3, :3] = r.clone()
            t0 = (tempJ[:,pi]).reshape(N,3)
            tr[:,:-1, -1] = t0

            # print('r',r)

            transformL[:,idx + 1] = tr
            Gp = transformG[:,self.parents[idx + 1]].reshape(N,4,4)
            transformG[:,idx + 1] = transformL[:,idx + 1] @ Gp
            transformLmano[:,idx + 1] = torch.inverse(Gp) @ transformL[:,idx + 1] @ Gp

            for child in childern[pi]:
                t1 = (tempJ[:,child] - tempJ[:,pi]).reshape(N,3,1)
                tempJ[:,child] = (transformL[:,idx + 1] @ getHomo3D(t1)).reshape(N,4,1)[:,:-1,0]


        local_trans = transformLmano[:, 1:, :3, :3].reshape(N, 15, 3, 3)
        wrist_trans = transformLmano[:, 0, :3, :3].reshape(N, 1, 3, 3)

        if(restrainFingerDOF==1):
            self.newjoints=joint_gt.clone()

        outjoints = rotate2joint(wrist_trans, local_trans, tempJori, self.parents).reshape(N,21,3)
        assert (torch.mean(torch.sqrt(torch.sum((outjoints-tempJ)**2,dim=2)))<2),"outjoints and tempJ epe should be small"+str(torch.mean(torch.sqrt(torch.sum((outjoints - tempJ) ** 2, dim=2))))
        #print(torch.mean(torch.sqrt(torch.sum((outjoints - tempJ) ** 2, dim=2))))

        outjoints = outjoints + oriWrist
        # print('oriWrist',oriWrist)
        # print('innr edu',torch.mean(torch.sqrt(torch.sum((outjoints-orijoint_gt)**2,dim=2))))
        # print('innr2 edu',torch.mean(torch.sqrt(torch.sum(((outjoints+oriWrist)-(orijoint_gt))**2,dim=2))))


        if(restrainFingerDOF==3):
            return wrist_trans, local_trans, outjoints,loss
        return wrist_trans,local_trans,outjoints


    def matchTemplate2VerticesGreedy(self,vertices_gt:np.ndarray,tempJ=None,tempV=None):
        N = vertices_gt.shape[0]
        vertices_gt = vertices_gt.reshape(N, 778, 3)
        if (not torch.is_tensor(vertices_gt)):
            vertices_gt = torch.tensor(vertices_gt, device='cpu', dtype=torch.float32)
        device = vertices_gt.device

        # first make wrist to zero
        # orijoint_gt = joint_gt.clone()
        # oriWrist = orijoint_gt[:, 0:1, :].clone()
        # joint_gt = joint_gt - oriWrist.clone()


        transformG = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                                 1).reshape(N,
                                                                                                            16, 4, 4)
        transformL = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                                 1).reshape(N,
                                                                                                            16, 4, 4)
        transformLmano = torch.eye(4, dtype=torch.float32, device=device).reshape(1, 1, 4, 4).repeat(N, 16, 1,
                                                                                                 1).reshape(N,
                                                                                                            16, 4, 4)

        weight_coeff=getHardWeightCoeff(self.weight)

        if(tempV is None):
            tempV = self.v_template.reshape(1, 778, 3).clone().repeat(N, 1, 1).reshape(N, 778, 3).to(device)
            tempJ = self.bJ.reshape(1, 21, 3).clone().repeat(N, 1, 1).reshape(N, 21, 3).to(device)
        else:
            #print("use external template")
            if(not torch.is_tensor(tempV)):tempV=torch.tensor(tempV,dtype=torch.float32,device=device)
            if(len(tempV.shape)==3):
                tempV=tempV.reshape(N, 778, 3)
            else:
                tempV = tempV.reshape(1, 778, 3).clone().repeat(N, 1, 1).reshape(N, 778, 3)


        tempJori = tempJ.clone()
        R,t = getRotationBetweenTwoMeshBone(tempV, vertices_gt,weight_coeff[0],rt=True)
        transformG[:, 0, :3, :3] = R
        transformL[:, 0, :3, :3] = R
        transformLmano[:, 0, :3, :3] = R
        transformG[:, 0, :3, 3] = t
        transformL[:, 0, :3, 3] = t
        transformLmano[:, 0, :3, 3] = t

        #assert (torch.sum(vertices_gt[:,0]-tempJ[:,0])<1e-5),"wrist joint should be same!"

        childern = [[1, 2, 3, 17, 4, 5, 6, 18, 7, 8, 9, 20, 10, 11, 12, 19, 13, 14, 15, 16],
                    [2, 3, 17], [3, 17], [17],
                    [5, 6, 18], [6, 18], [18],
                    [8, 9, 20], [9, 20], [20],
                    [11, 12, 19], [12, 19], [19],
                    [14, 15, 16], [15, 16], [16]]
        bonelist= [[1,2,3],[2,3],[3],
                   [4,5,6],[5,6],[6],
                   [7, 8, 9], [8, 9], [9],
                   [10, 11, 12], [11, 12], [12],
                   [13, 14, 15], [14, 15], [15],
                   ]#for idx index, start from finger instead from wrist

        for child in childern[0]:
            t1 = (tempJ[:,child] - tempJ[:,0]).reshape(N,3,1)
            tempJ[:,child] = (transformL[:,0] @ getHomo3D(t1)).reshape(N,4,1)[:,:-1,0]

        tempV=(transformL[:,0].reshape(N,4,4)@getHomo3D(tempV).reshape(N,778,4,1))[:,:,:-1,0]

        manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        jidx = [[0], [1, 2, 3, 17], [4, 5, 6, 18], [10, 11, 12, 19], [7, 8, 9, 20], [13, 14, 15, 16]]
        for idx, i in enumerate(manoidx):
            pi = manopdx[idx]
            #v0 = (tempJ[:, i] - tempJ[:, pi]).reshape(N, 3)

            tr = torch.eye(4, dtype=torch.float32,device=device).reshape(1, 4, 4).repeat(N,1,1)
            #print(tempV.shape,vertices_gt.shape,weight_coeff[pi].shape)
            #print(np.arange(778)[weight_coeff[pi]>.9])
            r,t = getRotationBetweenTwoMeshBone(tempV, vertices_gt,weight_coeff[pi].clone(),rt=True)
            pret = (tempJ[:, pi]).reshape(N, 1,3)
            # tempV, vertices_gt = tempV - pret, vertices_gt - pret
            # r0 = getRotationBetweenTwoMeshBone(tempV, vertices_gt,weight_coeff[pi].clone())
            # tempV, vertices_gt = tempV + pret, vertices_gt + pret
            # r1 = getRotationBetweenTwoMeshBone(tempV, vertices_gt, weight_coeff[pi].clone())
            # if manoidx[idx]>=16:r=r0
            # print(r, r0, r1)
            # if manoidx[idx] >= 16:
            #     r, t = ICP(tempV.clone(), vertices_gt.clone(), weight_coeff[0].clone(),pret)

            tr[:,:3, :3] = r
            #print(r0,r)
            t0 = (tempJ[:,pi]).reshape(N,3)
            tr[:,:-1, -1] = t0


            transformL[:,idx + 1] = tr
            Gp = transformG[:,self.parents[idx + 1]].reshape(N,4,4)
            transformG[:,idx + 1] = transformL[:,idx + 1] @ Gp
            transformLmano[:,idx + 1] = torch.inverse(Gp) @ transformL[:,idx + 1] @ Gp

            for child in childern[pi]:
                t1 = (tempJ[:,child] - tempJ[:,pi]).reshape(N,3,1)
                tempJ[:,child] = (transformL[:,idx + 1] @ getHomo3D(t1)).reshape(N,4,1)[:,:-1,0]

            curweight=torch.sum(weight_coeff[bonelist[idx]],dim=0).reshape(N,778,1,1)
            curidx=(torch.sum(weight_coeff[bonelist[idx]],dim=0).reshape(N,778)>0.7)
            tempV[curidx] = (r.reshape(N, 3, 3) @ (tempV.reshape(N, 778, 3, 1)*curweight))[:, :, :, 0][curidx]


        local_trans = transformLmano[:, 1:, :3, :3].reshape(N, 15, 3, 3)
        wrist_trans = transformLmano[:, 0, :3, :3].reshape(N, 1, 3, 3)

        outjoints = rotate2joint(wrist_trans, local_trans, tempJori, self.parents).reshape(N,21,3)
        assert (torch.mean(torch.sqrt(torch.sum((outjoints-tempJ)**2,dim=2)))<2),"outjoints and tempJ epe should be small"+str(torch.mean(torch.sqrt(torch.sum((outjoints - tempJ) ** 2, dim=2))))
        #print(torch.mean(torch.sqrt(torch.sum((outjoints - tempJ) ** 2, dim=2))))


        return wrist_trans,local_trans,outjoints


    def matchTemplate2Vertices(self,vertices_gt,tempV=None,tempJ=None):
        device=vertices_gt.device
        N=vertices_gt.shape[0]
        vertices_gt = vertices_gt.reshape(N,778, 3)
        if (not torch.is_tensor(vertices_gt)):
            vertices_gt = torch.tensor(vertices_gt,device=device,dtype=torch.float32)
        N = vertices_gt.shape[0]
        transformG = torch.eye(4, dtype=torch.float32,device=device).reshape(1,1, 4, 4).repeat(N,16,1,1).reshape(N,16, 4, 4)
        transformL = torch.eye(4, dtype=torch.float32,device=device).reshape(1,1, 4, 4).repeat(N,16,1,1).reshape(N,16, 4, 4)

        weight_coeff=getUsefulWeightCoeff(self.weight.clone())
        weight_coeff=getPortionWeightCoeff(self.weight.clone())
        weight_coeff=getUsefulWeightCoeff(self.weight.clone())
        #weight_coeff=getHardWeightCoeff(self.weight.clone())
        weight_coeff=getOriWeightCoeff(self.weight.clone())

        if (tempV is None):
            tempV = self.v_template.reshape(1, 778, 3).clone().repeat(N, 1, 1).reshape(N, 778, 3).to(device)
            tempJ = self.bJ.reshape(1, 21, 3).clone().repeat(N, 1, 1).reshape(N, 21, 3).to(device)
        else:
            if (not torch.is_tensor(tempV)): tempV = torch.tensor(tempV, dtype=torch.float32, device=device)
            if (len(tempV.shape) == 3):
                tempV = tempV.reshape(N, 778, 3)
            else:
                tempV = tempV.reshape(1, 778, 3).clone().repeat(N, 1, 1).reshape(N, 778, 3)
        R,t=getRotationBetweenTwoMeshBone(tempV,vertices_gt,weight_coeff[0],rt=True)

        transformG[:,0,:3,:3],transformL[:,0,:3,:3],transformG[:, 0, :3, 3],transformL[:, 0, :3, 3] = R,R,t,t

        manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        for idx, i in enumerate(manoidx):
            pi = manopdx[idx]

            tr = torch.eye(4, dtype=torch.float32,device=device).reshape(1, 4, 4).repeat(N,1,1)
            r,t = getRotationBetweenTwoMeshBone(tempV, vertices_gt,weight_coeff[pi],rt=True)
            # if manoidx[idx]>=16:
            #     r, t = ICP(tempV.clone(), vertices_gt.clone(), weight_coeff[0].clone())
            tr[:,:3, :3],tr[:,:3,3]=r,t

            invp = torch.inverse(transformG[:,self.parents[idx + 1]])
            transformG[:,idx + 1] = tr.clone().reshape(N,4,4)
            local = invp @ tr

            transformL[:,idx + 1] = local.clone().reshape(N,4,4)

        local_trans = transformL[:,1:, :3, :3].reshape(N,15,3,3)
        wrist_trans = transformL[:,0, :3, :3].reshape(N,1,3,3)

        outjoints=rotate2joint(wrist_trans,local_trans,tempJ,self.parents)
        return wrist_trans,local_trans,outjoints.reshape(N,21,3)



    def matchTemplate2Joints(self,joint_gt,tempJ=None):
        device=joint_gt.device
        N=joint_gt.shape[0]
        joint_gt = joint_gt.reshape(N,21, 3)
        if (not torch.is_tensor(joint_gt)):
            joint_gt = torch.tensor(joint_gt,device=device,dtype=torch.float32)
        N = joint_gt.shape[0]
        transformG = torch.eye(4, dtype=torch.float32,device=device).reshape(1,1, 4, 4).repeat(N,16,1,1).reshape(N,16, 4, 4)
        transformL = torch.eye(4, dtype=torch.float32,device=device).reshape(1,1, 4, 4).repeat(N,16,1,1).reshape(N,16, 4, 4)
        transformG[:,0, :3, 3] = joint_gt[:,0].clone()
        transformL[:,0, :3, 3] = joint_gt[:,0].clone()

        if (tempJ is None):
            tempJ = self.bJ.reshape(1, 21, 3).clone().repeat(N, 1, 1).reshape(N, 21, 3).cuda()
        else:
            if (not torch.is_tensor(tempJ)): tempJ = torch.tensor(tempJ, dtype=torch.float32, device=device)
            if (len(tempJ.shape) == 3):
                tempJ = tempJ.reshape(N, 21, 3)
            else:
                tempJ = tempJ.reshape(1, 21, 3).clone().repeat(N, 1, 1).reshape(N, 21, 3)
        R=wristRotTorch(tempJ,joint_gt)
        transformG[:,0, :3, :3] = R
        transformL[:,0, :3, :3] = R

        manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        for idx, i in enumerate(manoidx):
            pi = manopdx[idx]
            v0 = tempJ[:,i] - tempJ[:,pi]
            v1 = joint_gt[:,i] - joint_gt[:,pi]
            tr = torch.eye(4, dtype=torch.float32,device=device).reshape(1, 4, 4).repeat(N,1,1)
            r = getRotationBetweenTwoVector(v0, v1).reshape(N,3,3)
            tr[:,:3, :3] = r.clone()
            tr = getBatchTransitionMatrix3D(joint_gt[:,pi]) @ tr @ getBatchTransitionMatrix3D(-tempJ[:,pi])


            invp = torch.inverse(transformG[:,self.parents[idx + 1]])
            transformG[:,idx + 1] = tr.clone().reshape(N,4,4)
            local = invp @ tr

            transformL[:,idx + 1] = local.clone().reshape(N,4,4)

        local_trans = transformL[:,1:, :3, :3].reshape(N,15,3,3)
        wrist_trans = transformL[:,0, :3, :3].reshape(N,1,3,3)

        outjoints=rotate2joint(wrist_trans,local_trans,tempJ,self.parents)
        return wrist_trans,local_trans,outjoints.reshape(N,21,3)


    #@staticmethod
    def rotate2joint(self,wrist_trans,local_trans,template):
        device = wrist_trans.device
        Rs = torch.cat([wrist_trans, local_trans], dim=1)
        N = Rs.shape[0]
        root_rotation = Rs[:, 0, :, :]
        Js = torch.unsqueeze(template, -1)

        def make_A(R, t):
            R_homo = F.pad(R, [0, 0, 0, 1, 0, 0])
            ones_homo = Variable(torch.ones(N, 1, 1))
            ones_homo = ones_homo.to(device)
            t_homo = torch.cat([t, ones_homo], dim=1)
            return torch.cat([R_homo, t_homo], 2)

        A0 = make_A(root_rotation, Js[:, 0])
        results = [A0]
        newjs = template.clone()

        newjsones = torch.ones([newjs.shape[0], 21, 1]).to(device)
        newjs = torch.cat([newjs, newjsones], dim=2).reshape(N, 21, 4, 1)
        orijs = newjs.clone()
        transidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        cpidx = [1, 4, 7, 10, 13]

        for i in range(5):
            a = minusHomoVectors(orijs[:, cpidx[i]], orijs[:, 0]).reshape(N,4,1)
            newjs[:, cpidx[i]] = (A0 @ a)

        for i in range(1, self.parents.shape[0]):
            j_here = Js[:, i] - Js[:, self.parents[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[self.parents[i]], A_here)

            a = minusHomoVectors(orijs[:, transidx[i - 1]], orijs[:, i])
            newjs[:, transidx[i - 1]] = (res_here @ a).reshape(N,4,1)
            results.append(res_here)

        return newjs[:,:,:-1].reshape(-1,21,3)

