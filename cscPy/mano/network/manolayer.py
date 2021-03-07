import pickle

import torch.nn as nn
import torchgeometry as tgm
import torch
import torch.functional as F


class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)

        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)

        return torch.stack([b1, b2, b3], dim=-1)


class VPoser(nn.Module):
    def __init__(self,inshape=21,dim=3):
        super(VPoser, self).__init__()
        latentD=64
        self.latentD = latentD
        self.use_cont_repr = True
        num_neurons=128*2
        n_features = inshape*dim
        self.strange=2
        self.num_joints = 21
        self.dim=dim
        if dim==2:
            self.camerafc = nn.Linear(num_neurons,1)
            self.trans2dfc = nn.Linear(num_neurons, 2)
        self.bodyprior_enc_bn1 = nn.BatchNorm1d(n_features)
        self.bodyprior_enc_fc1 = nn.Linear(n_features, num_neurons)
        self.bodyprior_enc_bn2 = nn.BatchNorm1d(num_neurons)
        self.bodyprior_enc_fc2 = nn.Linear(num_neurons, num_neurons)
        self.bodyprior_enc_mu = nn.Linear(num_neurons, latentD)
        self.bodyprior_enc_logvar = nn.Linear(num_neurons, latentD)
        #tr, sca, sh
        self.transitionfc=nn.Linear(num_neurons,3)
        self.scalefc=nn.Linear(num_neurons,1)
        self.shapefc=nn.Linear(num_neurons,10)


        self.bodyprior_dec_fc1 = nn.Linear(latentD, num_neurons)
        self.bodyprior_dec_fc2 = nn.Linear(num_neurons, num_neurons)
        self.leakyrate=0.2

        if self.use_cont_repr:
            self.rot_decoder = ContinousRotReprDecoder()

        self.bodyprior_dec_out = nn.Linear(num_neurons, 16* 6)

    def encode(self, Pin):
        '''
        :param Pin: Nx(numjoints*3)
        :param rep_type: 'matrot'/'aa' for matrix rotations or axis-angle
        :return:
        '''

        Xout = Pin.view(Pin.shape[0], -1)  # flatten input
        Xout = self.bodyprior_enc_bn1(Xout)

        Xout = F.leaky_relu(self.bodyprior_enc_fc1(Xout), negative_slope=self.leakyrate)
        #Xout = self.bodyprior_enc_bn2(Xout)
        Xout = F.leaky_relu(self.bodyprior_enc_fc2(Xout), negative_slope=self.leakyrate)
        transition=self.transitionfc(Xout)
        scale = self.scalefc(Xout)
        sh = self.shapefc(Xout)
        if self.dim==2:
            cam=self.camerafc(Xout)
            trans2d=self.trans2dfc(Xout)
            return torch.distributions.normal.Normal(self.bodyprior_enc_mu(Xout),
                                                     F.softplus(self.bodyprior_enc_logvar(Xout))), transition, scale, sh,cam,trans2d
        else:
            #return torch.distributions.normal.Normal(self.bodyprior_enc_mu(Xout), F.softplus(self.bodyprior_enc_logvar(Xout))),transition,scale,sh
            return self.bodyprior_enc_mu(Xout),transition,scale,sh

    def decode(self, Zin, output_type='matrot'):
        assert output_type in ['matrot', 'aa']

        Xout = F.leaky_relu(self.bodyprior_dec_fc1(Zin), negative_slope=self.leakyrate)
        Xout = F.leaky_relu(self.bodyprior_dec_fc2(Xout), negative_slope=self.leakyrate)
        Xout = self.bodyprior_dec_out(Xout)
        if self.use_cont_repr:
            Xout = self.rot_decoder(Xout)
        else:
            Xout = torch.tanh(Xout)

        Xout = Xout.view([-1, 16, 9])
        if output_type == 'aa': return VPoser.matrot2aa(Xout)
        return Xout

    def forward(self, Pin, output_type='aa'):
        '''
        :param Pin: aa: Nx1xnum_jointsx3 / matrot: Nx1xnum_jointsx9
        :param input_type: matrot / aa for matrix rotations or axis angles
        :param output_type: matrot / aa
        :return:
        '''
        assert output_type in ['matrot', 'aa']
        # if input_type == 'aa': Pin = VPoser.aa2matrot(Pin)
        # if Pin.size(3) == 3: Pin = VPoser.aa2matrot(Pin)
        if(self.dim==3):q_z,transition,scale,sh = self.encode(Pin)
        else:q_z,transition,scale,sh,cam,trans2d = self.encode(Pin)

        #q_z_sample = q_z.rsample()
        #Prec = self.decode(q_z_sample)
        Prec = self.decode(q_z)

        #results = {'mean':q_z.mean, 'std':q_z.scale,'transition':transition,"scale":scale,"shape":sh}
        results = {'transition':transition,"scale":scale,"shape":sh}
        if(self.dim==2):
            results['cam']=cam
            results['trans2d']=trans2d

        if output_type == 'aa': results['pose_aa'] = VPoser.matrot2aa(Prec)
        else: results['pose_matrot'] = Prec
        return results

    def sample_poses(self, num_poses, output_type='aa', seed=None):
        np.random.seed(seed)
        dtype = self.bodyprior_dec_fc1.weight.dtype
        device = self.bodyprior_dec_fc1.weight.device
        self.eval()
        with torch.no_grad():
            Zgen = torch.tensor(np.random.normal(0., 1., size=(num_poses, self.latentD)), dtype=dtype).to(device)
        return self.decode(Zgen, output_type=output_type)

    @staticmethod
    def matrot2aa(pose_matrot):
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        batch_size = pose_matrot.size(0)
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 16, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        batch_size = pose.size(0)
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous().view(batch_size, 1, -1, 9)
        return pose_body_matrot



# joint mapping indices from mano to bighand
mano2bighand_skeidx = [0, 13, 1, 4, 10, 7, 14, 15, 16, 2, 3, 17, 5, 6, 18, 11, 12, 19, 8, 9, 20]

from cscPy.mano.network.utils import *

class MANO_SMPL(nn.Module):
    def __init__(self, mano_pkl_path, ncomps = 10, flat_hand_mean=False,cuda=True,bighandorder=False,device='cuda'):
        super(MANO_SMPL, self).__init__()
        self.bighandorder=bighandorder
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
        print('self.parents',self.parents)

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

        if(self.bighandorder):
            joints = joints[:, mano2bighand_skeidx, :]

        # mmcp is 3th joint in bighand order
        if mmcp_center and self.bighandorder:
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
        #newjs=JsAll.clone().reshape(N,21,3)

        #newjsones=torch.ones([N,21,1])
        #newjs=torch.cat([newjs,newjsones],dim=2).reshape(N,21,4,1)
        #orijs = newjs.clone().reshape(N,21,4)
        # transidx=[2,3,17,   5, 6, 18,    8, 9, 20,    11, 12, 19,   14, 15, 16]
        # transpdx=[1,2,3,    4, 5, 6,     7, 8, 9,     10, 11, 12,   13, 14, 15]
        #manopdx=[1,2,3,   4,5, 6,    7,8, 9,    10,11, 12,   13,14, 15]
        #parent: 012 045 078 01011 01314
        #cpidx=[1,4,7,10,13]
        #for i in range(len(cpidx)):
            #a=minusHomoVectors(orijs[:, cpidx[i]],orijs[:, 0]).reshape(N,4,1)
            #newjs[:,cpidx[i]]=(A0@a)


        for i in range(1, parent.shape[0]):
            j_here = Js[:, i] - Js[:, parent[i]]
            A_here = make_A(Rs[:, i], j_here)
            res_here = torch.matmul(results[parent[i]], A_here)

            #a = minusHomoVectors(orijs[:,transidx[i-1]], orijs[:,transpdx[i-1]]).reshape(N,4,1)
            #newjs[:,transidx[i-1]]=(res_here@a)
            results.append(res_here)

        #self.newjs=newjs
        results = torch.stack(results, dim=1)

        new_J = results[:, :, :3, 3] #did not use later
        ones_homo = Variable(torch.zeros(N, 16, 1, 1))
        if self.is_cuda:
            ones_homo = ones_homo.cuda()
        Js_w0 = torch.cat([Js, ones_homo], dim=2)
        init_bone = torch.matmul(results, Js_w0)
        init_bone = F.pad(init_bone, [3, 0, 0, 0, 0, 0, 0, 0])
        A = results - init_bone

        return new_J, A



    def restrainFingerDirectly(self,joint_gt:np.ndarray):
        N=joint_gt.shape[0]
        jidx = [[1, 2, 3, 17], [4, 5, 6, 18], [10, 11, 12, 19], [7, 8, 9, 20], [13, 14, 15, 16]]
        from itertools import combinations
        for finger in jidx:
            subsets = list(combinations(finger, 3))
            vlist=[]
            for subset in subsets:
                v0=joint_gt[:,subset[0]]-joint_gt[:,subset[1]]
                v1=joint_gt[:,subset[1]]-joint_gt[:,subset[2]]
                vh=torch.cross(v0,v1,dim=1)
                vlist.append(vh.reshape(1,N,3))
            vh=torch.mean(torch.cat(vlist,dim=0),dim=0).reshape(N,1,3)
            subj=joint_gt[:,finger]
            vd=torch.mean(-torch.sum(subj*vh,dim=2),dim=1)
            for idx in range(4):
                joint_gt[:,finger[idx]]=projectPoint2Plane(joint_gt[:,finger[idx]],vh,vd)
        return joint_gt

    def get_palm_norm(self,joint_gt:np.ndarray):
        palmNorm = unit_vector(torch.cross(joint_gt[:, 4] - joint_gt[:, 0], joint_gt[:, 7] - joint_gt[:, 4], dim=1))
        return palmNorm

    def restrainFingerAngle(self,joint_gt:np.ndarray):
        N = joint_gt.shape[0]
        joint_gt=self.restrainFingerDirectly(joint_gt)
        palmNorm=self.get_palm_norm(joint_gt).reshape(N,3)
        vecWristMcp=unit_vector(joint_gt[:, 4] - joint_gt[:, 0]).reshape(N,3)
        stdFingerNorm=unit_vector(torch.cross(palmNorm,vecWristMcp,dim=1))
        jidx = [[0,1, 2, 3, 17], [0,4, 5, 6, 18], [0,10, 11, 12, 19], [0,7, 8, 9, 20]]
        loss=0
        for finger in jidx:
            angleThreshold=[3.14/(2*12),3.14/(2*6)]
            for i in range(1,3):
                print('finger i',finger,i)
                a0,a1,a2=joint_gt[:,finger[i]],joint_gt[:,finger[i+1]],joint_gt[:,finger[i+2]].reshape(N,3)
                a,b=unit_vector(a1-a0),unit_vector(a2-a1)
                fingernorm=unit_vector(torch.cross(a,b,dim=1))
                sign=torch.sum(fingernorm*stdFingerNorm,dim=1).reshape(N)
                print('sign',sign)
                angle = torch.acos(torch.sum(a * b, dim=1))
                loss += torch.sum((sign < 0) & (torch.abs(angle) > angleThreshold[i - 1]))
                print('loss', (sign < 0) & (torch.abs(angle) > angleThreshold[i - 1]))
        return joint_gt,loss





    def matchTemplate2JointsGreedy(self,joint_gt:np.ndarray,tempJ=None,restrainFingerDOF=0):
        #restrainFingerDOF=1 forward
        #restrainFingerDOF=2 backward
        #restrainFingerDOF=3 backward+finger angle



        device = joint_gt.device
        N = joint_gt.shape[0]
        joint_gt = joint_gt.reshape(N, 21, 3)
        if (not torch.is_tensor(joint_gt)):
            joint_gt = torch.tensor(joint_gt, device=device, dtype=torch.float32)
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
        R = wristRotTorch(tempJ, joint_gt)
        transformG[:, 0, :3, :3] = R
        transformL[:, 0, :3, :3] = R
        transformLmano[:, 0, :3, :3] = R

        assert (torch.sum(joint_gt[:,0]-tempJ[:,0])<1e-5),"wrist joint should be same!"

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
                projectedPoint=projectPoint2Plane(points=joint_gt[:,i],planeNorm=palmNorm,planeD=palmd)

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
                    projectedjoint1 = projectPoint2Plane(points=joint_gt[mask, manoidx[idx+1]], planeNorm=fingerNorm, planeD=FingerD)
                    projectedjoint2 = projectPoint2Plane(points=joint_gt[mask, manoidx[idx+2]], planeNorm=fingerNorm, planeD=FingerD)
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

        if(restrainFingerDOF==3):
            return wrist_trans, local_trans, outjoints,loss
        return wrist_trans,local_trans,outjoints



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

