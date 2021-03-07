import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
STB2Bighand_skeidx = [0, 17, 13, 9, 5, 1, 18, 19, 20, 14, 15, 16, 10, 11, 12, 6, 7, 8, 2, 3, 4]
Bighand2mano_skeidx = [0, 2, 9, 10, 3, 12, 13, 5, 18, 19, 4, 15, 16, 1, 6, 7, 8, 11, 14, 17, 20]
RHD2Bighand_skeidx = [0,4,8,12,16,20,3,2,1,7,6,5,11,10,9,15,14,13,19,18,17]
SynthHands2Bighand_skeidx=[0,1,5,9,13,17,2,6,10,14,18,3,7,11,15,19,4,8,12,16,20]

def minusHomoVectors(v0, v1):
    v = v0 - v1
    if (v.shape[-1] == 1):
        v[..., -1, 0] = 1
    else:
        v[..., -1] = 1
    return v

def getRtMatrix2D(theta,x=0,y=0):
    return np.array([[np.cos(theta), -np.sin(theta),x],
                     [np.sin(theta), np.cos(theta),y],
                     [0,0,1]])

def comebineRt3D(r,t):
    return np.array([[r[0,0], r[0,1],r[0,2],t[0]],
                     [r[1,0], r[1,1],r[1,2],t[1]],
                     [r[2,0], r[2,1],r[2,2],t[2]],
                     [0,0,0,1]])


def getTransitionMatrix2D(x=0,y=0):
    return np.array([[1, 0,x],[0, 1,y],[0,0,1]])

def getInhomogeneousLine(lines:np.ndarray):
    l=lines.copy()
    l=l.reshape(-1,l.shape[-1])[:,:-1]
    return l

def get32fTensor(a):
    return torch.tensor(a,dtype=torch.float32)


def getBoneLen(joints_gt):
    if (torch.is_tensor(joints_gt)): joints_gt = joints_gt.detach().cpu().numpy().copy()
    joints_gt=joints_gt.reshape(21,3).copy()
    out=[]
    manopdx = [1, 2, 3, 17,   4, 5, 6, 18,  7, 8, 9, 20,  10, 11, 12,19,  13, 14, 15,16]
    manoppx = [0, 1, 2, 3,  0, 4, 5, 6,  0, 7, 8, 9,   0, 10, 11,12,  0, 13, 14,15]
    for i in range(len(manoppx)):
        ppi = manoppx[i]
        pi = manopdx[i]
        d=np.linalg.norm(joints_gt[pi] - joints_gt[ppi])+1e-8
        out.append(d)
    return np.array(out).reshape(20)

def getTemplateFrom(boneLen,manoTemplate):
    if(torch.is_tensor(boneLen)):boneLen=boneLen.clone().detach().cpu().numpy().copy()
    boneLen=boneLen.copy().reshape(20)
    boneLen=np.concatenate([[0],boneLen],axis=0).reshape(21)
    OriManotempJ = manoTemplate.reshape(21, 3)
    manotempJ = OriManotempJ.copy()

    boneidxpalm=[1,5,9,13,17]
    manopalm=[1,4,7,10,13]
    for i in range(5):
        ci=manopalm[i]
        dm=np.linalg.norm(OriManotempJ[ci] - OriManotempJ[0]) + 1e-8
        manotempJ[ci]=manotempJ[0]+(OriManotempJ[ci]-OriManotempJ[0])/dm*boneLen[boneidxpalm[i]]


    manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
    manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    boneidxfinger=[2,3,4,6,7,8,10,11,12,14,15,16,18,19,20]

    for idx in range(len(manoidx)):
        ci = manoidx[idx]
        pi = manopdx[idx]
        ppi = manoppx[idx]
        dp = boneLen[boneidxfinger[idx]] + 1e-8
        dm = np.linalg.norm(OriManotempJ[pi] - OriManotempJ[ppi]) + 1e-8
        manotempJ[ci] = manotempJ[pi] + (manotempJ[pi] - manotempJ[ppi]) / dm * dp
    return manotempJ



def getRefJoints(joint_gt):
    if(torch.is_tensor(joint_gt)):
        N=joint_gt.shape[0]
        OriManotempJ = torch.tensor(joint_gt.reshape(N, 21, 3),dtype=torch.float32)
        manotempJ = OriManotempJ.clone()
        manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        for idx in range(len(manoidx)):
            ci = manoidx[idx]
            pi = manopdx[idx]
            ppi = manoppx[idx]
            dp = torch.norm(OriManotempJ[:, ci] - OriManotempJ[:, pi], dim=-1, keepdim=True)+1e-8
            dm = torch.norm(OriManotempJ[:, pi] - OriManotempJ[:, ppi], dim=-1, keepdim=True)+1e-8
            manotempJ[:, ci] = manotempJ[:, pi] + (manotempJ[:, pi] - manotempJ[:, ppi]) / dm * dp
        return manotempJ
    else:
        OriManotempJ = joint_gt.reshape(21, 3)
        manotempJ = OriManotempJ.copy()
        manoidx = [2, 3, 17, 5, 6, 18, 8, 9, 20, 11, 12, 19, 14, 15, 16]
        manopdx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        manoppx = [0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
        for idx in range(len(manoidx)):
            ci = manoidx[idx]
            pi = manopdx[idx]
            ppi = manoppx[idx]
            dp = np.linalg.norm(OriManotempJ[ci] - OriManotempJ[pi]) + 1e-8
            dm = np.linalg.norm(OriManotempJ[pi] - OriManotempJ[ppi]) + 1e-8
            manotempJ[ci] = manotempJ[pi] + (manotempJ[pi] - manotempJ[ppi]) / dm * dp
        return manotempJ

def getHomo3D(x):
    if(torch.is_tensor(x)):
        if(x.shape[-1]==4):return x
        if(x.shape[-1]==1 and x.shape[-2]==4):return x
        if(x.shape[-1]==1 and x.shape[-2]==3):
            return torch.cat([x, torch.ones([*(x.shape[:-2])] + [1,1], dtype=torch.float32,device=x.device)], dim=-2)
        if(x.shape[-1]==3):
            return torch.cat([x, torch.ones([*(x.shape[:-1])] + [1], dtype=torch.float32,device=x.device)], dim=-1)
    if(x.shape[-1]==3):
        return np.concatenate([x,np.ones([*(x.shape[:-1])]+[1],dtype=np.float64)],axis=-1)
    return x

def projectPoint2Plane(points,planeNorm,planeD):
    N=points.shape[0]
    # print(planeNorm.reshape(N,3))
    # print(torch.sum((points * planeNorm.reshape(N,3)).reshape(N, 3), dim=1, keepdim=True).shape)
    # print(planeD.reshape(N,1).shape)
    dis = (torch.sum((points * planeNorm.reshape(N,3)).reshape(N, 3), dim=1, keepdim=True) + planeD.reshape(N,1)).reshape(N, 1)
    projectedPoint = (points - dis*planeNorm.reshape(N,3)).reshape(N, 3)
    return projectedPoint



def getBatchTransitionMatrix3D(x):
    bs=x.shape[0]
    x=x.reshape(bs,3)
    device=x.device
    a=torch.eye(4, dtype=torch.float32, device=device).reshape(1, 4, 4).repeat(bs, 1, 1)
    a[:,:-1,-1]=x
    return a

def getTransitionMatrix3D(x=0,y=0,z=0):
    return np.array([[1, 0,0,x],[0, 1,0,y],[0, 0,1,z],[0,0,0,1]])

def AxisRotMat(angles,rotation_axis):
    ''' rotation matrix from rotation around axis
        see https://en.wikipedia.org/wiki/Rotation_matrix#Axis_and_angle
         [[cos+self.xx*(1-cos), self.xy*(1-cos)-self.z*sin, self.xz*(1-cos)+self.y*sin, 0.0],
          [self.xy*(1-cos)+self.z*sin, cos+self.yy*(1-cos), self.yz*(1-cos)-self.x*sin, 0.0],
          [self.xz*(1-cos)-self.y*sin, self.yz*(1-cos)+self.x*sin, cos+self.zz*(1-cos), 0.0],
          [0.0, 0.0, 0.0, 1.0]]
    '''
    x,y,z=rotation_axis
    xx,xy,xz,yy,yz,zz=x*x,x*y,x*z,y*y,y*z,z*z
    c = np.cos(angles)
    s = np.sin(angles)
    i = 1 - c
    rot_mats=np.eye(4).astype(np.float32)
    rot_mats[0,0] =  xx * i + c
    rot_mats[0,1] =  xy * i -  z * s
    rot_mats[0,2] =  xz * i +  y * s

    rot_mats[1,0] =  xy * i +  z * s
    rot_mats[1,1] =  yy * i + c
    rot_mats[1,2] =  yz * i -  x * s

    rot_mats[2,0] =  xz * i -  y * s
    rot_mats[2,1] =  yz * i +  x * s
    rot_mats[2,2] =  zz * i + c
    rot_mats[3,3]=1
    return rot_mats

def getRotationMatrix3D(thetax,thetay,thetaz):
    rx=AxisRotMat(thetax,[1,0,0])
    ry=AxisRotMat(thetay,[0,1,0])
    rz=AxisRotMat(thetaz,[0,0,1])
    return rx@ry@rz


def unit_vector(vec):
    if(torch.is_tensor(vec)):
        bs=vec.shape[0]
        vec=vec.reshape(bs,3)
        return vec / (torch.norm(vec,dim=1,keepdim=True)+1e-8)
    return vec / (np.linalg.norm(vec)+1e-8)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if (torch.is_tensor(v1_u)):
        bs=v1_u.shape[0]
        v1_u=v1_u.reshape(bs,3)
        return torch.acos(torch.clamp(torch.sum(v1_u*v2_u,dim=1), -1.0, 1.0))
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotate2joint(wrist_trans,local_trans,template,parent):
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

    for i in range(1, parent.shape[0]):
        j_here = Js[:, i] - Js[:, parent[i]]
        A_here = make_A(Rs[:, i], j_here)
        res_here = torch.matmul(results[parent[i]], A_here)

        a = minusHomoVectors(orijs[:, transidx[i - 1]], orijs[:, i])
        newjs[:, transidx[i - 1]] = (res_here @ a).reshape(N,4,1)
        results.append(res_here)

    return newjs[:,:,:-1].reshape(N,21,3)


def wristRotTorch(tempJ,joint_gt):
    N=tempJ.shape[0]
    palm0 = torch.cross(tempJ[:, 4] - tempJ[:, 0], tempJ[:, 7] - tempJ[:, 4], dim=1)
    palm1 = torch.cross(joint_gt[:, 4] - joint_gt[:, 0], joint_gt[:, 7] - joint_gt[:, 4], dim=1)
    r = getRotationBetweenTwoVector(palm0, palm1)
    pl0 = r @ (tempJ[:, 4] - tempJ[:, 0]).reshape(N, 3, 1)
    pl0 = pl0.reshape(N, 3)
    pl1 = joint_gt[:, 4] - joint_gt[:, 0]
    pl1 = pl1.reshape(N, 3)
    r2 = getRotationBetweenTwoVector(pl0, pl1)
    return r2@r

def wristRot(tempJ,joint_gt):
    # pal=np.array([unit_vector(tempJ[1]-tempJ[0]),
    #               unit_vector(tempJ[4]-tempJ[0]),
    #               # unit_vector(tempJ[7] - tempJ[0]),
    #               # unit_vector(tempJ[10] - tempJ[0]),
    #               unit_vector(tempJ[7] - tempJ[1]),
    #               unit_vector(tempJ[10] - tempJ[4]),])
    # U, S, Vh = np.linalg.svd(pal)
    # palm0 = Vh[-1]
    #
    # pal = np.array([unit_vector(joint_gt[1] - joint_gt[0]),
    #                 unit_vector(joint_gt[4] - joint_gt[0]),
    #                 # unit_vector(joint_gt[7] - joint_gt[0]),
    #                 # unit_vector(joint_gt[10] - joint_gt[0]),
    #                 unit_vector(joint_gt[7] - joint_gt[1]),
    #                 unit_vector(joint_gt[10] - joint_gt[4]),])
    # U, S, Vh = np.linalg.svd(pal)
    # palm1 = Vh[-1]

    palm0 = np.cross(tempJ[4] - tempJ[0], tempJ[7] - tempJ[4])
    palm1 = np.cross(joint_gt[4] - joint_gt[0], joint_gt[7] - joint_gt[4])
    r = getRotationBetweenTwoVector(palm0.copy(), palm1.copy())
    pl0 = r @ (tempJ[4] - tempJ[0])
    pl1 = joint_gt[4] - joint_gt[0]
    r2 = getRotationBetweenTwoVector(pl0, pl1)
    return r2 @ r

def getRotationBetweenTwoVector(a,b):
    if(torch.is_tensor(a)):
        #print('a,b',a,b)
        device=a.device
        bs = a.shape[0]
        a = unit_vector(a)
        b = unit_vector(b)
        a=a.reshape(bs,3)
        G=torch.eye(3,dtype=torch.float32,device=device).reshape(1,3,3).repeat(bs,1,1).reshape(bs,3,3)
        G[:, 0, 0] = torch.sum(a * b, dim=1)
        G[:, 0, 1] = -torch.norm(torch.cross(a,b,dim=1), dim=1)
        G[:, 1, 0] = torch.norm(torch.cross(a,b,dim=1), dim=1)
        G[:, 1, 1] = torch.sum(a * b, dim=1)
        u=a.clone()
        v=b-torch.sum(a*b,dim=1,keepdim=True)*a
        v = unit_vector(v)
        F = torch.zeros([bs,3,3],dtype=torch.float32,device=device)
        F[:,:, 0], F[:,:, 1], F[:,:, 2] = u, v, unit_vector(torch.cross(b, a, dim=1))

        f = F.cpu()
        #print('f',f)
        #print(np.linalg.matrix_rank(f))
        rf = (torch.sum(torch.svd(f)[1]>1e-4,dim=1) == 3)
        if(rf.device!=device):rf=rf.to(device)
        R = torch.eye(3, dtype=torch.float32, device=device).reshape(1, 3, 3).repeat(bs, 1, 1).reshape(bs, 3, 3)
        R[rf] = F[rf] @ G[rf] @ torch.inverse(F[rf])
        return R
    else:
        a=unit_vector(a).copy()
        b=unit_vector(b).copy()
        if (np.linalg.norm(a - b) < 1e-5): return np.eye(3)
        G=np.array([[np.dot(a,b),- np.linalg.norm(np.cross(a,b)),0],[ np.linalg.norm(np.cross(a,b)),np.dot(a,b),0],[0,0,1]])
        u=a.copy()
        v=b-(np.dot(a,b))*a
        v=unit_vector(v)
        F=np.zeros([3,3],dtype=np.float64)
        F[:,0],F[:,1],F[:,2]=u,v,unit_vector(np.cross(b,a))
        R=F@G@np.linalg.inv(F)
        return R



if __name__ == "__main__":
    a=getRotationBetweenTwoVector(torch.tensor([[0,1,0]],dtype=torch.float32),
                                  torch.tensor([[1,0,0]],dtype=torch.float32))
    b=getRotationBetweenTwoVector(torch.tensor([[1,0,0]],dtype=torch.float32),
                                  torch.tensor([[0,1,0]],dtype=torch.float32))
    print(a,b)
