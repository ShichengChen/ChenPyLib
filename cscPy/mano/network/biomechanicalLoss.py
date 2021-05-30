import pickle

import torch.nn as nn

from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.mano.network.Const import boneSpace

class BiomechanicalLayer(nn.Module):
    def __init__(self, fingerPlaneLoss=False,fingerFlexLoss=False,FingerAbductionLoss=False):
        super(BiomechanicalLayer, self).__init__()
        self.fingerPlaneLoss=fingerPlaneLoss
        self.fingerFlexLoss=fingerFlexLoss
        self.FingerAbductionLoss=FingerAbductionLoss

    def forward(self,joint_gt):
        loss=0
        if(self.fingerPlaneLoss):
            pass
            #loss+=self.restrainFingerAngle()
        if(self.fingerFlexLoss):
            dir,joint_gt=self.restrainFingerDirectly(joint_gt)
            print("plane sum dis",torch.sum(dir))
            loss+=self.restrainFlexAngle(joint_gt)
        if(self.FingerAbductionLoss):
            pass
        return loss

    def restrainFingerDirectly(self, joint_gt: torch.Tensor)->(torch.Tensor,torch.Tensor):
        N = joint_gt.shape[0]
        scale = 1  # from meter to milimeter
        joint_gt = joint_gt.reshape(N, 21, 3)
        if (torch.mean(torch.sum(joint_gt[:, 0] - joint_gt[:, 1])) < 1): scale = 1000
        newjoints_gt=joint_gt.clone()
        distance=torch.zeros([N,21],device=joint_gt.device,requires_grad=True,dtype=joint_gt.dtype)
        jidx = [[1, 2, 3, 17], [4, 5, 6, 18], [10, 11, 12, 19], [7, 8, 9, 20], [13, 14, 15, 16]]
        from itertools import combinations
        for finger in jidx:
            subsets = list(combinations(finger, 3))
            vlist = []
            for subset in subsets:
                v0 = joint_gt[:, subset[0]] - joint_gt[:, subset[1]]
                v1 = joint_gt[:, subset[1]] - joint_gt[:, subset[2]]
                vh = torch.cross(v0, v1, dim=1)
                vlist.append(vh.reshape(1, N, 3))
            vh = torch.mean(torch.cat(vlist, dim=0), dim=0).reshape(N, 1, 3)
            subj = joint_gt[:, finger]
            vd = torch.mean(-torch.sum(subj * vh, dim=2), dim=1).reshape(N,1)
            for idx in range(4):
                distance[:,finger[idx]], newjoints_gt[:, finger[idx]] = \
                    projectPoint2Plane(joint_gt[:, finger[idx]], vh, vd)
        return distance*scale, newjoints_gt

    #n0=wrist tmcp imcp
    #n1=wrist imcp mmcp
    #n2=wrist mmcp rmcp
    #n3=wrist rmcp pmcp
    def getPalmNormByIndex(self,joint_gt: torch.Tensor,idx:int) -> torch.Tensor:
        if(idx==-1):return self.getPalmNorm(joint_gt)
        assert 0<=idx<4,"bad index"
        c=[(13,1),(1,4),(4,10),(10,7)]
        return unit_vector(torch.cross(joint_gt[:, c[idx][0]] - joint_gt[:, 0], joint_gt[:, c[idx][1]] - joint_gt[:, c[idx][0]], dim=1))


    def getPalmNorm(self, joint_gt: torch.Tensor,) -> torch.Tensor:
        palmNorm = unit_vector(torch.cross(joint_gt[:, 4] - joint_gt[:, 0], joint_gt[:, 7] - joint_gt[:, 4], dim=1))
        return palmNorm


    def restrainFlexAngle(self, joints: torch.Tensor)->torch.Tensor:
        scale=1#from meter to milimeter
        N = joints.shape[0]
        joints = joints.reshape(N, 21, 3)
        if(torch.mean(torch.sum(joints[:,0]-joints[:,1]))<1):scale=1000
        normidx=[-1,-1,-1,-1,0] #index,middle,ringy,pinky,thumb
        mcpidx=[1,4,10,7,13]
        stdFingerNorms=[]
        for i in range(5):
            palmNorm = self.getPalmNormByIndex(joints,normidx[i]).reshape(N, 3) # palm up
            vecWristMcp = unit_vector(joints[:, mcpidx[i]] - joints[:, 0]).reshape(N, 3) # wirst to mmcp
            stdFingerNorm = unit_vector(torch.cross(vecWristMcp,palmNorm, dim=1)) #direction from pmcp to imcp
            stdFingerNorms.append(stdFingerNorm.clone())
        jidx = [[0, 1, 2, 3, 17], [0, 4, 5, 6, 18], [0, 10, 11, 12, 19], [0, 7, 8, 9, 20],[0, 13,14,15,16]]
        fidces = [[0, 1, 2], [0, 1, 2],[0, 1, 2],[0, 1, 2],[1, 2],]
        loss = 0
        for fidx,finger in enumerate(jidx):
            angleP = torch.tensor([np.pi*3.2 / 4]*3,device=joints.device,dtype=joints.dtype)
            angleN = torch.tensor([3.14 / 2,3.14 / 18 ,3.14 / 4],device=joints.device,dtype=joints.dtype)
            for i in fidces[fidx]:
                a0, a1, a2 = joints[:, finger[i]], joints[:, finger[i + 1]], joints[:, finger[i + 2]].reshape(N, 3)
                a, b = unit_vector(a1 - a0), unit_vector(a2 - a1)
                disb=torch.cdist(a1,a2).reshape(N)
                fingernorm = unit_vector(torch.cross(a, b, dim=1))
                sign = torch.sum(fingernorm * stdFingerNorms[fidx], dim=1).reshape(N)
                angle = torch.acos(torch.sum(a * b, dim=1))
                print(finger[i:i+3],angle,sign,disb)
                angle=torch.abs(angle)
                maskpositive=(sign>=0)
                masknegative=(sign<0)
                if(torch.sum(maskpositive)):
                    print(torch.sum(torch.max(angle[maskpositive]-angleP[i],
                                                  torch.zeros_like(angle[maskpositive]))*disb[maskpositive]*scale))
                    loss += torch.sum(torch.max(angle[maskpositive]-angleP[i],
                                                    torch.zeros_like(angle[maskpositive]))*disb[maskpositive]*scale)
                if(torch.sum(masknegative)):
                    print(torch.sum(torch.max(angle[masknegative]-angleN[i],
                                                  torch.zeros_like(angle[masknegative]))*disb[masknegative]*scale))
                    loss += torch.sum(torch.max(angle[masknegative]-angleN[i],
                                                    torch.zeros_like(angle[masknegative]))*disb[masknegative]*scale)
                # print('loss', (sign < 0) & (torch.abs(angle) > angleThreshold[i - 1]))
        return loss



if __name__ == "__main__":
    from cscPy.mano.network.manolayer import MANO_SMPL
    from cscPy.mano.network.utils import *
    import trimesh
    biolayer=BiomechanicalLayer(fingerFlexLoss=True)

    mano_right = MANO_SMPL('/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl', ncomps=45, oriorder=True,
                           cuda='cpu')
    rootr=torch.tensor(np.random.uniform(-0,0,[3]).astype(np.float32))
    pose=torch.tensor([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                       [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                       [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],dtype=torch.float32)

    pose[0,2]-=1.57
    pose[1,2]+=1.57/2
    pose[2,2]+=1.57/2
    pose[3,2]+=1.57/2
    pose[4,2]-=1.57/2
    # for i in range(12):
    #     pose[i,2]+=1.57/1.5

    vertex_gt, joint_gt = \
                mano_right.get_mano_vertices(rootr.view(1, 1, 3),
                                             pose.view(1, 45),
                                             torch.zeros([10]).view(1, 10),
                                             torch.ones([1]).view(1, 1), torch.tensor([[0, 0, 0]]).view(1, 3),
                                             pose_type='euler', mmcp_center=False)

    print(biolayer(joint_gt))

    v = trimesh.Trimesh(vertices=vertex_gt[0].cpu().numpy(),faces=mano_right.faces)
    scene = trimesh.Scene(v)
    scene.show()


