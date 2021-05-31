import pickle

import torch.nn as nn

from cscPy.mano.network.utils import *
from cscPy.mano.network.utilsSmallFunctions import *
from cscPy.mano.network.Const import boneSpace

class BiomechanicalLayer(nn.Module):
    def __init__(self, fingerPlaneLoss=False,fingerFlexLoss=False,fingerAbductionLoss=False):
        super(BiomechanicalLayer, self).__init__()
        ##only works for right hand!!!
        self.fingerPlaneLoss=fingerPlaneLoss
        self.fingerFlexLoss=fingerFlexLoss
        self.fingerAbductionLoss=fingerAbductionLoss

    def forward(self,joint_gt):
        loss=0
        if(self.fingerPlaneLoss):
            pass
            #loss+=self.restrainFingerAngle()
        if(self.fingerFlexLoss):
            dir,joint_gt=self.restrainFingerDirectly(joint_gt)
            print("plane sum dis",torch.sum(dir))
            loss+=self.restrainFlexAngle(joint_gt)
            print("flex loss",loss)
        if(self.fingerAbductionLoss):
            dir, joint_gt = self.restrainFingerDirectly(joint_gt)
            curloss=self.restrainAbductionAngle(joint_gt)
            print("abduction loss",curloss)
            loss +=curloss
        return loss

    def jointCheck(self, joint_gt: torch.Tensor)->int:
        N = joint_gt.shape[0]
        scale = 1  # from meter to milimeter
        joint_gt = joint_gt.reshape(N, 21, 3)
        if (torch.mean(torch.sum(joint_gt[:, 0] - joint_gt[:, 1])) < 1): scale = 1000
        return scale

    def restrainFingerDirectly(self, joint_gt: torch.Tensor)->(torch.Tensor,torch.Tensor):
        N = joint_gt.shape[0]
        scale=self.jointCheck(joint_gt)
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
            subj = joint_gt[:, finger].reshape(N,4,3)
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

    def restrainAbductionAngle(self,joints: torch.Tensor)->torch.Tensor:
        N = joints.shape[0]
        scale=self.jointCheck(joints)
        normidx = [1, 2, 2, 3]  # index,middle,ringy,pinky,thumb
        mcpidx = [1, 4, 10, 7]
        pipidx = [2, 5, 11, 8]
        loss=0
        r=6
        angleP = torch.tensor([np.pi/r+0.1890,np.pi/r+0.1331,np.pi/r-0.1491,np.pi/r+0.0347], device=joints.device, dtype=joints.dtype)
        angleN = torch.tensor([np.pi/r-0.1890,np.pi/r-0.1331,np.pi/r+0.1491,np.pi/r-0.0347], device=joints.device, dtype=joints.dtype)
        for i in range(4):
            palmNorm = self.getPalmNormByIndex(joints, normidx[i]).reshape(N, 3)  # palm up
            vh = palmNorm.reshape(N, 3)
            mcp = joints[:, mcpidx[i]].reshape(N,3)
            vd = -torch.sum(mcp * vh, dim=1).reshape(N, 1)
            pip = joints[:, pipidx[i]].reshape(N,3)
            projpip=projectPoint2Plane(pip,vh,vd)[1].reshape(N,3)
            dis=torch.cdist(mcp,pip).reshape(N,1)
            flexRatio=torch.cdist(projpip,mcp)/dis
            flexRatio[flexRatio<0.15]=0
            #remove influence of perpendicular fingers
            a=unit_vector(joints[:,mcpidx[i]]-joints[:,0]).reshape(N,3)
            b=unit_vector(projpip-mcp).reshape(N,3)
            sign=torch.sum(torch.cross(a,b,dim=1)*palmNorm,dim=1)
            maskP=sign>=0
            maskN=sign<0
            angle = torch.acos(torch.sum(a * b, dim=1))
            maskOver90=angle > 3.14 / 2
            angle[maskOver90] = 3.14 - angle[maskOver90]
            if(torch.sum(maskP)):
                curloss=torch.mean(torch.max(angle[maskP]-angleP[i],
                                                  torch.zeros_like(angle[maskP]))*dis[maskP]*scale*flexRatio[maskP])
            if(torch.sum(maskN)):
                curloss = torch.mean(torch.max(angle[maskN] - angleN[i],
                                               torch.zeros_like(angle[maskN])) * dis[maskN] * scale * flexRatio[maskN])
            loss+=curloss
            print("angle,idx,loss",angle,angle/3.14*180,mcpidx[i],curloss)
            print('flexRatio,sign,maskOver90',flexRatio,sign,maskOver90)
        return loss

    def restrainFlexAngle(self, joints: torch.Tensor)->torch.Tensor:
        N = joints.shape[0]
        scale=self.jointCheck(joints)
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
            angleNthumb = torch.tensor([3.14 / 2,3.14 / 4 ,3.14 / 4],device=joints.device,dtype=joints.dtype)
            if(fidx==4):angleN=angleNthumb
            for i in fidces[fidx]:
                a0, a1, a2 = joints[:, finger[i]], joints[:, finger[i + 1]], joints[:, finger[i + 2]].reshape(N, 3)
                a, b = unit_vector(a1 - a0), unit_vector(a2 - a1)
                disb=torch.cdist(a1,a2).reshape(N)
                fingernorm = unit_vector(torch.cross(a, b, dim=1))
                sign = torch.sum(fingernorm * stdFingerNorms[fidx], dim=1).reshape(N)
                angle = torch.acos(torch.sum(a * b, dim=1)).reshape(N)
                print(finger[i:i+3],angle,angle/3.14*180,sign,disb)
                angle=torch.abs(angle)

                maskpositive=(sign>=0)
                masknegative=(sign<0)
                if(torch.sum(maskpositive)):
                    print(torch.mean(torch.max(angle[maskpositive]-angleP[i],
                                                  torch.zeros_like(angle[maskpositive]))*disb[maskpositive]*scale))
                    loss += torch.mean(torch.max(angle[maskpositive]-angleP[i],
                                                    torch.zeros_like(angle[maskpositive]))*disb[maskpositive]*scale)
                if(torch.sum(masknegative)):
                    print(torch.mean(torch.max(angle[masknegative]-angleN[i],
                                                  torch.zeros_like(angle[masknegative]))*disb[masknegative]*scale))
                    loss += torch.mean(torch.max(angle[masknegative]-angleN[i],
                                                    torch.zeros_like(angle[masknegative]))*disb[masknegative]*scale)
                # print('loss', (sign < 0) & (torch.abs(angle) > angleThreshold[i - 1]))
        return loss



if __name__ == "__main__":
    from cscPy.mano.network.manolayer import MANO_SMPL
    from cscPy.mano.network.utils import *
    import trimesh
    biolayer=BiomechanicalLayer(fingerFlexLoss=True,fingerAbductionLoss=True)

    mano_right = MANO_SMPL('/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl', ncomps=45, oriorder=True,
                           cuda='cpu')
    rootr=torch.tensor(np.random.uniform(-0,0,[3]).astype(np.float32))
    pose=torch.tensor([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                       [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],
                       [0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],dtype=torch.float32)

    # pose[0,2]-=1.57
    # pose[1,2]+=1.57/2
    # pose[2,2]-=1.57/2
    # pose[3,2]-=1.57
    # pose[4,2]-=1.57/2
    # for i in range(12):
    #     pose[i,2]+=1.57/1.5
    # pose[0, 1]+=np.pi/2/3
    # pose[3, 1]+=np.pi/2/3
    # pose[6, 1]+=np.pi/2/3
    # pose[9, 1]+=np.pi/2/3
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


