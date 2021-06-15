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
        self.nocheckScale=True

    def forward(self,joint_gt: torch.Tensor,scale:torch.Tensor):
        N=joint_gt.shape[0]
        bonelen=scale.reshape(N)
        loss,disEudloss=0,0
        if(self.fingerPlaneLoss):
            curloss,disEud,_=self.restrainFingerDirectly(joint_gt,bonelen)
            # print("plane loss",disEud*1000)
            loss+=curloss
            disEudloss+=disEud
        if(self.fingerFlexLoss):
            _,_,joint_gt=self.restrainFingerDirectly(joint_gt,bonelen)
            curloss,disEud=self.restrainFlexAngle(joint_gt,bonelen)
            # print("flex loss",disEud*1000)
            loss+=curloss
            disEudloss += disEud
            #print("flex loss",loss)
        if(self.fingerAbductionLoss):
            _,_,joint_gt = self.restrainFingerDirectly(joint_gt,bonelen)
            curloss,disEud=self.restrainAbductionAngle(joint_gt,bonelen)
            # print("abduction loss",disEud*1000)
            loss +=curloss
            disEudloss += disEud
        return loss,disEudloss


    def restrainFingerDirectly(self, joint_gt: torch.Tensor,bonelen: torch.Tensor,)\
            ->(torch.Tensor,torch.Tensor,torch.Tensor):
        N = joint_gt.shape[0]
        newjoints_gt=joint_gt.clone()
        distance=torch.zeros([N,21,1],device=joint_gt.device,requires_grad=False,dtype=joint_gt.dtype)
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
        cur=torch.mean(distance, dim=(1, 2))
        return torch.mean(cur),torch.mean(cur*bonelen), newjoints_gt

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

    def restrainAbductionAngle(self,joints: torch.Tensor,bonelen: torch.Tensor,)->torch.Tensor:
        N = joints.shape[0]
        normidx = [1, 2, 2, 3]  # index,middle,ringy,pinky,thumb
        mcpidx = [1, 4, 10, 7]
        pipidx = [2, 5, 11, 8]
        loss,disEud=0,0
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
            dis=euDist(mcp,pip).reshape(N)
            flexRatio=euDist(projpip,mcp).reshape(N)/dis
            flexRatio[flexRatio<0.15]=0
            #remove influence of perpendicular fingers
            a=unit_vector(joints[:,mcpidx[i]]-joints[:,0]).reshape(N,3)
            b=unit_vector(projpip-mcp).reshape(N,3)
            sign=torch.sum(torch.cross(a,b,dim=1)*palmNorm,dim=1)
            maskP=sign>=0
            maskN=sign<0
            angle = torch.acos(torch.sum(a * b, dim=1)).reshape(N)
            maskOver90=angle > 3.14 / 2
            angle[maskOver90] = 3.14 - angle[maskOver90]
            if(torch.sum(maskP)):
                cur=torch.max(angle[maskP] - angleP[i],
                          torch.zeros_like(angle[maskP])) * dis[maskP] * flexRatio[maskP]
                loss += torch.mean(cur)
                disEud+=torch.mean(cur*bonelen[maskP])
            if(torch.sum(maskN)):
                cur=torch.max(angle[maskN] - angleN[i],
                          torch.zeros_like(angle[maskN])) * dis[maskN] * flexRatio[maskN]
                loss += torch.mean(cur)
                disEud += torch.mean(cur * bonelen[maskN])
            #print("angle,idx,loss",angle,angle/3.14*180,mcpidx[i],curloss)
            #print('flexRatio,sign,maskOver90',flexRatio,sign,maskOver90)
        return loss/4,disEud/4 #constraint for 4 fingers

    def restrainFlexAngle(self, joints: torch.Tensor,bonelen: torch.Tensor,)->torch.Tensor:
        N = joints.shape[0]
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
        loss,disEud = 0,0
        angleP = torch.tensor([np.pi * 3.2 / 4] * 3, device=joints.device, dtype=joints.dtype)
        angleN = torch.tensor([3.14 / 2, 3.14 / 18, 3.14 / 4], device=joints.device, dtype=joints.dtype)
        angleNthumb = torch.tensor([3.14 / 2, 3.14 / 4, 3.14 / 4], device=joints.device, dtype=joints.dtype)
        for fidx,finger in enumerate(jidx):
            if(fidx==4):angleN=angleNthumb
            for i in fidces[fidx]:
                a0, a1, a2 = joints[:, finger[i]], joints[:, finger[i + 1]], joints[:, finger[i + 2]].reshape(N, 3)
                a, b = unit_vector(a1 - a0), unit_vector(a2 - a1)
                disb=euDist(a1,a2).reshape(N)
                fingernorm = unit_vector(torch.cross(a, b, dim=1))

                sign = torch.sum(fingernorm * stdFingerNorms[fidx], dim=1).reshape(N)
                angle = torch.acos(torch.sum(a * b, dim=1)).reshape(N)
                #print(finger[i:i+3],angle,angle/3.14*180,sign,disb)
                angle=torch.abs(angle)
                #print("sign",sign)
                maskpositive=(sign>=0)
                masknegative=(sign<0)
                if(torch.sum(maskpositive)):
                    cur=torch.max(angle[maskpositive]-angleP[i],
                                                    torch.zeros_like(angle[maskpositive]))*disb[maskpositive]
                    loss+=torch.mean(cur)
                    disEud+=torch.mean(cur*bonelen[maskpositive])
                    #print(finger,i,torch.mean(cur*bonelen[maskpositive])*1000)
                    cur0=torch.max(angle[maskpositive]-angleP[i],torch.zeros_like(angle[maskpositive]))
                    cur1=disb[maskpositive]
                    #print(torch.mean(bonelen[maskpositive])*cur1*1000,cur0,angle,"pos")
                if(torch.sum(masknegative)):
                    #print(torch.mean(torch.max(angle[masknegative]-angleN[i],
                    #                              torch.zeros_like(angle[masknegative]))*disb[masknegative]))
                    cur=torch.max(angle[masknegative]-angleN[i],
                                                    torch.zeros_like(angle[masknegative]))*disb[masknegative]

                    loss += torch.mean(cur)
                    disEud+=torch.mean(cur*bonelen[masknegative])
                    #print(finger, i, torch.mean(cur * bonelen[masknegative])*1000)
                    cur0 = torch.max(angle[masknegative] - angleP[i], torch.zeros_like(angle[masknegative]))
                    cur1 = disb[masknegative]
                    #print(torch.mean(bonelen[masknegative])*cur1*1000,cur0,angle,"neg")
                #print('disEud',disEud*1000)
                # print('loss', (sign < 0) & (torch.abs(angle) > angleThreshold[i - 1]))
        return loss/15,disEud/15#15 joints constraints



if __name__ == "__main__":
    from cscPy.mano.network.manolayer import MANO_SMPL
    from cscPy.mano.network.utils import *
    import trimesh
    biolayer=BiomechanicalLayer(fingerFlexLoss=True,fingerAbductionLoss=True)

    mano_right = MANO_SMPL('/home/csc/MANO-hand-model-toolkit/mano/models/MANO_RIGHT.pkl', ncomps=45, oriorder=True,
                           device='cpu')
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


