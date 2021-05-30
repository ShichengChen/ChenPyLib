import sys
input = sys.stdin.readline

############ ---- Input Functions ---- ############
def inp():
    return(int(input()))
def inlt():
    return(list(map(int,input().split())))
def insr():
    s = input()
    return(list(s[:len(s) - 1]))
def invr():
    return(map(int,input().split()))

class Infix:
    def __init__(self, function):
        self.function = function
    def __ror__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __or__(self, other):
        return self.function(other)
    def __rlshift__(self, other):
        return Infix(lambda x, self=self, other=other: self.function(other, x))
    def __rshift__(self, other):
        return self.function(other)
    def __call__(self, value1, value2):
        return self.function(value1, value2)

# Examples
MOD=(1000000007)
# simple multiplication
def fun(x,y):
    #return (x**2+MOD*x+y**3+y*998244353)
    #print()
    h1=hash(str(y)+"508")
    if h1 < 0:
        h1 += sys.maxsize
    return int(str(hash(str(x)+'101'))+str(h1))
#x=Infix(lambda x,y: x*y)
op=Infix(fun)
# print(eval("1|op|2"))
# print(eval("2|op|1"))
# print(eval("11|op|2"))
# print(eval("1|op|12"))
def solve():
    n=inp()
    arr=[eval(input()[:-1].replace('#', '|op|')) for i in range(n)]
    ma={}
    ans=[]
    cnt=1
    for i in range(len(arr)):
        if(arr[i] in ma):ans.append(ma[arr[i]])
        else:
            ans.append(cnt)
            ma[arr[i]]=cnt
            cnt+=1
    print(" ".join(str(x) for x in ans))
if __name__ == "__main__":
    t=inp()
    for i in range(t):
        print("Case #"+str(i+1)+": ", end ="")
        solve()