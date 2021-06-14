# import sys
#
# def solution(s):
#     h = str(hex(int(s))[2:])
#     ans=""
#     for i in h:
#         if(i=='0'):ans+='O'
#         elif(i=='1'):ans+='I'
#         else: ans+=i
#     # print(ans)
#     for i in ans:
#         if i not in ['A','B','C','D','E','F','I','O']:
#             return 'ERROR'
#     return ans
#
# solution('257')

from itertools import groupby



